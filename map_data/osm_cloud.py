#!/usr/bin/env python3
"""
ROS2 node for publishing OSM map data as a point cloud.

This module provides the OSMCloud class which converts parsed MapData
into ROS2 PointCloud2 and MarkerArray messages for visualization.
"""

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from numpy.lib.recfunctions import unstructured_to_structured
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from ros2_numpy import msgify, numpify
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformException,
)
from visualization_msgs.msg import Marker, MarkerArray

import map_data.map_data as md
from map_data.annotations import (
    NO_ANNOTATIONS,
    annotation_path_for,
    load_mapdata_with_annotations,
)
from map_data.traversability import resolve_traversability_path
from map_data.utils.geodesy import apply_transform, ecef_to_latlon, utm_to_local_via_ecef
from map_data.utils.way import NON_ROUTABLE_HIGHWAY_VALUES

CLOUD_COLS = 4
TOLERANCE = 1e-3
TRANSFORM_MODES = ("tf", "auto", "geodetic")
# Smallest plausible norm of an ECEF position [m]. Every point on Earth is ~6.37e6 m from
# the ECEF origin, so a much smaller one means the earth_frame -> local_frame TF carries no
# real origin yet: the Fixposition unit publishes FP_ECEF -> FP_ENU0 as all zeros until it
# has a fusion fix, which puts FP_ENU0 at the centre of the Earth. Placing the map with
# that transform shrinks the whole map to a metre-wide box and the grid comes out empty.
MIN_ECEF_NORM = 1.0e6
# How far the placement transform has to move before the map is rebuilt. /tf_static is
# re-broadcast periodically on Helhest, so the common case is the very same numbers again.
PLACEMENT_EPS = 1e-6
# highway_types value -> MapData.get_ways() key, as in route_planner's graph planner.
HIGHWAY_TYPE_KEYS = {"footway": "footways", "road": "roads"}


class OSMCloud(Node):
    """ROS2 node that publishes OSM data as point clouds and markers."""

    def __init__(self) -> None:
        super().__init__("osm_cloud")
        self.utm_frame: str = self.declare_parameter("utm_frame", "utm").value
        self.local_frame: str = self.declare_parameter("local_frame", "map").value
        self.earth_frame: str = self.declare_parameter("earth_frame", "FP_ECEF").value
        # How UTM map data is placed in ``local_frame``:
        #   "tf"       - look up utm_frame -> local_frame in TF (legacy default)
        #   "auto"     - local frame at the map centre; publishes utm_frame -> local_frame
        #   "geodetic" - UTM -> lat/lon -> ECEF, then the earth_frame -> local_frame TF
        #                (exact for GNSS/INS stacks such as Fixposition: FP_ECEF -> FP_ENU0)
        self.transform_mode: str = self.declare_parameter("transform_mode", "tf").value
        self.mapdata_file: str | None = self.declare_parameter(
            "mapdata_file",
            rclpy.Parameter.Type.STRING,
        ).value
        self.gpx_file: str | None = self.declare_parameter(
            "gpx_file",
            rclpy.Parameter.Type.STRING,
        ).value
        # Annotation store merged into mapdata_file, with route_planner's semantics:
        # "auto" = <map>.annotations.json next to the map, "none" = the unedited map,
        # or a path to a store file. The planner and this node must see the same map,
        # or a junction on a retagged/deleted way publishes a ring nothing routes over.
        self.annotations: str = self.declare_parameter("annotations", "auto").value
        # highway= values dropped from the map: stairs are footways in OSM, but the
        # planner does not route over them, so they get no rings and no road cost.
        self.exclude_highway: list[str] = list(
            self.declare_parameter("exclude_highway", sorted(NON_ROUTABLE_HIGHWAY_VALUES)).value
        )
        # Tag rules (stairs, grass, bridges, ...) deciding which ways the robot may drive
        # on, as in route_planner: "" = the package's config/traversability.yaml. The two
        # nodes must use the same file, or the rings describe a network nothing routes on.
        self.traversability_file: str = self.declare_parameter("traversability_file", "").value
        # Way types the grid is drawn from: footway and/or road, as route_planner's
        # highway_types. Match the planner's, or the grid shows ways it does not route on.
        self.highway_types: list[str] = self._valid_highway_types(
            self.declare_parameter("highway_types", ["footway"]).value
        )
        self.save_mapdata: bool = self.declare_parameter("save_mapdata", False).value
        self.max_path_dist: float = self.declare_parameter("max_path_dist", 1.0).value
        self.neighbor_cost: str = self.declare_parameter("neighbor_cost", "linear").value
        self.grid_res: float = self.declare_parameter("grid_res", 0.25).value
        self.grid_max: list[float] = self.declare_parameter("grid_max", [0.0, 0.0]).value
        self.grid_min: list[float] = self.declare_parameter("grid_min", [0.0, 0.0]).value
        # True once the bounds are auto-calculated from the map's query bbox. Ways that
        # cross the bbox are downloaded whole, so the network (and the crossroads on it)
        # reaches past those bounds: only *explicit* bounds may clip the intersections.
        self.grid_bounds_auto: bool = False
        self.publish_intersections: bool = self.declare_parameter(
            "publish_intersections",
            False,
        ).value
        # Publishers are latched (transient local): late subscribers receive the last
        # message, so a periodic re-publish of the (large) grid cloud is not needed.
        # Set > 0 to additionally re-publish every N seconds.
        self.republish_period: float = self.declare_parameter("republish_period", 0.0).value

        # Topic parameters
        self.grid_topic: str = self.declare_parameter("grid_topic", "grid").value
        self.intersections_topic: str = self.declare_parameter(
            "intersections_topic",
            "intersections",
        ).value
        self.intersection_markers_topic: str = self.declare_parameter(
            "intersection_markers_topic",
            "intersection_markers",
        ).value

        # Register parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_grid = self.create_publisher(PointCloud2, self.grid_topic, qos)

        if self.publish_intersections:
            self.pub_poses = self.create_publisher(PoseArray, self.intersections_topic, qos)
            self.pub_markers = self.create_publisher(
                MarkerArray,
                self.intersection_markers_topic,
                qos,
            )

        self.tf = Buffer()
        self.tf_sub = None
        self.tf_static_pub = StaticTransformBroadcaster(self)
        # True once the map has been built and published once, so that _tf_static_cb()
        # rebuilds it rather than racing the start-up lookup that drives the callback.
        self.initialized = False
        # True when the grid came out empty although the map has ways, i.e. the map is
        # misplaced. Nothing is published then: a latched empty cloud would hide the map
        # from every late subscriber until the node was restarted.
        self.cloud_degenerate = False

        self.utm_to_local: np.ndarray | None = None
        self.ecef_to_local: np.ndarray | None = None
        self.poses: PoseArray | None = None
        self.markers: MarkerArray | None = None

        if self.mapdata_file:
            self.map_data = self.load_map_data(self.mapdata_file)
        elif self.gpx_file:
            self.map_data = md.MapData(self.gpx_file)
            self.map_data.run_all(save=self.save_mapdata)
        else:
            self.get_logger().error("No map data or gpx file provided")
            sys.exit(1)
        self.get_logger().info(str(self.map_data))

        if self.transform_mode not in TRANSFORM_MODES:
            self.get_logger().warning(
                f"Unknown transform_mode '{self.transform_mode}', falling back to 'tf'"
            )
            self.transform_mode = "tf"

        # Only the static placement transform is looked up here, so one /tf_static
        # subscription is enough: a full TransformListener would also digest the /tf
        # firehose (hundreds of Hz on Helhest). It is kept for the node's life, so that a
        # placement transform published or corrected later rebuilds and re-publishes the
        # map instead of needing the node restarted -- the ENU0 origin only exists once
        # the GNSS/INS unit has a fusion fix, and moves whenever its driver restarts.
        if self.transform_mode != "auto":
            static_qos = QoSProfile(depth=100, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            self.tf_sub = self.create_subscription(
                TFMessage,
                "/tf_static",
                self._tf_static_cb,
                static_qos,
            )

        if self.transform_mode == "geodetic":
            self.get_ecef_to_local()
        elif self.transform_mode == "auto":
            self.get_logger().info("Auto-calculating UTM to local transform from map center")
            center_x = (self.map_data.min_x + self.map_data.max_x) / 2
            center_y = (self.map_data.min_y + self.map_data.max_y) / 2
            self.utm_to_local = np.eye(4)
            self.utm_to_local[0, 3] = -center_x
            self.utm_to_local[1, 3] = -center_y

            # Publish the static transform (utm -> local_utm)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.utm_frame
            t.child_frame_id = self.local_frame
            t.transform.translation.x = center_x
            t.transform.translation.y = center_y
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_static_pub.sendTransform(t)
        else:
            self.get_utm_to_local()

        if self.transform_mode == "geodetic":
            self.get_logger().info(
                f"Using geodetic placement via {self.earth_frame} -> {self.local_frame}: "
                f"{self.ecef_to_local}"
            )
        else:
            self.get_logger().info(f"Using UTM to local transform: {self.utm_to_local}")

        if all(v == 0.0 for v in self.grid_min) and all(v == 0.0 for v in self.grid_max):
            self.grid_bounds_auto = True
            self._auto_grid_bounds()

        self.grid_cloud: PointCloud2 = self.get_cloud()
        if self.publish_intersections:
            self.poses, self.markers = self.get_intersections()

        self.publish_cb()
        if self.republish_period > 0:
            self.create_timer(self.republish_period, self.publish_cb)
        # From here on a /tf_static message rebuilds the map, instead of only filling the
        # buffer for the start-up lookup that drove the callback until now.
        self.initialized = True
        self.get_logger().info("Initialized OSM cloud")

    def _auto_grid_bounds(self) -> None:
        """Set ``grid_min`` / ``grid_max`` from the map's bounding box in ``local_frame``."""
        self.get_logger().info("Auto-calculating grid bounds from map data")
        # Transform all four corners of the UTM bounding box to the local frame
        # (the local frame may be rotated relative to UTM).
        xs = (self.map_data.min_x, self.map_data.max_x)
        ys = (self.map_data.min_y, self.map_data.max_y)
        corners = {
            i: np.array([x, y, 0.0]).reshape(3, 1)
            for i, (x, y) in enumerate((x, y) for x in xs for y in ys)
        }
        # The transform is None only if its lookup was interrupted by rclpy shutdown
        # mid-startup; skip the auto-calc (get_cloud() raises a clear error in that case).
        if not self._transform_ready():
            self.get_logger().error(
                "Map placement transform unavailable; skipping grid-bounds auto-calc"
            )
            return
        bounds_arr = np.array([p.ravel() for p in self._to_local(corners).values()])
        self.grid_min = [np.min(bounds_arr[:, 0]), np.min(bounds_arr[:, 1])]
        self.grid_max = [np.max(bounds_arr[:, 0]), np.max(bounds_arr[:, 1])]
        self.get_logger().info(f"Calculated grid bounds: min={self.grid_min}, max={self.grid_max}")

    def load_map_data(self, path: str) -> "md.MapData":
        """
        Load ``path`` with the same merge the planner uses.

        The annotation store selected by the ``annotations`` parameter is merged
        in, and the ways the ``traversability_file`` rules (and ``exclude_highway``)
        refuse are dropped, so the rings and the cost grid published here describe
        the network ``route_planner`` routes on.
        """
        ann = None if self.annotations in ("", "auto") else self.annotations
        map_data, store = load_mapdata_with_annotations(
            path,
            ann,
            exclude_highway=self.exclude_highway,
            traversability=self.traversability_file or None,
        )
        if ann == NO_ANNOTATIONS:
            store_name = "none"
        else:
            ann_path = Path(ann).expanduser() if ann else annotation_path_for(path)
            store_name = ann_path.name if ann_path.is_file() else "no store"
        removed = getattr(map_data, "traversability_removed", {})
        trav_path = resolve_traversability_path(self.traversability_file)
        self.get_logger().info(
            f"loaded {Path(path).name}: {len(map_data.footways_list)} footways, "
            f"{len(map_data.roads_list)} roads, {len(map_data.crossroads_list)} crossroads; "
            f"annotations={store_name} ({len(store.get('deleted_ways', []))} deleted ways, "
            f"{len(store.get('annotations', []))} drawn), "
            f"excluded highway={','.join(self.exclude_highway) or 'none'}, "
            f"traversability={trav_path.name if trav_path else 'none'} ("
            + (
                ", ".join(f"{reason} {count}" for reason, count in removed.items())
                or "nothing removed"
            )
            + ")"
        )
        return map_data

    def _valid_highway_types(self, values: list[str]) -> list[str]:
        """Return *values* without unknown way types, warning about each one dropped."""
        values = list(values)
        unknown = [v for v in values if v not in HIGHWAY_TYPE_KEYS]
        if unknown:
            self.get_logger().warning(
                f"Ignoring unknown highway_types {unknown}; expected any of "
                f"{sorted(HIGHWAY_TYPE_KEYS)}"
            )
        return [v for v in values if v in HIGHWAY_TYPE_KEYS]

    def parameter_callback(self, params: list[rclpy.Parameter]) -> SetParametersResult:
        rebuild_cloud = False
        rebuild_intersections = False
        for param in params:
            if param.name == "max_path_dist":
                self.max_path_dist = param.value
                rebuild_cloud = True
            elif param.name == "neighbor_cost":
                self.neighbor_cost = param.value
                rebuild_cloud = True
            elif param.name == "highway_types":
                self.highway_types = self._valid_highway_types(param.value)
                rebuild_cloud = True
            elif param.name == "grid_res":
                self.grid_res = param.value
                rebuild_cloud = True
            elif param.name == "grid_max":
                self.grid_max = param.value
                self.grid_bounds_auto = False
                rebuild_cloud = True
                rebuild_intersections = True
            elif param.name == "grid_min":
                self.grid_min = param.value
                self.grid_bounds_auto = False
                rebuild_cloud = True
                rebuild_intersections = True
            elif param.name == "publish_intersections":
                self.publish_intersections = param.value
                if self.publish_intersections and not hasattr(self, "pub_poses"):
                    qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
                    self.pub_poses = self.create_publisher(PoseArray, self.intersections_topic, qos)
                    self.pub_markers = self.create_publisher(
                        MarkerArray,
                        self.intersection_markers_topic,
                        qos,
                    )
                rebuild_intersections = True

        if rebuild_cloud:
            self.get_logger().info("Rebuilding grid cloud due to parameter change")
            try:
                self.grid_cloud = self.get_cloud()
            except (ValueError, TypeError, RuntimeError) as e:
                self.get_logger().error(f"Failed to rebuild grid cloud: {e}")
                return SetParametersResult(successful=False, reason=str(e))

        if rebuild_intersections and self.publish_intersections:
            self.get_logger().info("Rebuilding intersections due to parameter change")
            try:
                self.poses, self.markers = self.get_intersections()
            except (ValueError, TypeError, RuntimeError) as e:
                self.get_logger().error(f"Failed to rebuild intersections: {e}")
                return SetParametersResult(successful=False, reason=str(e))

        if rebuild_cloud or rebuild_intersections:
            self.publish_cb()

        return SetParametersResult(successful=True)

    def publish_cb(self) -> None:
        """
        Timer callback to publish the grid cloud and intersections.
        """
        # The crossroads come from the same placement as the grid, so a degenerate grid
        # means their rings would be just as misplaced: hold everything back rather than
        # latch it onto every late subscriber.
        if self.cloud_degenerate:
            self.get_logger().error(
                "Not publishing an empty grid and its intersections; "
                "waiting for a usable map placement",
                throttle_duration_sec=30.0,
            )
            return

        now = self.get_clock().now().to_msg()
        self.grid_cloud.header.stamp = now
        self.pub_grid.publish(self.grid_cloud)

        if self.publish_intersections and self.poses is not None and self.markers is not None:
            self.poses.header.stamp = now
            for marker in self.markers.markers:
                marker.header.stamp = now
            self.pub_poses.publish(self.poses)
            self.pub_markers.publish(self.markers)

        self.get_logger().info("Published OSM data", throttle_duration_sec=60.0)

    def _tf_static_cb(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            self.tf.set_transform_static(t, "osm_cloud")
        # Before the map is published, _poll_tf() is what drives this callback and it
        # picks the new transform up on its next lookup; only one arriving afterwards
        # has to rebuild what was already published.
        if self.initialized:
            self._replace_placement()

    def _replace_placement(self) -> None:
        """
        Rebuild and re-publish the map if the placement transform has changed.

        ``/tf_static`` is re-broadcast periodically on Helhest, so nearly every call sees
        the same transform again and returns immediately. A genuinely different one means
        the published map is now in the wrong place: the ENU0 origin appears only with the
        GNSS/INS unit's first fusion fix, and moves whenever its driver restarts.
        """
        source = self.earth_frame if self.transform_mode == "geodetic" else self.utm_frame
        try:
            tf_msg = self.tf.lookup_transform(self.local_frame, source, rclpy.time.Time())
        except (TransformException, RuntimeError, TypeError, ValueError):
            return
        matrix = numpify(tf_msg.transform)
        reason = self._placement_error(matrix)
        if reason:
            self.get_logger().warning(
                f"Ignoring {source} -> {self.local_frame} transform: {reason}",
                throttle_duration_sec=10.0,
            )
            return
        current = self.ecef_to_local if self.transform_mode == "geodetic" else self.utm_to_local
        if current is not None and np.allclose(matrix, current, rtol=0.0, atol=PLACEMENT_EPS):
            return
        self.get_logger().info(f"{source} -> {self.local_frame} changed; rebuilding the map")
        if self.transform_mode == "geodetic":
            self.ecef_to_local = matrix
            self._log_local_origin()
        else:
            self.utm_to_local = matrix
            self.get_logger().info(f"Using UTM to local transform: {self.utm_to_local}")
        if self.grid_bounds_auto:
            self._auto_grid_bounds()
        try:
            self.grid_cloud = self.get_cloud()
            if self.publish_intersections:
                self.poses, self.markers = self.get_intersections()
        except (ValueError, TypeError, RuntimeError) as e:
            self.get_logger().error(f"Failed to rebuild the map: {e}")
            return
        self.publish_cb()

    def _placement_error(self, matrix: np.ndarray) -> str:
        """
        Return why *matrix* cannot place the map yet, or ``""`` when it can.

        Only the geodetic mode can tell: ``earth_frame`` is ECEF, so ``local_frame``'s
        origin has to sit on the Earth's surface. A Fixposition unit without a fusion fix
        publishes ``FP_ECEF -> FP_ENU0`` as all zeros, which puts that origin at the centre
        of the Earth; ``utm_to_local_via_ecef`` then works at an altitude of -6378 km and
        collapses the whole map into a metre-wide box, so the grid comes out empty.
        """
        if self.transform_mode != "geodetic":
            return ""
        try:
            origin = np.linalg.inv(matrix)[:3, 3]
        except np.linalg.LinAlgError:
            return "the transform is singular"
        norm = float(np.linalg.norm(origin))
        if norm < MIN_ECEF_NORM:
            return (
                f"{self.local_frame} origin is only {norm:.0f} m from the centre of the "
                f"Earth, so {self.earth_frame} does not carry a position yet"
            )
        return ""

    def _log_local_origin(self) -> None:
        """Log the ``local_frame`` origin of the geodetic placement as lat/lon/alt."""
        origin = np.linalg.inv(self.ecef_to_local)[:3, 3]  # type: ignore[arg-type]
        lat, lon, alt = ecef_to_latlon(*origin)
        self.get_logger().info(
            f"Got {self.earth_frame} -> {self.local_frame} transform; "
            f"{self.local_frame} origin at lat={lat:.7f} lon={lon:.7f} alt={alt:.1f}"
        )

    def _poll_tf(
        self,
        target: str,
        source: str,
        validate: Callable[[np.ndarray], str] | None = None,
    ) -> np.ndarray | None:
        """
        Block until a usable ``source -> target`` TF transform is available.

        While rclpy is not shutdown, retry every second until successful. *validate*
        returns why a transform cannot be used yet (``""`` when it can), so a placeholder
        one is waited out rather than latched for the node's life.
        Returns ``None`` only if rclpy is shut down while waiting.
        """
        while rclpy.ok():
            try:
                # Zero timeout: the spin_once() below is what delivers /tf_static,
                # so a blocking wait here would only sleep.
                tf_msg = self.tf.lookup_transform(target, source, rclpy.time.Time())
                matrix = numpify(tf_msg.transform)
                reason = "" if validate is None else validate(matrix)
                if not reason:
                    return matrix
            except (TransformException, RuntimeError, TypeError, ValueError) as e:
                reason = str(e)
            self.get_logger().warning(
                f"Failed to get {source} -> {target} transform: {reason}",
                throttle_duration_sec=10.0,
            )
            rclpy.spin_once(self, timeout_sec=1.0)
        return None

    def get_utm_to_local(self) -> None:
        """Poll for the UTM to local coordinate transform."""
        self.utm_to_local = self._poll_tf(self.local_frame, self.utm_frame)
        self.get_logger().info(f"Got UTM to local transform: {self.utm_to_local}")

    def get_ecef_to_local(self) -> None:
        """
        Poll for the ``earth_frame -> local_frame`` transform (ECEF to local ENU).

        The result is stored as a 4x4 matrix mapping ECEF points into ``local_frame``.
        """
        self.ecef_to_local = self._poll_tf(
            self.local_frame,
            self.earth_frame,
            self._placement_error,
        )
        if self.ecef_to_local is not None:
            self._log_local_origin()

    def _transform_ready(self) -> bool:
        if self.transform_mode == "geodetic":
            return self.ecef_to_local is not None
        return self.utm_to_local is not None

    def _to_local(self, points: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """
        Place UTM points (id -> (3, 1) array) in ``local_frame`` with z set to 0.
        """
        if not self._transform_ready():
            raise RuntimeError(
                "Map placement transform unavailable; the node was shutting down "
                "before it could be resolved."
            )
        if self.transform_mode != "geodetic":
            return transform_points(points, self.utm_to_local, 0.0)  # type: ignore[arg-type]
        if not points:
            return {}
        ids = list(points)
        arr = np.array([points[i].ravel()[:2] for i in ids], dtype=float)
        local = utm_to_local_via_ecef(
            arr[:, 0],
            arr[:, 1],
            self.map_data.zone_number,
            self.map_data.zone_letter,
            self.ecef_to_local,  # type: ignore[arg-type]
        )
        local[:, 2] = 0.0
        return {pid: local[k].reshape(3, 1) for k, pid in enumerate(ids)}

    def get_cloud(self) -> PointCloud2:
        """
        Return a point cloud from the map data.

        Returns
        -------
        cloud : sensor_msgs.PointCloud2
            Created point cloud.

        """
        points = self._to_local(self.map_data.get_points())
        grid = create_grid(tuple(self.grid_min), tuple(self.grid_max), self.grid_res)
        waypoints = split_ways_to_points(
            points, self.map_data.get_ways(), self.grid_res, self.highway_types
        )

        # The nearest-neighbour search is 2-D (a third all-zero column only slows the
        # kd-tree down); the z=0 column create_cloud expects is put back afterwards.
        grid = np.insert(points_near_ref(grid, waypoints, self.max_path_dist), 2, 0.0, axis=1)
        if self.neighbor_cost == "linear":
            pass
        elif self.neighbor_cost == "quadratic":
            grid[:, 3] = grid[:, 3] ** 2
        else:
            if self.neighbor_cost != "zero" and self.neighbor_cost != "linear":
                self.get_logger().warning(f"Unknown neighbor cost: {self.neighbor_cost}")
            grid[:, 3] = 0.0
        # An empty grid over a map that has ways means the bounds and the way points do
        # not overlap, i.e. the map is misplaced -- publishing it would latch an empty
        # cloud onto every late subscriber, so publish_cb() holds it back instead.
        self.cloud_degenerate = grid.shape[0] == 0 and waypoints.shape[0] > 0
        if self.cloud_degenerate:
            self.get_logger().error(
                f"Grid is empty although the map has {waypoints.shape[0]} way points: "
                f"bounds min={self.grid_min} max={self.grid_max} do not cover it. "
                f"Check the {self.local_frame} placement transform."
            )
        cloud = create_cloud(grid)
        self.get_logger().info(str(grid.shape))
        cloud.header.frame_id = self.local_frame
        cloud.header.stamp = self.get_clock().now().to_msg()

        return cloud

    def get_intersections(self) -> tuple[PoseArray, MarkerArray]:
        """
        Create PoseArray and MarkerArray from intersections.
        """
        ways = self.map_data.get_ways()
        crossroads = ways.get("crossroads", [])

        points_to_transform = {}
        for way in crossroads:
            # way.line is a buffered Point (Polygon); crossroad Ways are always built
            # with a concrete line in MapData.parse_intersections, never None.
            centroid = way.line.centroid  # type: ignore[union-attr]
            points_to_transform[way.id] = np.array([centroid.x, centroid.y, 0.0]).reshape(3, 1)

        transformed_points = self._to_local(points_to_transform)

        pose_array = PoseArray()
        pose_array.header.frame_id = self.local_frame

        marker_array = MarkerArray()

        marker_id = 0
        for point in transformed_points.values():
            p = point.ravel()

            # Spatial filtering based on local frame coordinates. Skipped for
            # auto-calculated bounds: they come from the map's query bbox, while the
            # footway network the route is planned on extends past it, so clipping
            # here silently hides the crossroads the road follower switches at.
            if not self.grid_bounds_auto and not (
                self.grid_min[0] <= p[0] <= self.grid_max[0]
                and self.grid_min[1] <= p[1] <= self.grid_max[1]
            ):
                continue

            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = 0.0
            pose_array.poses.append(pose)

            marker = Marker()
            marker.header.frame_id = self.local_frame
            marker.ns = "intersections"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(p[0])
            marker.pose.position.y = float(p[1])
            marker.pose.position.z = 0.0
            marker.scale.x = 2.0
            marker.scale.y = 2.0
            marker.scale.z = 2.0
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)

        return pose_array, marker_array


def create_grid(
    low: tuple[float, ...],
    high: tuple[float, ...],
    cell_size: float = 0.25,
) -> np.ndarray:
    """
    Create a grid of points.

    Parameters
    ----------
    low : tuple
        Lower bounds of the grid.
    high : tuple
        Upper bounds of the grid.
    cell_size : float
        Size of the cell.

    Returns
    -------
    grid : np.array
        Grid of points.

    """
    low_arr = np.round(low)
    high_arr = np.round(high)
    xs = np.arange(int(low_arr[0]), int(high_arr[0]), cell_size)
    ys = np.arange(int(low_arr[1]), int(high_arr[1]), cell_size)
    return np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)


def create_cloud(points: np.ndarray) -> PointCloud2:
    """
    Create a point cloud from points.

    Parameters
    ----------
    points : np.array
        Points in a grid to create the cloud from.

    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if points.ndim != 2:
        msg = f"points must be a 2-D array, got {points.ndim}-D"
        raise ValueError(msg)
    if points.shape[1] != CLOUD_COLS:
        msg = f"points must have {CLOUD_COLS} columns (x, y, z, cost), got {points.shape[1]}"
        raise ValueError(msg)

    points_f32 = points.astype(np.float32)
    cloud: PointCloud2 = msgify(
        PointCloud2,
        unstructured_to_structured(points_f32, names=["x", "y", "z", "cost"]),
    )
    return cloud


def points_near_ref(points: np.ndarray, reference: np.ndarray, max_dist: float = 1.0) -> np.ndarray:
    """
    Get points near reference points and set linear distance as cost.

    Parameters
    ----------
    points : np.array
        Points to check.
    reference : np.array
        Reference points.
    max_dist : float
        Maximum distance to check.

    Returns
    -------
    points : np.array
        All points with a cost based on distance to reference points.

    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if not isinstance(reference, np.ndarray):
        reference = np.array(reference)

    tree = cKDTree(reference, compact_nodes=False, balanced_tree=False)
    dists, _ = tree.query(points, distance_upper_bound=max_dist, workers=-1)
    mask = dists < max_dist
    filtered_points = points[mask]
    filtered_dists = dists[mask]

    return np.hstack([filtered_points, (filtered_dists / max_dist).reshape(-1, 1)])


def transform_points(
    points: dict[int, np.ndarray],
    transform: np.ndarray,
    z: float | None = None,
) -> dict[int, np.ndarray]:
    """
    Apply a 4x4 transform to every point in *points* (each a (3, 1) column vector).

    Parameters
    ----------
    points : dict
        Points to transform.
    transform : np.array
        Transformation matrix.
    z : float
        Z value to set on every transformed point, overriding the transform's.

    Returns
    -------
    transformed : dict
        Dictionary id: transformed (3, 1) point.

    """
    if not points:
        return {}
    ids = list(points)
    stacked = np.stack([np.asarray(points[i]).reshape(3) for i in ids])
    transformed = apply_transform(stacked, transform)
    if z is not None:
        transformed[:, 2] = z
    return {pid: transformed[k].reshape(3, 1) for k, pid in enumerate(ids)}


def split_ways_to_points(
    points: dict[int, np.ndarray],
    ways: dict[str, list[Any]],
    max_dist: float = 0.25,
    highway_types: Sequence[str] = ("footway",),
) -> np.ndarray:
    """
    Split OSM ways into equidistant points.

    Equidistantly split ways into points with a maximal step size. Only the way types in
    *highway_types* are used, as the robot is not allowed to leave them.

    Parameters
    ----------
    points : dict
        Points to split ways on.
    ways : dict
        Ways to split.
    max_dist : float
        Maximal step size.
    highway_types : sequence of str
        Way types to split: any of ``"footway"``, ``"road"``.

    Returns
    -------
    waypoints : np.array
        Waypoints created from the ways.

    """
    waypoints = []
    selected = [
        way
        for highway_type in dict.fromkeys(highway_types)
        if highway_type in HIGHWAY_TYPE_KEYS
        for way in ways.get(HIGHWAY_TYPE_KEYS[highway_type], [])
    ]
    for way in selected:
        ids = [getattr(n, "id", n) for n in way.nodes]
        if len(ids) < 2:
            continue
        nodes = np.array([points[i].ravel()[:2] for i in ids])
        starts, ends = nodes[:-1], nodes[1:]
        dists = np.linalg.norm(ends - starts, axis=1)

        waypoints.append(nodes[:1])
        for point0, point1, dist in zip(starts, ends, dists):
            if dist <= TOLERANCE:
                waypoints.append(point1[None])
                continue

            num = int(np.ceil(dist / max_dist))
            steps = np.arange(1, num + 1) / num
            waypoints.append(point0 + steps[:, None] * (point1 - point0))

    return np.concatenate(waypoints) if waypoints else np.empty((0, 2))


def main() -> None:
    rclpy.init()
    osm_cloud = OSMCloud()
    rclpy.spin(osm_cloud)
    osm_cloud.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
