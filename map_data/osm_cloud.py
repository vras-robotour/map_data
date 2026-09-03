#!/usr/bin/env python3
"""
ROS2 node for publishing OSM map data as a point cloud.

This module provides the OSMCloud class which converts parsed MapData
into ROS2 PointCloud2 and MarkerArray messages for visualization.
"""

import sys
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
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformException,
    TransformListener,
)
from visualization_msgs.msg import Marker, MarkerArray

import map_data.map_data as md
from map_data.utils.geodesy import ecef_to_latlon, utm_to_local_via_ecef

CLOUD_COLS = 4
TOLERANCE = 1e-3
TRANSFORM_MODES = ("tf", "auto", "geodetic")
# lookup_transform() does not spin the node, so TF messages only arrive in the
# spin_once() between attempts; keep the blocking wait short.
TF_POLL_TIMEOUT = 0.5


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
        self.utm_to_local_param: list[float] | None = self.declare_parameter(
            "utm_to_local",
            rclpy.Parameter.Type.DOUBLE_ARRAY,
        ).value
        self.mapdata_file: str | None = self.declare_parameter(
            "mapdata_file",
            rclpy.Parameter.Type.STRING,
        ).value
        self.gpx_file: str | None = self.declare_parameter(
            "gpx_file",
            rclpy.Parameter.Type.STRING,
        ).value
        self.save_mapdata: bool = self.declare_parameter("save_mapdata", False).value
        self.max_path_dist: float = self.declare_parameter("max_path_dist", 1.0).value
        self.neighbor_cost: str = self.declare_parameter("neighbor_cost", "linear").value
        self.grid_res: float = self.declare_parameter("grid_res", 0.25).value
        self.grid_max: list[float] = self.declare_parameter("grid_max", [0.0, 0.0]).value
        self.grid_min: list[float] = self.declare_parameter("grid_min", [0.0, 0.0]).value
        self.auto_utm: bool = self.declare_parameter("auto_utm", False).value
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
        self.tf_sub = TransformListener(self.tf, self, spin_thread=True)
        self.tf_static_pub = StaticTransformBroadcaster(self)

        self.utm_to_local: np.ndarray | None = None
        self.ecef_to_local: np.ndarray | None = None
        self.poses: PoseArray | None = None
        self.markers: MarkerArray | None = None

        if self.mapdata_file:
            self.map_data = md.MapData.load(self.mapdata_file)
        elif self.gpx_file:
            self.map_data = md.MapData(self.gpx_file)
            self.map_data.run_all(save=self.save_mapdata)
        else:
            self.get_logger().error("No map data or gpx file provided")
            sys.exit(1)
        self.get_logger().info(str(self.map_data))

        if self.transform_mode not in TRANSFORM_MODES:
            self.get_logger().warn(
                f"Unknown transform_mode '{self.transform_mode}', falling back to 'tf'"
            )
            self.transform_mode = "tf"
        if self.auto_utm and self.transform_mode == "tf":
            self.transform_mode = "auto"  # backward-compatible alias

        if self.utm_to_local_param is not None:
            self.utm_to_local = np.array(self.utm_to_local_param).reshape(4, 4)
        elif self.transform_mode == "geodetic":
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
            self.get_logger().info("Auto-calculating grid bounds from map data")
            # Transform all four corners of the UTM bounding box to the local frame
            # (the local frame may be rotated relative to UTM).
            xs = (self.map_data.min_x, self.map_data.max_x)
            ys = (self.map_data.min_y, self.map_data.max_y)
            corners = {
                i: np.array([x, y, 0.0]).reshape(3, 1)
                for i, (x, y) in enumerate((x, y) for x in xs for y in ys)
            }
            # The transform is None only if its lookup was interrupted by rclpy
            # shutdown mid-startup; skip the auto-calc (get_cloud() below raises a
            # clear error in that case).
            if not self._transform_ready():
                self.get_logger().error(
                    "Map placement transform unavailable; skipping grid-bounds auto-calc"
                )
            else:
                bounds_arr = np.array([p.ravel() for p in self._to_local(corners).values()])
                self.grid_min = [np.min(bounds_arr[:, 0]), np.min(bounds_arr[:, 1])]
                self.grid_max = [np.max(bounds_arr[:, 0]), np.max(bounds_arr[:, 1])]
                self.get_logger().info(
                    f"Calculated grid bounds: min={self.grid_min}, max={self.grid_max}"
                )

        self.grid_cloud: PointCloud2 = self.get_cloud()
        if self.publish_intersections:
            self.poses, self.markers = self.get_intersections()

        self.publish_cb()
        if self.republish_period > 0:
            self.create_timer(self.republish_period, self.publish_cb)
        self.get_logger().info("Initialized OSM cloud")

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
            elif param.name == "grid_res":
                self.grid_res = param.value
                rebuild_cloud = True
            elif param.name == "grid_max":
                self.grid_max = param.value
                rebuild_cloud = True
                rebuild_intersections = True
            elif param.name == "grid_min":
                self.grid_min = param.value
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

    def get_utm_to_local(self) -> None:
        """
        Poll for the UTM to local coordinate transform.

        While rclpy is not shutdown, try to get the UTM to local transform every second or
        until successful.
        """
        while rclpy.ok():
            try:
                utm_to_local = self.tf.lookup_transform(
                    self.local_frame,
                    self.utm_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=TF_POLL_TIMEOUT),
                )
                self.utm_to_local = numpify(utm_to_local.transform)
                self.get_logger().info(f"Got UTM to local transform: {self.utm_to_local}")
                break
            except (TransformException, RuntimeError, TypeError, ValueError) as e:
                self.get_logger().warning(
                    f"Failed to get UTM to local transform: {e}", throttle_duration_sec=10.0
                )
                rclpy.spin_once(self, timeout_sec=1.0)

    def get_ecef_to_local(self) -> None:
        """
        Poll for the ``earth_frame -> local_frame`` transform (ECEF to local ENU).

        The result is stored as a 4x4 matrix mapping ECEF points into ``local_frame``.
        """
        while rclpy.ok():
            try:
                tf_msg = self.tf.lookup_transform(
                    self.local_frame,
                    self.earth_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=TF_POLL_TIMEOUT),
                )
                self.ecef_to_local = numpify(tf_msg.transform)
                origin = np.linalg.inv(self.ecef_to_local)[:3, 3]
                lat, lon, alt = ecef_to_latlon(*origin)
                self.get_logger().info(
                    f"Got {self.earth_frame} -> {self.local_frame} transform; "
                    f"{self.local_frame} origin at lat={lat:.7f} lon={lon:.7f} alt={alt:.1f}"
                )
                break
            except (TransformException, RuntimeError, TypeError, ValueError) as e:
                self.get_logger().warning(
                    f"Failed to get {self.earth_frame} -> {self.local_frame} transform: {e}",
                    throttle_duration_sec=10.0,
                )
                rclpy.spin_once(self, timeout_sec=1.0)

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
        grid = np.pad(
            create_grid(tuple(self.grid_min), tuple(self.grid_max), self.grid_res),
            ((0, 0), (0, 1)),
        )
        waypoints = np.pad(
            split_ways_to_points(points, self.map_data.get_ways(), self.grid_res),
            ((0, 0), (0, 1)),
        )

        grid = points_near_ref(grid, waypoints, self.max_path_dist)
        if self.neighbor_cost == "linear":
            pass
        elif self.neighbor_cost == "quadratic":
            grid[:, 3] = grid[:, 3] ** 2
        else:
            if self.neighbor_cost != "zero" and self.neighbor_cost != "linear":
                self.get_logger().warn(f"Unknown neighbor cost: {self.neighbor_cost}")
            grid[:, 3] = 0.0
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

            # Spatial filtering based on local frame coordinates
            if not (
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
    dists, _ = tree.query(points, distance_upper_bound=max_dist)
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
    Transform points.

    Parameters
    ----------
    points : dict
        Points to transform.
    transform : np.array
        Transformation matrix.
    z : float
        Z value to set.

    Returns
    -------
    transformed : dict
        Dictionary id: transformed point.

    """

    def transform_point(point: np.ndarray, transform_mat: np.ndarray) -> np.ndarray:
        """
        Transform a point with a transformation matrix.

        Parameters
        ----------
        point : np.array
            Point to transform.
        transform_mat : np.array
            Transformation matrix.

        Returns
        -------
        point : np.array
            Transformed point.

        """
        if not isinstance(point, np.ndarray):
            msg = f"point must be np.ndarray, got {type(point).__name__}"
            raise TypeError(msg)
        if not isinstance(transform_mat, np.ndarray):
            msg = f"transform_mat must be np.ndarray, got {type(transform_mat).__name__}"
            raise TypeError(msg)

        return np.dot(transform_mat[:3, :3], point) + transform_mat[:3, 3:]

    transformed = {}
    for pid, point in points.items():
        transformed[pid] = transform_point(point, transform)
        if z is not None:
            transformed[pid][2] = z
    return transformed


def split_ways_to_points(
    points: dict[int, np.ndarray],
    ways: dict[str, list[Any]],
    max_dist: float = 0.25,
) -> np.ndarray:
    """
    Split OSM ways into equidistant points.

    Equidistantly split ways into points with a maximal step size. Also only use footways
    from map data, as we are not allowed to leave the footways.

    Parameters
    ----------
    points : dict
    ...
        Points to split ways on.
    ways : dict
        Ways to split.
    max_dist : float
        Maximal step size.

    Returns
    -------
    waypoints : np.array
        Waypoints created from the ways.

    """
    waypoints = []
    for way in ways.get("footways", []):
        for i, (n0, n1) in enumerate(zip(way.nodes, way.nodes[1:])):
            id0 = getattr(n0, "id", n0)
            id1 = getattr(n1, "id", n1)
            point0 = points[id0].ravel()[:2]
            point1 = points[id1].ravel()[:2]

            if i == 0:
                waypoints.append(point0)

            dist = float(np.linalg.norm(point1 - point0))

            if dist <= TOLERANCE:
                waypoints.append(point1)
                continue

            vec = (point1 - point0) / dist
            num = int(np.ceil(dist / max_dist))
            step = dist / num
            waypoints.extend(point0 + (j + 1) * step * vec for j in range(num))

    return np.array(waypoints) if waypoints else np.empty((0, 2))


def main() -> None:
    rclpy.init()
    osm_cloud = OSMCloud()
    rclpy.spin(osm_cloud)
    osm_cloud.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
