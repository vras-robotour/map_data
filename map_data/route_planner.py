#!/usr/bin/env python3
"""
``route_planner``: the viewer's Planner screen as a ROS 2 action server.

Serves ``map_data_interfaces/action/PlanRoute`` on ``~/plan_route``. A goal
names a ``.mapdata`` file (annotations merged, like the viewer), an ordered
list of lat/lon waypoints and the planner parameters; the result is the route
as a ``geographic_msgs/GeoPath`` plus the same route as a ``nav_msgs/Path`` in
``local_frame`` (through the ``earth_frame -> local_frame`` TF, exactly as
``osm_cloud`` places the map), and the route is written to a GPX track for
the record. Both paths are also published latched on ``~/route`` and
``~/route_path`` so the tracker/viewer and a waypoint follower can pick them
up without being action clients.

With ``start_from_robot`` the latest GNSS fix (``gps_fix_topic``) is used as
the first waypoint, so a single goal, e.g. from a Robotour QR code, is enough.
The same happens for a ``geographic_msgs/GeoPointStamped`` published on
``~/goal`` (the topic hook for a QR reader), using the node's default
parameters.

Parameters
----------
mapdata_file : str
    Default ``.mapdata`` (name in ``data_dir`` or absolute path).
data_dir : str
    Directory the file names are resolved against (default: the package's
    installed ``share/map_data/data``).
mission_dir : str
    Where GPX routes are written (default ``~/missions``).
gps_fix_topic : str
    ``sensor_msgs/NavSatFix`` used for ``start_from_robot`` (default
    ``/fixposition/odometry_llh``).
earth_frame, local_frame : str
    ECEF and local frames for ``route_local`` (``FP_ECEF`` -> ``FP_ENU0``).
algorithm, highway_types, spacing, max_snap_distance, cell_size,
inflate_obstacles, simplify_path, smooth_path
    Planner defaults applied to goals that leave the field empty/zero.
fix_max_age : float
    Seconds after which the last fix is considered stale (0 = never).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geographic_msgs.msg import GeoPath, GeoPointStamped, GeoPoseStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, qos_profile_sensor_data
from ros2_numpy import numpify
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException

from map_data.annotations import NO_ANNOTATIONS, annotation_path_for, load_mapdata_with_annotations
from map_data.pathsolver.graph_planner import DEFAULT_MAX_SNAP_DISTANCE, GraphPlanner
from map_data.pathsolver.route import (
    GRAPH_ALGORITHM,
    RoutePlanningError,
    RouteResult,
    plan_route,
)
from map_data.utils.geodesy import latlon_to_ecef
from map_data.utils.gpx import create_gpx_track
from map_data_interfaces.action import PlanRoute

WGS84_FRAME = "wgs84"


def _default_data_dir() -> str:
    try:
        from ament_index_python.resources import get_resource

        _, pkg = get_resource("packages", "map_data")
        return str(Path(pkg) / "share" / "map_data" / "data")
    except (ImportError, LookupError):
        return str((Path(__file__).parent / ".." / "data").resolve())


class RoutePlanner(Node):
    def __init__(self) -> None:
        super().__init__("route_planner")
        p = self.declare_parameter
        self.mapdata_file = p("mapdata_file", "").value
        self.data_dir = p("data_dir", _default_data_dir()).value
        # "auto" = <mapdata>.annotations.json next to the map, "none" = the unedited map,
        # or an explicit store file.
        self.annotations = p("annotations", "auto").value
        # Load mapdata_file and build its footway graph at startup (~20 MB) so the first
        # goal does not pay for it; graph planners are cached per map / way set anyway.
        self.preload = bool(p("preload", True).value)
        self.mission_dir = p("mission_dir", str(Path.home() / "missions")).value
        self.gps_fix_topic = p("gps_fix_topic", "/fixposition/odometry_llh").value
        self.goal_topic = p("goal_topic", "~/goal").value
        self.route_topic = p("route_topic", "~/route").value
        self.route_path_topic = p("route_path_topic", "~/route_path").value
        self.status_topic = p("status_topic", "~/status").value
        self.earth_frame = p("earth_frame", "FP_ECEF").value
        self.local_frame = p("local_frame", "FP_ENU0").value
        self.fix_max_age = float(p("fix_max_age", 10.0).value)

        # Planner defaults (a goal field left empty/zero takes these).
        self.default_algorithm = p("algorithm", GRAPH_ALGORITHM).value
        self.default_highway_types = list(p("highway_types", ["footway"]).value)
        self.default_spacing = float(p("spacing", 3.0).value)
        self.default_max_snap = float(p("max_snap_distance", DEFAULT_MAX_SNAP_DISTANCE).value)
        self.default_cell_size = float(p("cell_size", 0.25).value)
        self.default_inflate = float(p("inflate_obstacles", 0.25).value)
        self.default_simplify = bool(p("simplify_path", True).value)
        self.default_smooth = bool(p("smooth_path", False).value)

        self._fix: NavSatFix | None = None
        self._fix_time = 0.0
        self._lock = threading.Lock()  # one plan at a time
        # (key, mtime, MapData)
        self._map_cache: tuple[tuple[str, str], float, object] | None = None
        # (map key, mtime, highway types, snap distance) -> GraphPlanner; small LRU
        self._planner_cache: OrderedDict[tuple, GraphPlanner] = OrderedDict()
        self._planner_cache_size = 4

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_route = self.create_publisher(GeoPath, self.route_topic, latched)
        self.pub_route_path = self.create_publisher(PathMsg, self.route_path_topic, latched)
        self.pub_status = self.create_publisher(String, self.status_topic, latched)

        # Only the static earth->local transform is needed; a full TransformListener would
        # also digest the /tf firehose (hundreds of Hz on Helhest) and starve the planner.
        self.tf = Buffer()
        static_qos = QoSProfile(depth=100, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(TFMessage, "/tf_static", self._tf_static_cb, static_qos)

        cbg = ReentrantCallbackGroup()
        self.create_subscription(
            NavSatFix, self.gps_fix_topic, self._fix_cb, qos_profile_sensor_data, callback_group=cbg
        )
        self.create_subscription(
            GeoPointStamped, self.goal_topic, self._goal_topic_cb, latched, callback_group=cbg
        )
        if self.preload and self.mapdata_file:
            self._preload_timer = self.create_timer(0.1, self._preload_once)
        self._server = ActionServer(
            self,
            PlanRoute,
            "~/plan_route",
            execute_callback=self._execute,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=cbg,
        )
        self.get_logger().info(
            f"route_planner ready: mapdata_file='{self.mapdata_file}', data_dir={self.data_dir}, "
            f"mission_dir={self.mission_dir}, fix from {self.gps_fix_topic}, "
            f"routes on {self.route_topic} / {self.route_path_topic} ({self.local_frame})"
        )

    # ------------------------------------------------------------------ inputs
    def _fix_cb(self, msg: NavSatFix) -> None:
        if np.isfinite(msg.latitude) and np.isfinite(msg.longitude):
            self._fix = msg
            self._fix_time = time.monotonic()

    def _current_fix(self) -> tuple[float, float] | None:
        if self._fix is None:
            return None
        if self.fix_max_age > 0 and (time.monotonic() - self._fix_time) > self.fix_max_age:
            return None
        return self._fix.latitude, self._fix.longitude

    def _goal_topic_cb(self, msg: GeoPointStamped) -> None:
        """A single goal (QR code, operator) planned from the robot with the defaults."""
        goal = PlanRoute.Goal()
        goal.waypoints = [msg.position]
        goal.start_from_robot = True
        self.get_logger().info(
            f"Goal on {self.goal_topic}: {msg.position.latitude:.7f}, {msg.position.longitude:.7f}"
        )
        result = self._plan(goal, feedback=None)
        self._log_result(result)

    def _goal_cb(self, goal: PlanRoute.Goal) -> GoalResponse:
        if not goal.waypoints:
            self.get_logger().warning("PlanRoute goal rejected: no waypoints")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _handle) -> CancelResponse:
        return CancelResponse.ACCEPT  # planning is short; the result is still delivered

    def _execute(self, handle) -> PlanRoute.Result:
        def feedback(stage: str) -> None:
            fb = PlanRoute.Feedback()
            fb.stage = stage
            handle.publish_feedback(fb)

        result = self._plan(handle.request, feedback)
        self._log_result(result)
        if result.success:
            handle.succeed()
        else:
            handle.abort()
        return result

    def _preload_once(self) -> None:
        """One-shot timer: load the default map and build its default graph planner."""
        self._preload_timer.cancel()
        path = self._resolve_mapdata("")
        if path is None:
            self.get_logger().warning(
                f"preload: '{self.mapdata_file}' not found in {self.data_dir}"
            )
            return
        t0 = time.monotonic()
        try:
            with self._lock:
                md = self._load_map(path)
                self._graph_planner(path, md, self.default_highway_types, self.default_max_snap)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"preload of {path.name} failed: {e}")
            return
        self.get_logger().info(
            f"preloaded {path.name} and its graph in {time.monotonic() - t0:.2f} s"
        )

    def _graph_planner(self, path: Path, md, highway_types, max_snap: float) -> GraphPlanner:
        """Cached GraphPlanner for (map file, mtime, way set, snap distance)."""
        cache = self._map_cache
        if cache is None:  # every caller runs _load_map first, which fills the cache
            raise RuntimeError("_graph_planner called before the map was loaded")
        key = (cache[0], cache[1], tuple(highway_types), float(max_snap))
        planner = self._planner_cache.get(key)
        if planner is not None and planner.map_data is md:
            self._planner_cache.move_to_end(key)
            return planner
        t0 = time.monotonic()
        planner = GraphPlanner(md, highway_types=list(highway_types), max_snap_distance=max_snap)
        self._planner_cache[key] = planner
        while len(self._planner_cache) > self._planner_cache_size:
            self._planner_cache.popitem(last=False)
        self.get_logger().info(
            f"built graph planner for {path.name} ({'/'.join(highway_types)}) in "
            f"{time.monotonic() - t0:.2f} s; {len(self._planner_cache)} cached"
        )
        return planner

    def _tf_static_cb(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            self.tf.set_transform_static(t, "route_planner")

    # ------------------------------------------------------------------ planning
    def _resolve_mapdata(self, name: str) -> Path | None:
        name = name or self.mapdata_file
        if not name:
            return None
        p = Path(name).expanduser()
        if not p.is_absolute():
            p = Path(self.data_dir).expanduser() / name
        return p if p.is_file() else None

    def _load_map(self, path: Path):
        ann = None if self.annotations in ("", "auto") else self.annotations
        ann_path = None if ann in (None, NO_ANNOTATIONS) else Path(ann).expanduser()
        if ann is None:
            ann_path = annotation_path_for(path)
        mtime = path.stat().st_mtime
        if ann_path is not None and ann_path.is_file():
            mtime += ann_path.stat().st_mtime
        key = (str(path), str(ann))
        if self._map_cache and self._map_cache[0] == key and self._map_cache[1] == mtime:
            return self._map_cache[2]
        md, store = load_mapdata_with_annotations(path, ann)
        if ann == NO_ANNOTATIONS:
            store_name = "none"
        elif ann_path is not None and ann_path.is_file():
            store_name = ann_path.name
        else:
            store_name = "no store"
        self.get_logger().info(
            f"loaded {path.name}: {len(md.footways_list)} footways, {len(md.roads_list)} roads, "
            f"annotations={store_name} ({len(store.get('deleted_ways', []))} deleted ways, "
            f"{len(store.get('annotations', []))} drawn)"
        )
        self._map_cache = (key, mtime, md)
        return md

    def _plan(self, goal: PlanRoute.Goal, feedback) -> PlanRoute.Result:
        res = PlanRoute.Result()
        res.route.header.frame_id = WGS84_FRAME
        res.route_local.header.frame_id = self.local_frame

        def fail(reason: str, message: str) -> PlanRoute.Result:
            res.success = False
            res.reason = reason
            res.message = message
            self._publish_status(f"FAILED {reason}: {message}")
            return res

        with self._lock:
            path = self._resolve_mapdata(goal.mapdata_file)
            if path is None:
                name = goal.mapdata_file or self.mapdata_file
                return fail(
                    "file_not_found", f"map data file '{name}' not found in {self.data_dir}"
                )

            points: list[tuple[float, float]] = []
            if goal.start_from_robot:
                fix = self._current_fix()
                if fix is None:
                    return fail("no_fix", f"no recent GNSS fix on {self.gps_fix_topic}")
                points.append(fix)
            points.extend((wp.latitude, wp.longitude) for wp in goal.waypoints)
            if len(points) < 2:
                return fail(
                    "too_few_points", "give at least two waypoints, or one with start_from_robot"
                )

            if feedback:
                feedback("loading map")
            try:
                md = self._load_map(path)
            except Exception as e:  # corrupt file, wrong format
                return fail("file_not_found", f"could not load {path}: {e}")

            algorithm = goal.algorithm or self.default_algorithm
            highway_types = list(goal.highway_types) or self.default_highway_types
            spacing = self.default_spacing if goal.spacing == 0.0 else max(goal.spacing, 0.0)
            max_snap = goal.max_snap_distance or self.default_max_snap
            planner = None
            if algorithm == GRAPH_ALGORITHM:
                planner = self._graph_planner(path, md, highway_types, max_snap)
            if feedback:
                feedback(f"planning ({algorithm}, {'/'.join(highway_types)})")
            t0 = time.monotonic()
            try:
                route = plan_route(
                    md,
                    points,
                    algorithm=algorithm,
                    highway_types=highway_types,
                    spacing=spacing,
                    max_snap_distance=max_snap,
                    planner=planner,
                    cell_size=goal.cell_size or self.default_cell_size,
                    inflate_obstacles=goal.inflate_obstacles or self.default_inflate,
                    simplify_path=goal.simplify_path or self.default_simplify,
                    smooth_path=goal.smooth_path or self.default_smooth,
                )
            except RoutePlanningError as e:
                return fail(e.reason, e.message)
            dt = time.monotonic() - t0

            if feedback:
                feedback("publishing")
            stamp = self.get_clock().now().to_msg()
            res.route = self._geo_path(route, stamp)
            res.route_local = self._local_path(route, stamp)
            res.length_m = float(route.length_m)
            res.snap_distances = [float(d) for d in route.snap_distances]
            res.gpx_path = self._save_gpx(route, goal.save_gpx)
            res.success = True
            res.reason = ""
            res.message = (
                f"{len(route.latlon)} waypoints, {route.length_m:.0f} m ({algorithm}, {dt:.2f} s)"
                + (f", saved {res.gpx_path}" if res.gpx_path else "")
            )
            self.pub_route.publish(res.route)
            if res.route_local.poses:
                self.pub_route_path.publish(res.route_local)
            self._publish_status(f"PLANNED {res.message}")
            return res

    # ------------------------------------------------------------------ outputs
    def _geo_path(self, route: RouteResult, stamp: TimeMsg) -> GeoPath:
        msg = GeoPath()
        msg.header.stamp = stamp
        msg.header.frame_id = WGS84_FRAME
        for lat, lon in route.latlon:
            gp = GeoPoseStamped()
            gp.header = msg.header
            gp.pose.position.latitude = float(lat)
            gp.pose.position.longitude = float(lon)
            gp.pose.orientation.w = 1.0
            msg.poses.append(gp)
        return msg

    def _local_path(self, route: RouteResult, stamp: TimeMsg) -> PathMsg:
        """The route in ``local_frame`` via lat/lon -> ECEF -> TF (exact, like osm_cloud)."""
        msg = PathMsg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.local_frame
        try:
            tf_msg = self.tf.lookup_transform(
                self.local_frame,
                self.earth_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.5),
            )
        except (TransformException, RuntimeError, TypeError, ValueError) as e:
            self.get_logger().warning(
                f"{self.earth_frame} -> {self.local_frame} unavailable, route_local left empty: {e}"
            )
            return msg
        m = numpify(tf_msg.transform)
        lats = [lat for lat, _ in route.latlon]
        lons = [lon for _, lon in route.latlon]
        ecef = latlon_to_ecef(lats, lons, 0.0)  # (N, 3)
        local = (m[:3, :3] @ ecef.T).T + m[:3, 3]
        for x, y, _z in local:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def _save_gpx(self, route: RouteResult, save_gpx: str) -> str:
        if save_gpx == "-":
            return ""
        if not save_gpx:
            save_gpx = time.strftime("route_%Y%m%d-%H%M%S.gpx")
        out = Path(save_gpx).expanduser()
        if not out.is_absolute():
            out = Path(self.mission_dir).expanduser() / out
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(create_gpx_track(route.latlon, name=out.stem))
        except OSError as e:
            self.get_logger().error(f"Could not write {out}: {e}")
            return ""
        return str(out)

    def _publish_status(self, text: str) -> None:
        self.pub_status.publish(String(data=text))

    def _log_result(self, result: PlanRoute.Result) -> None:
        if result.success:
            self.get_logger().info(f"Route planned: {result.message}")
        else:
            self.get_logger().error(f"Route planning failed ({result.reason}): {result.message}")


def main() -> None:
    rclpy.init()
    node = RoutePlanner()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
