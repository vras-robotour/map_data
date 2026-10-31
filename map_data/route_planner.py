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
algorithm, highway_types, spacing, max_snap_distance, cell_size, inflate_obstacles
    Planner defaults for goals that leave the field empty/zero
    (:func:`~map_data.pathsolver.route.goal_plan_kwargs`).
simplify_path, smooth_path : bool
    Used for ``~/goal`` goals only: a PlanRoute goal's bools are taken as sent.
    They are also the initial state of the viewer's planner toggles.
exclude_highway : str[]
    ``highway`` tag values never routed over (default ``["steps"]``: a wheeled
    robot cannot take stairs).
traversability_file : str
    Tag rules deciding which ways may be driven on and what they cost
    (``""`` = the package's ``config/traversability.yaml``). The parameter is
    read at startup; edits to the file are picked up by the next goal.
keep_goal : bool
    End the route at the requested goal coordinate rather than at its projection
    onto the network (default ``True``).
goal_max_snap_distance : float
    Reject a goal farther than this (m) from every allowed way with
    ``snap_too_far`` (default 30 m). The start keeps ``max_snap_distance``.
outside_map_tolerance : float
    Reject a start or goal farther than this (m) outside the map's area with
    ``start_outside_map`` / ``goal_outside_map`` (default 100 m); the message names the
    loaded file, as a start off the map usually means the wrong one.
fix_max_age : float
    Seconds after which the last fix is considered stale (default 10 s, 0 = never).

The eight planner parameters (``algorithm`` ... ``smooth_path``, see
:data:`PLANNER_PARAMETERS`) are read on every goal, so ``ros2 param set`` applies to
the next one, and their names are a stable interface for clients such as the viewer.
``cell_size``, ``inflate_obstacles``, ``simplify_path`` and ``smooth_path`` default to
``config/planner_defaults.yaml``.

Cancelling a PlanRoute goal stops its grid search (the goal UUID is its
``transfer_id``); the goal ends as canceled with reason ``cancelled``.

Besides the :mod:`~map_data.pathsolver.route` reasons, a goal fails with ``busy``,
``no_fix``, ``file_not_found`` or ``internal_error`` (a bug, logged with its traceback).
A goal may only name files inside ``data_dir`` (``mapdata_file``) and ``mission_dir``
(``save_gpx``); one with more than :data:`MAX_GOAL_WAYPOINTS` waypoints is rejected.
"""

from __future__ import annotations

import threading
import time
import traceback
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

from map_data.annotations import (
    NO_ANNOTATIONS,
    annotation_path_for,
    load_mapdata_with_annotations,
    load_summary,
)
from map_data.pathsolver.graph_planner import DEFAULT_MAX_SNAP_DISTANCE, GraphPlanner
from map_data.pathsolver.replan import cancel_replan_backend, grid_defaults
from map_data.pathsolver.route import (
    GRAPH_ALGORITHM,
    OUTSIDE_MAP_TOLERANCE_M,
    RoutePlanningError,
    RouteResult,
    check_in_map,
    goal_plan_kwargs,
    latlon_to_utm_path,
    plan_route,
)
from map_data.traversability import resolve_traversability_path
from map_data.utils.config import package_share, resolve_client_data_file, resolve_data_file
from map_data.utils.geodesy import latlon_to_ecef
from map_data.utils.gpx import create_gpx_track, mission_gpx_path
from map_data.utils.way import NON_ROUTABLE_HIGHWAY_VALUES
from map_data_interfaces.action import PlanRoute

WGS84_FRAME = "wgs84"
#: Parameters a goal falls back to (goal_plan_kwargs); the viewer reads them by name.
PLANNER_PARAMETERS = (
    "algorithm",
    "highway_types",
    "spacing",
    "max_snap_distance",
    "cell_size",
    "inflate_obstacles",
    "simplify_path",
    "smooth_path",
)
#: PlanRoute goals with more waypoints than this are rejected.
MAX_GOAL_WAYPOINTS = 100
#: GraphPlanners kept per (map, way set, snap distance); a small LRU.
PLANNER_CACHE_SIZE = 4


class RoutePlanner(Node):
    def __init__(self) -> None:
        super().__init__("route_planner")
        p = self.declare_parameter
        self.mapdata_file = p("mapdata_file", "").value
        self.data_dir = p("data_dir", str(package_share("data"))).value
        # "auto" = <mapdata>.annotations.json next to the map, "none" = the unedited map,
        # or an explicit store file.
        self.annotations = p("annotations", "auto").value
        # highway= values dropped from the map and from the graph: stairs are footways
        # in OSM and in the saved .mapdata, but a wheeled robot cannot take them.
        self.exclude_highway = list(p("exclude_highway", sorted(NON_ROUTABLE_HIGHWAY_VALUES)).value)
        # Tag rules (stairs, grass, bridges, ...) deciding what may be driven on and what
        # it costs; "" = the package's config/traversability.yaml. File edits reload on the
        # next goal.
        self.traversability_file = p("traversability_file", "").value
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

        # Planner defaults (a goal field left empty/zero takes these), see _planner_defaults.
        grid = grid_defaults()
        p("algorithm", GRAPH_ALGORITHM)
        p("highway_types", ["footway"])
        p("spacing", 3.0)
        p("max_snap_distance", DEFAULT_MAX_SNAP_DISTANCE)
        # Unset, these four take config/planner_defaults.yaml, like map_data_plan and the viewer.
        p("cell_size", float(grid["cell_size"]))
        p("inflate_obstacles", float(grid["inflate_obstacles"]))
        p("simplify_path", bool(grid["simplify_path"]))
        p("smooth_path", bool(grid["smooth_path"]))
        # The start is the robot's own fix and may be far from any way; the goal is a
        # place someone wants to reach, and a goal snapped 50 m away is a typo, not a
        # route. Off-network goals are driven as a final straight leg (keep_goal).
        self.default_keep_goal = bool(p("keep_goal", True).value)
        self.default_goal_max_snap = float(p("goal_max_snap_distance", 30.0).value)
        self.outside_map_tolerance = float(
            p("outside_map_tolerance", OUTSIDE_MAP_TOLERANCE_M).value
        )

        self._fix: NavSatFix | None = None
        self._fix_time = 0.0
        self._lock = threading.Lock()  # one plan at a time; a second one fails "busy"
        # (key, mtime, MapData)
        self._map_cache: tuple[tuple[str, str, str], float, object] | None = None
        # (map key, mtime, highway types, snap distance) -> GraphPlanner; small LRU
        self._planner_cache: OrderedDict[tuple, GraphPlanner] = OrderedDict()

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

    def _planner_defaults(self) -> dict:
        """The planner parameters' current values (the goal_plan_kwargs defaults)."""
        return {name: self.get_parameter(name).value for name in PLANNER_PARAMETERS}

    def _goal_topic_cb(self, msg: GeoPointStamped) -> None:
        """A single goal (QR code, operator) planned from the robot with the defaults."""
        try:
            defaults = self._planner_defaults()
            goal = PlanRoute.Goal()
            goal.waypoints = [msg.position]
            goal.start_from_robot = True
            goal.simplify_path = defaults["simplify_path"]
            goal.smooth_path = defaults["smooth_path"]
            self.get_logger().info(
                f"Goal on {self.goal_topic}: "
                f"{msg.position.latitude:.7f}, {msg.position.longitude:.7f}"
            )
            result = self._plan(goal, feedback=None)
            self._log_result(result)
        except Exception:  # an exception here would stop the executor
            self.get_logger().error(f"Goal on {self.goal_topic} failed:\n{traceback.format_exc()}")

    def _goal_cb(self, goal: PlanRoute.Goal) -> GoalResponse:
        if not goal.waypoints:
            self.get_logger().warning("PlanRoute goal rejected: no waypoints")
            return GoalResponse.REJECT
        if len(goal.waypoints) > MAX_GOAL_WAYPOINTS:
            self.get_logger().warning(
                f"PlanRoute goal rejected: {len(goal.waypoints)} waypoints, "
                f"more than {MAX_GOAL_WAYPOINTS}"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, handle) -> CancelResponse:
        # Stops a grid search in progress.
        cancel_replan_backend(_transfer_id(handle))
        return CancelResponse.ACCEPT

    def _execute(self, handle) -> PlanRoute.Result:
        def feedback(stage: str) -> None:
            fb = PlanRoute.Feedback()
            fb.stage = stage
            handle.publish_feedback(fb)

        result = self._plan(
            handle.request,
            feedback,
            transfer_id=_transfer_id(handle),
            cancel_requested=lambda: handle.is_cancel_requested,
        )
        self._log_result(result)
        if result.reason == "cancelled" and handle.is_cancel_requested:
            handle.canceled()
        elif result.success:
            handle.succeed()
        else:
            handle.abort()
        return result

    def _preload_once(self) -> None:
        """One-shot timer: load the default map and build its default graph planner."""
        self._preload_timer.cancel()
        self.destroy_timer(self._preload_timer)
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
                defaults = self._planner_defaults()
                self._graph_planner(
                    path, md, defaults["highway_types"], defaults["max_snap_distance"]
                )
        except Exception as e:
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
        # exclude_highway and traversability_file are fixed at init; the latter is
        # also in the map key (cache[0]).
        key = (cache[0], cache[1], tuple(highway_types), float(max_snap))
        planner = self._planner_cache.get(key)
        if planner is not None and planner.map_data is md:
            self._planner_cache.move_to_end(key)
            return planner
        t0 = time.monotonic()
        planner = GraphPlanner(
            md,
            highway_types=list(highway_types),
            max_snap_distance=max_snap,
            exclude_highway=self.exclude_highway,
            traversability=self.traversability_file or None,
        )
        self._planner_cache[key] = planner
        while len(self._planner_cache) > PLANNER_CACHE_SIZE:
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
        """A goal's *name* (only inside ``data_dir``), or else the ``mapdata_file`` parameter."""
        if name:
            return resolve_client_data_file(name, self.data_dir)
        if not self.mapdata_file:
            return None
        return resolve_data_file(self.mapdata_file, self.data_dir)

    def _load_map(self, path: Path):
        ann = None if self.annotations in ("", "auto") else self.annotations
        ann_path = None if ann in (None, NO_ANNOTATIONS) else Path(ann).expanduser()
        if ann is None:
            ann_path = annotation_path_for(path)
        mtime = path.stat().st_mtime
        if ann_path is not None and ann_path.is_file():
            mtime += ann_path.stat().st_mtime
        # An edited rule file invalidates the cached map, like an edited map or store.
        trav_path = resolve_traversability_path(self.traversability_file)
        if trav_path is not None:
            mtime += trav_path.stat().st_mtime
        key = (str(path), str(ann), str(self.traversability_file))
        if self._map_cache and self._map_cache[0] == key and self._map_cache[1] == mtime:
            return self._map_cache[2]
        md, store = load_mapdata_with_annotations(
            path,
            ann,
            exclude_highway=self.exclude_highway,
            traversability=self.traversability_file or None,
        )
        self.get_logger().info(load_summary(path, md, store, ann, self.exclude_highway, trav_path))
        self._map_cache = (key, mtime, md)
        return md

    def _plan(
        self,
        goal: PlanRoute.Goal,
        feedback,
        transfer_id: str | None = None,
        cancel_requested=lambda: False,
    ) -> PlanRoute.Result:
        res = PlanRoute.Result()
        res.route.header.frame_id = WGS84_FRAME
        res.route_local.header.frame_id = self.local_frame

        def fail(reason: str, message: str) -> PlanRoute.Result:
            res.success = False
            res.reason = reason
            res.message = message
            self._publish_status(f"FAILED {reason}: {message!r}")
            return res

        # Never wait for the lock: a blocked callback holds an executor thread, and enough of
        # them leave none for cancels and GNSS fixes.
        if not self._lock.acquire(blocking=False):
            return fail("busy", "another plan is running; try again when it has finished")
        try:
            path = self._resolve_mapdata(goal.mapdata_file)
            if path is None:
                name = goal.mapdata_file or self.mapdata_file
                return fail(
                    "file_not_found",
                    f"map data file {name!r} not found in {self.data_dir} "
                    "(a goal may only name a file inside it)",
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

            kwargs = goal_plan_kwargs(goal, self._planner_defaults())
            algorithm = kwargs["algorithm"]
            highway_types = kwargs["highway_types"]
            # Before the snap check, so a point off the map is reported as such; the file is
            # named because a start off the map is nearly always the wrong one loaded.
            try:
                start_utm, goal_utm = latlon_to_utm_path(
                    [points[0], points[-1]], md.zone_number, md.zone_letter
                )
                check_in_map(md, start_utm, "start", self.outside_map_tolerance)
                check_in_map(md, goal_utm, "goal", self.outside_map_tolerance)
            except RoutePlanningError as e:
                return fail(e.reason, f"{e.message}: {path}")
            planner = None
            if algorithm == GRAPH_ALGORITHM:
                planner = self._graph_planner(path, md, highway_types, kwargs["max_snap_distance"])
                goal_snap = planner.snap_distance(goal_utm)
                if not goal_snap <= self.default_goal_max_snap:  # also NaN
                    return fail(
                        "snap_too_far",
                        f"the goal is {goal_snap:.1f} m from the nearest "
                        f"{'/'.join(highway_types)} (goal limit "
                        f"{self.default_goal_max_snap:.0f} m); pick a point nearer a path",
                    )
            if feedback:
                feedback(f"planning ({algorithm}, {'/'.join(highway_types)})")
            t0 = time.monotonic()
            try:
                route = plan_route(
                    md,
                    points,
                    **kwargs,
                    planner=planner,
                    transfer_id=transfer_id,
                    # The robot is where it is: keep its GNSS fix as the first
                    # route point instead of jumping to the nearest path.
                    keep_start=goal.start_from_robot,
                    # ... and end at the goal itself: its projection can be tens of
                    # metres short of the coordinate the mission asks for.
                    keep_goal=self.default_keep_goal,
                    outside_map_tolerance=self.outside_map_tolerance,
                )
            except RoutePlanningError as e:
                return fail(e.reason, e.message)
            dt = time.monotonic() - t0
            if cancel_requested():
                return fail("cancelled", "the goal was cancelled")

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
        except Exception as e:  # a bug must fail the goal, not kill the node
            self.get_logger().error(f"PlanRoute failed internally:\n{traceback.format_exc()}")
            return fail("internal_error", f"{type(e).__name__}: {e}")
        finally:
            self._lock.release()

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
                rclpy.duration.Duration(seconds=0),
            )
        except (TransformException, RuntimeError, TypeError, ValueError) as e:
            self.get_logger().warning(
                f"{self.earth_frame} -> {self.local_frame} unavailable, "
                f"route_local left empty: {e}",
                throttle_duration_sec=30.0,
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
        out = mission_gpx_path(save_gpx, self.mission_dir)
        if out is None:
            self.get_logger().error(
                f"Not saving the route: save_gpx {save_gpx!r} is not a .gpx file name "
                f"in {self.mission_dir}"
            )
            return ""
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
            self.get_logger().error(f"Route planning failed ({result.reason}): {result.message!r}")


def _transfer_id(handle) -> str:
    """The goal's UUID, under which a cancel reaches the grid planner."""
    return bytes(handle.goal_id.uuid).hex()


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
