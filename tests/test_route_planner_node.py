"""route_planner's PlanRoute goal handling: cancels reach the grid planner by goal UUID."""

import threading
import uuid
from types import SimpleNamespace

import pytest
import utm

route_planner = pytest.importorskip(
    "map_data.route_planner", reason="needs ROS 2 (rclpy, map_data_interfaces, ...)"
)
from geographic_msgs.msg import GeoPoint  # noqa: E402

from map_data.annotations import load_mapdata_with_annotations  # noqa: E402
from map_data.pathsolver import replan  # noqa: E402
from map_data.pathsolver.graph_planner import GraphPlanner  # noqa: E402


def _latlon(lat0, lon0, dx, dy):
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    return utm.to_latlon(e0 + dx, n0 + dy, zn, zl)


class _Handle:
    def __init__(self):
        self.goal_id = SimpleNamespace(uuid=list(uuid.uuid4().bytes))
        self.request = object()
        self.is_cancel_requested = False
        self.state = None

    def publish_feedback(self, _fb):
        pass

    def canceled(self):
        self.state = "canceled"

    def succeed(self):
        self.state = "succeeded"

    def abort(self):
        self.state = "aborted"


def test_cancel_reaches_the_grid_planner_by_goal_uuid():
    handle = _Handle()
    tid = route_planner._transfer_id(handle)
    try:
        route_planner.RoutePlanner._cancel_cb(None, handle)
        assert replan._is_cancelled(tid)
    finally:
        replan._discard_cancelled(tid)


@pytest.mark.parametrize(
    ("reason", "success", "cancel_requested", "state"),
    [
        ("cancelled", False, True, "canceled"),
        ("", True, False, "succeeded"),
        ("no_path", False, False, "aborted"),
    ],
)
def test_execute_ends_the_goal_by_its_result(reason, success, cancel_requested, state):
    handle = _Handle()
    handle.is_cancel_requested = cancel_requested
    seen = {}

    def plan(goal, feedback, transfer_id=None, cancel_requested=None):
        seen["transfer_id"] = transfer_id
        return SimpleNamespace(reason=reason, success=success)

    node = SimpleNamespace(_plan=plan, _log_result=lambda _r: None)
    route_planner.RoutePlanner._execute(node, handle)
    assert handle.state == state
    assert seen["transfer_id"] == route_planner._transfer_id(handle)


def _plan_node(md=None, path=None):
    return SimpleNamespace(
        _lock=threading.Lock(),
        local_frame="map",
        _publish_status=lambda _text: None,
        _resolve_mapdata=lambda _name: path,
        _load_map=lambda _path: md,
        _planner_defaults=lambda: {
            "algorithm": "graph",
            "highway_types": ["footway"],
            "spacing": 3.0,
            "max_snap_distance": 100.0,
            "cell_size": 0.25,
            "inflate_obstacles": 0.25,
            "simplify_path": True,
            "smooth_path": True,
        },
        _graph_planner=lambda _p, md, types, snap: GraphPlanner(
            md, highway_types=types, max_snap_distance=snap
        ),
        outside_map_tolerance=100.0,
        default_goal_max_snap=30.0,
    )


def _goal(*points):
    goal = route_planner.PlanRoute.Goal()
    goal.waypoints = [GeoPoint(latitude=lat, longitude=lon) for lat, lon in points]
    return goal


def test_plan_fails_busy_instead_of_waiting_for_the_running_plan():
    """A plan waiting for the lock holds an executor thread, starving cancels and fixes."""
    node = _plan_node()
    node._lock.acquire()
    result = route_planner.RoutePlanner._plan(node, _goal((50.0, 14.0), (50.0, 14.001)), None)
    assert (result.success, result.reason) == (False, "busy")
    assert node._lock.locked()  # still the running plan's


def test_plan_goal_too_far_from_any_footway_fails_with_snap_too_far(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md, _ = load_mapdata_with_annotations(path)
    start, goal = _latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 200.0, 50.0)  # 50 m off
    node = _plan_node(md, path)
    result = route_planner.RoutePlanner._plan(node, _goal(start, goal), None)
    assert (result.success, result.reason) == (False, "snap_too_far")
    assert "goal limit 30 m" in result.message
    assert not node._lock.locked()
