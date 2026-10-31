"""route_planner's PlanRoute goal handling: cancels reach the grid planner by goal UUID."""

import uuid
from types import SimpleNamespace

import pytest

route_planner = pytest.importorskip(
    "map_data.route_planner", reason="needs ROS 2 (rclpy, map_data_interfaces, ...)"
)
from map_data.pathsolver import replan  # noqa: E402


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
