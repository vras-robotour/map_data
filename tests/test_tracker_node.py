"""TrackerNode in a real rclpy context: config, subscription types, live reconfiguration."""

import time

import pytest

from map_data.viewer import ros_node
from map_data.viewer.tracker_config import SETTINGS, TrackerConfigError, topic_type

pytestmark = pytest.mark.skipif(
    not ros_node.ROS_AVAILABLE, reason="needs ROS 2 (rclpy, nav2_msgs, ...)"
)

TOPICS = [s.name for s in SETTINGS if s.name.endswith("_topic")]


@pytest.fixture
def make_node():
    # ros_node's own rclpy: test_osm_cloud replaces sys.modules["rclpy"] with a mock
    rclpy = ros_node.rclpy
    started = not rclpy.ok()
    if started:
        rclpy.init()
    nodes = []

    def make(config=None):
        nodes.append(ros_node.TrackerNode(config))
        return nodes[-1]

    yield make
    for node in nodes:
        node.destroy_node()
    if started:
        rclpy.shutdown()


def _subscribed(node):
    """Topic -> message type (``pkg/msg/Name``) of the tracker's own subscriptions."""
    out = {}
    for sub in node._tracker_subscriptions:
        pkg, kind = sub.msg_type.__module__.split(".")[:2]
        out[sub.topic_name] = f"{pkg}/{kind}/{sub.msg_type.__name__}"
    return out


def test_subscribes_with_the_types_the_web_app_shows(make_node):
    every_topic = {name: f"/t/{name}" for name in TOPICS}
    # heading_topic, battery_state_topic, temperature_topic and commander_state_topic replace
    # a fallback topic, so the fallbacks are only subscribed with those empty
    fallbacks = {
        "heading_topic": "",
        "battery_state_topic": "",
        "temperature_topic": "",
        "commander_state_topic": "",
    }
    seen = set()
    for config in ({**every_topic, "heading_type": "yaw_vector3"}, {**every_topic, **fallbacks}):
        node = make_node(config)
        for topic, msg_type in _subscribed(node).items():
            name = topic.removeprefix("/t/")
            assert msg_type == topic_type(name, node.settings()), name
            seen.add(name)
    assert seen == set(TOPICS)


def test_config_values_become_parameters(make_node):
    node = make_node({"gps_fix_topic": "/fix", "heading_type": "odometry", "trail_length": 7})
    assert node.get_parameter("gps_fix_topic").value == "/fix"
    assert node.settings()["heading_type"] == "odometry"
    assert node.trail.maxlen == 7
    assert "/fix" in _subscribed(node)


def test_apply_settings_resubscribes_and_restarts_telemetry(make_node):
    node = make_node({"path_topic": "/old_path", "goal_topic": ""})
    with node._lock:
        node.waypoints_gps = [{"lat": 50.0, "lon": 14.0}]
    node.get_telemetry()

    assert node.apply_settings({"path_topic": "", "goal_topic": "/goal"}) is True
    subscribed = _subscribed(node)
    assert "/old_path" not in subscribed
    assert subscribed["/goal"] == "geometry_msgs/msg/PoseStamped"
    assert node.get_parameter("goal_topic").value == "/goal"
    telemetry = node.get_telemetry()
    assert telemetry["enabled_features"]["path"] is False
    assert telemetry["enabled_features"]["goal"] is True
    assert telemetry["mission"]["waypoints"] == []

    assert node.apply_settings({"goal_topic": "/goal"}) is False


@pytest.mark.parametrize(
    "bad",
    [
        {"path_topic": "/1bad"},
        {"goal_topic": "/a//b"},
        {"heading_type": "gps"},
        {"trail_length": "3"},
    ],
)
def test_apply_settings_changes_nothing_when_invalid(make_node, bad):
    node = make_node({"path_topic": "/p"})
    before = _subscribed(node)
    with pytest.raises(TrackerConfigError):
        node.apply_settings({"joy_topic": "/joy", **bad})
    assert _subscribed(node) == before
    assert node.settings()["joy_topic"] == ""


def test_plan_clears_on_empty_path_and_when_the_planner_exits(make_node):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    node = make_node({"path_topic": "/plan", "goal_topic": "/goal", "earth_frame": ""})
    pub = node.create_publisher(Path, "/plan", 10)
    goal_pub = node.create_publisher(PoseStamped, "/goal", 10)
    stale = ros_node.PLAN_STALE_TIMEOUT + 1
    with node._lock:
        node.waypoints_gps = [{"lat": 50.0, "lon": 14.0}]
        node.goal_gps = {"lat": 50.0, "lon": 14.0}
        node._last_plan_time = node._last_goal_time = time.time()
    # crl_commander sends empty plans at 20 Hz between goals: one does not clear the plan
    node._path_callback(Path())
    node._drop_orphaned_plan()
    assert node.waypoints_gps and node.goal_gps
    # ... but empties persisting past PLAN_STALE_TIMEOUT do, and take a quiet goal with them
    with node._lock:
        node._last_plan_time -= stale
    node._drop_orphaned_plan()
    assert node.waypoints_gps == []
    assert node.goal_gps  # goal still republished
    with node._lock:
        node._last_goal_time -= stale
    node._drop_orphaned_plan()
    assert node.goal_gps is None

    # silence alone (nav2 publishes a plan once) never clears it
    with node._lock:
        node.waypoints_gps = [{"lat": 50.0, "lon": 14.0}]
        node._last_plan_time = node._last_empty_plan_time = 0.0
    node._drop_orphaned_plan()
    assert node.waypoints_gps  # still published
    node.destroy_publisher(pub)
    node.destroy_publisher(goal_pub)
    node._drop_orphaned_plan()
    assert node.waypoints_gps == []


def test_route_keeps_its_shape_and_long_ones_are_thinned(make_node):
    """A mission route is sparse already: drawing it must not cut its corners."""
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    node = make_node({"sequence_path_topic": "/route", "earth_frame": "", "utm_frame": "utm"})

    def route(n):
        msg = Path()
        msg.header.frame_id = "utm"  # no TF lookup needed
        for i in range(n):
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y = 500000.0 + i, 5551000.0 + i
            msg.poses.append(pose)
        return msg

    node._sequence_path_callback(route(98))  # a Stromovka-sized route: every waypoint kept
    assert len(node.sequence_gps) == 98
    node._sequence_path_callback(route(5000))
    assert 200 <= len(node.sequence_gps) <= 201  # thinned, last point kept
