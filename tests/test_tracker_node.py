"""TrackerNode in a real rclpy context: config, subscription types, live reconfiguration."""

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
