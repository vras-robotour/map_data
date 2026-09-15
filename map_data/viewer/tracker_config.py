"""
Topic configuration of the viewer's Tracker (:class:`map_data.viewer.ros_node.TrackerNode`).

ROS-free, so the web API and the tests use it without a ROS 2 context. The config file is an
ordinary ROS 2 parameter file (parameters under ``map_data_tracker`` / ``/map_data_tracker``,
on top of ``/**``), so the same file works with ``map_data_viewer --config`` and with
``--ros-args --params-file``.

:func:`save_tracker_config` edits that file line by line, so its comments and layout survive
a save from the web app.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

NODE_NAME = "map_data_tracker"
DEFAULT_CONFIG_NAME = "tracker.yaml"

# heading_type -> message type expected on heading_topic
HEADING_TYPES: dict[str, str] = {
    "imu": "sensor_msgs/msg/Imu",  # orientation quaternion
    "yaw_vector3": "geometry_msgs/msg/Vector3Stamped",  # x = yaw rad, e.g. /fixposition/ypr
    "odometry": "nav_msgs/msg/Odometry",  # pose orientation
}


class TrackerConfigError(ValueError):
    """A tracker config file or setting that cannot be used."""


@dataclass(frozen=True)
class Setting:
    """One tracker parameter the web app can change at runtime."""

    name: str
    default: str
    section: str
    description: str
    msg_type: str = ""  # topics: the type TrackerNode subscribes with ("" = see heading_type)
    choices: tuple[str, ...] = ()


_POS = "Position / heading"
_HW = "Hardware"
_NAV = "Navigation state"
_GEO = "Map geometry"

# In the order the web app shows them
SETTINGS: tuple[Setting, ...] = (
    Setting("earth_frame", "", "Frames", "ECEF TF frame for exact pose conversion (FP_ECEF); empty = UTM fallback"),
    Setting("utm_frame", "utm", "Frames", "UTM TF frame used when earth_frame is empty"),
    Setting("gps_fix_topic", "/gps/fix", _POS, "Raw GPS fix", "sensor_msgs/msg/NavSatFix"),
    Setting("gps_filtered_topic", "/gps/filtered", _POS, "EKF-fused GPS position", "sensor_msgs/msg/NavSatFix"),
    Setting("heading_type", "imu", _POS, "Message type on heading_topic", choices=tuple(HEADING_TYPES)),
    Setting("heading_topic", "", _POS, "Heading source, typed by heading_type"),
    Setting("azimuth_topic", "/gps/azimuth_imu", _POS, "Legacy IMU heading, used when heading_topic is empty", "sensor_msgs/msg/Imu"),
    Setting("odom_topic", "/odom_2d", _POS, "Odometry for the speed row", "nav_msgs/msg/Odometry"),
    Setting("battery_state_topic", "", _HW, "Voltage, current and percentage; replaces the two Float32 topics", "sensor_msgs/msg/BatteryState"),
    Setting("bus_voltage_topic", "/bus_voltage", _HW, "Battery voltage", "std_msgs/msg/Float32"),
    Setting("bus_current_topic", "/bus_current", _HW, "Battery current", "std_msgs/msg/Float32"),
    Setting("temperature_topic", "", _HW, "Temperatures, one per frame_id; replaces teensy_temp_topic", "sensor_msgs/msg/Temperature"),
    Setting("teensy_temp_topic", "/teensy_temp", _HW, "Controller temperature", "std_msgs/msg/Float32"),
    Setting("odrv_error_topic", "/odrv_error", _HW, "ODrive error code", "std_msgs/msg/UInt64"),
    Setting("motors_enabled_topic", "/motors_enabled", _HW, "Motor enable state", "std_msgs/msg/Bool"),
    Setting("estop_topic", "", _HW, "Emergency stop state", "std_msgs/msg/Bool"),
    Setting("diagnostics_topic", "", _HW, "Diagnostics summary", "diagnostic_msgs/msg/DiagnosticArray"),
    Setting("commander_state_topic", "", _NAV, "Navigation state string (crl_commander); replaces bt_log_topic", "std_msgs/msg/String"),
    Setting("bt_log_topic", "/behavior_tree_log", _NAV, "Nav2 behavior tree log", "nav2_msgs/msg/BehaviorTreeLog"),
    Setting("follower_state_topic", "", _NAV, "road_follower state (latched)", "std_msgs/msg/String"),
    Setting("speed_limit_topic", "/speed_limit", _NAV, "Active speed limit", "nav2_msgs/msg/SpeedLimit"),
    Setting("collision_monitor_state_topic", "/collision_monitor_state", _NAV, "Collision monitor state", "nav2_msgs/msg/CollisionMonitorState"),
    Setting("recovery_heartbeat_topic", "/recovery/heartbeat", _NAV, "Recovery behavior heartbeat", "std_msgs/msg/Header"),
    Setting("teleop_topic", "/cmd_vel_teleop", _NAV, "Teleop velocity command", "geometry_msgs/msg/TwistStamped"),
    Setting("joy_topic", "", _NAV, "Joystick; any input counts as teleop", "sensor_msgs/msg/Joy"),
    Setting("speak_info_topic", "/speak/info", _NAV, "Info speech (the warn/error topics need it set)", "std_msgs/msg/String"),
    Setting("speak_warn_topic", "/speak/warn", _NAV, "Warning speech", "std_msgs/msg/String"),
    Setting("speak_error_topic", "/speak/err", _NAV, "Error speech", "std_msgs/msg/String"),
    Setting("path_topic", "/path", _GEO, "Planned path", "nav_msgs/msg/Path"),
    Setting("goal_topic", "", _GEO, "Current navigation goal", "geometry_msgs/msg/PoseStamped"),
    Setting("sequence_path_topic", "", _GEO, "Waypoint sequence (latched)", "nav_msgs/msg/Path"),
    Setting("sequence_poses_topic", "", _GEO, "Waypoint window (latched)", "geometry_msgs/msg/PoseArray"),
    Setting("road_path_topic", "", _GEO, "Visual road-following path", "nav_msgs/msg/Path"),
    Setting("intersections_topic", "", _GEO, "OSM intersections (latched)", "geometry_msgs/msg/PoseArray"),
    Setting("active_intersection_topic", "", _GEO, "Intersection that triggered GPS mode (latched)", "geometry_msgs/msg/PoseStamped"),
    Setting("nav_through_poses_feedback_topic", "/navigate_through_poses/_action/feedback", _GEO, "Nav2 NavigateThroughPoses feedback", "nav2_msgs/action/NavigateThroughPoses_Feedback"),
    Setting("follow_gps_waypoints_feedback_topic", "/follow_gps_waypoints/_action/feedback", _GEO, "Nav2 FollowGPSWaypoints feedback", "nav2_msgs/action/FollowGPSWaypoints_Feedback"),
    Setting("follow_waypoints_feedback_topic", "/follow_waypoints/_action/feedback", _GEO, "Nav2 FollowWaypoints feedback", "nav2_msgs/action/FollowWaypoints_Feedback"),
)  # fmt: skip

SETTING_DEFAULTS: dict[str, str] = {s.name: s.default for s in SETTINGS}

# Parameters only the config file sets (not editable in the web app)
TUNING_DEFAULTS: dict[str, float | int] = {
    "battery_low_voltage": 22.0,
    "stale_after": 3.0,
    "trail_length": 500,
    "trail_min_step": 0.5,
    "intersection_enter_threshold": 5.0,
    "intersection_exit_threshold": 6.0,
}

DEFAULTS: dict[str, Any] = {**SETTING_DEFAULTS, **TUNING_DEFAULTS}


def settings_metadata() -> list[dict[str, Any]]:
    """The editable settings as JSON-ready dicts, in display order."""
    return [{**asdict(s), "choices": list(s.choices)} for s in SETTINGS]


def topic_type(name: str, settings: Mapping[str, str]) -> str:
    """Message type TrackerNode subscribes to setting ``name`` with."""
    if name == "heading_topic":
        return HEADING_TYPES.get(settings.get("heading_type", ""), "")
    return next((s.msg_type for s in SETTINGS if s.name == name), "")


def validate_settings(values: Any) -> dict[str, str]:
    """
    Check web-app settings: known names, whitespace-free strings, a valid ``heading_type``.

    Returns the values stripped. Topic *names* are checked by the node, which knows the
    ROS naming rules.

    Raises
    ------
    TrackerConfigError
        On the first problem found.

    """
    if not isinstance(values, Mapping):
        raise TrackerConfigError("settings must be an object of name: value")
    unknown = sorted(set(values) - set(SETTING_DEFAULTS))
    if unknown:
        raise TrackerConfigError(f"unknown settings: {', '.join(unknown)}")
    out = {}
    for name, value in values.items():
        if not isinstance(value, str):
            raise TrackerConfigError(f"{name} must be a string")
        value = value.strip()
        if any(c.isspace() for c in value):
            raise TrackerConfigError(f"{name} must not contain whitespace: {value!r}")
        out[name] = value
    if "heading_type" in out and out["heading_type"] not in HEADING_TYPES:
        raise TrackerConfigError(f"heading_type must be one of {', '.join(HEADING_TYPES)}")
    return out


def node_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    All tracker parameters: :data:`DEFAULTS` overridden by ``config``, typed like the defaults.

    Unrelated keys in ``config`` (e.g. ``use_sim_time`` from ``/**``) are ignored; a YAML
    null topic means "disabled".

    Raises
    ------
    TrackerConfigError
        If a value has the wrong type or an editable setting is invalid.

    """
    params = dict(DEFAULTS)
    for name, default in DEFAULTS.items():
        if name not in config:
            continue
        value = config[name]
        if isinstance(default, str):
            value = "" if value is None else value
            ok = isinstance(value, str)
        elif isinstance(default, float):
            ok = isinstance(value, int | float) and not isinstance(value, bool)
            value = float(value) if ok else value
        else:
            ok = isinstance(value, int) and not isinstance(value, bool)
        if not ok:
            raise TrackerConfigError(f"{name} must be a {type(default).__name__}, got {value!r}")
        params[name] = value
    params.update(validate_settings({k: params[k] for k in SETTING_DEFAULTS}))
    return params


def _parameters_in(doc: Any, source: str) -> dict[str, Any]:
    """The tracker's ``ros__parameters`` of a parsed parameter file (``/**`` first)."""
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise TrackerConfigError(f"{source}: expected a ROS 2 parameter file mapping")
    merged: dict[str, Any] = {}
    for key in ("/**", NODE_NAME, f"/{NODE_NAME}"):
        section = doc.get(key)
        if section is None:
            continue
        params = section.get("ros__parameters") if isinstance(section, dict) else None
        if not isinstance(params, dict):
            raise TrackerConfigError(f"{source}: {key} has no ros__parameters mapping")
        merged.update(params)
    return merged


def load_tracker_config(path: str | Path) -> dict[str, Any]:
    """
    The tracker parameters a config file sets (``{}`` if the file does not exist).

    Raises
    ------
    TrackerConfigError
        If the file is not valid YAML or not shaped like a parameter file.

    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise TrackerConfigError(f"{path}: {e}") from e
    return _parameters_in(doc, str(path))


# ---------------------------------------------------------------------- comment-preserving save
_NODE_HEADER = re.compile(rf"^/?{NODE_NAME}:\s*(#.*)?$")
_ROS_PARAMETERS = re.compile(r"^\s+ros__parameters:\s*(#.*)?$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_content(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _split_comment(rest: str) -> tuple[str, str]:
    """``'"/x"   # note'`` -> ``('"/x"', '# note')``; a ``#`` inside quotes is not a comment."""
    quote = None
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1].isspace()):
            return rest[:i].rstrip(), rest[i:]
    return rest.rstrip(), ""


def _set_parameter_line(lines: list[str], name: str, value: Any) -> list[str]:
    """Set ``name: value`` in the tracker's ``ros__parameters`` block, adding what is missing."""
    scalar = json.dumps(value)  # a JSON scalar is a valid YAML flow scalar
    lines = list(lines)
    header = next((i for i, line in enumerate(lines) if _NODE_HEADER.match(line)), None)
    if header is None:
        if lines and lines[-1].strip():
            lines.append("")
        return [*lines, f"{NODE_NAME}:", "  ros__parameters:", f"    {name}: {scalar}"]

    # The node block ends at the next top-level key
    end = next(
        (i for i in range(header + 1, len(lines)) if _is_content(lines[i]) and not _indent(lines[i])),
        len(lines),
    )
    rp = next((i for i in range(header + 1, end) if _ROS_PARAMETERS.match(lines[i])), None)
    if rp is None:
        lines.insert(header + 1, "  ros__parameters:")
        rp, end = header + 1, end + 1
    rp_indent = _indent(lines[rp])
    rp_end = next(
        (i for i in range(rp + 1, end) if _is_content(lines[i]) and _indent(lines[i]) <= rp_indent),
        end,
    )
    params = [i for i in range(rp + 1, rp_end) if _is_content(lines[i])]
    param_indent = _indent(lines[params[0]]) if params else rp_indent + 2

    key = re.compile(rf"^(\s*){re.escape(name)}:(\s*)(.*)$")
    for i in params:
        m = key.match(lines[i])
        if not m or len(m.group(1)) != param_indent:
            continue
        _, comment = _split_comment(m.group(3))
        new = f"{m.group(1)}{name}:{m.group(2) or ' '}{scalar}"
        if comment:
            # Keep the comment in its column when the value fits
            column = len(lines[i]) - len(comment)
            new += " " * max(1, column - len(new)) + comment
        lines[i] = new
        return lines

    lines.insert(params[-1] + 1 if params else rp + 1, f"{' ' * param_indent}{name}: {scalar}")
    return lines


def save_tracker_config(path: str | Path, settings: Mapping[str, Any]) -> Path:
    """
    Write ``settings`` into a tracker config file, keeping its comments and layout.

    Only values that differ from what the file already yields are touched; missing keys
    (or the whole ``map_data_tracker`` block, or the file) are added. A symlink is written
    through, so saving via a ``colcon --symlink-install`` share directory updates the
    source file. Returns the path written.

    Raises
    ------
    TrackerConfigError
        If the file is malformed or the edit would not produce the requested values.
    OSError
        If the file cannot be written.

    """
    target = Path(path).resolve()
    text = target.read_text() if target.is_file() else ""
    try:
        current = node_parameters(_parameters_in(yaml.safe_load(text), str(target)))
    except yaml.YAMLError as e:
        raise TrackerConfigError(f"{target}: {e}") from e

    lines = text.splitlines()
    changed = [name for name, value in settings.items() if current.get(name) != value]
    for name in changed:
        lines = _set_parameter_line(lines, name, settings[name])
    if not changed:
        return target
    new_text = "\n".join(lines) + "\n"

    try:
        written = node_parameters(_parameters_in(yaml.safe_load(new_text), str(target)))
    except yaml.YAMLError as e:
        raise TrackerConfigError(f"editing {target} broke its YAML ({e}); edit it by hand") from e
    wrong = [name for name, value in settings.items() if written.get(name) != value]
    if wrong:
        raise TrackerConfigError(f"could not set {', '.join(wrong)} in {target}; edit it by hand")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(new_text)
    tmp.replace(target)
    return target
