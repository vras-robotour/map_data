"""
ROS 2 side of the viewer's Tracker mode.

``TrackerNode`` subscribes to a configurable set of robot topics and turns them into one
JSON-serialisable telemetry snapshot (``get_telemetry``) that ``app.py`` pushes to the
browser over a WebSocket. Every topic is a parameter; an empty topic disables the feature
and hides its UI row.

Geometry (paths, goals) may arrive in any TF frame. With ``earth_frame`` set (e.g.
``FP_ECEF`` on a Fixposition stack) poses are transformed into that ECEF frame through TF
and converted to lat/lon exactly; with ``earth_frame`` empty the legacy behaviour
(transform into ``utm_frame`` and ``utm.to_latlon``) is used.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import utm

from ..utils.geodesy import ecef_to_latlon_array

try:
    import rclpy
    import rclpy.duration
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from geometry_msgs.msg import PoseArray, PoseStamped, TwistStamped, Vector3Stamped
    from nav2_msgs.action import (
        FollowGPSWaypoints,
        FollowWaypoints,
        NavigateThroughPoses,
    )
    from nav2_msgs.msg import BehaviorTreeLog, CollisionMonitorState, SpeedLimit
    from nav_msgs.msg import Odometry, Path
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from ros2_numpy import numpify
    from sensor_msgs.msg import BatteryState, Imu, Joy, NavSatFix, Temperature
    from std_msgs.msg import Bool, Float32, Header, String, UInt64
    from tf2_ros import TransformException
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

RECOVERY_TIMEOUT = 5.0
TELEOP_TIMEOUT = 2.0
PATH_SUBSAMPLE = 10  # keep every n-th pose of long paths (plus the last one)
ROAD_PATH_SUBSAMPLE = 5

# Ordered like nav2's SpeedLimit / CollisionMonitor enums
_COLLISION_ACTIONS: dict[int, str] = {0: "STOP", 1: "SLOWDOWN", 2: "LIMIT", 3: "PASSTHROUGH"}


def heading_from_yaw(yaw_rad: float) -> float:
    """ENU yaw (rad, CCW from east) -> compass heading (deg, CW from north)."""
    return round((90.0 - math.degrees(yaw_rad)) % 360.0, 1)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def xyz_to_latlon(xyz: np.ndarray, to_ecef: np.ndarray | None) -> list[dict[str, float]]:
    """
    Points (N, 3) in some frame -> ``[{"lat", "lon"}, ...]``.

    ``to_ecef`` is the 4x4 matrix mapping the points' frame into ECEF (``None`` when the
    points already are ECEF).
    """
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return []
    if to_ecef is not None:
        pts = pts @ to_ecef[:3, :3].T + to_ecef[:3, 3]
    lla = ecef_to_latlon_array(pts)
    return [{"lat": float(lat), "lon": float(lon)} for lat, lon, _ in lla]


def subsample(points: list[Any], every: int) -> list[Any]:
    if len(points) <= every:
        return list(points)
    out = points[::every]
    if points[-1] is not out[-1]:
        out.append(points[-1])
    return out


class TrackerNode(Node if ROS_AVAILABLE else object):  # type: ignore[misc] # dynamic base class depending on optional rclpy availability; not statically resolvable
    _TOPIC_DEFAULTS: ClassVar[dict[str, str]] = {
        # --- position / heading
        "gps_fix_topic": "/gps/fix",
        "gps_filtered_topic": "/gps/filtered",
        "heading_topic": "",  # see heading_type
        "azimuth_topic": "/gps/azimuth_imu",  # legacy alias: heading_topic + heading_type imu
        "odom_topic": "/odom_2d",
        # --- hardware
        "bus_voltage_topic": "/bus_voltage",
        "bus_current_topic": "/bus_current",
        "battery_state_topic": "",
        "teensy_temp_topic": "/teensy_temp",
        "temperature_topic": "",
        "odrv_error_topic": "/odrv_error",
        "motors_enabled_topic": "/motors_enabled",
        "estop_topic": "",
        "diagnostics_topic": "",
        # --- navigation
        "speed_limit_topic": "/speed_limit",
        "collision_monitor_state_topic": "/collision_monitor_state",
        "recovery_heartbeat_topic": "/recovery/heartbeat",
        "bt_log_topic": "/behavior_tree_log",
        "commander_state_topic": "",
        "follower_state_topic": "",
        "teleop_topic": "/cmd_vel_teleop",
        "joy_topic": "",
        "speak_info_topic": "/speak/info",
        "speak_warn_topic": "/speak/warn",
        "speak_error_topic": "/speak/err",
        # --- geometry drawn on the map
        "path_topic": "/path",
        "goal_topic": "",
        "sequence_path_topic": "",
        "sequence_poses_topic": "",
        "road_path_topic": "",
        "nav_through_poses_feedback_topic": "/navigate_through_poses/_action/feedback",
        "follow_gps_waypoints_feedback_topic": "/follow_gps_waypoints/_action/feedback",
        "follow_waypoints_feedback_topic": "/follow_waypoints/_action/feedback",
    }

    def __init__(self) -> None:
        if not ROS_AVAILABLE:
            return
        super().__init__("map_data_tracker")

        for name, default in self._TOPIC_DEFAULTS.items():
            self.declare_parameter(name, default)
        # "imu" (sensor_msgs/Imu orientation), "yaw_vector3" (Vector3Stamped, x = yaw rad,
        # e.g. Fixposition /fixposition/ypr) or "odometry" (nav_msgs/Odometry orientation)
        self.declare_parameter("heading_type", "imu")
        # ECEF frame for exact pose -> lat/lon conversion; empty = legacy utm_frame path
        self.declare_parameter("earth_frame", "")
        self.declare_parameter("utm_frame", "utm")
        self.declare_parameter("battery_low_voltage", 22.0)

        self.earth_frame: str = self.get_parameter("earth_frame").value
        self.utm_frame: str = self.get_parameter("utm_frame").value
        self.battery_low_voltage: float = float(self.get_parameter("battery_low_voltage").value)

        # Track which features are enabled (topic name is not empty)
        self.enabled_features: dict[str, bool] = {}

        self.current_waypoint = 0
        self.num_waypoints = 0
        self.waypoints_gps: list[dict[str, float]] = []
        self.sequence_gps: list[dict[str, float]] = []
        self.road_path_gps: list[dict[str, float]] = []
        self.goal_gps: dict[str, float] | None = None
        self.pose_gps: dict[str, float] | None = None
        self.pose_ekf: dict[str, float] | None = None
        self.current_heading: float | None = None

        self.bus_voltage: float | None = None
        self.bus_current: float | None = None
        self.battery_percentage: float | None = None
        self.motors_enabled: bool | None = None
        self.estop_active: bool | None = None
        self.gps_fix_status: int | None = None
        self.teensy_temp: float | None = None
        self.temperatures: dict[str, float] = {}
        self.speed: float | None = None

        self.motor_error = 0
        self.speed_limit: dict[str, float | bool] | None = None
        self.collision_action: str | None = None
        self._last_recovery_time = 0.0
        self._last_teleop_time = 0.0
        self.nav_state: str | None = None
        self.follower_state: str | None = None
        self.localization_state = None
        self.last_speech: dict[str, str] | None = None
        self.diagnostics: dict[str, Any] | None = None

        # Guards all state fields against concurrent access from ROS spin and broadcaster threads
        self._lock = threading.Lock()
        self._dirty = True  # start dirty so the first poll always emits

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Use Best Effort QoS for telemetry to match bags and typical sensor publishers
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Latched publishers (goal, sequence): receive the last message when starting late
        qos_latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        def subscribe_if_enabled(
            topic_param: str,
            msg_type: type,
            callback: Callable[..., Any],
            qos: int | QoSProfile = 10,
            feature_name: str | None = None,
        ) -> bool:
            topic = self.get_parameter(topic_param).value
            if topic:
                self.create_subscription(msg_type, topic, callback, qos)
                if feature_name:
                    self.enabled_features[feature_name] = True
                return True
            if feature_name:
                self.enabled_features[feature_name] = False
            return False

        # ---------------------------------------------------------------- position / heading
        subscribe_if_enabled(
            "gps_fix_topic", NavSatFix, self._gps_callback, qos_best_effort, "gps_fix"
        )
        subscribe_if_enabled(
            "gps_filtered_topic", NavSatFix, self._ekf_callback, qos_best_effort, "gps_ekf"
        )

        heading_type = self.get_parameter("heading_type").value
        heading_topic = self.get_parameter("heading_topic").value
        if heading_topic:
            if heading_type == "yaw_vector3":
                self.create_subscription(
                    Vector3Stamped, heading_topic, self._yaw_vector_callback, qos_best_effort
                )
            elif heading_type == "odometry":
                self.create_subscription(
                    Odometry, heading_topic, self._odom_heading_callback, qos_best_effort
                )
            else:
                self.create_subscription(
                    Imu, heading_topic, self._azimuth_callback, qos_best_effort
                )
            self.enabled_features["heading"] = True
        else:
            subscribe_if_enabled(
                "azimuth_topic", Imu, self._azimuth_callback, qos_best_effort, "heading"
            )

        subscribe_if_enabled(
            "odom_topic", Odometry, self._odom_speed_callback, qos_best_effort, "speed"
        )

        # ---------------------------------------------------------------- hardware
        if subscribe_if_enabled(
            "battery_state_topic",
            BatteryState,
            self._battery_state_callback,
            qos_best_effort,
            "battery",
        ):
            self.enabled_features["battery_percentage"] = True
        else:
            self.enabled_features["battery_percentage"] = False
            subscribe_if_enabled(
                "bus_voltage_topic", Float32, self._voltage_callback, qos_best_effort, "battery"
            )
            subscribe_if_enabled(
                "bus_current_topic", Float32, self._current_callback, qos_best_effort
            )
        if not subscribe_if_enabled(
            "temperature_topic", Temperature, self._temperature_callback, qos_best_effort, "temp"
        ):
            subscribe_if_enabled(
                "teensy_temp_topic", Float32, self._temp_callback, qos_best_effort, "temp"
            )
        subscribe_if_enabled(
            "odrv_error_topic", UInt64, self._odrv_error_callback, qos_best_effort, "motor_error"
        )
        subscribe_if_enabled(
            "motors_enabled_topic", Bool, self._motors_callback, qos_best_effort, "motors"
        )
        subscribe_if_enabled("estop_topic", Bool, self._estop_callback, qos_best_effort, "estop")
        subscribe_if_enabled(
            "diagnostics_topic",
            DiagnosticArray,
            self._diagnostics_callback,
            qos_best_effort,
            "diagnostics",
        )

        # ---------------------------------------------------------------- navigation state
        subscribe_if_enabled(
            "speed_limit_topic",
            SpeedLimit,
            self._speed_limit_callback,
            qos_best_effort,
            "speed_limit",
        )
        subscribe_if_enabled(
            "collision_monitor_state_topic",
            CollisionMonitorState,
            self._collision_callback,
            qos_best_effort,
            "collision",
        )
        subscribe_if_enabled(
            "recovery_heartbeat_topic", Header, self._recovery_callback, qos_best_effort, "recovery"
        )
        if not subscribe_if_enabled(
            "commander_state_topic", String, self._commander_state_callback, 10, "nav_state"
        ):
            subscribe_if_enabled(
                "bt_log_topic", BehaviorTreeLog, self._bt_callback, qos_best_effort, "nav_state"
            )
        subscribe_if_enabled(
            "follower_state_topic",
            String,
            self._follower_state_callback,
            qos_latched,
            "follower_state",
        )
        teleop = subscribe_if_enabled(
            "teleop_topic", TwistStamped, self._teleop_callback, qos_best_effort, "teleop"
        )
        if subscribe_if_enabled("joy_topic", Joy, self._joy_callback, qos_best_effort) or teleop:
            self.enabled_features["teleop"] = True

        if subscribe_if_enabled(
            "speak_info_topic",
            String,
            lambda m: self._speech_callback(m, "info"),
            qos_best_effort,
            "speech",
        ):
            subscribe_if_enabled(
                "speak_warn_topic",
                String,
                lambda m: self._speech_callback(m, "warn"),
                qos_best_effort,
            )
            subscribe_if_enabled(
                "speak_error_topic",
                String,
                lambda m: self._speech_callback(m, "error"),
                qos_best_effort,
            )

        # ---------------------------------------------------------------- geometry
        subscribe_if_enabled("path_topic", Path, self._path_callback, 10, "path")
        # crl_commander publishes commander/goal volatile; a volatile subscriber also matches
        # latched publishers (e.g. /goal_waypoint), only without history.
        subscribe_if_enabled("goal_topic", PoseStamped, self._goal_callback, 10, "goal")
        seq = subscribe_if_enabled(
            "sequence_path_topic", Path, self._sequence_path_callback, qos_latched, "sequence"
        )
        if (
            subscribe_if_enabled(
                "sequence_poses_topic", PoseArray, self._sequence_poses_callback, qos_latched
            )
            or seq
        ):
            self.enabled_features["sequence"] = True
        subscribe_if_enabled("road_path_topic", Path, self._road_path_callback, 10, "road_path")

        subscribe_if_enabled(
            "nav_through_poses_feedback_topic",
            NavigateThroughPoses.Feedback,
            self._feedback_callback,
            10,
            "actions",
        )
        subscribe_if_enabled(
            "follow_gps_waypoints_feedback_topic",
            FollowGPSWaypoints.Feedback,
            self._feedback_callback,
            10,
        )
        subscribe_if_enabled(
            "follow_waypoints_feedback_topic", FollowWaypoints.Feedback, self._feedback_callback, 10
        )

        enabled = [k for k, v in self.enabled_features.items() if v]
        self.get_logger().info(
            f"Tracker ready: earth_frame={self.earth_frame!r}, utm_frame={self.utm_frame!r}, "
            f"enabled={enabled}"
        )

    # ------------------------------------------------------------------ telemetry snapshot
    def _build_status_locked(self) -> dict[str, Any]:
        """
        Build status snapshot. Caller must hold self._lock.
        """
        now = time.time()
        temp_max = None
        if self.temperatures:
            name, value = max(self.temperatures.items(), key=lambda kv: kv[1])
            temp_max = {"name": name, "value": value}
        return {
            "battery": {
                "voltage": self.bus_voltage,
                "current": self.bus_current,
                "percentage": self.battery_percentage,
                "low_voltage": self.battery_low_voltage,
            },
            "motors_enabled": self.motors_enabled,
            "estop_active": self.estop_active,
            "motor_error": self.motor_error,
            "gps_fix": self.gps_fix_status,
            "teensy_temp": self.teensy_temp,
            "temp_max": temp_max,
            "temperatures": dict(self.temperatures),
            "speed": self.speed,
            "speed_limit": dict(self.speed_limit) if self.speed_limit else None,
            "collision_action": self.collision_action,
            "recovery_active": self._last_recovery_time > 0
            and (now - self._last_recovery_time) < RECOVERY_TIMEOUT,
            "teleop_active": self._last_teleop_time > 0
            and (now - self._last_teleop_time) < TELEOP_TIMEOUT,
            "nav_state": self.nav_state,
            "follower_state": self.follower_state,
            "localization_state": self.localization_state,
            "last_speech": dict(self.last_speech) if self.last_speech else None,
            "diagnostics": dict(self.diagnostics) if self.diagnostics else None,
        }

    def get_telemetry(self) -> dict[str, Any] | None:
        if not ROS_AVAILABLE:
            return None
        with self._lock:
            if not self._dirty:
                return None
            self._dirty = False
            return {
                "enabled_features": self.enabled_features,
                "position": {
                    "gps": dict(self.pose_gps) if self.pose_gps else {},
                    "ekf": dict(self.pose_ekf) if self.pose_ekf else {},
                    "goal": dict(self.goal_gps) if self.goal_gps else None,
                },
                "mission": {
                    "waypoints": list(self.waypoints_gps),
                    "current_waypoint_index": self.current_waypoint,
                    "sequence": list(self.sequence_gps),
                    "road_path": list(self.road_path_gps),
                },
                "status": self._build_status_locked(),
            }

    # ------------------------------------------------------------------ geometry -> lat/lon
    def _lookup_matrix(self, target: str, source: str) -> np.ndarray | None:
        """Latest 4x4 ``target <- source`` transform, or None (non-blocking)."""
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time(), rclpy.duration.Duration(seconds=0)
            )
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {target} <- {source}: {ex}", throttle_duration_sec=5.0
            )
            return None
        return numpify(tf_msg.transform)

    def _points_to_latlon(self, frame_id: str, xyz: np.ndarray) -> list[dict[str, float]] | None:
        """Points in ``frame_id`` -> lat/lon list, or None if the TF is not available yet."""
        if not len(xyz):
            return []
        if self.earth_frame:
            if frame_id == self.earth_frame:
                return xyz_to_latlon(xyz, None)
            m = self._lookup_matrix(self.earth_frame, frame_id)
            return None if m is None else xyz_to_latlon(xyz, m)

        # Legacy: UTM frame in TF, zone inferred from the current fix
        if frame_id != self.utm_frame:
            m = self._lookup_matrix(self.utm_frame, frame_id)
            if m is None:
                return None
            xyz = xyz @ m[:3, :3].T + m[:3, 3]
        with self._lock:
            ref_pos = self.pose_gps
        if ref_pos:
            _, _, zone_number, zone_letter = utm.from_latlon(ref_pos["lat"], ref_pos["lon"])
        else:
            zone_number, zone_letter = 33, "U"
        lat, lon = utm.to_latlon(xyz[:, 0], xyz[:, 1], zone_number, zone_letter, strict=False)
        return [{"lat": float(a), "lon": float(b)} for a, b in zip(lat, lon, strict=True)]

    @staticmethod
    def _poses_xyz(poses: list[Any]) -> np.ndarray:
        return np.array(
            [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in poses], dtype=float
        ).reshape(-1, 3)

    def _convert_path_latlon(
        self,
        msg: Path | NavigateThroughPoses.Goal | FollowWaypoints.Goal | FollowGPSWaypoints.Goal,
    ) -> list[dict[str, float]] | None:
        if isinstance(msg, FollowGPSWaypoints.Goal):
            return [
                {"lat": pose.position.latitude, "lon": pose.position.longitude}
                for pose in msg.gps_poses
            ]
        if isinstance(msg, Path):
            poses = msg.poses
        elif isinstance(msg, NavigateThroughPoses.Goal):
            poses = msg.poses.goals
        elif isinstance(msg, FollowWaypoints.Goal):
            poses = msg.poses
        else:
            return None
        return self._points_to_latlon(msg.header.frame_id, self._poses_xyz(poses))

    # ------------------------------------------------------------------ callbacks: position
    def _gps_callback(self, msg: NavSatFix) -> None:
        with self._lock:
            first = self.pose_gps is None
            self.gps_fix_status = int(msg.status.status)
            self.pose_gps = {"lat": msg.latitude, "lon": msg.longitude}
            if self.current_heading is not None:
                self.pose_gps["heading"] = self.current_heading
            self._dirty = True
        if first:
            self.get_logger().info(f"First GPS fix received: {msg.latitude}, {msg.longitude}")

    def _ekf_callback(self, msg: NavSatFix) -> None:
        with self._lock:
            first = self.pose_ekf is None
            self.pose_ekf = {"lat": msg.latitude, "lon": msg.longitude}
            if self.current_heading is not None:
                self.pose_ekf["heading"] = self.current_heading
            self._dirty = True
        if first:
            self.get_logger().info(f"First EKF pose received: {msg.latitude}, {msg.longitude}")

    def _set_heading(self, heading: float) -> None:
        with self._lock:
            self.current_heading = heading
            # Keep heading in sync with existing pose dicts immediately
            if self.pose_gps is not None:
                self.pose_gps["heading"] = heading
            if self.pose_ekf is not None:
                self.pose_ekf["heading"] = heading
            self._dirty = True

    def _azimuth_callback(self, msg: Imu) -> None:
        q = msg.orientation
        self._set_heading(heading_from_yaw(yaw_from_quaternion(q.x, q.y, q.z, q.w)))

    def _yaw_vector_callback(self, msg: Vector3Stamped) -> None:
        self._set_heading(heading_from_yaw(msg.vector.x))

    def _odom_heading_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self._set_heading(heading_from_yaw(yaw_from_quaternion(q.x, q.y, q.z, q.w)))

    def _odom_speed_callback(self, msg: Odometry) -> None:
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        with self._lock:
            self.speed = round(math.sqrt(vx * vx + vy * vy), 2)
            self._dirty = True

    # ------------------------------------------------------------------ callbacks: geometry
    def _path_callback(self, msg: Path) -> None:
        waypoints = self._convert_path_latlon(msg)
        if waypoints:
            with self._lock:
                self.waypoints_gps = subsample(waypoints, PATH_SUBSAMPLE)
                self.num_waypoints = len(msg.poses)
                self._dirty = True

    def _goal_callback(self, msg: PoseStamped) -> None:
        pts = self._points_to_latlon(msg.header.frame_id, self._poses_xyz([msg]))
        if pts:
            with self._lock:
                self.goal_gps = pts[0]
                self._dirty = True

    def _sequence_path_callback(self, msg: Path) -> None:
        pts = self._points_to_latlon(msg.header.frame_id, self._poses_xyz(msg.poses))
        if pts is not None:
            with self._lock:
                self.sequence_gps = pts
                self._dirty = True

    def _sequence_poses_callback(self, msg: PoseArray) -> None:
        xyz = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses]).reshape(
            -1, 3
        )
        pts = self._points_to_latlon(msg.header.frame_id, xyz)
        if pts is not None:
            with self._lock:
                self.sequence_gps = pts
                self._dirty = True

    def _road_path_callback(self, msg: Path) -> None:
        pts = self._points_to_latlon(msg.header.frame_id, self._poses_xyz(msg.poses))
        if pts is not None:
            with self._lock:
                self.road_path_gps = subsample(pts, ROAD_PATH_SUBSAMPLE)
                self._dirty = True

    def _feedback_callback(
        self,
        msg: NavigateThroughPoses.Feedback | FollowWaypoints.Feedback | FollowGPSWaypoints.Feedback,
    ) -> None:
        with self._lock:
            if isinstance(msg, NavigateThroughPoses.Feedback):
                self.current_waypoint = self.num_waypoints - msg.number_of_poses_remaining
            else:
                self.current_waypoint = msg.current_waypoint
            self._dirty = True

    # ------------------------------------------------------------------ callbacks: hardware
    def _voltage_callback(self, msg: Float32) -> None:
        with self._lock:
            self.bus_voltage = round(float(msg.data), 2)
            self._dirty = True

    def _current_callback(self, msg: Float32) -> None:
        with self._lock:
            self.bus_current = round(float(msg.data), 2)
            self._dirty = True

    def _battery_state_callback(self, msg: BatteryState) -> None:
        with self._lock:
            self.bus_voltage = round(float(msg.voltage), 2) if math.isfinite(msg.voltage) else None
            self.bus_current = round(float(msg.current), 2) if math.isfinite(msg.current) else None
            # sensor_msgs says 0..1; some drivers publish 0..100 (Helhest does) - normalise
            pct = float(msg.percentage)
            if math.isfinite(pct):
                self.battery_percentage = round(pct * 100.0 if pct <= 1.0 else pct, 1)
            else:
                self.battery_percentage = None
            self._dirty = True

    def _motors_callback(self, msg: Bool) -> None:
        with self._lock:
            self.motors_enabled = bool(msg.data)
            self._dirty = True

    def _estop_callback(self, msg: Bool) -> None:
        with self._lock:
            changed = self.estop_active != bool(msg.data)
            self.estop_active = bool(msg.data)
            self._dirty = self._dirty or changed

    def _temp_callback(self, msg: Float32) -> None:
        with self._lock:
            self.teensy_temp = round(float(msg.data), 1)
            self._dirty = True

    def _temperature_callback(self, msg: Temperature) -> None:
        with self._lock:
            self.temperatures[msg.header.frame_id or "temperature"] = round(
                float(msg.temperature), 1
            )
            self._dirty = True

    def _odrv_error_callback(self, msg: UInt64) -> None:
        with self._lock:
            self.motor_error = int(msg.data)
            self._dirty = True

    def _diagnostics_callback(self, msg: DiagnosticArray) -> None:
        errors = [s for s in msg.status if s.level >= DiagnosticStatus.ERROR]
        warnings = [s for s in msg.status if s.level == DiagnosticStatus.WARN]
        worst = errors[0] if errors else (warnings[0] if warnings else None)
        summary = {
            "errors": len(errors),
            "warnings": len(warnings),
            "worst": {"name": worst.name, "message": worst.message} if worst else None,
        }
        with self._lock:
            if summary != self.diagnostics:
                self.diagnostics = summary
                self._dirty = True

    # ------------------------------------------------------------------ callbacks: navigation
    def _speed_limit_callback(self, msg: SpeedLimit) -> None:
        with self._lock:
            self.speed_limit = {
                "value": round(msg.speed_limit, 2),
                "percentage": bool(msg.percentage),
            }
            self._dirty = True

    def _collision_callback(self, msg: CollisionMonitorState) -> None:
        with self._lock:
            self.collision_action = _COLLISION_ACTIONS.get(msg.action_type, str(msg.action_type))
            self._dirty = True

    def _recovery_callback(self, _msg: Header) -> None:
        with self._lock:
            self._last_recovery_time = time.time()
            self._dirty = True

    def _teleop_callback(self, msg: TwistStamped) -> None:
        lv, av = msg.twist.linear, msg.twist.angular
        if any(v != 0.0 for v in (lv.x, lv.y, lv.z, av.x, av.y, av.z)):
            with self._lock:
                self._last_teleop_time = time.time()
                self._dirty = True

    def _joy_callback(self, msg: Joy) -> None:
        if any(abs(a) > 0.05 for a in msg.axes) or any(msg.buttons):
            with self._lock:
                self._last_teleop_time = time.time()
                self._dirty = True

    def _bt_callback(self, msg: BehaviorTreeLog) -> None:
        running = [e for e in msg.event_log if e.current_status == "RUNNING"]
        nav_state = running[-1].node_name if running else None
        with self._lock:
            self.nav_state = nav_state
            self._dirty = True

    def _commander_state_callback(self, msg: String) -> None:
        with self._lock:
            if msg.data != self.nav_state:
                self.nav_state = msg.data
                self._dirty = True

    def _follower_state_callback(self, msg: String) -> None:
        with self._lock:
            if msg.data != self.follower_state:
                self.follower_state = msg.data
                self._dirty = True

    def _speech_callback(self, msg: String, level: str) -> None:
        with self._lock:
            self.last_speech = {"level": level, "text": msg.data}
            self._dirty = True
