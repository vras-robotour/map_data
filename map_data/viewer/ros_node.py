from __future__ import annotations

import math
import threading
import time

import utm

try:
    import rclpy
    import rclpy.duration
    from geometry_msgs.msg import TwistStamped
    from nav2_msgs.action import (
        FollowGPSWaypoints,
        FollowWaypoints,
        NavigateThroughPoses,
    )
    from nav2_msgs.msg import BehaviorTreeLog, CollisionMonitorState, SpeedLimit
    from nav_msgs.msg import Odometry, Path
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, NavSatFix
    from std_msgs.msg import Bool, Float32, Header, String, UInt64
    from tf2_geometry_msgs import do_transform_pose_stamped
    from tf2_ros import TransformException
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class TrackerNode(Node if ROS_AVAILABLE else object):
    def __init__(self):
        if not ROS_AVAILABLE:
            return
        super().__init__("map_data_tracker")

        self.declare_parameter("robot_id", "robot")
        self.declare_parameter("path_topic", "/path")

        # New topic parameters
        self.declare_parameter("gps_fix_topic", "/gps/fix")
        self.declare_parameter("gps_filtered_topic", "/gps/filtered")
        self.declare_parameter("bus_voltage_topic", "/bus_voltage")
        self.declare_parameter("bus_current_topic", "/bus_current")
        self.declare_parameter("teensy_temp_topic", "/teensy_temp")
        self.declare_parameter("odrv_error_topic", "/odrv_error")
        self.declare_parameter("azimuth_topic", "/gps/azimuth_imu")
        self.declare_parameter("odom_topic", "/odom_2d")
        self.declare_parameter("motors_enabled_topic", "/motors_enabled")
        self.declare_parameter("speed_limit_topic", "/speed_limit")
        self.declare_parameter("collision_monitor_state_topic", "/collision_monitor_state")
        self.declare_parameter("recovery_heartbeat_topic", "/recovery/heartbeat")
        self.declare_parameter("bt_log_topic", "/behavior_tree_log")
        self.declare_parameter("teleop_topic", "/cmd_vel_teleop")
        self.declare_parameter("speak_info_topic", "/speak/info")
        self.declare_parameter("speak_warn_topic", "/speak/warn")
        self.declare_parameter("speak_error_topic", "/speak/err")
        self.declare_parameter(
            "nav_through_poses_feedback_topic",
            "/navigate_through_poses/_action/feedback",
        )
        self.declare_parameter(
            "follow_gps_waypoints_feedback_topic",
            "/follow_gps_waypoints/_action/feedback",
        )
        self.declare_parameter(
            "follow_waypoints_feedback_topic", "/follow_waypoints/_action/feedback"
        )

        self.robot_id = self.get_parameter("robot_id").value

        # Track which features are enabled (topic name is not empty)
        self.enabled_features = {}

        self.current_waypoint = 0
        self.num_waypoints = 0
        self.waypoints_gps = []
        self.pose_gps = None
        self.pose_ekf = None
        self.current_heading = None

        self.bus_voltage = None
        self.bus_current = None
        self.motors_enabled = None
        self.gps_fix_status = None
        self.teensy_temp = None
        self.speed = None

        self.motor_error = 0
        self.speed_limit = None
        self.collision_action = None
        self._last_recovery_time = 0.0
        self._last_teleop_time = 0.0
        self.nav_state = None
        self.localization_state = None
        self.last_speech = None

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

        def subscribe_if_enabled(topic_param, msg_type, callback, qos=10, feature_name=None):
            topic = self.get_parameter(topic_param).value
            if topic:
                self.create_subscription(msg_type, topic, callback, qos)
                if feature_name:
                    self.enabled_features[feature_name] = True
                return True
            else:
                if feature_name:
                    self.enabled_features[feature_name] = False
                return False

        subscribe_if_enabled("path_topic", Path, self._path_callback, 10, "path")

        subscribe_if_enabled(
            "gps_fix_topic", NavSatFix, self._gps_callback, qos_best_effort, "gps_fix"
        )
        subscribe_if_enabled(
            "gps_filtered_topic",
            NavSatFix,
            self._ekf_callback,
            qos_best_effort,
            "gps_ekf",
        )

        subscribe_if_enabled(
            "bus_voltage_topic",
            Float32,
            self._voltage_callback,
            qos_best_effort,
            "battery",
        )
        subscribe_if_enabled("bus_current_topic", Float32, self._current_callback, qos_best_effort)
        subscribe_if_enabled(
            "teensy_temp_topic", Float32, self._temp_callback, qos_best_effort, "temp"
        )
        subscribe_if_enabled(
            "odrv_error_topic",
            UInt64,
            self._odrv_error_callback,
            qos_best_effort,
            "motor_error",
        )

        subscribe_if_enabled(
            "azimuth_topic", Imu, self._azimuth_callback, qos_best_effort, "heading"
        )
        subscribe_if_enabled(
            "odom_topic", Odometry, self._odom_speed_callback, qos_best_effort, "speed"
        )

        subscribe_if_enabled(
            "motors_enabled_topic",
            Bool,
            self._motors_callback,
            qos_best_effort,
            "motors",
        )
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
            "recovery_heartbeat_topic",
            Header,
            self._recovery_callback,
            qos_best_effort,
            "recovery",
        )
        subscribe_if_enabled(
            "bt_log_topic",
            BehaviorTreeLog,
            self._bt_callback,
            qos_best_effort,
            "nav_state",
        )

        subscribe_if_enabled(
            "teleop_topic",
            TwistStamped,
            self._teleop_callback,
            qos_best_effort,
            "teleop",
        )

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
            "follow_waypoints_feedback_topic",
            FollowWaypoints.Feedback,
            self._feedback_callback,
            10,
        )

    _COLLISION_ACTIONS = {0: "STOP", 1: "SLOWDOWN", 2: "LIMIT", 3: "PASSTHROUGH"}

    def _build_status_locked(self):
        """
        Build status snapshot. Caller must hold self._lock.
        """
        now = time.time()
        return {
            "battery": {
                "voltage": self.bus_voltage,
                "current": self.bus_current,
            },
            "motors_enabled": self.motors_enabled,
            "motor_error": self.motor_error,
            "gps_fix": self.gps_fix_status,
            "teensy_temp": self.teensy_temp,
            "speed": self.speed,
            "speed_limit": dict(self.speed_limit) if self.speed_limit else None,
            "collision_action": self.collision_action,
            "recovery_active": self._last_recovery_time > 0
            and (now - self._last_recovery_time) < 5.0,
            "teleop_active": self._last_teleop_time > 0 and (now - self._last_teleop_time) < 2.0,
            "nav_state": self.nav_state,
            "localization_state": self.localization_state,
            "last_speech": dict(self.last_speech) if self.last_speech else None,
        }

    def get_telemetry(self):
        if not ROS_AVAILABLE:
            return None
        with self._lock:
            if not self._dirty:
                return None
            self._dirty = False
            return {
                "robot_id": self.robot_id,
                "enabled_features": self.enabled_features,
                "position": {
                    "gps": dict(self.pose_gps) if self.pose_gps else {},
                    "ekf": dict(self.pose_ekf) if self.pose_ekf else {},
                },
                "mission": {
                    "waypoints": list(self.waypoints_gps),
                    "current_waypoint_index": self.current_waypoint,
                },
                "status": self._build_status_locked(),
            }

    def _convert_path_latlon(
        self,
        msg: Path | NavigateThroughPoses.Goal | FollowWaypoints.Goal | FollowGPSWaypoints.Goal,
    ):
        waypoints_gps = []
        if ROS_AVAILABLE and isinstance(msg, FollowGPSWaypoints.Goal):
            for pose in msg.gps_poses:
                waypoints_gps.append(
                    {"lat": pose.position.latitude, "lon": pose.position.longitude}
                )
        elif ROS_AVAILABLE:
            try:
                # Non-blocking: use latest cached transform rather than waiting up to 5s
                transform = self.tf_buffer.lookup_transform(
                    "utm",
                    msg.header.frame_id,
                    msg.header.stamp,
                    rclpy.duration.Duration(seconds=0),
                )
            except TransformException as ex:
                self.get_logger().info(f"Could not transform utm to {msg.header.frame_id}: {ex}")
                return None

            if isinstance(msg, Path):
                poses = msg.poses
            elif isinstance(msg, NavigateThroughPoses.Goal):
                poses = msg.poses.goals
            elif isinstance(msg, FollowWaypoints.Goal):
                poses = msg.poses
            else:
                return None

            # Infer UTM zone from current GPS fix instead of hardcoding zone 33U
            with self._lock:
                ref_pos = self.pose_gps
            if ref_pos:
                _, _, zone_number, zone_letter = utm.from_latlon(ref_pos["lat"], ref_pos["lon"])
            else:
                zone_number, zone_letter = 33, "U"

            for pose in poses:
                pose_utm = do_transform_pose_stamped(pose, transform)
                ret_gps = utm.to_latlon(
                    pose_utm.pose.position.x,
                    pose_utm.pose.position.y,
                    zone_number,
                    zone_letter,
                )
                waypoints_gps.append({"lat": ret_gps[0], "lon": ret_gps[1]})

        return waypoints_gps

    def _gps_callback(self, msg):
        with self._lock:
            first = self.pose_gps is None
            self.gps_fix_status = int(msg.status.status)
            self.pose_gps = {"lat": msg.latitude, "lon": msg.longitude}
            if self.current_heading is not None:
                self.pose_gps["heading"] = self.current_heading
            self._dirty = True
        if first:
            self.get_logger().info(f"First GPS fix received: {msg.latitude}, {msg.longitude}")

    def _ekf_callback(self, msg):
        with self._lock:
            first = self.pose_ekf is None
            self.pose_ekf = {"lat": msg.latitude, "lon": msg.longitude}
            if self.current_heading is not None:
                self.pose_ekf["heading"] = self.current_heading
            self._dirty = True
        if first:
            self.get_logger().info(f"First EKF pose received: {msg.latitude}, {msg.longitude}")

    def _path_callback(self, msg: Path):
        waypoints = self._convert_path_latlon(msg)
        if waypoints:
            subsampled = waypoints[::10]
            if waypoints[-1] not in subsampled:
                subsampled.append(waypoints[-1])
            with self._lock:
                self.waypoints_gps = subsampled
                self._dirty = True

    def _feedback_callback(
        self,
        msg: NavigateThroughPoses.Feedback | FollowWaypoints.Feedback | FollowGPSWaypoints.Feedback,
    ):
        if not ROS_AVAILABLE:
            return
        with self._lock:
            if isinstance(msg, NavigateThroughPoses.Feedback):
                self.current_waypoint = self.num_waypoints - msg.number_of_poses_remaining
            else:
                self.current_waypoint = msg.current_waypoint
            self._dirty = True

    def _voltage_callback(self, msg: Float32):
        with self._lock:
            self.bus_voltage = round(float(msg.data), 2)
            self._dirty = True

    def _current_callback(self, msg: Float32):
        with self._lock:
            self.bus_current = round(float(msg.data), 2)
            self._dirty = True

    def _motors_callback(self, msg: Bool):
        with self._lock:
            self.motors_enabled = bool(msg.data)
            self._dirty = True

    def _azimuth_callback(self, msg: Imu):
        q = msg.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        heading = round((90.0 - math.degrees(yaw)) % 360.0, 1)
        with self._lock:
            self.current_heading = heading
            # Keep heading in sync with existing pose dicts immediately
            if self.pose_gps is not None:
                self.pose_gps["heading"] = heading
            if self.pose_ekf is not None:
                self.pose_ekf["heading"] = heading
            self._dirty = True

    def _temp_callback(self, msg: Float32):
        with self._lock:
            self.teensy_temp = round(float(msg.data), 1)
            self._dirty = True

    def _odom_speed_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        with self._lock:
            self.speed = round(math.sqrt(vx * vx + vy * vy), 2)
            self._dirty = True

    def _odrv_error_callback(self, msg: UInt64):
        with self._lock:
            self.motor_error = int(msg.data)
            self._dirty = True

    def _speed_limit_callback(self, msg: SpeedLimit):
        with self._lock:
            self.speed_limit = {
                "value": round(msg.speed_limit, 2),
                "percentage": bool(msg.percentage),
            }
            self._dirty = True

    def _collision_callback(self, msg: CollisionMonitorState):
        with self._lock:
            self.collision_action = self._COLLISION_ACTIONS.get(
                msg.action_type, str(msg.action_type)
            )
            self._dirty = True

    def _recovery_callback(self, msg: Header):
        with self._lock:
            self._last_recovery_time = time.time()
            self._dirty = True

    def _teleop_callback(self, msg: TwistStamped):
        lv, av = msg.twist.linear, msg.twist.angular
        if any(v != 0.0 for v in (lv.x, lv.y, lv.z, av.x, av.y, av.z)):
            with self._lock:
                self._last_teleop_time = time.time()
                self._dirty = True

    def _bt_callback(self, msg: BehaviorTreeLog):
        running = [e for e in msg.event_log if e.current_status == "RUNNING"]
        nav_state = running[-1].node_name if running else None
        with self._lock:
            self.nav_state = nav_state
            self._dirty = True

    def _speech_callback(self, msg: String, level: str):
        with self._lock:
            self.last_speech = {"level": level, "text": msg.data}
            self._dirty = True
