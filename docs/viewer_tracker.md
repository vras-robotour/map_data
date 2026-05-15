# Tracker

The **Tracker** mode provides a real-time view of a robot's position and hardware status.
It requires the viewer to be launched inside a sourced ROS2 workspace so that the
`TrackerNode` can subscribe to the robot's topics.

!!! note "ROS2 required"
    The Tracker tab is only shown when a ROS2 context is available. Launching the viewer
    with `map_data_viewer` inside a sourced workspace is sufficient — no additional nodes
    need to be started.

## Robot Marker

When telemetry is received, a green arrow marker appears on the map at the robot's current
position. The marker rotates to reflect the robot's heading (derived from the IMU azimuth topic).
The robot marker is visible in all three modes as long as the **Robot** layer is enabled in
the Layers panel.

## Sidebar Controls

| Control | Description |
|---------|-------------|
| **Center Robot** | Pan and zoom the map to the robot's current position. |
| **Follow** | Continuously pan the map to keep the robot centered as it moves. |

## Status Display

The sidebar shows live telemetry grouped into three sections. Each section is hidden automatically
if the corresponding ROS2 topics are not configured.

### Hardware

| Field | Description |
|-------|-------------|
| Battery | Bus voltage (V) and current (A). Shown in red when voltage drops below 22 V. |
| Motors | Motor enable state (`ENABLED` / `DISABLED`). |
| Temp | Teensy microcontroller temperature in °C. |
| Motor Error | ODrive error code (displayed in hex when non-zero). |

### Localization

| Field | Description |
|-------|-------------|
| Localization | GPS fix quality: **Fixed** (RTK fixed), **Float** (RTK float), or **No Fix**. |
| Speed | Current robot speed in m/s (from odometry). |
| Limit | Active speed limit value and unit (m/s or %). |

### Navigation

| Field | Description |
|-------|-------------|
| State | Active Nav2 behavior tree node name, or `IDLE`. |
| Collision | Active collision monitor action (`STOP`, `SLOWDOWN`, `LIMIT`). Hidden when passthrough. |
| Recovery Active | Shown in red when a recovery behavior is running. |
| Teleop Active | Shown in yellow when a non-zero teleop command was received in the last 2 s. |

The sidebar also shows the robot's **last speech message** (info / warn / error level) when the
speech topics are configured.

## Mission Waypoints

When a Nav2 action (NavigateThroughPoses, FollowWaypoints, or FollowGPSWaypoints) is active,
the planned path is drawn on the map as a dashed green polyline. The polyline updates automatically
when the waypoint set changes.

## ROS2 Topic Configuration

The `TrackerNode` subscribes to a set of configurable topics. Set a topic parameter to an empty
string to disable the corresponding feature and hide its UI row.

| Parameter | Default topic | Description |
|-----------|--------------|-------------|
| `gps_fix_topic` | `/gps/fix` | Raw GPS fix (`NavSatFix`) |
| `gps_filtered_topic` | `/gps/filtered` | EKF-fused GPS position (`NavSatFix`) |
| `azimuth_topic` | `/gps/azimuth_imu` | IMU heading for marker rotation (`Imu`) |
| `odom_topic` | `/odom_2d` | Odometry for speed display (`Odometry`) |
| `bus_voltage_topic` | `/bus_voltage` | Battery voltage (`Float32`) |
| `bus_current_topic` | `/bus_current` | Battery current (`Float32`) |
| `teensy_temp_topic` | `/teensy_temp` | Controller temperature (`Float32`) |
| `odrv_error_topic` | `/odrv_error` | ODrive error code (`UInt64`) |
| `motors_enabled_topic` | `/motors_enabled` | Motor enable state (`Bool`) |
| `speed_limit_topic` | `/speed_limit` | Active speed limit (`SpeedLimit`) |
| `collision_monitor_state_topic` | `/collision_monitor_state` | Collision monitor state (`CollisionMonitorState`) |
| `recovery_heartbeat_topic` | `/recovery/heartbeat` | Recovery behavior heartbeat (`Header`) |
| `bt_log_topic` | `/behavior_tree_log` | Nav2 behavior tree log (`BehaviorTreeLog`) |
| `teleop_topic` | `/cmd_vel_teleop` | Teleop velocity command (`TwistStamped`) |
| `speak_info_topic` | `/speak/info` | Info speech messages (`String`) |
| `speak_warn_topic` | `/speak/warn` | Warning speech messages (`String`) |
| `speak_error_topic` | `/speak/err` | Error speech messages (`String`) |
| `path_topic` | `/path` | Planned path for map overlay (`Path`) |

Telemetry is polled at **2 Hz** and pushed to the browser over a WebSocket.
