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

The sidebar shows live telemetry grouped into three sections. Each row is hidden automatically
if the corresponding ROS2 topic is not configured.

### Hardware

| Field | Description |
|-------|-------------|
| Battery | Voltage (V), current (A) and, with `battery_state_topic`, the pack percentage. Red below `battery_low_voltage`. |
| E-Stop | `ACTIVE` (red) / `released` from `estop_topic`. |
| Motors | Motor enable state (`ENABLED` / `DISABLED`). |
| Temp | Hottest `sensor_msgs/Temperature` (name = its `frame_id`; hover for all), or the Teensy temperature. |
| Motor Error | ODrive error code (displayed in hex when non-zero). |

### Localization

| Field | Description |
|-------|-------------|
| Localization | GPS fix quality: **Fixed** (RTK fixed), **Float** (RTK float), or **No Fix**. |
| Fix age | Seconds since the last fix; red `STALE` (and a greyed marker) after `stale_after`. |
| Speed | Current robot speed in m/s (from odometry). |
| Limit | Active speed limit value and unit (m/s or %). |

### Navigation

| Field | Description |
|-------|-------------|
| State | `commander_state_topic` string (crl_commander mode, `STUCK` in red) or the active Nav2 behavior tree node, or `IDLE`. |
| Follower | `road_follower` state: `ROAD` or `GPS:<reason>` (yellow). |
| Diagnostics | `OK`, or the number of ERROR/WARN statuses and the worst one (hover for its message). |
| Collision | Active collision monitor action (`STOP`, `SLOWDOWN`, `LIMIT`). Hidden when passthrough. |
| Recovery Active | Shown in red when a recovery behavior is running. |
| Teleop Active | Shown in yellow when a non-zero teleop twist or any joystick input arrived in the last 2 s. |

The sidebar also shows the robot's **last speech message** (info / warn / error level) when the
speech topics are configured.

## Map Layers

All robot geometry is part of the **Robot** layer:

| Layer | Source | Style |
|-------|--------|-------|
| Planned path | `path_topic` or an active Nav2 action | green dashed polyline |
| Waypoint sequence | `sequence_path_topic` (`Path`, e.g. the commander's loaded sequence) | blue dotted polyline |
| Waypoint window | `sequence_poses_topic` (`PoseArray`, e.g. `road_follower`'s `/goal_sequence`) | light-blue polyline |
| Road path | `road_path_topic` (`path_centerline` prediction) | cyan polyline |
| Goal | `goal_topic` | orange circle |
| Trail | last `trail_length` fixes, one every `trail_min_step` m | thin green polyline |
| Intersections | `intersections_topic` (`PoseArray`, `osm_cloud`) | magenta rings |
| Active intersection | `active_intersection_topic` (`road_follower`) with the `intersection_enter_threshold` (red) and `intersection_exit_threshold` (yellow dashed) radii | circles in metres |

Poses may be stamped in **any TF frame**. With `earth_frame` set (e.g. `FP_ECEF` on the
Fixposition stack) they are transformed into that ECEF frame through TF and converted to
lat/lon exactly. With `earth_frame` empty the legacy path is used: transform into `utm_frame`
and convert with the UTM zone of the current fix.

## ROS2 Topic Configuration

The `TrackerNode` subscribes to a set of configurable topics. Set a topic parameter to an empty
string to disable the corresponding feature and hide its UI row. `config/helhest.yaml` is the
full configuration for the Helhest field stack (Fixposition + crl_commander + road_follower):

```bash
map_data_viewer --ros-args --params-file config/helhest.yaml
```

| Parameter | Default topic | Description |
|-----------|--------------|-------------|
| `earth_frame` | `""` | ECEF TF frame for exact pose conversion (`FP_ECEF`); empty = UTM fallback |
| `utm_frame` | `utm` | UTM TF frame used when `earth_frame` is empty |
| `gps_fix_topic` | `/gps/fix` | Raw GPS fix (`NavSatFix`) |
| `gps_filtered_topic` | `/gps/filtered` | EKF-fused GPS position (`NavSatFix`) |
| `heading_topic` / `heading_type` | `""` / `imu` | Heading source: `imu` (`Imu`), `yaw_vector3` (`Vector3Stamped`, x = yaw rad, e.g. `/fixposition/ypr`), `odometry` (`Odometry`) |
| `azimuth_topic` | `/gps/azimuth_imu` | Legacy IMU heading topic, used when `heading_topic` is empty |
| `odom_topic` | `/odom_2d` | Odometry for speed display (`Odometry`) |
| `battery_state_topic` | `""` | `BatteryState` (voltage, current, percentage); replaces the two Float32 topics |
| `battery_low_voltage` | `22.0` | Voltage below which the battery row turns red |
| `bus_voltage_topic` | `/bus_voltage` | Battery voltage (`Float32`) |
| `bus_current_topic` | `/bus_current` | Battery current (`Float32`) |
| `temperature_topic` | `""` | `Temperature` messages, one row per `frame_id`; replaces `teensy_temp_topic` |
| `teensy_temp_topic` | `/teensy_temp` | Controller temperature (`Float32`) |
| `odrv_error_topic` | `/odrv_error` | ODrive error code (`UInt64`) |
| `motors_enabled_topic` | `/motors_enabled` | Motor enable state (`Bool`) |
| `estop_topic` | `""` | Emergency stop state (`Bool`) |
| `diagnostics_topic` | `""` | `DiagnosticArray` summary |
| `commander_state_topic` | `""` | Navigation state string (`String`, crl_commander); replaces `bt_log_topic` |
| `follower_state_topic` | `""` | `road_follower` state (`String`, latched) |
| `bt_log_topic` | `/behavior_tree_log` | Nav2 behavior tree log (`BehaviorTreeLog`) |
| `speed_limit_topic` | `/speed_limit` | Active speed limit (`SpeedLimit`) |
| `collision_monitor_state_topic` | `/collision_monitor_state` | Collision monitor state (`CollisionMonitorState`) |
| `recovery_heartbeat_topic` | `/recovery/heartbeat` | Recovery behavior heartbeat (`Header`) |
| `teleop_topic` | `/cmd_vel_teleop` | Teleop velocity command (`TwistStamped`) |
| `joy_topic` | `""` | Joystick (`Joy`); any input counts as teleop |
| `speak_info_topic` | `/speak/info` | Info speech messages (`String`) |
| `speak_warn_topic` | `/speak/warn` | Warning speech messages (`String`) |
| `speak_error_topic` | `/speak/err` | Error speech messages (`String`) |
| `path_topic` | `/path` | Planned path for map overlay (`Path`) |
| `goal_topic` | `""` | Current navigation goal (`PoseStamped`, latched) |
| `sequence_path_topic` | `""` | Waypoint sequence (`Path`, latched) |
| `sequence_poses_topic` | `""` | Waypoint window (`PoseArray`, latched) |
| `road_path_topic` | `""` | Visual road-following path (`Path`) |
| `intersections_topic` | `""` | Intersections (`PoseArray`, latched) |
| `active_intersection_topic` | `""` | Intersection that triggered GPS mode (`PoseStamped`, latched; empty `frame_id` = none) |
| `intersection_enter_threshold` / `intersection_exit_threshold` | `5.0` / `6.0` | Radii drawn around the active intersection (m) |
| `trail_length` / `trail_min_step` | `500` / `0.5` | Robot trail size and spacing (m) |
| `stale_after` | `3.0` | Fix age (s) after which the position is flagged stale |
| `*_feedback_topic` | nav2 action feedback | Current waypoint index of Nav2 actions |

Telemetry is polled at **2 Hz** by default and pushed to the browser over a WebSocket.
The rate is configurable via `map_data_viewer --telemetry-rate <Hz>` (e.g. `10` for
smoother tracking of a fast-moving robot).
