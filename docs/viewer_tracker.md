# Tracker

The **Tracker** mode provides a real-time view of a robot's position and hardware status.
It requires the viewer to be launched inside a sourced ROS 2 workspace so that the
`TrackerNode` can subscribe to the robot's topics.

!!! note "ROS 2 required"
    The Tracker tab is only shown when a ROS 2 context is available. Launching the viewer
    with `map_data_viewer` inside a sourced workspace is sufficient — no additional nodes
    need to be started.

## Robot marker

When telemetry is received, a green arrow marker appears on the map at the robot's current
position. The marker rotates to the robot's heading (`heading_topic`, or `azimuth_topic` when that is empty).
The robot marker is visible in all three modes as long as the **Robot** layer is enabled in
the Layers panel.

## Sidebar controls

| Control | Description |
|---------|-------------|
| **Center Robot** | Pan and zoom the map to the robot's current position |
| **Clear Trail** | Clear the recorded trail (dropped on the tracker node, so it does not reappear with the next fix) |
| **Follow** | Continuously pan the map to keep the robot centered as it moves |

## Status display

The sidebar shows live telemetry grouped into three sections. Each row is hidden automatically
if the corresponding ROS 2 topic is not configured.

### Hardware

| Field | Description |
|-------|-------------|
| Battery | Voltage (V), current (A) and, with `battery_state_topic`, pack percentage; turns red below `battery_low_voltage` |
| E-Stop | `ACTIVE` (red) / `released` from `estop_topic` |
| Motors | Motor enable state (`ENABLED` / `DISABLED`) |
| Temp | Hottest `sensor_msgs/Temperature` (name = its `frame_id`), or controller temperature |
| Motor Error | ODrive error code (displayed in hex when non-zero) |

### Localization

| Field | Description |
|-------|-------------|
| Localization | GPS fix quality: **Fixed** (RTK fixed), **Float** (RTK float), or **No Fix** |
| Fix age | Seconds since the last fix; displays red `STALE` after `stale_after` |
| Speed | Current robot speed in m/s (from odometry) |
| Limit | Active speed limit value and unit (m/s or %) |

### Navigation

| Field | Description |
|-------|-------------|
| State | `commander_state_topic` string (crl_commander mode), active Nav2 behavior tree node, or `IDLE` |
| Follower | `road_follower` state: `ROAD` or `GPS:<reason>` |
| Diagnostics | `OK`, or the number of ERROR/WARN statuses and the worst status message |
| Collision | Active collision monitor action (`STOP`, `SLOWDOWN`, `APPROACH`, `LIMIT`); hidden for `DO_NOTHING` |
| Recovery Active | Displayed in red when a recovery behavior is running |
| Teleop Active | Displayed in yellow when teleop velocity commands or joystick inputs were received within 2 s |

The sidebar also displays the **last five speech messages** (info / warn / error level) when speech topics are configured.

## Map layers

All robot geometry is part of the **Robot** layer; the Tracker sidebar includes a **Map legend**
listing active layers:

| Layer | Source | Style |
|-------|--------|-------|
| Planned path | `path_topic` or active Nav2 action | green dashed polyline |
| Route / waypoint sequence | `sequence_path_topic` (`Path`, e.g. `road_follower`'s `~/route_path`) | blue dotted polyline |
| Waypoint window | `sequence_poses_topic` (`PoseArray`, e.g. `/goal_sequence`) | light-blue polyline |
| Road path | `road_path_topic` (`path_centerline` prediction) | cyan polyline |
| Goal | `goal_topic` | orange circle |
| Trail | last `trail_length` fixes of `gps_filtered_topic` (else `gps_fix_topic`), spaced by `trail_min_step` m; no-fix messages are skipped | thin green polyline |
| Intersections | `intersections_topic` (`PoseArray`, from `osm_cloud`) | magenta rings |
| Active intersection | `active_intersection_topic` with `intersection_enter_threshold` and `intersection_exit_threshold` | colored radius circles |

Poses may be stamped in **any TF frame**. With `earth_frame` set (such as `FP_ECEF`),
coordinates are transformed via TF into ECEF and converted to exact lat/lon. If `earth_frame`
is empty, coordinates are transformed into `utm_frame` and converted with the UTM zone of the current fix.

## ROS 2 topic configuration

The `TrackerNode` subscribes to configurable topics. Set any topic parameter to an empty
string (`""`) to disable the corresponding feature and hide its UI row.

### Config file

Topic configurations are defined in a ROS 2 parameter file under a `map_data_tracker` section.
By default the viewer reads `config/tracker.yaml` (generic Nav2 topics); `--config` selects
another file, such as `config/helhest_jr.yaml`:

```bash
map_data_viewer --config config/helhest_jr.yaml
```

The config file is a standard ROS 2 parameter file, so `--ros-args --params-file` can also be used.

### Changing topics in the web app

The **Topics…** dialog in the Tracker sidebar lists all topic settings and frame parameters.
Topic fields indicate live publisher status (green = expected type, yellow = different type,
grey = not published, hollow = disabled).

- **Apply** updates active subscriptions in the running tracker.
- **Save to file** applies changes and updates the active configuration YAML file.
- **Reload from file** restores settings from the configuration file on disk.

### Tracker parameters

| Parameter | Default topic | Description |
|-----------|--------------|-------------|
| `earth_frame` | `""` | ECEF TF frame for exact coordinate conversion (`FP_ECEF`); empty = UTM fallback |
| `utm_frame` | `utm` | UTM TF frame used when `earth_frame` is empty |
| `tf_static_only` | `true` | Read frames from `/tf_static` only; set `false` if frames are published on dynamic `/tf` |
| `gps_fix_topic` | `/gps/fix` | Raw GPS fix (`sensor_msgs/NavSatFix`) |
| `gps_filtered_topic` | `/gps/filtered` | EKF-fused GPS position (`sensor_msgs/NavSatFix`) |
| `heading_topic` / `heading_type` | `""` / `imu` | Heading source: `imu` (`Imu`), `yaw_vector3` (`Vector3Stamped`), `odometry` (`Odometry`) |
| `azimuth_topic` | `/gps/azimuth_imu` | Legacy IMU heading topic, used when `heading_topic` is empty |
| `odom_topic` | `/odom_2d` | Odometry for speed display (`nav_msgs/Odometry`) |
| `battery_state_topic` | `""` | `sensor_msgs/BatteryState` (voltage, current, percentage) |
| `battery_low_voltage` | `22.0` | Voltage below which the battery indicator turns red |
| `bus_voltage_topic` | `/bus_voltage` | Battery voltage (`std_msgs/Float32`) |
| `bus_current_topic` | `/bus_current` | Battery current (`std_msgs/Float32`) |
| `temperature_topic` | `""` | `sensor_msgs/Temperature` messages, one row per `frame_id` |
| `teensy_temp_topic` | `/teensy_temp` | Controller temperature (`std_msgs/Float32`) |
| `odrv_error_topic` | `/odrv_error` | ODrive error code (`std_msgs/UInt64`) |
| `motors_enabled_topic` | `/motors_enabled` | Motor enable state (`std_msgs/Bool`) |
| `estop_topic` | `""` | Emergency stop state (`std_msgs/Bool`) |
| `diagnostics_topic` | `""` | `diagnostic_msgs/DiagnosticArray` summary |
| `commander_state_topic` | `""` | Navigation state string (`std_msgs/String`) |
| `follower_state_topic` | `""` | Road follower state string (`std_msgs/String`, latched) |
| `bt_log_topic` | `/behavior_tree_log` | Nav2 behavior tree log (`nav2_msgs/BehaviorTreeLog`) |
| `speed_limit_topic` | `/speed_limit` | Active speed limit (`nav2_msgs/SpeedLimit`) |
| `collision_monitor_state_topic` | `/collision_monitor_state` | Collision monitor state |
| `recovery_heartbeat_topic` | `/recovery/heartbeat` | Recovery behavior heartbeat (`std_msgs/Header`) |
| `teleop_topic` | `/cmd_vel_teleop` | Teleop velocity command (`geometry_msgs/TwistStamped`) |
| `joy_topic` | `""` | Joystick (`sensor_msgs/Joy`); any input indicates active teleoperation |
| `speak_info_topic` | `/speak/info` | Info speech messages (`std_msgs/String`) |
| `speak_warn_topic` | `/speak/warn` | Warning speech messages (`std_msgs/String`) |
| `speak_error_topic` | `/speak/err` | Error speech messages (`std_msgs/String`) |
| `path_topic` | `/path` | Planned path for map overlay (`nav_msgs/Path`) |
| `goal_topic` | `""` | Current navigation goal (`geometry_msgs/PoseStamped`, latched) |
| `sequence_path_topic` | `""` | Route sequence being followed (`nav_msgs/Path`, latched) |
| `sequence_poses_topic` | `""` | Waypoint window poses (`geometry_msgs/PoseArray`, latched) |
| `road_path_topic` | `""` | Visual road path prediction (`nav_msgs/Path`) |
| `intersections_topic` | `""` | Intersections (`geometry_msgs/PoseArray`, latched) |
| `active_intersection_topic` | `""` | Active intersection (`geometry_msgs/PoseStamped`, latched) |
| `intersection_enter_threshold` / `intersection_exit_threshold` | `5.0` / `6.0` | Radii drawn around active intersection (m) |
| `trail_length` / `trail_min_step` | `500` / `0.5` | Robot trail size (points) and step distance (m) |
| `stale_after` | `3.0` | Fix age threshold (s) after which position is marked stale |
| `planner_node` | `/route_planner` | Node name whose ROS 2 parameters seed Planner toggles (`""` disables syncing) |
| `*_feedback_topic` | Nav2 action feedback | Action feedback topics reporting current waypoint index |

Telemetry is broadcast to the browser over WebSocket at **2 Hz** by default, configurable
via `map_data_viewer --telemetry-rate <Hz>`.
