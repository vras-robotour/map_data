#!/usr/bin/env python3
"""
Launch the ``route_planner`` action server (PlanRoute on /route_planner/plan_route).

    ros2 launch map_data route_planner.launch.py mapdata_file:=stromovka.mapdata

Then, for example:

    ros2 action send_goal /route_planner/plan_route map_data_interfaces/action/PlanRoute \\
        "{waypoints: [{latitude: 50.1067, longitude: 14.4193}], start_from_robot: true}"
"""

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "mapdata_file",
            default_value="",
            description="Default .mapdata (name in mapdata_path or absolute path).",
        ),
        DeclareLaunchArgument(
            "mapdata_path",
            default_value=PathJoinSubstitution([FindPackageShare("map_data"), "data"]),
            description="Directory .mapdata names are resolved against.",
        ),
        DeclareLaunchArgument(
            "mission_dir",
            default_value="~/missions",
            description="Where planned routes are written as GPX tracks.",
        ),
        DeclareLaunchArgument("gps_fix_topic", default_value="/fixposition/odometry_llh"),
        DeclareLaunchArgument("earth_frame", default_value="FP_ECEF"),
        DeclareLaunchArgument("local_frame", default_value="FP_ENU0"),
        DeclareLaunchArgument(
            "algorithm", default_value="graph", description="graph (paths only) | astar | rrt"
        ),
        DeclareLaunchArgument(
            "spacing", default_value="3.0", description="m between output waypoints (0 = raw)"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]
    node = Node(
        package="map_data",
        executable="route_planner",
        name="route_planner",
        output="screen",
        parameters=[
            {
                "mapdata_file": LaunchConfiguration("mapdata_file"),
                "data_dir": LaunchConfiguration("mapdata_path"),
                "mission_dir": LaunchConfiguration("mission_dir"),
                "gps_fix_topic": LaunchConfiguration("gps_fix_topic"),
                "earth_frame": LaunchConfiguration("earth_frame"),
                "local_frame": LaunchConfiguration("local_frame"),
                "algorithm": LaunchConfiguration("algorithm"),
                "spacing": LaunchConfiguration("spacing"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
    )
    return LaunchDescription([*args, node])
