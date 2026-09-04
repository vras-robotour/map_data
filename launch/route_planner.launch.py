#!/usr/bin/env python3
"""
Launch the ``route_planner`` action server (PlanRoute on /route_planner/plan_route).

    ros2 launch map_data route_planner.launch.py
    ros2 launch map_data route_planner.launch.py mapdata_file:=KN.mapdata \
        highway_types:=footway,road
    ros2 launch map_data route_planner.launch.py params_file:=/home/robot/kn_planner.yaml

Every parameter of the node lives in ``config/route_planner.yaml`` (or the ``params_file``
given above); the launch arguments below only override what they are given, so an argument
left unset keeps the value from the file.

    ros2 action send_goal /route_planner/plan_route map_data_interfaces/action/PlanRoute \\
        "{waypoints: [{latitude: 50.1067, longitude: 14.4193}], start_from_robot: true}"
"""

from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription
from map_data.utils.launch import flag, resolve_config_file, way_types

DEFAULT_PARAMS_FILE = "route_planner.yaml"


def _arguments():
    return [
        DeclareLaunchArgument(
            "params_file",
            default_value=DEFAULT_PARAMS_FILE,
            description="Parameter file: an absolute path or a name in map_data/config.",
        ),
        # Everything below defaults to empty: unset = keep what params_file says.
        DeclareLaunchArgument(
            "mapdata_file",
            default_value="",
            description="Default .mapdata (name in mapdata_path or absolute path).",
        ),
        DeclareLaunchArgument(
            "mapdata_path",
            default_value="",
            description="Directory .mapdata names are resolved against "
            "(default: the package's share/map_data/data).",
        ),
        DeclareLaunchArgument(
            "annotations",
            default_value="",
            description="auto = <map>.annotations.json next to the map, none = unedited map, "
            "or a store file.",
        ),
        DeclareLaunchArgument(
            "mission_dir", default_value="", description="Where routes are written as GPX."
        ),
        DeclareLaunchArgument("gps_fix_topic", default_value=""),
        DeclareLaunchArgument("earth_frame", default_value=""),
        DeclareLaunchArgument("local_frame", default_value=""),
        DeclareLaunchArgument(
            "algorithm", default_value="", description="graph (ways only) | astar | rrt"
        ),
        DeclareLaunchArgument(
            "highway_types",
            default_value="",
            description="Way types the graph planner may route over: footway, road, or both "
            '("footway,road"). A PlanRoute goal can override it per request.',
        ),
        DeclareLaunchArgument(
            "spacing", default_value="", description="m between output waypoints (0 = raw)"
        ),
        DeclareLaunchArgument(
            "preload", default_value="", description="load the map and its graph at startup"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]


def launch_setup(context, *args, **kwargs):
    def given(name, key=None, cast=str):
        value = LaunchConfiguration(name).perform(context).strip()
        return {} if not value else {key or name: cast(value)}

    # Later parameter sources win in ROS 2, so the file first and the arguments on top.
    overrides = {
        **given("mapdata_file"),
        **given("mapdata_path", "data_dir"),
        **given("annotations"),
        **given("mission_dir"),
        **given("gps_fix_topic"),
        **given("earth_frame"),
        **given("local_frame"),
        **given("algorithm"),
        **given("highway_types", cast=way_types),
        **given("spacing", cast=float),
        **given("preload", cast=flag),
        "use_sim_time": flag(LaunchConfiguration("use_sim_time").perform(context)),
    }
    node = Node(
        package="map_data",
        executable="route_planner",
        name="route_planner",
        output="screen",
        parameters=[
            resolve_config_file(LaunchConfiguration("params_file").perform(context)),
            overrides,
        ],
    )
    return [node]


def generate_launch_description():
    return LaunchDescription([*_arguments(), OpaqueFunction(function=launch_setup)])
