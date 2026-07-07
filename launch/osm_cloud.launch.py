#!/usr/bin/env python3

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def launch_setup(context, *args, **kwargs):
    mapdata_path = LaunchConfiguration("mapdata_path")
    mapdata_file = LaunchConfiguration("mapdata_file")
    gpx_file = LaunchConfiguration("gpx_file")
    config_file = LaunchConfiguration("config_file").perform(context)
    osm_grid_params = LaunchConfiguration("osm_grid_params").perform(context)

    # Resolve config_file if it's just a filename
    config_path = Path(config_file)
    if not config_path.is_absolute():
        package_share = get_package_share_directory("map_data")
        potential_path = Path(package_share) / "config" / config_file
        if potential_path.exists():
            config_file = str(potential_path)
        else:
            # Try in current directory
            potential_path = config_path.resolve()
            if potential_path.exists():
                config_file = str(potential_path)

    # Resolve osm_grid_params if it's just a filename
    osm_path = Path(osm_grid_params)
    if not osm_path.is_absolute():
        package_share = get_package_share_directory("map_data")
        potential_path = Path(package_share) / "config" / osm_grid_params
        if potential_path.exists():
            osm_grid_params = str(potential_path)
        else:
            # Try in current directory
            potential_path = osm_path.resolve()
            if potential_path.exists():
                osm_grid_params = str(potential_path)

    # Define the osm_cloud node
    osm_cloud_node = Node(
        package="map_data",
        executable="osm_cloud",
        name="osm_cloud",
        output="screen",
        respawn=True,
        respawn_delay=1.0,
        parameters=[
            config_file,
            osm_grid_params,
            {
                "mapdata_file": PathJoinSubstitution([mapdata_path, mapdata_file]),
                "gpx_file": PathJoinSubstitution([mapdata_path, gpx_file]),
            },
        ],
    )

    return [osm_cloud_node]


def generate_launch_description():
    # Declare launch arguments
    mapdata_path_arg = DeclareLaunchArgument(
        "mapdata_path",
        default_value=PathJoinSubstitution([FindPackageShare("map_data"), "data"]),
        description="Path to the directory with map data.",
    )
    mapdata_file_arg = DeclareLaunchArgument(
        "mapdata_file",
        default_value="stromovka.mapdata",
        description="File with preprocessed OSM map data.",
    )
    gpx_file_arg = DeclareLaunchArgument(
        "gpx_file",
        default_value="stromovka.gpx",
        description="File with gpx coords denoting area to be processed.",
    )
    grid_topic_arg = DeclareLaunchArgument(
        "grid_topic",
        default_value="osm_grid",
        description="Name of the topic to which the grid will be published.",
    )
    publish_static_tf_arg = DeclareLaunchArgument(
        "publish_static_tf",
        default_value="false",
        description="Whether to publish static transforms for utm and map.",
    )
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value="helhest.yaml",
        description="Path or name (in config/) of the yaml file with topic names.",
    )
    osm_grid_params_arg = DeclareLaunchArgument(
        "osm_grid_params",
        default_value="osm_grid.yaml",
        description="Path or name (in config/) of the yaml file with OSM grid parameters.",
    )

    return LaunchDescription(
        [
            mapdata_path_arg,
            mapdata_file_arg,
            gpx_file_arg,
            grid_topic_arg,
            publish_static_tf_arg,
            config_file_arg,
            osm_grid_params_arg,
            OpaqueFunction(function=launch_setup),
        ],
    )
