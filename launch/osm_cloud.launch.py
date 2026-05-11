#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    mapdata_path = DeclareLaunchArgument(
        "mapdata_path",
        default_value=PathJoinSubstitution([FindPackageShare("map_data"), "data"]),
        description="Path to the directory with map data.",
    )
    mapdata_file = DeclareLaunchArgument(
        "mapdata_file",
        default_value="stromovka.mapdata",
        description="File with preprocessed OSM map data.",
    )
    gpx_file = DeclareLaunchArgument(
        "gpx_file",
        default_value="stromovka.gpx",
        description="File with gpx coords denoting area to be processed.",
    )
    grid_topic = DeclareLaunchArgument(
        "grid_topic",
        default_value="osm_grid",
        description="Name of the topic to which the grid will be published.",
    )
    publish_static_tf = DeclareLaunchArgument(
        "publish_static_tf",
        default_value="false",
        description="Whether to publish static transforms for utm and map.",
    )

    # Define the osm_cloud node
    osm_cloud_node = Node(
        package="map_data",
        executable="osm_cloud",
        name="osm_cloud",
        output="screen",
        respawn=True,
        respawn_delay=1.0,
        remappings=[("grid", LaunchConfiguration("grid_topic"))],
        parameters=[
            {
                "utm_frame": "utm",
                "local_frame": "local_utm",
                "mapdata_file": PathJoinSubstitution(
                    [
                        LaunchConfiguration("mapdata_path"),
                        LaunchConfiguration("mapdata_file"),
                    ]
                ),
                "gpx_file": PathJoinSubstitution(
                    [
                        LaunchConfiguration("mapdata_path"),
                        LaunchConfiguration("gpx_file"),
                    ]
                ),
                "save_mapdata": False,
                "max_path_dist": 2.5,
                "neighbor_cost": "linear",  # zero, linear, quadratic
                "grid_res": 0.4,
                "grid_max": [0.0, 0.0],
                "grid_min": [0.0, 0.0],
                "publish_intersections": True,
            }
        ],
    )

    return LaunchDescription(
        [
            mapdata_path,
            mapdata_file,
            gpx_file,
            grid_topic,
            publish_static_tf,
            osm_cloud_node,
        ]
    )
