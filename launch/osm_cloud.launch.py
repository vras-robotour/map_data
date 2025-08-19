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
        default_value="buchlovice.mapdata",
        description="File with preprocessed OSM map data.",
    )
    gpx_file = DeclareLaunchArgument(
        "gpx_file",
        default_value="buchlovice.gpx",
        description="File with gpx coords denoting area to be processed.",
    )
    grid_topic = DeclareLaunchArgument(
        "grid_topic",
        default_value="osm_grid",
        description="Name of the topic to which the grid will be published.",
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
                # "mapdata_file": PathJoinSubstitution(
                #     [
                #         LaunchConfiguration("mapdata_path"),
                #         LaunchConfiguration("mapdata_file"),
                #     ]
                # ),
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
                "grid_max": [250.0, 250.0],
                "grid_min": [-250.0, -250.0],
            }
        ],
    )

    local_utm_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="local_utm_transform",
        output="screen",
        arguments=[
            "670667.0",
            "5439425.14",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "utm",
            "local_utm",
        ],
    )

    map_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_transform",
        output="screen",
        arguments=[
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "map",
            "local_utm",
        ],
    )

    base_link_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_transform",
        output="screen",
        arguments=[
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "map",
            "base_link",
        ],
    )

    return LaunchDescription(
        [
            mapdata_path,
            mapdata_file,
            gpx_file,
            grid_topic,
            osm_cloud_node,
            local_utm_node,
            map_node,
            base_link_node,
        ]
    )
