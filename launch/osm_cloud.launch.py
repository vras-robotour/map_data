#!/usr/bin/env python3

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
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
                "auto_utm": True,
            }
        ],
    )

    local_utm_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="local_utm_transform",
        output="screen",
        arguments=[
            "--x",
            # "670667.0", # buchlovice
            "458378.63",  # stromovka
            "--y",
            # "5439425.14", # buchlovice
            "5550538.10",  # stromovka
            "--z",
            "0.0",
            "--yaw",
            "0.0",
            "--pitch",
            "0.0",
            "--roll",
            "0.0",
            "--frame-id",
            "utm",
            "--child-frame-id",
            "local_utm",
        ],
        condition=IfCondition(LaunchConfiguration("publish_static_tf")),
    )

    map_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_transform",
        output="screen",
        arguments=[
            "--x",
            "0.0",
            "--y",
            "0.0",
            "--z",
            "0.0",
            "--yaw",
            "0.0",
            "--pitch",
            "0.0",
            "--roll",
            "0.0",
            "--frame-id",
            "map",
            "--child-frame-id",
            "local_utm",
        ],
        condition=IfCondition(LaunchConfiguration("publish_static_tf")),
    )

    base_link_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_transform",
        output="screen",
        arguments=[
            "--x",
            "0.0",
            "--y",
            "0.0",
            "--z",
            "0.0",
            "--yaw",
            "0.0",
            "--pitch",
            "0.0",
            "--roll",
            "0.0",
            "--frame-id",
            "map",
            "--child-frame-id",
            "base_link",
        ],
        condition=IfCondition(LaunchConfiguration("publish_static_tf")),
    )

    # Define the osm_intersections node
    osm_intersections_node = Node(
        package="map_data",
        executable="osm_intersections",
        name="osm_intersections",
        output="screen",
        respawn=True,
        respawn_delay=1.0,
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
                "grid_max": [500.0, 500.0],
                "grid_min": [-500.0, -500.0],
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
            osm_intersections_node,
            local_utm_node,
            map_node,
            base_link_node,
        ]
    )
