#!/usr/bin/env python3

from pathlib import Path

from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from map_data.utils.launch import given, resolve_config_file, way_types


def _resolve_data_file(name: str, mapdata_path: str) -> str:
    """Resolve a map-data filename against ``mapdata_path``.

    Empty values and absolute paths are returned unchanged; a bare filename is
    joined onto ``mapdata_path`` (the data directory) when the resulting file
    exists, otherwise it is passed through untouched so downstream error
    handling can report the missing file.
    """
    if not name:
        return name
    candidate = Path(name)
    if candidate.is_absolute():
        return name
    joined = Path(mapdata_path) / name
    return str(joined) if joined.exists() else name


def launch_setup(context, *args, **kwargs):
    config_file = resolve_config_file(LaunchConfiguration("config_file").perform(context))
    osm_grid_params = resolve_config_file(LaunchConfiguration("osm_grid_params").perform(context))

    # Resolve mapdata_file / gpx_file against the mapdata_path data directory
    # when they are given as bare filenames.
    mapdata_path = LaunchConfiguration("mapdata_path").perform(context)
    mapdata_file = _resolve_data_file(
        LaunchConfiguration("mapdata_file").perform(context), mapdata_path
    )
    gpx_file = _resolve_data_file(LaunchConfiguration("gpx_file").perform(context), mapdata_path)

    # Frame / placement / traversability overrides: only forwarded when set, so the
    # yaml values apply otherwise (later parameter sources win in ROS 2).
    optional_overrides = {
        **given(context, "local_frame"),
        **given(context, "utm_frame"),
        **given(context, "earth_frame"),
        **given(context, "transform_mode"),
        **given(context, "traversability", "traversability_file"),
        **given(context, "highway_types", cast=way_types),
    }

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
                "mapdata_file": mapdata_file,
                "gpx_file": gpx_file,
                "grid_topic": LaunchConfiguration("grid_topic"),
                # Always forwarded (default "auto"), so it wins over the yaml files:
                # set it on the launch line to plan and publish on the unedited map.
                "annotations": LaunchConfiguration("annotations"),
                **optional_overrides,
            },
        ],
    )

    return [osm_cloud_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mapdata_path",
                default_value=PathJoinSubstitution([FindPackageShare("map_data"), "data"]),
                description="Path to the directory with map data.",
            ),
            DeclareLaunchArgument(
                "mapdata_file",
                default_value="",
                description="File with preprocessed OSM map data.",
            ),
            DeclareLaunchArgument(
                "gpx_file",
                default_value="",
                description="File with gpx coords denoting area to be processed.",
            ),
            DeclareLaunchArgument(
                "grid_topic",
                default_value="osm_grid",
                description="Name of the topic to which the grid will be published.",
            ),
            DeclareLaunchArgument(
                "local_frame",
                default_value="",
                description="Frame the grid and intersections are published in "
                "(empty = value from osm_grid_params, e.g. FP_ENU0).",
            ),
            DeclareLaunchArgument(
                "utm_frame",
                default_value="",
                description="UTM frame name used by transform_mode 'tf'/'auto' "
                "(empty = from yaml).",
            ),
            DeclareLaunchArgument(
                "earth_frame",
                default_value="",
                description="ECEF frame name used by transform_mode 'geodetic' "
                "(empty = from yaml, e.g. FP_ECEF).",
            ),
            DeclareLaunchArgument(
                "transform_mode",
                default_value="",
                description="How map data is placed in local_frame: 'tf', 'auto' or 'geodetic' "
                "(empty = from yaml).",
            ),
            DeclareLaunchArgument(
                "annotations",
                default_value="auto",
                description="Annotation store merged into the map, as in route_planner: "
                "'auto' = <map>.annotations.json next to it, 'none' = the unedited map, or a "
                "path to a store file. The planner and this node must see the same map.",
            ),
            DeclareLaunchArgument(
                "traversability",
                default_value="",
                description="Tag rule file deciding which ways the robot may drive on "
                "(empty = the value from osm_grid_params, i.e. the package's "
                "config/traversability.yaml). Must match route_planner's.",
            ),
            DeclareLaunchArgument(
                "highway_types",
                default_value="",
                description="Way types the grid is drawn from: footway, road, or both "
                '("footway,road"); empty = the value from osm_grid_params. Match route_planner\'s.',
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value="helhest_jr.yaml",
                description="Path or name (in config/) of the yaml file with topic names.",
            ),
            DeclareLaunchArgument(
                "osm_grid_params",
                default_value="osm_grid.yaml",
                description="Path or name (in config/) of the yaml file with OSM grid parameters.",
            ),
            OpaqueFunction(function=launch_setup),
        ],
    )
