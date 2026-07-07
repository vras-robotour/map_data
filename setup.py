from pathlib import Path

from setuptools import find_packages, setup

package_name = "map_data"

# Version and dependencies are defined in pyproject.toml; this file only
# carries the ROS/ament-specific metadata (data_files, entry points).
setup(
    name=package_name,
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            str(Path("share") / package_name / "launch"),
            [str(p) for p in Path("launch").glob("*.launch.py")],
        ),
        (
            str(Path("share") / package_name / "config"),
            [str(p) for p in Path("config").glob("*.yaml")],
        ),
        (
            str(Path("share") / package_name / "data"),
            [str(p) for p in Path("data").glob("*.gpx")],
        ),
        (
            str(Path("share") / package_name / "parameters"),
            [str(p) for p in Path("parameters").glob("*.csv")],
        ),
    ],
    package_data={
        "map_data": [
            "viewer/templates/*.html",
            "viewer/static/css/*.css",
            "viewer/static/js/*.js",
        ]
    },
    zip_safe=False,
    maintainer="vlkjan6",
    maintainer_email="vlkjan6@fel.cvut.cz",
    description="ROS2 package for downloading, parsing, and visualizing OSM map data with\
                 integrated A* and RRT* path planners.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "osm_cloud = map_data.osm_cloud:main",
            "create_mapdata = map_data.create_mapdata:main",
            "map_data_viewer = map_data.viewer.app:main",
            "map_data_info = map_data.info:main",
        ],
    },
)
