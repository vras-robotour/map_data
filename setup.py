from pathlib import Path

from setuptools import find_namespace_packages, setup

package_name = "map_data"

# Version and dependencies are defined in pyproject.toml; this file only
# carries the ROS/ament-specific metadata (data_files, entry points).
setup(
    name=package_name,
    python_requires=">=3.12",
    # Namespace discovery lists the static/template dirs as packages, so their
    # package_data installs without setuptools' "importable package" warning.
    packages=find_namespace_packages(include=["map_data", "map_data.*"], exclude=["*.__pycache__"]),
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
            [
                str(p)
                for pat in ("*.gpx", "*.mapdata", "*.annotations.json")
                for p in Path("demo").glob(pat)
            ],
        ),
        (
            str(Path("share") / package_name / "parameters"),
            [str(p) for p in Path("parameters").glob("*.csv")],
        ),
    ],
    package_data={
        "map_data.viewer.templates": ["*.html"],
        "map_data.viewer.static.css": ["*.css"],
        "map_data.viewer.static.js": ["*.js"],
    },
    zip_safe=False,
    maintainer="vlkjan6",
    maintainer_email="vlkjan6@fel.cvut.cz",
    description=(
        "ROS2 package for downloading, parsing, and visualizing OSM map data "
        "with integrated A* and RRT* path planners."
    ),
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "osm_cloud = map_data.osm_cloud:main",
            "create_mapdata = map_data.create_mapdata:main",
            "map_data_viewer = map_data.viewer.app:main",
            "map_data_info = map_data.info:main",
            "map_data_plan = map_data.plan_route_cli:main",
            "route_planner = map_data.route_planner:main",
        ],
    },
)
