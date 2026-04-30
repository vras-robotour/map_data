import os
from glob import glob
from setuptools import find_packages, setup

package_name = "map_data"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "data"), glob("data/*.gpx")),
        (os.path.join("share", package_name, "data"), glob("data/*.mapdata")),
        (os.path.join("share", package_name, "parameters"), glob("parameters/*.csv")),
    ],
    install_requires=[
        "setuptools",
        "flask",
        "gpxpy",
        "numpy",
        "overpy",
        "shapely",
        "utm",
        "tqdm",
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
    description="TODO: Package description",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "osm_cloud = map_data.osm_cloud:main",
            "create_mapdata = map_data.create_mapdata:main",
            "visualize_mapdata = map_data.visualize_mapdata:main",
            "map_data_viewer = map_data.viewer.app:main",
        ],
    },
)
