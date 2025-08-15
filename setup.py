from setuptools import find_packages, setup

package_name = "map_data"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/osm_cloud.launch.py"]),
        ("share/" + package_name + "/data", ["data/buchlovice.gpx"]),
        ("share/" + package_name + "/parameters", ["parameters/anti_barrier_tags.csv"]),
        (
            "share/" + package_name + "/parameters",
            ["parameters/anti_obstacle_tags.csv"],
        ),
        ("share/" + package_name + "/parameters", ["parameters/barrier_tags.csv"]),
        ("share/" + package_name + "/parameters", ["parameters/not_barrier_tags.csv"]),
        ("share/" + package_name + "/parameters", ["parameters/not_obstacle_tags.csv"]),
        ("share/" + package_name + "/parameters", ["parameters/obstacle_tags.csv"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
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
        ],
    },
)
