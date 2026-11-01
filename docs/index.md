# Overview

ROS 2 tools to work with OSM data and perform path planning.

## About

This package parses a `.gpx` file with GPS waypoints into a Python class, queries [OpenStreetMap](https://www.openstreetmap.org) for map features within the area, and serializes the result as a `.mapdata` file.

Parsed features are classified into three categories:

- **barriers** — obstacles assumed to be untraversable (walls, buildings, fences, water…)
- **footways** — paths intended for pedestrians
- **roads** — areas intended for vehicles

Additional tools let you visualize the data, annotate it interactively, perform path planning, and publish a cost-aware footway point cloud for autonomous navigation.

The `MapData` class, the path planning modules, the `create_mapdata` CLI, and the interactive viewer work **standalone** without ROS 2 for data parsing, annotation, and planning. ROS 2 is required for `osm_cloud`, `route_planner`, and the viewer's live Tracker mode.

The package targets **ROS 2 Jazzy** or later on Ubuntu 24.04, and Python >= 3.12 for standalone use.

A sample GPX file and its `.mapdata` are in `demo/` (`buchlovice.gpx`, `buchlovice.mapdata`).

## Installation

### Standalone Python package (uv)

```bash
git clone https://github.com/vras-robotour/map_data.git
cd map_data
uv sync
uv run python -m pytest
```

Alternatively, install as an editable package with pip:

```bash
pip install -e .
```

### With ROS 2 (colcon)

The `map_data_interfaces` package is nested inside this repository. Because `colcon` does not discover nested packages from a plain clone, run `scripts/setup_ws.sh` to create the workspace symlink before building:

```bash
# From your colcon workspace root (with this repo in src/map_data):
bash src/map_data/scripts/setup_ws.sh
colcon build --packages-up-to map_data
source install/setup.bash
```

!!! note "`ros2_numpy` must be built from source"
    The ROS 2 nodes depend on [`ros2_numpy`](https://github.com/Box-Robotics/ros2_numpy),
    which is not released as a binary package in any ROS distribution — `rosdep install`
    cannot resolve it. Clone its `jazzy` branch into your colcon workspace `src/` and build it
    alongside this package.

`map_data_viewer` also wants `simple-websocket` (`pip install simple-websocket`), which
`rosdep` cannot install; without it the browser polls over HTTP instead of a WebSocket
(see [Tracker](viewer_tracker.md)).

To check the ROS 2 runtime nodes, run the smoke test script:

```bash
bash scripts/ros_smoke.sh
```

## Python library

`MapData` and `Way` can be used directly in Python without a running ROS 2 node.
The only requirement is that the package is installed (e.g. via `pip install -e .`,
`uv sync`, or `colcon build`) so the `parameters/` CSV files are accessible.

### Examples

**Parse a GPX file and save a `.mapdata` file:**

```python
from map_data.map_data import MapData

md = MapData("./demo/buchlovice.gpx")
md.run_all(save=True)  # queries OSM, parses, saves demo/buchlovice.mapdata
```

**Load an existing `.mapdata` file and access parsed features:**

```python
from map_data.map_data import MapData

md = MapData.load("demo/buchlovice.mapdata")

print(len(md.roads_list))  # list of Way objects
print(len(md.footways_list))
print(len(md.barriers_list))
```

## License

BSD 3-Clause License. See the repository `LICENSE` file for full terms.
