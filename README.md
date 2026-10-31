# map_data

ROS 2 tools to work with OSM data and perform path planning.

[![CI](https://github.com/vras-robotour/map_data/actions/workflows/ci.yml/badge.svg)](https://github.com/vras-robotour/map_data/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://vras-robotour.github.io/map_data/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/vras-robotour/map_data/blob/master/LICENSE)

## Overview

This package parses a `.gpx` file with GPS waypoints into a Python class, queries [OpenStreetMap](https://www.openstreetmap.org) for map features, and performs path planning. It provides ROS 2 nodes as well as standalone Python tools.

## Features

- Parse GPX waypoints and download OSM map features (barriers, footways, roads)
- Serialize and reload map data as `.mapdata` files
- Interactive web-based viewer for inspection, annotation, and mission planning
- Path planning via Graph A\*, Grid A\*, and RRT\*
- ROS 2 nodes for route planning and publishing cost-aware footway point clouds
- Standalone Python library — ROS 2 is required for `osm_cloud`, `route_planner`, and the viewer's live Tracker mode; parsing, CLI planning (`map_data_plan`), inspection (`map_data_info`), and the web viewer's Viewer and Planner modes run standalone without ROS 2

## Documentation

Full documentation, including installation guides, CLI usage, and API reference, is available at:

**[https://vras-robotour.github.io/map_data/](https://vras-robotour.github.io/map_data/)**

## Quick start

### Prerequisites

- Python >= 3.12
- ROS 2 Jazzy or later (required for `osm_cloud`, `route_planner`, and the viewer's Tracker mode)

### Installation

#### Standalone Python workflow (uv)

```bash
git clone https://github.com/vras-robotour/map_data.git
cd map_data

# Install dependencies and sync the virtual environment
uv sync

# Run the test suite
uv run python -m pytest
```

Alternatively, install as an editable package with pip:

```bash
pip install -e .
```

#### ROS 2 workspace (colcon)

The `map_data_interfaces` package is nested inside this repository. Because `colcon` does not discover nested packages from a plain clone, create the workspace symlink using `scripts/setup_ws.sh` before building:

```bash
# From your colcon workspace root (with this repo in src/map_data):
bash src/map_data/scripts/setup_ws.sh
colcon build --packages-up-to map_data
source install/setup.bash
```

The ROS 2 nodes additionally depend on [`ros2_numpy`](https://github.com/Box-Robotics/ros2_numpy), which is not released as a binary package — clone its `jazzy` branch into your colcon workspace `src/` and build it from source.

To verify the ROS 2 runtime stack, run the smoke test script:

```bash
bash scripts/ros_smoke.sh
```

### Basic usage

Download and parse OSM data for a GPX file:

```bash
create_mapdata -d -f coords.gpx
# or, in a sourced ROS 2 workspace:
ros2 run map_data create_mapdata -d -f coords.gpx
```

Inspect a `.mapdata` file (statistics, or `--validate` for structural validation):

```bash
map_data_info coords.mapdata
map_data_info coords.mapdata --validate
```

Plan a route on a `.mapdata` file without the viewer (annotations merged), via CLI or the ROS 2 action:

```bash
map_data_plan -f demo/buchlovice.mapdata --start 49.084416,17.336309 --goal 49.083106,17.338527   # bundled demo, ~420 m
map_data_plan -f coords.mapdata --start 50.1038,14.4294 --goal geo:50.1067,14.4193 --spacing 3 --save route.gpx
ros2 launch map_data route_planner.launch.py mapdata_file:=coords.mapdata   # PlanRoute action on /route_planner/plan_route
```

Launch the interactive viewer:

```bash
map_data_viewer
# or with an explicit data directory, host, and port:
map_data_viewer --data-dir /path/to/data --host 0.0.0.0 --port 8080
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for more information.
