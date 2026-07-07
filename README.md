# map_data

ROS2 tools to work with OSM data and perform path planning.

[![CI](https://github.com/vras-robotour/map_data/actions/workflows/ci.yml/badge.svg)](https://github.com/vras-robotour/map_data/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://vras-robotour.github.io/map_data/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/vras-robotour/map_data/blob/master/LICENSE)

## Overview

This package parses a `.gpx` file with GPS waypoints into a Python class, queries [OpenStreetMap](https://www.openstreetmap.org) for map features, and performs path planning. It provides both ROS2 nodes and standalone Python tools.

## Features

- Parse GPX waypoints and download OSM map features (barriers, footways, roads)
- Serialize and reload map data as `.mapdata` files
- Interactive web-based viewer for inspection, annotation, and mission planning
- Path planning via Graph A\*, Grid A\*, and RRT\*
- ROS2 node for publishing cost-aware footway point clouds
- Standalone Python library — no ROS2 context required for parsing and planning

## Documentation

Full documentation, including installation guides, CLI usage, and API reference, is available at:

**[https://vras-robotour.github.io/map_data/](https://vras-robotour.github.io/map_data/)**

## Quick Start

### Prerequisites

- Python 3.12+
- ROS2 Jazzy or later (required only for the `osm_cloud` node)

### Installation

```bash
# Clone the repository
git clone https://github.com/vras-robotour/map_data.git
cd map_data

# Install as an editable package (includes all dependencies)
pip install -e .

# (Optional) Build with colcon if using ROS2
colcon build --packages-select map_data
```

The `osm_cloud` node additionally depends on [`ros2_numpy`](https://github.com/Box-Robotics/ros2_numpy),
which is not released as a binary package — clone its `jazzy` branch into your colcon
workspace and build it from source.

### Basic Usage

Download and parse OSM data for a GPX file:

```bash
create_mapdata -d -f coords.gpx
# or, in a sourced ROS2 workspace:
ros2 run map_data create_mapdata -d -f coords.gpx
```

Launch the interactive viewer:

```bash
map_data_viewer
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for more information.
