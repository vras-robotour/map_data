# map_data

ROS2 tools to work with OSM data and perform path planning.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://vras-robotour.github.io/map_data/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/vras-robotour/map_data/blob/master/LICENSE)

## Overview

This package parses a `.gpx` file with GPS waypoints into a Python class, queries [OpenStreetMap](https://www.openstreetmap.org) for map features, and performs path planning. It provides both ROS2 nodes and standalone Python tools.

## Documentation

Full documentation, including installation guides, CLI usage, and API reference, is available at:

👉 **[https://vras-robotour.github.io/map_data/](https://vras-robotour.github.io/map_data/)**

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vras-robotour/map_data.git
cd map_data

# Install dependencies
pip install -r requirements.txt

# (Optional) Build with colcon if using ROS2
colcon build --packages-select map_data
```

### Basic Usage

Download and parse OSM data for a GPX file:

```bash
ros2 run map_data create_mapdata -d -f coords.gpx
```

Launch the interactive viewer:

```bash
map_data_viewer
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for more information.
