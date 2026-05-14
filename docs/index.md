# map_data

ROS2 tools to work with OSM data and perform path planning.

## Overview

This package parses a `.gpx` file with GPS waypoints into a Python class, queries
[OpenStreetMap](https://www.openstreetmap.org) for map features within the area, and
serializes the result as a `.mapdata` file.

Parsed features are classified into three categories:

- **barriers** — obstacles assumed to be untraversable (walls, buildings, fences, water…)
- **footways** — paths intended for pedestrians
- **roads** — areas intended for vehicles

Additional tools let you visualize the data, annotate it interactively, perform path planning, and publish a
cost-aware footway point cloud for use in autonomous navigation.

The `MapData` class, the path planning modules, and the interactive viewer work **standalone** — no running ROS2
context is required for data parsing, annotation, or planning. ROS2 is only needed for the
`osm_cloud` publisher and the `create_mapdata` / `visualize_mapdata` CLI nodes.

The package targets **ROS2 Humble** or later on Ubuntu 22.04.

Sample `.gpx` files are provided in `./data/`.

## Using as a Python Library

`MapData` and `Way` can be used directly in Python without a running ROS2 node.
The only requirement is that the package is installed (e.g. via `pip install -e .`
or a `colcon build`) so the `parameters/` CSV files are accessible.

### Examples

**Parse a GPX file and save a `.mapdata` file:**

```python
from map_data.map_data import MapData

md = MapData("./data/coords.gpx")
md.run_all(save=True)  # queries OSM, parses, saves coords.mapdata
```

**Load an existing `.mapdata` file and access parsed features:**

```python
from map_data.map_data import MapData

md = MapData.load("coords.mapdata")

print(len(md.roads_list))    # list of Way objects
print(len(md.footways_list))
print(len(md.barriers_list))
```

**Plot the parsed data:**

```python
from map_data.map_data import MapData
from map_data.vis_utils import plot_map
import matplotlib.pyplot as plt

md = MapData.load("coords.mapdata")

plot_map(md)
plt.show()
```

## Project Structure

```text
map_data/
├── map_data.py              # MapData class — parses GPX + OSM into roads/footways/barriers
├── info.py                  # CLI tool to print information about a .mapdata file
├── create_mapdata.py        # ROS2 node / CLI: download and parse OSM data
├── visualize_mapdata.py     # ROS2 node / CLI: static matplotlib plots
├── osm_cloud.py             # ROS2 node: publishes footway grid and intersections
├── pathsolver/              # Path planning algorithms
│   ├── graph_planner.py     # Global A* planning on OSM ways
│   ├── replan.py            # Local/Grid-based replanning (Grid A*, RRT*)
│   ├── grid_astar.py        # Grid-based A* implementation
│   ├── rrt_star.py          # RRT* implementation
│   └── astar.py             # Generic A* search logic
├── utils/                   # Shared utility functions
│   ├── way.py               # Way class — represents a single OSM feature with geometry
│   ├── overpass.py          # OSM Overpass API client
│   ├── parsing.py           # OSM XML/JSON parsing logic
│   ├── serialization.py     # .mapdata file I/O
│   ├── background_map.py    # Raster map tile fetching
│   ├── vis_utils.py         # matplotlib plotting helpers
│   └── points_to_graph_points.py # Equidistant point interpolation
└── viewer/                  # Modular interactive viewer (Flask + Leaflet)
    ├── app.py               # App factory and server entry point
    ├── routes.py            # REST API endpoints and GeoJSON conversion
    ├── helpers.py           # Geometry and annotation utility functions
    ├── cache.py             # MapData object caching
    ├── templates/           # HTML templates
    └── static/              # External CSS and Modular JS assets
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](https://github.com/vras-robotour/map_data/blob/master/LICENSE) file for details.
