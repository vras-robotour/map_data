# map_data
ROS2 tools to work with OSM data.

- [Overview](#overview)
- [How to use](#how-to-use)
  - [Parsing and creating files](#parsing-and-creating-files)
  - [Interactive viewer](#interactive-viewer)
  - [Visualizing the parsed data](#visualizing-the-parsed-data)
  - [Publishing a point cloud of footways](#publishing-a-point-cloud-of-footways)
- [Importing as a Python package](#importing-as-a-python-package)
  - [Files](#files)
  - [Examples](#examples)
- [License](#license)

## Overview
This package parses a `.gpx` file with GPS waypoints into a Python class, queries
[OpenStreetMap](https://www.openstreetmap.org) for map features within the area, and
serializes the result as a `.mapdata` file.

Parsed features are classified into three categories:
- **barriers** — obstacles assumed to be untraversable (walls, buildings, fences, water…)
- **footways** — paths intended for pedestrians
- **roads** — areas intended for vehicles

Additional tools let you visualize the data, annotate it interactively, and publish a
cost-aware footway point cloud for use in path-planning nodes.

The `MapData` class and the interactive viewer work **standalone** — no running ROS2
context is required for data parsing or annotation. ROS2 is only needed for the
`osm_cloud` publisher and the `create_mapdata` / `visualize_mapdata` CLI nodes.

The package targets **ROS2 Humble** or later on Ubuntu 22.04.

Sample `.gpx` files are provided in `./data/`.

## How to use

### Parsing and creating files
`create_mapdata` creates a `.mapdata` file from a `.gpx` file, or re-parses an existing
`.mapdata` with the current tag configuration.

| Flag | Description |
|------|-------------|
| `-d` | Download fresh OSM data from the `.gpx` bounds and parse it |
| `-f <filename>` | `.gpx` file to parse (with `-d`), or `.mapdata` file to re-parse (without `-d`) |

Default input file when `-f` is omitted: `buchlovice.gpx`.

Download and parse OSM data for a GPX file:
```bash
ros2 run map_data create_mapdata -d -f coords.gpx
```
This creates `coords.mapdata` in the package data directory.

Re-parse an existing `.mapdata` file (e.g. after editing tag CSVs):
```bash
ros2 run map_data create_mapdata -f coords.mapdata
```

### Interactive viewer
`map_data_viewer` launches a local Flask web server with a Leaflet-based map UI.
It lets you inspect parsed features, view their OSM tags, and draw manual obstacle
annotations — without needing ROS2.

```bash
# After colcon build and sourcing the workspace:
map_data_viewer

# Explicit data directory and port:
map_data_viewer --data-dir /path/to/data --port 8080

# Standalone (outside a colcon workspace):
python -m map_data.viewer.app --data-dir ./data
```

Then open `http://127.0.0.1:5000` in a browser.

**Modes** (toolbar at the top):

| Mode | Action |
|------|--------|
| View | Click any feature to see its OSM tags in the sidebar |
| + Obstacle | Draw a polygon, rectangle, or circle to add a manual obstacle |
| ✕ Delete | Click an annotation to remove it |

Layer visibility (roads, footways, barriers, waypoints, annotations) can be toggled
independently in the sidebar.

Annotations are saved automatically to a sidecar `.annotations.json` file alongside
the `.mapdata` file, so they survive re-parsing.

### Visualizing the parsed data
`visualize_mapdata` generates static matplotlib plots from a `.mapdata` file.
It shows two figures: one with all parsed features (barriers, footways, roads) and
one with footways only.

| Flag | Description |
|------|-------------|
| `-f <filename>` | `.mapdata` file to visualize (default: `buchlovice.mapdata`) |
| `-sm` | Save the main map plot (default filename: `map.png`) |
| `-if <filename>` | Custom filename for the main plot (requires `-sm`) |
| `-sb` | Save the background tile image (default filename: `bgd_map.png`) |
| `-bf <filename>` | Custom filename for the background image (requires `-sb`) |

All saved images are written to the `./data/` directory.

Visualize and display interactively:
```bash
ros2 run map_data visualize_mapdata -f coords.mapdata
```

Visualize and save both images:
```bash
ros2 run map_data visualize_mapdata -f coords.mapdata -sm -sb
```

### Publishing a point cloud of footways
`osm_cloud` is a ROS2 node that publishes a `sensor_msgs/PointCloud2` on the `grid`
topic. Each point carries a `cost` field: `0.0` means on a footway, `1.0` means at
the maximum configured distance from any footway.

**ROS2 parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `utm_frame` | `"utm"` | TF frame name for UTM coordinates |
| `local_frame` | `"map"` | TF frame name for the local robot frame |
| `utm_to_local` | `None` | 4×4 transform matrix; looked up from TF if not set |
| `mapdata_file` | `None` | Absolute path to a `.mapdata` file |
| `gpx_file` | `None` | Absolute path to a `.gpx` file (used if no `.mapdata`) |
| `save_mapdata` | `false` | Save generated mapdata when loading from a `.gpx` |
| `max_path_dist` | `1.0` | Max distance (m) at which a grid point receives a cost |
| `neighbor_cost` | `"linear"` | Cost function: `linear`, `quadratic`, or `zero` |
| `grid_res` | `0.25` | Grid point spacing (m) |
| `grid_max` | `[250, 250]` | Upper bounds of the local-frame grid (m) |
| `grid_min` | `[-250, -250]` | Lower bounds of the local-frame grid (m) |

The recommended way to run the node is via the provided launch file, which also sets
up the required static transforms:

```bash
ros2 launch map_data osm_cloud.launch.py \
    mapdata_file:=/path/to/coords.mapdata \
    grid_topic:=osm_cloud
```

Launch file arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `mapdata_path` | package share/data | Directory containing the map file |
| `mapdata_file` | `buchlovice.mapdata` | Map data filename |
| `gpx_file` | `buchlovice.gpx` | GPX fallback filename |
| `grid_topic` | `osm_grid` | Topic name for the published point cloud |

## Importing as a Python package

`MapData` and `Way` can be used directly in Python without a running ROS2 node.
The only requirement is that the package is installed (e.g. via `pip install -e .`
or a `colcon build`) so the `parameters/` CSV files are accessible.

### Files

```
map_data/
├── map_data.py              # MapData class — parses GPX + OSM into roads/footways/barriers
├── way.py                   # Way class — represents a single OSM feature with geometry
├── background_map.py        # Fetches raster map tiles from Geoapify for visualizations
├── vis_utils.py             # matplotlib helpers for plotting parsed map data
├── points_to_graph_points.py# Utility: equidistant point interpolation along lines
├── create_mapdata.py        # ROS2 node / CLI: download and parse OSM data
├── visualize_mapdata.py     # ROS2 node / CLI: static matplotlib plots
├── osm_cloud.py             # ROS2 node: publishes footway cost point cloud
└── viewer/
    ├── app.py               # Flask backend — REST API + GeoJSON conversion
    └── templates/
        └── index.html       # Leaflet frontend — interactive map viewer
```

### Examples

Parse a GPX file and save a `.mapdata` file:
```python
from map_data.map_data import MapData

md = MapData("./data/coords.gpx")
md.run_all(save=True)  # queries OSM, parses, saves coords.mapdata
```

Load an existing `.mapdata` file and access parsed features:
```python
import pickle

with open("coords.mapdata", "rb") as fh:
    md = pickle.load(fh)

print(len(md.roads_list))    # list of Way objects
print(len(md.footways_list))
print(len(md.barriers_list))
```

Plot the parsed data:
```python
import pickle
from map_data.vis_utils import plot_map
import matplotlib.pyplot as plt

with open("coords.mapdata", "rb") as fh:
    md = pickle.load(fh)

plot_map(md)
plt.show()
```

## License
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/vras-robotour/map_data/blob/master/LICENSE)
