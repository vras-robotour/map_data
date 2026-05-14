# General Usage

This page covers the core CLI tools and ROS2 nodes for parsing, visualizing, and publishing map data.

!!! note "ROS2 requirement"
    `create_mapdata`, `visualize_mapdata`, and `osm_cloud` require a sourced ROS2 workspace.
    The `MapData` class, path planning modules, and the interactive viewer work **standalone**.

## Inspecting a .mapdata file

`map_data_info` prints statistics about a `.mapdata` file — feature counts, total footway
distance, covered area, UTM zone, and any stored annotations.

```bash
map_data_info coords.mapdata
```

Example output:

```text
========================================
MAP DATA STATISTICS: coords.mapdata
========================================
Source:      File: coords.gpx
UTM Zone:    33U
Bounds X:    [458200.0, 458900.0]
Bounds Y:    [5550100.0, 5550700.0]
Total Area:  420,000 m²
----------------------------------------
Roads:       12
Footways:    47
Barriers:    83
Total Footway Distance: 4823.6 m
Annotations: 3 (manual edits)
========================================
```

`map_data_info` is available after installing the package (standalone `pip install -e .` or `colcon build`).

## Parsing and creating files

`create_mapdata` creates a `.mapdata` file from a `.gpx` or `.yaml` waypoint file, or
re-parses an existing `.mapdata` with the current tag configuration.

| Flag | Description |
|------|-------------|
| `-d` | Download fresh OSM data from the input file's bounds and parse it |
| `-f <filename>` | `.gpx` or `.yaml` waypoint file to parse (with `-d`), or `.mapdata` file to re-parse (without `-d`) |

!!! tip "YAML waypoint format"
    Besides `.gpx` files, `create_mapdata` and `MapData` also accept a simple YAML format:

    ```yaml
    waypoints:
      - latitude: 50.1234
        longitude: 14.5678
        elevation: 200.0   # optional, defaults to 0
      - latitude: 50.1240
        longitude: 14.5690
    ```

    Save the file with a `.yaml` extension and use it wherever a `.gpx` file is accepted.

Download and parse OSM data for a GPX file:

```bash
ros2 run map_data create_mapdata -d -f coords.gpx
```

This creates `coords.mapdata` in the package data directory.

Re-parse an existing `.mapdata` file (e.g. after editing tag CSVs):

```bash
ros2 run map_data create_mapdata -f coords.mapdata
```

## Visualizing the parsed data

`visualize_mapdata` generates static matplotlib plots from a `.mapdata` file.
It shows two figures: one with all parsed features (barriers, footways, roads) and
one with footways only.

| Flag | Description |
|------|-------------|
| `-f <filename>` | `.mapdata` file to visualize |
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

## Publishing a point cloud of footways and intersections

`osm_cloud` is a ROS2 node that publishes a `sensor_msgs/PointCloud2` on the `grid`
topic (cost-aware footway grid) and optionally publishes intersections as a
`geometry_msgs/PoseArray` and `visualization_msgs/MarkerArray`.

### ROS2 parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `utm_frame` | `"utm"` | TF frame name for UTM coordinates |
| `local_frame` | `"local_utm"` | TF frame name for the local robot frame |
| `utm_to_local` | `None` | 4×4 transform matrix; looked up from TF if not set |
| `mapdata_file` | `None` | Absolute path to a `.mapdata` file |
| `gpx_file` | `None` | Absolute path to a `.gpx` file (used if no `.mapdata`) |
| `save_mapdata` | `false` | Save generated mapdata when loading from a `.gpx` |
| `max_path_dist` | `1.0` | Max distance (m) at which a grid point receives a cost |
| `neighbor_cost` | `"linear"` | Cost falloff for cells near ways: `"linear"` = proportional to distance, `"quadratic"` = distance squared, `"zero"` = constant regardless of distance |
| `grid_res` | `0.25` | Grid point spacing (m) |
| `grid_max` | `[0.0, 0.0]` | Upper bounds of the local-frame grid (m). `[0, 0]` triggers auto-calc. |
| `grid_min` | `[0.0, 0.0]` | Lower bounds of the local-frame grid (m). `[0, 0]` triggers auto-calc. |
| `auto_utm` | `false` | Auto-calculate UTM-to-local transform from map center |
| `publish_intersections` | `true` | Whether to publish footway intersections |

### Launching

The recommended way to run the node is via the provided launch file, which also sets
up the required static transforms and enables intersection publishing by default:

```bash
ros2 launch map_data osm_cloud.launch.py \
    mapdata_file:=/path/to/coords.mapdata \
    grid_topic:=osm_cloud
```

| Argument | Default | Description |
|----------|---------|-------------|
| `mapdata_path` | package share/data | Directory containing the map file |
| `mapdata_file` | `stromovka.mapdata` | Map data filename |
| `gpx_file` | `stromovka.gpx` | GPX fallback filename |
| `grid_topic` | `osm_grid` | Topic name for the published point cloud |
| `publish_static_tf` | `false` | Whether to publish static transforms for utm/map |
