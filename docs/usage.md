# General Usage

This page covers the core CLI tools and ROS2 nodes for parsing, visualizing, and publishing map data.

!!! note "ROS2 requirement"
    Only `osm_cloud` requires a sourced ROS2 workspace. The `create_mapdata` CLI,
    the `MapData` class, path planning modules, and the interactive viewer work **standalone**.

## What gets downloaded

When `create_mapdata` (or `MapData.run_all()`) is called, a **single**
[Overpass API](https://overpass-api.de/) query is sent, covering the bounding box of the
waypoints. It returns the union of four element sets:

| Set | Overpass fragment | What it fetches |
|-----|-------------------|-----------------|
| Ways | `way(bbox)->.w` | Every OSM way inside the bounding box |
| Their nodes | `.w >` | Recurse down — every node making up those ways |
| Their parents | `.w <` | Recurse up — every relation (and way) referencing them |
| Standalone nodes | `node(bbox)` | Every OSM node inside the box, including ones belonging to no way |

Combining these into one request instead of three concurrent ones means a single round trip
per mirror, one shared server-side timeout, and one cached response.

The downloaded data is then filtered and classified into `roads_list`, `footways_list`, and
`barriers_list` according to the tag values in the CSV files in `parameters/`. Ways and nodes
whose tags do not match any configured category are discarded during parsing, not during
download.

!!! tip "Responses are cached"
    The raw response is written to a `.osm_cache.json` file alongside the source waypoint
    file and reused on the next run when the stored bounding box matches, so re-parsing with
    changed tag CSVs does not re-query Overpass. Pass `use_cache=False` to `run_queries()`
    to force a fresh download. Map data built from a coordinate array rather than a file is
    not cached.

### Bounding box and margins

The waypoints are converted to UTM, and the query bounding box is their **axis-aligned**
min/max extent expanded by `grid_margin` on all four sides — **150 m beyond the outermost
waypoint** by default. That expanded box is converted back to WGS84 for the Overpass query.

A single margin serves both purposes:

- It sets the internal UTM planning bounds (`min_x`/`max_x`/`min_y`/`max_y`) used by the
  path planners.
- Because the query box is derived from those same bounds, it also guarantees that features
  near the route boundary — barriers and building footprints in particular — are downloaded
  in full even when their geometry extends past the last waypoint.

`grid_margin` is configurable in `config/planner_defaults.yaml`, per call via
`MapData(..., grid_margin=300)`, or from the viewer's fetch panel. Increase it for routes
near dense urban areas with large building footprints; decrease it to reduce query time and
file size for simple open-terrain routes. See
[Planner Configuration](dev/planner_config.md#top-level-parameters) for the full parameter
reference.

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

With `--validate`, the tool checks the file for structural issues instead of printing
statistics: missing metadata fields, ways without geometry, duplicate way IDs, nodes
missing from the node cache, and disconnected footway networks. It exits non-zero when
issues are found, so it can be used as a pre-flight check in scripts:

```bash
map_data_info coords.mapdata --validate
```

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
create_mapdata -d -f coords.gpx
# or, in a sourced ROS2 workspace:
ros2 run map_data create_mapdata -d -f coords.gpx
```

This creates `coords.mapdata` in the package data directory.

Re-parse an existing `.mapdata` file (e.g. after editing tag CSVs):

```bash
create_mapdata -f coords.mapdata
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
