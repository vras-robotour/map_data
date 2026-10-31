# General usage

This page covers the core CLI tools and ROS 2 nodes for parsing, inspecting, and publishing map data.

!!! note "ROS 2 requirement"
    ROS 2 is required for `osm_cloud` and `route_planner`. The `create_mapdata` CLI,
    `map_data_info`, the `MapData` class, path planning modules, and the interactive viewer
    (except the Tracker mode) work **standalone**.

## What gets downloaded

When `create_mapdata` (or `MapData.run_all()`) is called, a single Overpass API union query
is sent covering the bounding box of the GPX waypoints:

```text
[out:json][timeout:50];way(bbox)->.w;(.w; .w >; .w <; node(bbox););out;
```

This retrieves:

- Every OSM way intersecting the bounding box.
- All constituent nodes of those ways.
- Relations that reference those ways.
- Standalone nodes inside the bounding box.

The downloaded data is filtered and classified into `roads_list`, `footways_list`, and
`barriers_list` according to the tag values in the CSV files in `parameters/`. Ways and nodes
whose tags do not match any configured category are discarded during parsing.

### Bounding box calculation

The query bounding box is derived from the minimum and maximum UTM coordinates of the GPX
waypoints, expanded outward in all directions by `grid_margin` (default `150 m` in
`config/planner_defaults.yaml`). The four corner coordinates are converted to WGS-84
latitude/longitude to form the Overpass query bounding box `[min_lat, min_long, max_lat, max_long]`.

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

`map_data_info` is available after installing the package (standalone `pip install -e .`,
`uv sync`, or `colcon build`).

With `--validate`, the tool checks the file for structural issues instead of printing
statistics: missing metadata fields, ways without geometry, duplicate way IDs, nodes
missing from the node cache, and disconnected footway networks. It exits non-zero when
issues are found, so it can be used as a pre-flight check in scripts:

```bash
map_data_info coords.mapdata --validate
```

## Parsing and creating files

`create_mapdata` creates a `.mapdata` file from a `.gpx` waypoint file, or re-parses an
existing `.mapdata` file with current tag configurations.

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `-h, --help` | | Show help message and exit |
| `-f FILE, --file FILE` | `buchlovice.gpx` | GPX file name to parse (with `-d`), or `.mapdata` file to re-parse (without `-d`) |
| `-d, --download` | `false` | Download fresh data from OSM within the waypoint bounds and parse it |
| `--cache` | | Use the local OSM response cache (30-day expiry; ignored by `--no-cache`) |
| `--no-cache` | | Bypass cache and refetch fresh data from OSM |

### OSM caching

Downloaded OSM responses are cached locally keyed by the query hash and bounding box.
If `--cache` is active and a cache entry exists for the exact bounding box and query within
30 days (`OSM_CACHE_MAX_AGE = 30 days`), `create_mapdata` reuses the cached response without
querying Overpass again. Pass `--no-cache` to force a network refetch.

### Output file location

When saving, `create_mapdata` writes the `.mapdata` file next to the input GPX file, replacing
the `.gpx` extension with `.mapdata`.

Download and parse OSM data for a GPX file:

```bash
create_mapdata -d -f coords.gpx
# or, in a sourced ROS 2 workspace:
ros2 run map_data create_mapdata -d -f coords.gpx
```

Re-parse an existing `.mapdata` file (for example after editing tag CSVs):

```bash
create_mapdata -f coords.mapdata
```

## Publishing a point cloud of footways, roads, and intersections

`osm_cloud` is a ROS 2 node that publishes a `sensor_msgs/PointCloud2` on the configured
grid topic (`osm_grid` via launch, or `grid` on the node) with a cost-aware grid around
traversable ways, and optionally publishes intersections as a `geometry_msgs/PoseArray`
and `visualization_msgs/MarkerArray`.

When started without an input file (`mapdata_file` or `gpx_file`), `osm_cloud` logs an error
and idles rather than crashing, preventing a tight restart loop under launch respawn. Runtime
parameter changes via ROS 2 parameters are validated before applying; invalid changes are
rejected (`SetParametersResult(successful=False)`).

### ROS 2 parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mapdata_file` | `""` | Absolute path to a `.mapdata` file |
| `gpx_file` | `""` | Absolute path to a `.gpx` file (used when `mapdata_file` is empty) |
| `grid_topic` | `grid` | Topic name for the published point cloud (`osm_grid` in `osm_cloud.launch.py`) |
| `intersections_topic` | `intersections` | Topic name for intersection poses (`PoseArray`) |
| `intersection_markers_topic` | `intersection_markers` | Topic name for intersection markers (`MarkerArray`) |
| `local_frame` | `map` | TF frame the grid and intersections are published in (`FP_ENU0` in `osm_grid.yaml`) |
| `utm_frame` | `utm` | TF frame name for UTM coordinates (`transform_mode` `tf` or `auto`) |
| `earth_frame` | `FP_ECEF` | ECEF TF frame used by `transform_mode: geodetic` |
| `transform_mode` | `tf` | Placement mode: `tf` looks up `utm_frame -> local_frame` on `/tf_static`; `auto` calculates the local frame at the map center and broadcasts the static transform; `geodetic` converts UTM -> lat/lon -> ECEF and applies `earth_frame -> local_frame` from `/tf_static` (exact for GNSS/INS stacks; default in `osm_grid.yaml`) |
| `annotations` | `auto` | Annotation store merged into `mapdata_file`: `auto` = `<map>.annotations.json` next to it, `none` = unedited map, or a file path |
| `exclude_highway` | `["steps"]` | `highway=` tag values dropped from the map (non-routable, so they receive no intersection rings) |
| `traversability_file` | `""` | Tag rule file deciding which ways are traversable (`""` = package default `config/traversability.yaml`). [Traversability rules warning](planning.md#traversability-rules). Must match `route_planner`'s file |
| `highway_types` | `["footway"]` | Way types the grid is drawn from: `footway`, `road`, or both. Must match `route_planner`'s `highway_types` |
| `save_mapdata` | `false` | Save generated mapdata when loading from a `.gpx` |
| `max_path_dist` | `1.0` | Maximum distance (m) from way centerlines at which grid cells receive a cost |
| `neighbor_cost` | `linear` | Cost falloff for cells near ways: `linear`, `quadratic`, or `zero` |
| `grid_res` | `0.25` | Grid resolution / point spacing (m) |
| `grid_max` | `[0.0, 0.0]` | Upper bounds of the local-frame grid (m); `[0, 0]` enables auto-calculation |
| `grid_min` | `[0.0, 0.0]` | Lower bounds of the local-frame grid (m); `[0, 0]` enables auto-calculation |
| `publish_intersections` | `true` | Whether to publish footway intersections |
| `republish_period` | `0.0` | Periodic re-publish interval (s); `0.0` publishes once (latched) and on changes |

### Static transform placement

`osm_cloud` listens to `/tf_static` for its placement transform rather than subscribing to
the high-rate `/tf` stream. When the placement transform changes (for example when a GNSS/INS
driver restarts with a new ENU0 origin), the grid and intersections are rebuilt and re-published
automatically. In `geodetic` mode, all-zero transforms (such as an uninitialized fusion fix)
are rejected and waited out.

### Launching

Run the node via its launch file:

```bash
ros2 launch map_data osm_cloud.launch.py \
    mapdata_file:=/path/to/coords.mapdata \
    grid_topic:=osm_grid
```

| Argument | Default | Description |
|----------|---------|-------------|
| `config_file` | `helhest_jr.yaml` | Parameter file in `config/` (or path) with topic configurations |
| `osm_grid_params` | `osm_grid.yaml` | Parameter file in `config/` (or path) with OSM grid parameters |
| `mapdata_path` | package share `data` | Directory where bare map filenames are resolved |
| `mapdata_file` | `""` | Map data filename (resolved in `mapdata_path` or absolute) |
| `gpx_file` | `""` | GPX fallback filename (resolved in `mapdata_path` or absolute) |
| `grid_topic` | `osm_grid` | Topic name for the published point cloud |
| `local_frame` | `""` | Override frame for the grid/intersections (empty = value from `osm_grid_params`) |
| `utm_frame` | `""` | Override UTM frame name (empty = from yaml) |
| `earth_frame` | `""` | Override ECEF frame name in `geodetic` mode (empty = from yaml) |
| `transform_mode` | `""` | Override placement mode: `tf`, `auto`, or `geodetic` (empty = from yaml) |
| `annotations` | `auto` | Annotation store merged into the map: `auto`, `none`, or file path |
| `traversability` | `""` | Tag rule file deciding traversable ways (empty = from yaml) |
| `highway_types` | `""` | Way types drawn into the grid: `footway`, `road`, or `footway,road` (empty = from yaml) |
