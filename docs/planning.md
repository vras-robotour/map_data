# Path planning

The `pathsolver` module provides standalone planning capabilities — no ROS 2 context is required.
Use the `map_data_plan` CLI to process waypoints against a `.mapdata` file, import the
planners directly as a Python library, or run the `route_planner` ROS 2 action server.

## Offline route planning: `map_data_plan` and `route_planner`

The viewer's Planner screen (`POST /api/create_replan`), the `map_data_plan` command-line
tool, and the `route_planner` ROS 2 action server all call `map_data.pathsolver.route.plan_route`
on the same map data: the `.mapdata` file with its `<stem>.annotations.json` sidecar merged in
(`map_data.annotations.load_mapdata_with_annotations`). Whatever is drawn, deleted, or split
in the viewer is therefore reflected during planning.

Both tools support an explicit annotation choice: `--annotations auto|none|FILE` on the command
line, or the `annotations` parameter on the node. `none` plans on the unedited OSM map, which
is useful when an annotation store was pruned for another area.

Planner defaults come from `config/planner_defaults.yaml` (`cell_size`, `inflate_obstacles`,
`simplify_path`, `smooth_path`, `max_path_dist`, `grid_cost_weight`). `route_planner` exposes
the first four as ROS 2 parameters, re-read on every goal.

### Goal QR codes (Planner screen)

Robotour provides goals as QR codes with a `geo:lat,lon` payload. In the Planner screen,
right-click any waypoint and choose *QR code*, or press *QR goal* for the last waypoint:
the code opens full screen on a white background (size slider) so a laptop or tablet can be
held to the robot camera, with *Download SVG* (or *PNG*) for printing and *Copy geo text* for typing the
goal in with `qr_goal_send`. The code comes from `GET /api/qr.svg?lat=…&lon=…[&scale=12][&download=1][&caption=]`
as vector art, so it stays sharp at any size; `GET /api/qr?…` serves the same code as a PNG.

### Command line: `map_data_plan`

```bash
# Bundled demo map (the two waypoints of demo/buchlovice.gpx, ~420 m)
map_data_plan -f demo/buchlovice.mapdata --start 49.084416,17.336309 --goal 49.083106,17.338527

# Paths only (graph planner on footways), 3 m waypoint spacing, saved as a GPX track
map_data_plan -f coords.mapdata --start 50.1038,14.4294 --goal geo:50.1067,14.4193 \
    --spacing 3 --save route.gpx

# All-terrain (grid A*) through intermediate via points, printed as JSON
map_data_plan -f coords.mapdata -p 50.1038,14.4294 -p 50.1050,14.4250 -p 50.1067,14.4193 \
    --algorithm astar --cell-size 0.5 --json
```

#### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `-h, --help` | | Show help message and exit |
| `-f FILE, --file FILE` | *required* | `.mapdata` file path or filename in `data/` (or the source checkout's `demo/`) |
| `--annotations auto\|none\|FILE` | `auto` | Annotation store to merge: `auto` (`<file>.annotations.json`), `none`, or explicit file |
| `--start START` | | Start coordinates as `lat,lon` |
| `--goal GOAL` | | Goal coordinates as `lat,lon` or `geo:lat,lon` |
| `-p POINTS, --point POINTS` | | Intermediate via point `lat,lon` (repeat for multiple points between start and goal) |
| `--algorithm {graph,astar,rrt}` | `graph` | Planning algorithm: `graph` (on ways only), `astar` or `rrt` (all-terrain grid) |
| `--ways TYPE [TYPE ...]` | `footway` | Allowed way types: `footway`, `road`, or both (`--ways footway road`) |
| `--traversability FILE` | package default | Tag rule file determining drivable ways and costs (`config/traversability.yaml`) |
| `--no-traversability` | `false` | Ignore traversability rules entirely (stairs remain excluded) |
| `--spacing SPACING` | `0.0` | Maximum meters between output waypoints (`0.0` keeps planner vertices) |
| `--max-snap-distance DIST` | `100.0` | Graph: reject waypoints farther than this (m) from every allowed way |
| `--cell-size CELL_SIZE` | `0.25` | Grid: cell resolution in meters |
| `--inflate-obstacles DIST` | `0.25` | Grid: barrier inflation margin in meters |
| `--simplify`, `--no-simplify` | `--simplify` | Grid: simplify path vertices using Douglas-Peucker (`--no-simplify` retains all cells) |
| `--smooth`, `--no-smooth` | `--no-smooth` | Grid: smooth path using gradient-descent optimization |
| `--save SAVE` | | Write route to GPX file |
| `--wpt` | `false` | Write GPX `<wpt>` elements instead of a `<trk>` track |
| `--json` | `false` | Print planning result as JSON |
| `-v, --verbose` | `false` | Enable verbose debug logging |

### ROS 2 action server: `route_planner`

The `route_planner` node exposes an action server on `/route_planner/plan_route` (`map_data_interfaces/action/PlanRoute`).

```bash
ros2 launch map_data route_planner.launch.py mapdata_file:=coords.mapdata
ros2 launch map_data route_planner.launch.py mapdata_file:=coords.mapdata highway_types:=footway,road

ros2 action send_goal /route_planner/plan_route map_data_interfaces/action/PlanRoute \
    "{waypoints: [{latitude: 50.1067, longitude: 14.4193}], start_from_robot: true}"
```

#### Launch arguments (`route_planner.launch.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `params_file` | `route_planner.yaml` | Parameter file: absolute path or filename in `config/` |
| `mapdata_file` | `""` | Default `.mapdata` filename or path |
| `mapdata_path` | `""` | Directory where bare `.mapdata` names are resolved (default: package share `data`) |
| `annotations` | `""` | Annotation store: `auto`, `none`, or explicit file path |
| `traversability` | `""` | Tag rule file deciding traversable ways (empty = from `params_file`) |
| `mission_dir` | `""` | Directory where GPX routes are saved |
| `gps_fix_topic` | `""` | Topic for `sensor_msgs/NavSatFix` |
| `earth_frame` | `""` | ECEF frame name for exact TF transform |
| `local_frame` | `""` | Local frame name for `nav_msgs/Path` |
| `algorithm` | `""` | Default algorithm: `graph`, `astar`, or `rrt` |
| `highway_types` | `""` | Allowed way types: `footway`, `road`, or `footway,road` |
| `spacing` | `""` | Metres between output waypoints (0 = raw) |
| `preload` | `""` | Load map and graph at startup (`true` or `false`) |
| `use_sim_time` | `false` | Use simulation time |

#### ROS 2 parameters (`route_planner`)

Parameters in `PLANNER_PARAMETERS` (`algorithm`, `highway_types`, `spacing`, `max_snap_distance`,
`cell_size`, `inflate_obstacles`, `simplify_path`, `smooth_path`) are re-read on every goal,
allowing dynamic updates via `ros2 param set`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mapdata_file` | `""` | Default `.mapdata` filename or absolute path |
| `data_dir` | package share `data` | Directory where bare `.mapdata` filenames are resolved (then the package `demo/`) |
| `annotations` | `auto` | Annotation store merged into the map: `auto`, `none`, or file path |
| `exclude_highway` | `["steps"]` | Non-routable highway types excluded from graph and map |
| `traversability_file` | `""` | Tag rule file (`""` = package `config/traversability.yaml`). [Traversability rules warning](#traversability-rules) |
| `preload` | `true` | Load map and build graph at startup |
| `mission_dir` | `~/missions` | Directory where planned GPX tracks are saved |
| `gps_fix_topic` | `/fixposition/odometry_llh` | `NavSatFix` topic for `start_from_robot` |
| `goal_topic` | `~/goal` | `GeoPointStamped` topic that triggers planning from current robot fix |
| `route_topic` | `~/route` | Latched `geographic_msgs/GeoPath` topic |
| `route_path_topic` | `~/route_path` | Latched `nav_msgs/Path` topic in `local_frame` |
| `status_topic` | `~/status` | Latched `std_msgs/String` status topic |
| `earth_frame` | `FP_ECEF` | ECEF frame name for exact coordinate conversion |
| `local_frame` | `FP_ENU0` | Target local frame for `route_path` |
| `fix_max_age` | `10.0` | Maximum age (s) before GPS fix is considered stale |
| `algorithm` | `graph` | Fallback algorithm (`graph`, `astar`, `rrt`) |
| `highway_types` | `["footway"]` | Fallback way types (`footway`, `road`, or both) |
| `spacing` | `3.0` | Fallback waypoint spacing in meters |
| `max_snap_distance` | `100.0` | Maximum snap distance (m) from start to nearest way |
| `goal_max_snap_distance` | `30.0` | Maximum snap distance (m) for goal; farther fails with `snap_too_far` |
| `outside_map_tolerance` | `100.0` | Allowed distance (m) endpoints may lie outside map bounds |
| `keep_goal` | `true` | End route at exact goal coordinate rather than projected point |
| `cell_size` | `0.25` | Grid cell size (m), defaults from `planner_defaults.yaml` |
| `inflate_obstacles` | `0.25` | Obstacle inflation buffer (m), defaults from `planner_defaults.yaml` |
| `simplify_path` | `true` | Simplify grid path, defaults from `planner_defaults.yaml` |
| `smooth_path` | `false` | Smooth grid path, defaults from `planner_defaults.yaml` |

#### Goal execution, GPX saving, and waypoint spacing

- **Cancellable goals:** Action goals can be cancelled during execution. When cancelled, the action returns status `CANCELLED` and failure reason `cancelled`.
- **Boolean goal fields:** Goal booleans (`simplify_path`, `smooth_path`) are evaluated as sent without falling back to node defaults.
- **`save_gpx` field:**
    - Empty string (`""`): automatically writes a timestamped track to `mission_dir/route_<timestamp>.gpx`.
    - `"-"`: disables GPX file output.
    - Explicit filename or path: writes to that path (resolved under `mission_dir` if relative).
- **`spacing` semantics:**
    - `spacing > 0`: resamples the path so consecutive waypoints are at most `spacing` meters apart. Below `0.1` fails with `bad_parameter`.
    - `spacing == 0`: uses the node's default spacing (`3.0 m`).
    - Negative spacing (`spacing < 0`, e.g. `-1.0`): preserves raw planner vertices without densification.

#### Action failure reasons

When route planning fails, `PlanRoute.Result` sets `success = false` and provides a machine-readable `reason`:

| Reason | Meaning |
|--------|---------|
| `bad_parameter` | Unknown `algorithm` or `sub_algorithm`, non-positive `cell_size`, negative or non-finite `inflate_obstacles`, a positive `spacing` below `0.1`, or a waypoint UTM cannot express (latitude outside -80..84, non-finite) |
| `start_blocked` | Grid start coordinate lies inside an obstacle and cannot be snapped to free space |
| `goal_blocked` | Grid goal coordinate lies inside an obstacle and cannot be snapped to free space |
| `cancelled` | Planning request was cancelled |
| `busy` | Another plan is still running; the goal is not queued, send it again |
| `start_outside_map` | Start coordinate lies farther than `outside_map_tolerance` outside the map area |
| `goal_outside_map` | Goal coordinate lies farther than `outside_map_tolerance` outside the map area |
| `snap_too_far` | Waypoint exceeds maximum snap distance to the nearest allowed way |
| `unreachable` | Graph planner found no connection between endpoints in the way network |
| `no_path` | Grid planner found no collision-free path between endpoints |
| `grid_too_large` | Requested bounding area exceeds `max_grid_cells` |
| `too_few_points` | Fewer than two waypoints supplied without `start_from_robot` |
| `file_not_found` | Map data file could not be found or loaded |
| `no_fix` | `start_from_robot` requested but no recent GNSS fix was received on `gps_fix_topic` |

### Python library usage

```python
from map_data.annotations import load_mapdata_with_annotations
from map_data.pathsolver.route import RoutePlanningError, plan_route
from map_data.utils.gpx import create_gpx_track

md, _ = load_mapdata_with_annotations("demo/buchlovice.mapdata")
try:
    route = plan_route(md, [(49.084416, 17.336309), (49.083106, 17.338527)], spacing=3.0)
except RoutePlanningError as e:
    print(f"Planning failed ({e.reason}): {e.message}")
else:
    with open("route.gpx", "w") as f:
        f.write(create_gpx_track(route.latlon))
```

## Traversability rules

Which OSM ways the robot may drive on is decided from their tags by
`config/traversability.yaml`, the single file to edit when routing rules change. The rules are
applied when a map is loaded for planning (`load_mapdata_with_annotations`), ensuring that
`route_planner`, `osm_cloud`, and intersection generation all observe the same network.

!!! warning "Keep traversability rules in sync between nodes"
    Both `route_planner` and `osm_cloud` must be configured with the same traversability file. If they differ, `osm_cloud` may publish intersection rings and cost clouds for ways that `route_planner` refuses to route over. `route_planner` detects rule file modifications automatically on each new goal, whereas `osm_cloud` reads the rule file once at startup and must be restarted after edits.

```yaml
default:
  traversable: true
  cost: 0.0
rules:
  - match: {highway: steps}          # exact value
    traversable: false
    reason: stairs
  - match: {surface: [grass, mud, sand]}   # any of these values
    traversable: false
    reason: soft surface
  - match: {bridge: boardwalk}       # boardwalk bridges refused
    traversable: false
    reason: bridge
  - match: {smoothness: [bad, very_bad, horrible, very_horrible, impassable]}
    traversable: false
    reason: rough surface
  - match: {access: [no, private]}
    traversable: false
    reason: no access
  - match: {highway: path, informal: yes}   # multiple keys: all must match (AND)
    cost: 1.0
    reason: informal path
```

- Rules are evaluated in order and the **first match wins**. A way that matches no rule receives `default`.
- `traversable: false` removes the way: it is never routed over, its junctions receive no intersection rings, and it is excluded from `osm_cloud`.
- `cost:` is optional and **additive on top of** `highway`/`surface` costs from `config/planner_defaults.yaml`: graph edge weights are `(1 + way cost + rule cost) × length`.
- Rule and tag values are compared as strings (`informal: yes` matches `informal: "yes"`).
- Shipped defaults refuse stairs, soft surfaces (`grass`, `mud`, `sand`), boardwalk bridges (`bridge: boardwalk`), rough surfaces, and non-accessible ways.

## Available algorithms

### Graph planning

Plans a path by searching the OSM road and footway network using A\*. The route is constrained
to follow existing ways, making it suitable for on-road navigation. Its edge weights are
`length × (1 + way cost)` using `highway_costs`/`surface_costs` tables plus any additive cost
from [traversability rules](#traversability-rules).

Walkable areas — closed `area=yes` (or multipolygon) ways such as pedestrian squares — are
crossed directly rather than circumnavigated. The planner computes the shortest line-of-sight
path within the polygon boundary between entry and exit points.

### Grid A\*

Discretizes the area around the route into a uniform grid and runs A\* on it.
Each cell carries a cost (see [Cost grid](#cost-grid)); cells inside barrier polygons are impassable.
Grid A\* is complete (finds a path if one exists) and provides predictable runtime. Resolution
and obstacle inflation are tuned via `--cell-size` and `--inflate-obstacles`.

### RRT\*

Rapidly-exploring Random Tree Star is a sampling-based planner that builds a tree of
collision-free waypoints by randomly sampling the free space. RRT\* is well suited for large
open areas. If RRT\* exceeds its iteration budget without finding a path, Grid A\* runs on the
same grid as a fallback.

---

## Cost grid

`ReplanPath.fill_grid()` converts OSM map data into a 2-D NumPy array of floating-point costs,
one value per grid cell representing a `cell_size × cell_size` meter patch.

**Cell cost assignment:**

1. Barrier polygons (inflated by `inflate_obstacles`, minus the ways of `highway_types` that cross them, such as a footway through a gate) are rasterized as impassable (`cost = inf`).
2. For each remaining cell, the distance to the nearest way centerline is computed.
3. If that distance is $\le$ `max_path_dist`, the cell receives a blended cost:

    $$\text{cost} = \text{way\_cost} + (\text{default\_off\_path\_cost} - \text{way\_cost}) \times \left(\frac{\text{dist}}{\text{max\_path\_dist}}\right)^2$$

    where $\text{way\_cost} = \min(\text{path\_cost\_cap}, \text{highway\_cost} + \text{surface\_cost})$.

4. Cells beyond `max_path_dist` from any way receive `default_off_path_cost` (default `0.9`).

**Obstacle thresholds:**

- In **Grid A\***, only cells with cost `inf` (`np.isinf`) are treated as impassable obstacles. Cells with finite costs (even $\ge 1.0$) are traversable with weighted traversal cost.
- In **RRT\***, cells with cost $\ge$ `traversability_threshold` (default `10.0`) or `inf` are treated as obstacles.

---

## Performance tuning

### Grid resolution (`cell_size`)

`cell_size` is the dominant factor in memory consumption and planning time.

| `cell_size` | Grid cells per 100 m² | Typical use case |
|-------------|-----------------------|------------------|
| `0.5 m` | 400 | Coarse survey, large open areas |
| `0.25 m` | 1 600 | Default — balanced resolution for pedestrian routes |
| `0.1 m` | 10 000 | Narrow gates, tight corridors |

### Obstacle inflation (`inflate_obstacles`)

Increasing `inflate_obstacles` adds a safety margin around barriers at the expense of
potentially closing narrow corridors. If planning returns `None`, consider reducing this
value.

### Path post-processing

- `simplify_path` (Douglas-Peucker) drops collinear and redundant vertices, reducing waypoint count without altering path geometry.
- `smooth_path` applies gradient-descent smoothing to round sharp corners.

### Algorithm choice

| Situation | Recommended algorithm |
|-----------|-----------------------|
| Mapped pedestrian or road network, robot must stay on ways | Graph planning |
| Urban route with well-defined barrier obstacles | Grid A\* |
| Large open area, sparse obstacles | RRT\* |


---

## Using astar_search directly

`astar_search` is the generic A\* implementation `GraphPlanner` uses (`grid_astar` has its
own array-based loop). You can use it directly to build a custom planner for any graph-like problem.

```python
from map_data.pathsolver.astar import astar_search


# Example: plan over a simple weighted grid
def neighbors(node):
    r, c = node
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [(n, 1.0) for n in candidates if 0 <= n[0] < 10 and 0 <= n[1] < 10]


def heuristic(node):
    goal = (9, 9)
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


path = astar_search(
    start_node=(0, 0),
    goal_node=(9, 9),
    get_neighbors_func=neighbors,
    heuristic_func=heuristic,
)
# path is a list of (row, col) tuples from start to goal, or None if unreachable
```

`get_neighbors_func` must return an iterable of `(neighbor_node, edge_cost)` tuples.
`heuristic_func` must be admissible (never overestimate the true cost to the goal) for A\* to
return an optimal path.
