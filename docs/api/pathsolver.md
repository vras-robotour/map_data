# Pathsolvers

## Generic A\*

`astar_search` is the low-level A\* implementation shared by `GraphPlanner` and `grid_astar`.
It works on any hashable node type — you can use it directly to build a custom planner.
See [Using `astar_search` directly](../planning.md#using-astar_search-directly) for a worked example.

::: map_data.pathsolver.astar.astar_search
    options:
      show_source: true
      heading_level: 3

## Graph Planner

::: map_data.pathsolver.graph_planner.GraphPlanner
    options:
      show_root_heading: false
      members: false

::: map_data.pathsolver.graph_planner.GraphPlanner.__init__
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.graph_planner.GraphPlanner.plan
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.graph_planner.GraphPlanner.a_star
    options:
      show_source: true
      heading_level: 3

## Grid A*

::: map_data.pathsolver.grid_astar.grid_astar
    options:
      show_source: true
      heading_level: 3

## RRT*

::: map_data.pathsolver.rrt_star.RRTStar
    options:
      show_root_heading: false
      members: false

::: map_data.pathsolver.rrt_star.RRTStar.__init__
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.rrt_star.RRTStar.find_path
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.rrt_star.RRTStar._is_collision
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.rrt_star.RRTStar._sample_point
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.rrt_star.RRTStar._nearest_node
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.rrt_star.RRTStar._steer
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.rrt_star.RRTStar._get_near_nodes
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.rrt_star.RRTStar._segment_cost
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.rrt_star.RRTStar._reconstruct_path
    options:
      show_source: true
      heading_level: 3

::: map_data.pathsolver.rrt_star.RRTStar._simplify_path
    options:
      show_source: true
      heading_level: 3

## ReplanPath

`ReplanPath` is the local replanning engine used by the viewer's **Planner** mode and the
`replan` CLI tool. It discretizes the area around the input waypoints into a cost grid,
assigns traversal costs based on OSM highway and surface types, then finds a collision-free
path using either Grid A* or RRT*.

### Class attributes

These cost tables are loaded from `config/planner_defaults.yaml` at import time and can be
overridden per-instance:

| Attribute | Type | Description |
|-----------|------|-------------|
| `HIGHWAY_COSTS` | `dict[str, float]` | Per OSM `highway` value cost (0.0 = free, 1.0 = obstacle) |
| `SURFACE_COSTS` | `dict[str, float]` | Extra penalty per OSM `surface` value |
| `DEFAULT_OFF_PATH_COST` | `float` | Cost for cells not near any known way (default `0.9`) |
| `PATH_COST_CAP` | `float` | Maximum cost a way cell can have (default `0.85`, keeps ways preferred over off-path) |

### Constructor

```python
ReplanPath(args, obstacles=None, transfer_id=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `args` | `argparse.Namespace` | Planning parameters. Use `parse_args([])` to get defaults. |
| `obstacles` | `list[shapely.Geometry]` | Obstacle geometries (from `ways_to_shapely(md.barriers_list)`). |
| `transfer_id` | `str \| None` | Optional UUID for cancellation via `cancel_replan_backend()`. |

**`args` attributes used by `ReplanPath`:**

| Attribute | Default | Description |
|-----------|---------|-------------|
| `low` | — | `(min_x, min_y)` lower bound of the planning area in UTM metres |
| `high` | — | `(max_x, max_y)` upper bound of the planning area in UTM metres |
| `cell_size` | `0.25` | Grid resolution in metres |
| `inflate_obstacles` | `0.25` | Buffer added to obstacle geometries in metres |
| `simplify_path` | `True` | Apply Douglas-Peucker simplification to the output |
| `smooth_path` | `False` | Apply gradient-descent smoothing after planning |

### Methods

#### `fill_grid(map_data, highway_types=None, max_path_dist=2.0)`

Populate the cost grid from map data. Must be called before `replan()`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `map_data` | — | A loaded `MapData` object |
| `highway_types` | `["footway"]` | Which way categories to use. Pass `["footway", "road"]` to include roads. |
| `max_path_dist` | `2.0` | Cells within this distance (m) of a known way receive an interpolated cost. Cells beyond receive `DEFAULT_OFF_PATH_COST`. |

Cost model per cell: `way_cost + (DEFAULT_OFF_PATH_COST - way_cost) × (dist / max_path_dist)²`, where `way_cost = min(PATH_COST_CAP, HIGHWAY_COSTS[highway] + SURFACE_COSTS[surface])`. Obstacle cells are set to `inf`.

#### `replan(path, algorithm="astar")`

Plan a path through the cost grid.

| Parameter | Description |
|-----------|-------------|
| `path` | `np.ndarray` of shape `(N, 2+)` — input waypoints in UTM metres |
| `algorithm` | `"astar"` (Grid A*) or `"rrt"` (RRT*) |

Returns `np.ndarray` (replanned path), `None` (cancelled), or `False` (no path found).
Segments between consecutive waypoints are processed in parallel via `joblib`.

#### `visualize(path, old_path=None)`

Save a matplotlib debug plot as `replan.png` showing the cost grid, obstacles, and path.

### Cancellation

To cancel a running `replan()` call from another thread, pass a `transfer_id` UUID to the
constructor and call:

```python
from map_data.pathsolver.replan import cancel_replan_backend

cancel_replan_backend(transfer_id)
```

### Minimal usage example

```python
from map_data.map_data import MapData
from map_data.pathsolver.replan import ReplanPath, parse_args
from map_data.utils.parsing import ways_to_shapely
from map_data.utils.gpx import parse_path

md = MapData.load("coords.mapdata")
path_data = parse_path("waypoints.gpx")   # (utm_array, zone_num, zone_let)

args = parse_args([])
args.low  = (md.min_x, md.min_y)
args.high = (md.max_x, md.max_y)

replanner = ReplanPath(args, ways_to_shapely(md.barriers_list))
replanner.fill_grid(md, highway_types=["footway"], max_path_dist=2.0)

new_path = replanner.replan(path_data[0], algorithm="astar")
```
