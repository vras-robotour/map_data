# Path Planning

The `pathsolver` module provides standalone planning capabilities — no ROS2 context is required.
You can use the `replan` CLI tool to process existing GPX tracks, or import the planners directly as a Python library.

## CLI Tool: `replan`

The `replan` tool refines an existing GPX track using one of the available pathfinding algorithms
and the obstacle information stored in a `.mapdata` file.

```bash
# Replan a GPX track using Grid A* and save the result
python3 -m map_data.pathsolver.replan \
    --path data/coords.gpx \
    --file coords.mapdata \
    --save data/planned.gpx \
    --visualize
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--path <file>` | — | Input `.gpx` or `.yaml` file containing the initial path |
| `--file <file>` | — | `.mapdata` file used for obstacle and footway information |
| `--cell_size <meters>` | `0.25` | Grid resolution for all-terrain planning |
| `--inflate_obstacles <meters>` | `0.25` | Safety buffer added around barrier polygons |
| `--max_path_dist <meters>` | `2.0` | Max distance from a known way at which a cell receives a way-influenced cost |
| `--simplify_path` | `false` | Remove redundant waypoints using Douglas-Peucker simplification |
| `--smooth_path` | `false` | Apply gradient-descent smoothing to the planned path |
| `--visualize` | `false` | Show a matplotlib plot of the planned path and obstacles |
| `--save <file>` | — | Save the resulting path as a GPX file |

## Python Library

All planners work standalone — no ROS2 context required.

### GraphPlanner

Plans a route constrained to the OSM road and footway network using A*.

```python
from map_data.map_data import MapData
from map_data.pathsolver.graph_planner import GraphPlanner
import numpy as np

md = MapData.load("coords.mapdata")

start = np.array([md.min_x + 10, md.min_y + 10])
goal  = np.array([md.max_x - 10, md.max_y - 10])

planner = GraphPlanner(md, highway_types=["footway", "road"])
result  = planner.plan(np.array([start, goal]))  # np.ndarray or None
```

### ReplanPath

Grid-based local replanning around OSM barriers. See the [ReplanPath API reference](api/pathsolver.md#replanpath) for full details.

```python
from map_data.map_data import MapData
from map_data.pathsolver.replan import ReplanPath, parse_args
from map_data.utils.parsing import ways_to_shapely
from map_data.utils.gpx import parse_path, utm_path_to_latlon, create_gpx_content

md = MapData.load("coords.mapdata")
path_data = parse_path("waypoints.gpx")   # returns (utm_array, zone_num, zone_let)

args = parse_args([])
args.low  = (md.min_x, md.min_y)
args.high = (md.max_x, md.max_y)
args.cell_size = 0.25
args.inflate_obstacles = 0.25
args.simplify_path = True
args.smooth_path = False

replanner = ReplanPath(args, ways_to_shapely(md.barriers_list))
replanner.fill_grid(md, highway_types=["footway"], max_path_dist=2.0)
new_path = replanner.replan(path_data[0], algorithm="astar")

wgs84 = utm_path_to_latlon(new_path, path_data[1], path_data[2])
with open("planned.gpx", "w") as f:
    f.write(create_gpx_content(wgs84))
```

## Available Algorithms

### Graph Planning

Plans a path by searching the OSM road and footway network using A\*. The route is constrained
to follow existing ways, making it suitable for on-road or on-path navigation where staying on
designated routes is required. This algorithm is fast and produces geometrically clean results,
but cannot leave the road network to avoid obstacles.

### Grid A\*

Discretizes the area around the route into a uniform grid and runs A\* on it.
Each cell is marked free or occupied based on OSM barrier polygons (optionally inflated).
Grid A\* is the recommended all-terrain algorithm: it is complete (finds a path if one exists),
produces near-optimal paths, and has predictable runtime behavior. Grid resolution and obstacle
inflation are tunable via `--cell_size` and `--inflate_obstacles`.

### RRT\*

Rapidly-exploring Random Tree Star is a sampling-based planner that builds a tree of
collision-free waypoints by randomly sampling the free space. It is asymptotically optimal —
given enough iterations it converges to the shortest path — and handles irregular obstacle
shapes well. RRT\* is best suited for large open areas or when the grid resolution required
for Grid A\* would be prohibitively expensive.

---

## Cost grid

`ReplanPath.fill_grid()` converts OSM map data into a 2-D NumPy array of floating-point costs,
one value per grid cell. Each cell represents a `cell_size × cell_size` metre patch of ground.

**Cell cost assignment:**

1. Barrier polygons (inflated by `inflate_obstacles`) are rasterised as impassable (`cost = inf`).
2. For every remaining cell, the distance to the nearest way centre-line is calculated.
3. If that distance is ≤ `max_path_dist`, the cell receives a blended cost:

    ```
    cost = way_cost + (default_off_path_cost − way_cost) × (dist / max_path_dist)²
    ```

    where `way_cost = min(path_cost_cap, highway_cost + surface_cost)`.

4. Cells beyond `max_path_dist` from any way receive `default_off_path_cost` (default `0.9`).

Costs run from `0.0` (freely preferred, e.g. a pedestrian footway on asphalt) to `1.0`
(impassable). The planner treats cells with cost `≥ 1.0` as obstacles. See
[Planner Configuration](dev/planner_config.md) for the full cost tables.

**Traversability threshold.** Both Grid A\* and RRT\* consider a cell passable if its cost is
`< 1.0`. There is no separate traversability threshold parameter — adjust `inflate_obstacles` to
shrink or expand the impassable zone around physical barriers, and adjust `default_off_path_cost`
to make off-path terrain more or less discouraged.

---

## Performance tuning

### Grid resolution (`cell_size`)

`cell_size` is the dominant factor in both memory use and planning time.

| `cell_size` | Grid cells per 100 m² | Typical use case |
|------------|----------------------|-----------------|
| `0.5 m` | 400 | Coarse survey, large open areas |
| `0.25 m` | 1 600 | Default — good balance for urban pedestrian routes |
| `0.1 m` | 10 000 | Tight corridors, narrow gates |

Halving `cell_size` quadruples the number of cells and roughly doubles planning time. For routes
longer than a few hundred metres, prefer `0.25 m` or `0.5 m` and rely on `inflate_obstacles` to
enforce clearance rather than resolving every surface detail at fine resolution.

### Obstacle inflation (`inflate_obstacles`)

Increasing `inflate_obstacles` adds a safety margin around barriers at the cost of blocking
routes through narrow gaps. If the planner returns `False` (no path found), try reducing this
value before increasing grid resolution.

### Path post-processing

`simplify_path` (Douglas-Peucker) is cheap and almost always beneficial — it reduces the
waypoint count without noticeably changing the path shape. Enable it for any path that will be
sent to a navigation stack.

`smooth_path` applies gradient-descent smoothing. It rounds sharp corners but can shift
waypoints slightly off the low-cost cells produced by the planner. Use it when the downstream
controller benefits from smooth curvature and the small positional deviation is acceptable.

### Algorithm choice

| Situation | Recommended algorithm |
|-----------|----------------------|
| Well-mapped pedestrian network, path must stay on ways | `GraphPlanner` |
| Urban route, obstacles well-defined by OSM barriers | Grid A\* (`algorithm="astar"`) |
| Large open area, sparse obstacles | RRT\* (`algorithm="rrt"`) |
| Very large area where a fine Grid A\* grid is too expensive | RRT\* |

---

## Using `astar_search` directly

`astar_search` is the generic A\* implementation that powers both `GraphPlanner` and
`grid_astar`. You can use it directly to build a custom planner for any graph-like problem.

```python
from map_data.pathsolver.astar import astar_search

# Example: plan over a simple weighted grid
def neighbors(node):
    r, c = node
    candidates = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
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
