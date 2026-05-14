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
