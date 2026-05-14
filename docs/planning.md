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
| `--path <file>` | — | Input `.gpx` file containing the initial path |
| `--file <file>` | — | `.mapdata` file used for obstacle and footway information |
| `--cell_size <meters>` | `0.25` | Grid resolution for all-terrain planning |
| `--inflate_obstacles <meters>` | `0.25` | Safety buffer added around barrier polygons |
| `--visualize` | `false` | Show a matplotlib plot of the planned path and obstacles |
| `--save <file>` | — | Save the resulting path as a GPX file |

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
