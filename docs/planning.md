# Path Planning

The `pathsolver` module provides standalone planning capabilities. You can use the `replan` CLI tool to process existing GPX tracks.

## CLI Tool: `replan`

The `replan` tool allows you to refine an existing GPX track using various pathfinding algorithms and the information stored in a `.mapdata` file.

```bash
# Replan a GPX track using Grid A* and save the result
python3 -m map_data.pathsolver.replan --path data/coords.gpx --file coords.mapdata --save data/planned.gpx --visualize
```

### Parameters

| Flag | Description |
|------|-------------|
| `--path <file>` | Input `.gpx` file containing the initial path |
| `--file <file>` | `.mapdata` file used for obstacle information |
| `--cell_size` | Grid resolution for all-terrain planning (default: 0.25m) |
| `--inflate_obstacles` | Safety buffer around barriers (default: 0.25m) |
| `--visualize` | Show a matplotlib plot of the planned path and obstacles |
| `--save <file>` | Save the resulting path as a GPX file |

## Available Algorithms

- **Graph Planning:** Global A* planning on OSM ways.
- **Grid A\*:** Grid-based A* implementation for all-terrain planning.
- **RRT\*:** Rapidly-exploring Random Tree Star implementation.
