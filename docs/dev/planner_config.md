# Planner Configuration

All default planning parameters are stored in `config/planner_defaults.yaml`. The file is loaded once at import time by `map_data/map_data.py` (which reads `grid_margin`), `map_data/utils/parsing.py` (`obstacle_radius`, `buffer_widths`), `map_data/pathsolver/grid_astar.py` (`grid_cost_weight`), and by `ReplanPath` (which reads the cost tables, grid parameters, and the `rrt` settings). Values can be overridden at runtime through the viewer's Highway Costs modal or via `map_data_plan` CLI flags such as `--cell-size` and `--inflate-obstacles`.

---

## Full default file

```yaml
highway_costs:
  pedestrian: 0.0
  footway: 0.0
  path: 0.1
  living_street: 0.1
  track: 0.3
  service: 0.3
  residential: 0.5
  unclassified: 0.5
  tertiary: 0.7
  secondary: 0.9
  primary: 1.0
surface_costs:
  asphalt: 0.0
  paving_stones: 0.0
  concrete: 0.0
  fine_gravel: 0.1
  gravel: 0.2
  dirt: 0.3
  grass: 0.5
  sand: 0.4
default_off_path_cost: 0.9
max_path_dist: 2.0
cell_size: 0.25
inflate_obstacles: 0.25
simplify_path: true
smooth_path: false
grid_margin: 150
path_cost_cap: 0.85
grid_cost_weight: 5.0
rrt:
  informed: true
  improve_after_goal: true
  improve_iter: 200
  adaptive_radius: true
  # seed: 0
obstacle_radius: 2.0
buffer_widths:
  road: 7.0
  footway: 3.0
  barrier: 2.0
```

---

## `highway_costs`

Assigns a base traversal cost to each OSM `highway=*` value. The cost scale runs from `0.0` (freely preferred) to `1.0` (equivalent to an obstacle). Costs are capped at `path_cost_cap` so that even the most expensive way type remains cheaper than off-path terrain.

Both planners charge the same price, through `map_data.pathsolver.way_cost`: the grid planner blends it into the cell costs, and the graph planner weighs a way's edges `length × (1 + min(path_cost_cap, highway_cost + surface_cost))`. Whether a way may be driven on *at all* is a separate question, answered by [the traversability rules](../planning.md#traversability-rules).

| OSM highway type | Default cost | Interpretation |
|-----------------|-------------|----------------|
| `pedestrian` | 0.0 | Pedestrian-only street or square |
| `footway` | 0.0 | Dedicated footpath |
| `path` | 0.1 | Informal path |
| `living_street` | 0.1 | Shared pedestrian/vehicle area |
| `track` | 0.3 | Unpaved agricultural or forestry track |
| `service` | 0.3 | Private or access road |
| `residential` | 0.5 | Residential street |
| `unclassified` | 0.5 | Minor road, no specific classification |
| `tertiary` | 0.7 | Local connecting road |
| `secondary` | 0.9 | Regional road |
| `primary` | 1.0 | Major road (capped to `path_cost_cap` = 0.85 in practice) |

Highway types not listed in the YAML receive `default_off_path_cost`.

---

## `surface_costs`

An additive penalty applied on top of the highway cost when the way carries a `surface=*` tag. If no `surface` tag is present, the penalty is 0.0.

| OSM surface value | Default penalty | Notes |
|------------------|----------------|-------|
| `asphalt` | 0.0 | Hard, smooth |
| `paving_stones` | 0.0 | Hard, smooth |
| `concrete` | 0.0 | Hard, smooth |
| `fine_gravel` | 0.1 | Compact gravel |
| `gravel` | 0.2 | Loose gravel |
| `dirt` | 0.3 | Earthen track |
| `sand` | 0.4 | Soft, slow |
| `grass` | 0.5 | Natural grass |

Surface values not listed receive a penalty of 0.0.

---

## Top-level parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_off_path_cost` | float | 0.9 | Cost assigned to grid cells that are not within `max_path_dist` metres of any way. Represents hard off-road terrain. |
| `max_path_dist` | float (m) | 2.0 | Radius around each way centerline within which cells receive the way's highway/surface cost. Cells beyond this radius use `default_off_path_cost`. |
| `cell_size` | float (m) | 0.25 | Side length of each grid cell used by Grid A* and RRT*. Smaller values give finer paths but increase memory and computation time. |
| `inflate_obstacles` | float (m) | 0.25 | Safety buffer added around all barrier polygons before rasterisation. Increases the clearance between the planned path and physical obstacles. |
| `simplify_path` | bool | `true` | Apply Douglas-Peucker simplification to the output path after planning. Reduces the number of waypoints while preserving the overall shape. |
| `smooth_path` | bool | `false` | Apply gradient-descent smoothing after planning (and after simplification if enabled). Produces rounder curves but may shift the path slightly away from the original grid solution. |
| `grid_margin` | float (m) | 150 | Metres added to each side of the waypoint bounding box (`MapData.min_x/max_x/min_y/max_y`), used both for the Overpass API query area and to clip the planning grid. |
| `path_cost_cap` | float | 0.85 | Maximum cost a way cell can receive after adding highway and surface penalties. Ensures that all recognised way types remain cheaper than `default_off_path_cost` (0.9), so the planner always prefers a way over open terrain. |
| `grid_cost_weight` | float | 5.0 | Weight applied to a cell's traversal cost when computing edge costs in Grid A* and RRT* (`1 + grid_value × grid_cost_weight`). |
| `obstacle_radius` | float (m) | 2.0 | Radius used when buffering point obstacles (e.g. bollards) into polygons. |
| `buffer_widths` | dict | see above | Per-category buffer width (m) used when turning barrier ways into obstacle polygons (`road`, `footway`, `barrier`). |

---

## `rrt`

RRT*-only settings, read by `ReplanPath` and passed to `RRTStar` (see [RRT* API reference](../api/pathsolver.md#rrt)). Once the goal is first reached, `improve_after_goal` keeps refining the path for at most `improve_iter` more iterations (and never past `max_iter`).

Every `RRTStar` draws from its own `random.Random`, never the module-global one, so concurrent planners cannot interleave each other's draws. `seed` fixes that stream for the planners a `ReplanPath` builds; code can instead pass its own `random.Random` as `ReplanPath(..., rng=...)` or `RRTStar(..., rng=...)`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `informed` | bool | `true` | Once a solution exists, sample from the shrinking informed ellipse (Informed RRT*) instead of the full free space. |
| `improve_after_goal` | bool | `true` | Keep iterating after the goal is first reached to find a lower-cost path, instead of returning immediately. |
| `improve_iter` | int | `200` | Most extra iterations spent improving after the goal is first reached. Trades planning time for path cost. |
| `adaptive_radius` | bool | `true` | Shrink the rewiring radius as the tree grows per the RRT* asymptotic-optimality formula, instead of using a fixed radius. |
| `seed` | int | *(unset)* | Seed for the sampler, making a run reproducible. Unset (the default), every planner samples from its own unseeded stream. |

---

## Runtime overrides

### Viewer Highway Costs modal

The viewer exposes a modal panel where `highway_costs` values can be edited per session. Changes take effect immediately for the next path planning request without reloading the `.mapdata` file.

### CLI flags

`map_data_plan` (see [Offline CLI](../usage.md)) exposes `--cell-size` and `--inflate-obstacles`. These override the YAML defaults for that invocation only; the YAML file is not modified.

### Programmatic override

```python
import copy

from map_data.pathsolver.replan import DEFAULT_ARGS, ReplanPath

args = copy.copy(DEFAULT_ARGS)
args.cell_size = 0.5  # coarser grid for faster planning
args.inflate_obstacles = 0.5  # wider obstacle clearance

planner = ReplanPath(args, obstacles)
```

Any `args` attribute `ReplanPath` reads (see the constructor table above) takes precedence over the YAML defaults for that instance.
