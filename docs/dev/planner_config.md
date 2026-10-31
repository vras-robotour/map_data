# Planner configuration

`config/planner_defaults.yaml` holds the routing cost tables, the grid-planner defaults and the map-building defaults. It is the single source for these values: the library, `map_data_plan`, the `route_planner` node and the viewer all read it.

---

## Who reads what

| Keys | Read by | When |
|---|---|---|
| `highway_costs`, `surface_costs`, `path_cost_cap` | `load_cost_tables()` for `ReplanPath` and `GraphPlanner` | every planner construction |
| `default_off_path_cost`, `rrt` | `ReplanPath` | every construction |
| `cell_size`, `inflate_obstacles`, `simplify_path`, `smooth_path`, `max_path_dist`, `grid_cost_weight` | `grid_defaults()` in `map_data/pathsolver/replan.py` | every call |
| `grid_margin` | `MapData` (`GRID_MARGIN`) | import |
| `obstacle_radius`, `buffer_widths` | `map_data/utils/parsing.py` | import |
| everything | viewer `GET /api/planner_defaults` | every request |

`grid_defaults()` is the source for the `ReplanPath` keyword defaults, `plan_route`, the `map_data_plan` flag defaults and the defaults of the `route_planner` parameters `cell_size`, `inflate_obstacles`, `simplify_path` and `smooth_path`. Missing keys fall back to `GRID_DEFAULTS` in the same module. An edit to the file applies to the next plan; import-time keys need a restart.

---

## Default file

```yaml
--8<-- "config/planner_defaults.yaml"
```

---

## `highway_costs` and `surface_costs`

A way's cost is `highway_costs[highway] + surface_costs[surface]`, capped at `path_cost_cap` (`way_cost()` in `map_data/pathsolver/way_cost.py`). The scale runs from `0.0` (preferred) to `1.0`.

- A way without a `highway` tag is read as `path`; without a `surface` tag as `asphalt`.
- A `highway` value missing from the table costs `UNKNOWN_HIGHWAY_COST` (0.5).
- A `surface` value missing from the table adds 0.0.

Both planners charge the same price:

- the grid planner lowers a cell's cost towards the cost of the nearest way (see `max_path_dist`);
- the graph planner weighs an edge `length × (1 + way cost + rule cost)`, where the rule cost comes from the [traversability rules](../planning.md#traversability-rules).

Whether a way may be driven on at all is decided by the traversability rules, not by these tables.

---

## Top-level keys

| Key | Default | Description |
|-----|---------|-------------|
| `default_off_path_cost` | 0.9 | Cost of a grid cell farther than `max_path_dist` from every way. |
| `max_path_dist` | 2.0 m | Radius around a way centerline in which cells cost `c + (default_off_path_cost - c) × (d / max_path_dist)²`, where `c` is the way's cost and `d` the distance. |
| `cell_size` | 0.25 m | Grid cell side for Grid A* and RRT*. |
| `inflate_obstacles` | 0.25 m | Buffer added around every obstacle polygon before rasterizing and collision checks. |
| `simplify_path` | `true` | Drop vertices whose shortcut stays collision-free (`simplify_path_checked`). |
| `smooth_path` | `false` | Gradient-descent smoothing (`smooth_path`). Runs before simplification. |
| `grid_margin` | 150 m | Added to each side of the GPX waypoints' bounding box when a map is built; sets the Overpass query area and the map extent (`min_x` ... `max_y`). The grid planners use a separate box: the route's waypoints plus `GRID_MARGIN_M` (50 m), clipped to the map. |
| `path_cost_cap` | 0.85 | Upper bound of a way's cost. Below `default_off_path_cost`, so any mapped way is cheaper than open terrain. |
| `grid_cost_weight` | 5.0 | Grid A* and RRT* step cost is `length × (1 + cell cost × grid_cost_weight)`. |
| `obstacle_radius` | 2.0 m | Radius of the circle a point obstacle (bollard, tree) becomes. |
| `buffer_widths` | road 7, footway 3, barrier 2 (m) | Full width a road, footway or open barrier line is buffered to. |

Only cells inside an obstacle polygon are `inf` and block the grid search; every finite cost is passable.

---

## `rrt`

RRT* settings, read by `ReplanPath` and passed to `RRTStar` (see [RRT*](../api/pathsolver.md#rrt)).

| Key | Default | Description |
|-----|---------|-------------|
| `informed` | `true` | Once a solution exists, sample from the informed ellipse instead of the whole free space. |
| `improve_after_goal` | `true` | Keep iterating after the goal is first reached. |
| `improve_iter` | 200 | Most extra iterations spent improving after the goal is first reached (never past `max_iter`). |
| `adaptive_radius` | `true` | Shrink the rewiring radius as the tree grows. |
| `seed` | unset | Seed for the sampler, for reproducible runs. |

Each `RRTStar` draws from its own `random.Random`. `seed` fixes that stream for the planners one `ReplanPath` builds; code can pass its own generator as `ReplanPath(..., rng=...)` or `RRTStar(..., rng=...)`.

---

## Overrides

| Where | What |
|---|---|
| `map_data_plan` | `--cell-size`, `--inflate-obstacles`, `--simplify/--no-simplify`, `--smooth/--no-smooth` for one run. |
| `route_planner` | The parameters above, set in `config/route_planner.yaml` or with `ros2 param set`; read on every goal. A PlanRoute goal's `simplify_path`/`smooth_path` are used as sent. |
| Viewer | Planner fields and the *Highway Costs* modal (highway and surface tables) apply to the requests of the session; the file is not modified. Under ROS the toggles start from the `route_planner` node's parameters, see [Planner](../viewer_planner.md). |
| Viewer map download | `grid_margin`, `obstacle_radius`, `buffer_widths` in the *Advanced* section of the fetch and GPX dialogs. |
| Python | Any `ReplanPath` keyword (`cell_size`, `inflate_obstacles`, `highway_costs`, ...); `None` takes the file value. |
