# Pathsolvers

## Route planning

`plan_route` is the entry point the viewer, `map_data_plan` and `route_planner` share: graph or grid planning, request validation, waypoint spacing.

```python
from map_data.annotations import load_mapdata_with_annotations
from map_data.pathsolver.route import RoutePlanningError, plan_route

md, _ = load_mapdata_with_annotations("my_area.mapdata")
try:
    result = plan_route(md, [(50.1050, 14.4220), (50.1072, 14.4280)], algorithm="graph", spacing=3.0)
except RoutePlanningError as e:
    print(e.reason, e.message)
else:
    print(f"{result.length_m:.0f} m, {len(result.latlon)} waypoints")
```

::: map_data.pathsolver.route
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - plan_route
        - RouteResult
        - RoutePlanningError
        - goal_plan_kwargs
        - check_in_map

## Graph planner

::: map_data.pathsolver.graph_planner.GraphPlanner
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - __init__
        - plan
        - snap_distance
        - edge_factor

### Walkable areas

::: map_data.pathsolver.walkable_area
    options:
      show_root_heading: false
      heading_level: 4
      members: false

## Way costs

::: map_data.pathsolver.way_cost
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - way_cost
        - load_cost_tables

## ReplanPath

Grid planning (Grid A* or RRT*) over a box around the waypoints. `plan_route` builds one per request; use it directly to plan on a grid of your own.

```python
import numpy as np
from map_data.map_data import MapData
from map_data.pathsolver.replan import ReplanPath
from map_data.utils.parsing import ways_to_shapely

md = MapData.load("my_area.mapdata")
replanner = ReplanPath(
    ways_to_shapely(md.barriers_list), low=(md.min_x, md.min_y), high=(md.max_x, md.max_y)
)
replanner.fill_grid(md, highway_types=["footway"])

start = np.array([md.min_x + 10, md.min_y + 10])
goal = np.array([md.max_x - 10, md.max_y - 10])
path = replanner.replan(np.array([start, goal]), algorithm="astar")
if path is None:
    print(replanner.failure_reason)
```

Constructor keywords left `None` (`cell_size`, `inflate_obstacles`, `simplify_path`, `smooth_path`, `grid_cost_weight`, `highway_costs`, `surface_costs`) take the values of `config/planner_defaults.yaml`; see [Planner configuration](../dev/planner_config.md). The resolved tables are the instance attributes `highway_costs`, `surface_costs`, `path_cost_cap` and `default_off_path_cost`.

`replan()` returns the path, or `None` with `failure_reason` set to `no_path`, `start_blocked`, `goal_blocked` or `cancelled`. To cancel a run from another thread, construct it with a unique `transfer_id` and call `cancel_replan_backend(transfer_id)`.

::: map_data.pathsolver.replan
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - grid_defaults
        - cancel_replan_backend

::: map_data.pathsolver.replan.ReplanPath
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - replan
        - fill_grid
        - fill_grid_lazily

## Grid A*

::: map_data.pathsolver.grid_astar.grid_astar
    options:
      heading_level: 3

## RRT*

::: map_data.pathsolver.rrt_star.RRTStar
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - __init__
        - find_path

## Smoothing

::: map_data.pathsolver.smoothing.smooth_path
    options:
      heading_level: 3

## Generic A\*

`astar_search` works on any hashable node type; see [Using astar_search directly](../planning.md#using-astar_search-directly).

::: map_data.pathsolver.astar.astar_search
    options:
      heading_level: 3
