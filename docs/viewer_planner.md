# Planner

The **Planner** mode lets you design robot missions and compute planned paths directly in the
browser, using the same algorithms as the `map_data_plan` [CLI tool](planning.md).

## Workflow

1. Switch to **PLANNER** using the tab in the top-right corner.
2. Click on the map to place waypoints. Drag any waypoint to adjust it. With **Start at
   robot** ticked the first waypoint is the robot itself, so you only click the goals.
3. Select a planning mode and configure parameters in the sidebar.
4. Click **Replan Path** to compute the path between the placed waypoints.
5. Download the result as a GPX file or send it directly to the robot via **Wormhole**.

## Sidebar controls

### Import and export

| Button | Description |
|--------|-------------|
| **Import GPX** | Load an existing GPX track as the initial set of waypoints |
| **Download** | Export the planned path as a `.gpx` file |
| **Wormhole** | Send the planned path directly to the robot over the network |
| **Delete All** | Remove all placed waypoints |
| **Keep Endpoints** | Remove intermediate waypoints, keeping only start and end |
| **Start at robot** | Pin the first waypoint to the robot's live position and plan from there |

!!! info "Start at robot"
    The toggle requires a robot position, staying disabled until the tracker reports one
    (telemetry arrives in all modes, so the Robot layer does not need to be active). While
    enabled, waypoint 0 follows the robot and **Replan Path** refreshes it before planning.

### Planning mode

| Mode | Description |
|------|-------------|
| **All terrain — A\*** | Grid-based A\* that finds the optimal path while avoiding OSM barriers and manual annotations. Recommended for most off-road scenarios. |
| **All terrain — RRT\*** | Sampling-based planner. Suited for large open areas; falls back to Grid A\* if no path is found within the iteration budget. |
| **Paths only** | Constrains the route to the OSM road and footway network, including manually annotated paths. Produces on-network routes. |

!!! info "Path priority"
    OSM ways and annotated paths take **priority over obstacles**. A path overlapping a
    barrier polygon is treated as traversable, allowing you to route through obstacles
    (such as a gate in a fence).

### Routing options

- **Plan on Footways** — include pedestrian footways as traversable ways (default: enabled).
- **Plan on Roads** — include vehicle roads as traversable ways.

### Planner parameters

When the viewer runs under ROS 2 and the tracker's `planner_node` parameter is active
(default: `/route_planner`), initial toggles and parameter values are synchronized from the
running `/route_planner` node's current parameters. In this case, the UI displays "defaults from /route_planner".
When running standalone without ROS 2 or when `planner_node` is set to `""`, defaults are loaded
from `config/planner_defaults.yaml`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| Cell Size (m) | `0.25` | Grid resolution for all-terrain planning |
| Obstacle Inflation (m) | `0.25` | Safety buffer added around barrier polygons |
| Simplify Path | on | Drop vertices whose shortcut stays collision-free (Douglas-Peucker) |
| Smooth Path | off | Apply gradient-descent smoothing to the planned path |
| Grid Cost Weight | `5.0` | How strongly the cell cost weighs against path length in all-terrain planning |
| Show Cost Grid | off | Overlay the traversal cost grid of the current view on the map for debugging |

### Highway costs

The **Highway Costs** button opens a configuration dialog where you can assign base traversal
costs (0.0 = preferred, 1.0 = costly) to each OSM highway type and additional penalties per
surface material. These costs are taken into account during both **Paths only** graph planning
(where edge costs are weighted by way costs) and **all-terrain** grid planning.

### Traversability rules

The **Traversability Rules** button opens `config/traversability.yaml` in an editor.
See [Traversability rules](planning.md#traversability-rules) for rule syntax and synchronization requirements with other nodes.

- **Apply** checks the rules and applies them for **Paths only** planning in the current browser session.
- **Save to file** writes the edited rules to disk. `route_planner` picks up the saved file on its next goal; `osm_cloud` reads the file only at startup and requires a restart.
- **Reload from file** discards unsaved edits and reloads the file from disk.
- **Show only plannable ways** hides ways that **Paths only** will not route over (refused by rules, stairs, or unselected way types).
