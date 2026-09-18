# Planner

The **Planner** mode lets you design robot missions and compute planned paths directly in the
browser, using the same algorithms as the `replan` [CLI tool](planning.md).

## Workflow

1. Switch to **PLANNER** using the tab in the top-right corner.
2. Click on the map to place waypoints. Drag any waypoint to adjust it. With **Start at
   robot** ticked the first waypoint is the robot itself, so you only click the goals.
3. Select a planning mode and configure parameters in the sidebar.
4. Click **Replan Path** to compute the path between the placed waypoints.
5. Download the result as a GPX file or send it directly to the robot via **Wormhole**.

## Sidebar Controls

### Import / Export

| Button | Description |
|--------|-------------|
| **Import GPX** | Load an existing GPX track as the initial set of waypoints. |
| **Download** | Export the planned path as a `.gpx` file. |
| **Wormhole** | Send the planned path directly to the robot over the network. |
| **Delete All** | Remove all placed waypoints. |
| **Keep Endpoints** | Remove intermediate waypoints, keeping only start and end. |
| **Start at robot** | Pin the first waypoint to the robot's live position and plan from there. |

!!! info "Start at robot"
    The toggle needs a robot position, so it stays disabled until the tracker reports one
    (it needs no Tracker tab — the telemetry arrives in every mode, and the toggle works
    with the Robot layer switched off). While it is on, waypoint 0 follows the robot and
    **Replan Path** refreshes it first, so the route is planned from where the robot is at
    that moment. A planned path is left alone — its first pose is a result, not a waypoint —
    until the next replan re-pins it. Switching the toggle off leaves the waypoint behind as
    an ordinary one.

### Planning Mode

| Mode | Description |
|------|-------------|
| **All terrain — A\*** | Grid-based A\* that finds the optimal path while avoiding OSM barriers and manual annotations. Recommended for most use cases. |
| **All terrain — RRT\*** | Sampling-based planner. Faster in large open areas; converges to optimal asymptotically. |
| **Paths only** | Constrains the route to the OSM road and footway network, including manually annotated paths. Produces clean, on-path results but cannot leave designated ways. |

!!! info "Path priority"
    Both OSM ways and manually annotated paths take **priority over obstacles**. A path that
    physically overlaps a barrier polygon is still considered traversable — the planner will
    route through it. This lets you tunnel an annotated path through an OSM barrier (e.g. a
    gate through a fence) without modifying the underlying map data.

### Routing Options

- **Plan on Footways** — include pedestrian footways as traversable ways (enabled by default).
- **Plan on Roads** — include vehicle roads as traversable ways.

### Planner Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Cell Size (m) | `0.25` | Grid resolution for all-terrain planning. Smaller values give smoother paths at higher computation cost. |
| Obstacle Inflation (m) | `0.25` | Safety buffer added around barrier polygons. |
| Simplify Path | on | Remove redundant intermediate waypoints from the result. |
| Smooth Path | off | Apply spline smoothing to the planned path. |
| Show Cost Grid | off | Overlay the traversal cost grid on the map for debugging. Memory intensive on larger areas. |

### Highway Costs

The **Highway Costs** button opens a pop-up window where you can assign a base
traversal cost (0.0 = free, 1.0 = obstacle) to each OSM highway type and an
extra penalty per surface material. These costs influence which ways are
preferred during **all-terrain** planning (do not apply to paths only).

### Traversability Rules

The **Traversability Rules** button opens `config/traversability.yaml` (see
[Traversability rules](planning.md#traversability-rules)) in an editor. **Apply** checks
the rules and uses them for **Paths only** planning in this browser tab. **Save to file**
also writes them to the file, comments included. `route_planner` picks up the saved
file on its next goal; `osm_cloud` reads it only at startup, so restart it after saving. **Reload from file** throws away the edits.

**Show only plannable ways** hides the ways that **Paths only** will not route over:
ways the rules refuse, stairs, and roads or footways whose checkbox is off. The ways
come back when you leave the planner or clear the checkbox.
