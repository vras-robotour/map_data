# Interactive Viewer

`map_data_viewer` launches a local Flask web server with a Leaflet-based map UI.
It lets you inspect parsed features, view their OSM tags, draw manual annotations,
and even fetch new areas directly from the UI — without needing ROS2.

## Running the Viewer

```bash
# After colcon build and sourcing the workspace:
map_data_viewer

# Explicit data directory and port:
map_data_viewer --data-dir /path/to/data --port 8080

# Standalone (outside a colcon workspace):
python -m map_data.viewer.app --data-dir ./data
```

Then open `http://127.0.0.1:5000` in a browser.

## Main Modes

The modes can be switched using the tabs in the top-right corner:

- **VIEWER**: Inspect map features, manage layers, and draw manual annotations.
- **PLANNER**: Design missions and plan paths using graph-based or all-terrain algorithms.

## Viewer Tools

The toolbar at the top provides the following tools:

| Mode | Shortcut | Action |
|------|:---:|--------|
| **View** | `v` | Inspect map features. Click any object to see its properties in the sidebar. |
| **Edit** | `e` | Selection and reshaping of custom annotations. |
| **+ Obs** | `a` | Draw rectangles, polygons, or circles to add manual obstacles. |
| **+ Path** | `p` | Draw custom paths or footways. |
| **Del** | `d` | Remove custom annotations. |
| **Fetch** | `f` | Draw a bounding box to download and parse a new OSM area. |

## Planner Features

- **Interactive Waypoints:** Click on the map to add waypoints. Drag them to adjust the route.
- **Graph Planning:** Automatically snaps paths to the OSM road and footway network.
- **All-Terrain Planning:** Uses Grid A* or RRT* to find the shortest path while avoiding both OSM barriers and manual obstacle annotations.
- **GPX Support:** Import existing GPX tracks for refinement or export planned missions.
- **Live Replanning:** Adjust planning parameters (inflation, cell size, path simplification) and see the results instantly.
