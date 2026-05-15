# Interactive Viewer

`map_data_viewer` launches a local Flask web server with a Leaflet-based map UI.
It lets you inspect and edit parsed features, view their OSM tags, draw manual annotations,
and plan paths — without needing ROS2.

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

## Loading Map Data

The viewer loads files from its data directory (default: the package `data/` folder,
overridable with `--data-dir`). Supported file formats:

- **`.mapdata`** — pre-parsed map files created by `create_mapdata`. Loaded immediately with no network request.
- **`.gpx`** — raw GPS tracks. The viewer parses them on load by fetching OSM data for the track bounds.

You can also drag and drop a `.gpx` file directly onto the map.

## Modes

The viewer has three modes, switched via the tabs in the top-right corner:

| Mode | Description |
|------|-------------|
| [**Viewer**](viewer_view.md) | Inspect features, manage layers, draw manual annotations. |
| [**Planner**](viewer_planner.md) | Design missions and plan paths using graph-based or all-terrain algorithms. |
| [**Tracker**](viewer_tracker.md) | Monitor a live robot via ROS2 telemetry. Requires a running ROS2 context. |
