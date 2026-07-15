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

!!! warning "Trusted networks only"
    The viewer has no authentication and its API can read and write files in the data
    directory. Keep the default `--host 127.0.0.1`; only bind to other interfaces
    (e.g. `--host 0.0.0.0`) on networks where you trust every machine.

### Command-line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir <path>` | package `data/` folder | Directory to load `.mapdata`/`.gpx` files from and save new ones to |
| `--host <address>` | `127.0.0.1` | Address the Flask server binds to |
| `--port <port>` | `5000` | Port the Flask server binds to |
| `--telemetry-rate <Hz>` | `2.0` | Tracker telemetry broadcast rate; see [Tracker](viewer_tracker.md) |

### Tile Layer API Keys

The index page reads `THUNDERFOREST_API_KEY` and `SEZNAM_API_KEY` from the environment and makes
them available to the page for the corresponding optional basemap tile layers. Export the
relevant variable before launching the viewer to enable it:

```bash
export THUNDERFOREST_API_KEY=your_key_here
export SEZNAM_API_KEY=your_key_here
map_data_viewer
```

Both are optional — without them the viewer still works, using the built-in OpenStreetMap and
Satellite base layers (see [Viewer](viewer_view.md#layers-panel)).

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
