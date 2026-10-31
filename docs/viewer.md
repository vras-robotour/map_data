# Interactive viewer

`map_data_viewer` launches a local Flask web server with a Leaflet-based map UI.
It lets you inspect and edit parsed features, view their OSM tags, draw manual annotations,
and plan paths — without needing ROS 2.

!!! tip "Live demo"
    A read-only demo of the viewer (Buchlovice dataset) runs at
    <https://vras-robotour.github.io/map_data/demo/> — no installation needed.
    The demo is read-only: editing, OSM fetching, path planning, and tracker
    telemetry are disabled in static demo mode. Running `map_data_viewer` locally provides full functionality.

## Running the viewer

```bash
# Sourced ROS 2 workspace:
map_data_viewer

# Explicit data directory and port:
map_data_viewer --data-dir /path/to/data --port 8080

# Standalone Python environment:
python -m map_data.viewer.app --data-dir ./data
```

Then open `http://127.0.0.1:5000` in a browser.

!!! warning "Trusted networks only"
    By default the viewer has no authentication and its API can read and write files in
    the data directory. Keep the default `--host 127.0.0.1`; only bind to other interfaces
    (such as `--host 0.0.0.0`) on networks where you trust every machine, and see
    [Deployment security](#deployment-security) below for options to lock it down further.

### Command-line flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir <path>` | package `data/` folder | Directory containing `.mapdata` and `.gpx` files |
| `--host <address>` | `127.0.0.1` | Address the Flask server binds to |
| `--port <port>` | `5000` | Port the Flask server binds to |
| `--telemetry-rate <Hz>` | `2.0` | Tracker telemetry broadcast rate in Hz; see [Tracker](viewer_tracker.md) |
| `--config <path>` | package `config/tracker.yaml` | Tracker topics: a ROS 2 parameter file with a `map_data_tracker` section. See [Tracker](viewer_tracker.md#config-file) |

### API error handling and file safety

- **JSON error format:** All API error responses return JSON `{"error": "<message>"}` with the corresponding HTTP status code.
- **No file overwriting:** Map uploads (`/api/upload_mapdata`, `/api/upload_gpx`) and OSM area fetching (`/api/fetch_area`) never overwrite an existing `.mapdata` file. If a file with that name already exists in the data directory, a numerical suffix (`_1`, `_2`, etc.) is appended automatically.
- **Corrupt annotation store recovery:** If an annotation sidecar file (`<stem>.annotations.json`) cannot be loaded (due to corruption or invalid JSON), it is moved aside to `<path>.corrupt-<timestamp>` and answered with HTTP `409 Conflict`, preserving the damaged file for recovery without losing edits.

### Deployment security

The viewer binds to `127.0.0.1` (loopback only) by default, rejects file paths that attempt
directory traversal outside the data directory, and avoids deserializing untrusted data.
When bound to `--host 0.0.0.0` on a network, anyone who can reach the port has access to the API.

Two environment variables provide access control:

| Variable | Default | Effect |
|----------|---------|--------|
| `MAP_DATA_ACCESS_TOKEN` | unset (auth disabled) | When set, every HTTP request and Socket.IO connection must supply this token, or it is rejected with `401`. |
| `MAP_DATA_CORS_ORIGINS` | unset (same-origin only) | Controls which origins may open a Socket.IO connection to the server. |

**`MAP_DATA_ACCESS_TOKEN`** can be supplied in three ways:

- Header: `X-Access-Token: <token>` (for automated clients and robots).
- Query parameter: `http://host:5000/?access_token=<token>`. The viewer sets the cookie and redirects to the same URL without the token.
- Cookie: `map_data_access_token`, set automatically when authenticating via the query parameter above.

**CSRF protection:** To prevent cross-site request forgery, the cookie alone only
authenticates read requests (`GET`/`HEAD`/`OPTIONS`). State-changing requests
(`POST`/`PUT`/`DELETE`) must either supply the token via the `X-Access-Token` header or
carry the cookie together with an `X-Requested-With` header, which the viewer's frontend
attaches automatically. Request payloads are capped at 100 MB.

**`MAP_DATA_CORS_ORIGINS`** sets Socket.IO's allowed origins. Left unset, only the request's
own origin is accepted. Set to a comma-separated list of origins or `*` to widen access.

Example hardened deployment:

```bash
export MAP_DATA_ACCESS_TOKEN=$(openssl rand -hex 32)
export MAP_DATA_CORS_ORIGINS=http://192.168.1.50:5000
map_data_viewer --host 0.0.0.0 --data-dir /path/to/data
```

### Base layers

The layer control (bottom left) switches between OpenStreetMap and Esri satellite imagery; neither needs an API key. The choice is remembered in the browser.

## Loading map data

The file selector lists the `.mapdata` files in the data directory; opening one needs no network access except for the base layer tiles.

New maps come from:

- **GPX import** (`G`, or the GPX button): uploads a track, downloads the OSM data around it and saves a new `.mapdata` (`/api/upload_gpx`).
- **Fetch area**: downloads a drawn box of up to 25 km² in the background (`/api/fetch_area`).
- **Drag and drop**: a `.mapdata` file is uploaded as is; a `.gpx` file builds a new map, or in Planner mode loads its points as waypoints.

`.gpx` files in the data directory are not opened by the viewer.

## Modes

The viewer provides three modes via the tabs in the top-right corner:

| Mode | Description |
|------|-------------|
| [**Viewer**](viewer_view.md) | Inspect features, manage layer visibility, draw annotations, and edit ways. Supports [keyboard shortcuts](viewer_view.md#keyboard-shortcuts). |
| [**Planner**](viewer_planner.md) | Design missions and plan paths using graph-based or all-terrain grid algorithms. |
| [**Tracker**](viewer_tracker.md) | Monitor a live robot via ROS 2 telemetry. Requires a running ROS 2 context. |
