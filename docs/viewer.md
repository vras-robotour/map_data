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
    By default the viewer has no authentication and its API can read and write files in
    the data directory. Keep the default `--host 127.0.0.1`; only bind to other interfaces
    (e.g. `--host 0.0.0.0`) on networks where you trust every machine, and see
    [Deployment Security](#deployment-security) below for options to lock it down further.

### Command-line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir <path>` | package `data/` folder | Directory to load `.mapdata`/`.gpx` files from and save new ones to |
| `--host <address>` | `127.0.0.1` | Address the Flask server binds to |
| `--port <port>` | `5000` | Port the Flask server binds to |
| `--telemetry-rate <Hz>` | `2.0` | Tracker telemetry broadcast rate; see [Tracker](viewer_tracker.md) |

### Deployment Security

The viewer is safe by default: it binds to `127.0.0.1` (loopback only), rejects file paths
that try to escape the data directory, and never unpickles untrusted data. When you run it
with `--host 0.0.0.0` (or any other non-loopback address) so a robot or a teammate's machine
can reach it, keep in mind that **anyone who can reach the port has full use of the API** —
no authentication is required by default. Concretely, that surface includes:

- Creating/overwriting files in the data directory (`/api/fetch_area`, `/api/upload_gpx`,
  `/api/upload_mapdata`, and annotation/edit endpoints).
- Spawning a `wormhole send` subprocess (`/api/create_wormhole`).
- Triggering outbound Overpass API queries on your behalf (`/api/fetch_area`, `.gpx` loads).

Only bind to `0.0.0.0` on networks you trust (e.g. an isolated robot LAN). For extra
protection on top of that, two environment variables are available:

| Variable | Default | Effect |
|----------|---------|--------|
| `MAP_DATA_ACCESS_TOKEN` | unset (auth disabled) | When set, every HTTP request and SocketIO connection must supply this exact token, or it's rejected with `401`. |
| `MAP_DATA_CORS_ORIGINS` | unset (same-origin only) | Controls which origins may open a SocketIO connection to the server. |

**`MAP_DATA_ACCESS_TOKEN`** — opt-in, off by default so existing setups and tests are
unaffected. When set, the token can be supplied any of three ways:

- Header: `X-Access-Token: <token>` (the natural choice for headless/robot API clients).
- Query parameter: `http://host:5000/?access_token=<token>`.
- Cookie: `map_data_access_token`, set automatically after a request authenticates via the
  query parameter above — this lets the browser UI authenticate once by visiting the URL
  with `?access_token=...` and have the browser carry that cookie on all subsequent
  same-origin page, static asset, API, and SocketIO requests. No login UI is involved.

The cookie is set without `Secure` (so it also works over plain HTTP on a LAN) — if the
network itself isn't trusted, put a TLS-terminating reverse proxy in front of the viewer
rather than relying on the token alone.

**`MAP_DATA_CORS_ORIGINS`** — controls SocketIO's `cors_allowed_origins`. Left unset, only
the request's own origin is allowed (safe default, and it works out of the box for local
dev regardless of which port you pick, since it's derived per-request rather than
hardcoded). Set it to a comma-separated list of allowed origins to widen access (e.g. when
serving the frontend from a different host/port), or to `*` to explicitly allow any origin.
`*` is never the default — it must be requested explicitly.

Example hardened robot deployment:

```bash
export MAP_DATA_ACCESS_TOKEN=$(openssl rand -hex 32)
export MAP_DATA_CORS_ORIGINS=http://192.168.1.50:5000
map_data_viewer --host 0.0.0.0 --data-dir /path/to/data
```

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
