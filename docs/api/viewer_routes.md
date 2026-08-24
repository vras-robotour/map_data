# Viewer REST API

!!! note "Internal API"
    This API is consumed by the viewer's JavaScript frontend. It is documented here for
    developers who want to build custom tooling or integrate with the viewer programmatically.

All endpoints are served by the Flask app started with `map_data_viewer`. Data endpoints live
under `/api/`. Errors return standard HTTP status codes with a plain-text description.

---

## File Management

### `GET /api/files`

List available `.mapdata` and `.gpx` files in the data directory.

**Response**

```json
{
  "mapdata": ["stromovka.mapdata", "coords.mapdata"],
  "gpx": ["coords.gpx", "route.gpx"]
}
```

---

### `GET /api/mapdata?file=<name>`

Load all parsed features for a `.mapdata` file as a GeoJSON FeatureCollection. Stored
annotations (tag overrides, deletions, splits, node moves) are applied before the response
is returned.

**Query parameters**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `file` | yes | Filename in the data directory (e.g. `coords.mapdata`) |

**Response** — GeoJSON FeatureCollection. Each feature has:

| Property | Type | Description |
|----------|------|-------------|
| `id` | `int` or `"int:int"` | OSM way ID, or virtual ID `"123:0"` for split segments |
| `category` | `string` | `"road"`, `"footway"`, or `"barrier"` |
| `tags` | `object` | OSM tags, with any stored overrides merged in |
| `in_out` | `string \| null` | Direction hint (`"in"`, `"out"`, or `null`) |
| `is_node` | `bool` | `true` if the barrier is a point feature (node) rather than a polygon |

---

### `POST /api/fetch_area`

Start downloading and parsing a new OSM area, saving the result as a `.mapdata` file.

**Asynchronous.** The Overpass query runs in a background thread and this endpoint returns
immediately with a task ID — poll [`GET /api/fetch_area/<task_id>`](#get-apifetch_areatask_id)
for progress and the result.

**Request body**

```json
{
  "min_lat": 50.07,
  "max_lat": 50.09,
  "min_lon": 14.40,
  "max_lon": 14.43,
  "name": "my_area",
  "grid_margin": 150,
  "obstacle_radius": 2.0,
  "buffer_widths": { "road": 7.0, "footway": 3.0, "barrier": 2.0 }
}
```

The bounding box and `name` are required; `name` is sanitized to `[A-Za-z0-9_-]`. The last
three fields are optional and are passed through to `MapData` — see
[Planner Configuration](../dev/planner_config.md).

**Response 202/200**

```json
{ "task_id": "3f2b1c8e-..." }
```

**Error 400** — a required field is missing, the bounding box is degenerate or inverted,
`name` sanitizes to empty, or the requested area exceeds the **25 km²** per-fetch limit.

---

### `GET /api/fetch_area/<task_id>`

Poll the status of a fetch job started by `POST /api/fetch_area`.

**Response 200** — the task record. `status` moves through `pending` → `querying` →
`parsing` → a terminal `done` or `failed`:

```json
{ "status": "querying", "detail": "Querying overpass-api.de (attempt 1/3)" }
```

```json
{
  "status": "done",
  "result": {
    "filename": "my_area.mapdata",
    "roads": 14,
    "footways": 52,
    "barriers": 91,
    "crossroads": 23
  }
}
```

```json
{ "status": "failed", "error": "Overpass API unavailable — try again later" }
```

**Error 404** — unknown task ID: it never existed, or it completed and was already evicted.
Terminal tasks are retained for 60 s after the first poll that observes them as terminal,
so a client that stops polling does not leak the record.

---

### `POST /api/upload_gpx`

Upload a `.gpx` file, parse it (fetching OSM data for its bounds), and save the result as a
`.mapdata` file. Unlike `/api/fetch_area` this is **synchronous** — the request blocks until
the fetch and parse finish. The uploaded GPX itself is written to a temp file and deleted
afterwards; it is never persisted in the data directory.

**Request** — `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `file` | yes | The `.gpx` file |
| `name` | no | Base name for the output file (defaults to the uploaded filename) |

**Response 200** — the same shape as the `result` object of a completed fetch task:

```json
{
  "filename": "my_track.mapdata",
  "roads": 14,
  "footways": 52,
  "barriers": 91,
  "crossroads": 23
}
```

**Error 400** — no file given, `name` sanitizes to empty, or the track's surrounding area
exceeds the 25 km² limit.
**Error 503** — Overpass API unavailable; try again later.

---

### `POST /api/upload_mapdata`

Upload a pre-built `.mapdata` file directly to the data directory, skipping any Overpass
query.

**Request** — `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `file` | yes | The `.mapdata` file. Must have a `.mapdata` extension. |

The file is saved under its own basename. If that name is taken, a `_1`, `_2`, … suffix is
appended rather than overwriting. The saved file is then validated by loading it; an
unloadable file is deleted again rather than left behind.

**Response 200**

```json
{ "filename": "coords_1.mapdata" }
```

`filename` is the name actually saved under, which may differ from the uploaded name.

**Error 400** — no file part, wrong extension, or the file fails to load as valid map data.

---

### `GET /api/export?file=<name>`

Export a `.mapdata` file as a human-readable JSON file, with all stored annotations applied.

**Response** — file download, `Content-Type: application/json`, filename `<name>.exported.mapdata`.

---

### `GET /api/export/geojson?file=<name>`

Export a `.mapdata` file as a GeoJSON `FeatureCollection`, with all stored annotations applied —
the same merged data as `/api/export`, but in GeoJSON form for use in QGIS, geojson.io, or other
GIS tools.

**Response** — file download, `Content-Type: application/geo+json`, filename `<name>.geojson`.

---

## Annotations

All annotation endpoints require `?file=<name>`.

### `GET /api/annotations?file=<name>`

Return the raw annotation store for a file (the full contents of the `.annotations.json` sidecar).

---

### `POST /api/annotations?file=<name>`

Add a new drawn annotation (obstacle polygon or custom path).

**Request body**

```json
{
  "type": "obstacle",
  "geometry": { "type": "Polygon", "coordinates": [[[14.4, 50.07], ...]] },
  "properties": {}
}
```

`type` is `"obstacle"` or `"path"`.

**Response 201** — the created annotation with a generated `id` field.

---

### `PUT /api/annotations/<ann_id>?file=<name>`

Update an annotation's geometry, type, or properties.

**Request body** — same fields as POST.

**Response 200** — updated annotation object.

---

### `DELETE /api/annotations/<ann_id>?file=<name>`

Delete an annotation.

**Response 204.**

---

## Ways

### `GET /api/ways/<way_id>?file=<name>`

Get a single way as a GeoJSON Feature with all stored edits applied. Supports virtual IDs
(`123:0`, `123:1`) for individual segments of a split way.

**Response** — GeoJSON Feature (same properties as `/api/mapdata` features).

---

### `DELETE /api/ways/<way_id>?file=<name>`

Mark a way as deleted. The deletion is stored in the annotations sidecar and does not modify
the `.mapdata` file. Deleted ways are excluded from map display and planning.

**Request body** (optional)

```json
{ "category": "footway", "label": "Fence segment" }
```

**Response 204.**

---

### `PUT /api/ways/<way_id>/tags?file=<name>`

Override OSM tags for a way. The provided tags are merged on top of the original OSM tags on
every subsequent load.

**Request body**

```json
{ "tags": { "highway": "footway", "surface": "gravel" }, "category": "footway", "label": "..." }
```

**Response 204.**

---

### `DELETE /api/ways/<way_id>/tags?file=<name>`

Remove stored tag overrides for a way, reverting to the original OSM tags.

**Response 204.**

---

### `PUT /api/ways/<way_id>/hide?file=<name>`

Temporarily hide a way from the map. Hidden ways are still included in exports and planning.

**Response 204.**

---

### `PUT /api/ways/<way_id>/show?file=<name>`

Un-hide a previously hidden way.

**Response 204.**

---

### `PUT /api/ways/<way_id>/restore?file=<name>`

Restore a deleted way (undo a prior DELETE).

**Response 204.**

---

### `GET /api/ways/<way_id>/segments?file=<name>`

Get the individual segments of a split way as GeoJSON Features.

**Response**

```json
{ "segments": [ { "type": "Feature", ... }, ... ] }
```

---

### `POST /api/ways/split?file=<name>`

Split a way at a node, creating two independently editable segments.

**Request body**

```json
{ "way_id": 123456, "node_id": 789012 }
```

**Response 200**

```json
{ "success": true, "segments": [ ... ] }
```

---

### `DELETE /api/ways/split?file=<name>&way_id=<id>&node_id=<id>`

Undo a way split.

**Response 200** — `{"segments": [...]}`

---

## Nodes

### `GET /api/way_nodes?file=<name>&way_id=<id>`

Get the ordered node list for a way, including WGS84 coordinates.

**Response**

```json
{
  "way_id": 123456,
  "nodes": [
    { "id": 111, "lat": 50.071, "lon": 14.401, "tags": {} },
    { "id": 112, "lat": 50.072, "lon": 14.402, "tags": {} }
  ]
}
```

---

### `POST /api/way_node?file=<name>&way_id=<id>`

Insert a new node into a way, immediately after an existing one.

The new node is synthetic — it has no OSM counterpart — so it is assigned a **negative** ID,
one less than the smallest synthetic ID already recorded for this file (starting at `-1`).
Real OSM node IDs are always positive, so the sign alone distinguishes the two.

**Request body**

```json
{ "after_node_id": 111, "lat": 50.0715, "lon": 14.4015 }
```

`after_node_id` may itself be a previously added synthetic node, so nodes can be chained.

**Response 200**

```json
{ "id": -1, "lat": 50.0715, "lon": 14.4015 }
```

**Error 400** — `file` or `way_id` missing, `way_id`'s non-segment part is not an integer,
or the body is missing `after_node_id`, `lat`, or `lon`.

---

### `DELETE /api/way_node?file=<name>`

Delete a single node from a way.

**Request body**

```json
{ "way_id": 123456, "node_id": 111, "category": "footway", "label": "..." }
```

**Response 204.**

---

### `PUT /api/way_node/restore?file=<name>`

Restore a deleted node.

**Request body**

```json
{ "way_id": 123456, "node_id": 111 }
```

**Response 204.**

---

### `PUT /api/way_nodes/move?file=<name>`

Move one or more nodes to new WGS84 positions (used by drag-and-drop editing).

**Request body**

```json
{
  "way_id": 123456,
  "nodes": [
    { "id": 111, "lat": 50.0715, "lon": 14.4015 }
  ]
}
```

**Response 200** — the updated way as a GeoJSON Feature.

---

### `DELETE /api/way_nodes/move?file=<name>&way_id=<id>`

Reset all node position overrides for a way, restoring original OSM positions.

**Response 204.**

---

## Path Planning

### `GET /api/planner_defaults`

Return the planner default configuration from `config/planner_defaults.yaml`.

**Response** — the YAML file contents as a JSON object (see [Planner Configuration](../dev/planner_config.md)).

---

### `GET /api/cost_grid?file=<name>&min_lat=&min_lon=&max_lat=&max_lon=`

Compute the traversal cost grid for a bounding box. Used by the **Show Cost Grid** overlay
in the Planner. Cells have cost 0.0 (free path) to 1.0 (obstacle boundary); values above
1.0 indicate hard obstacles.

**Query parameters**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `file` | yes | Map filename |
| `min_lat`, `min_lon`, `max_lat`, `max_lon` | yes | Bounding box in WGS84 degrees |
| `highway_costs` | no | JSON-encoded cost override dict (e.g. `{"footway":0.0,"residential":0.5}`) |
| `surface_costs` | no | JSON-encoded surface cost override dict |

**Response** — GeoJSON FeatureCollection of Point features, each with property `cost`.

---

### `POST /api/create_replan`

Run path planning on a set of waypoints. The main planning endpoint.

**Request body**

```json
{
  "file": "coords.mapdata",
  "points": [[50.071, 14.401], [50.075, 14.408]],
  "algorithm": "astar",
  "sub_algorithm": "astar",
  "allowed_ways": ["footway"],
  "cell_size": 0.25,
  "inflate_obstacles": 0.25,
  "simplify_path": true,
  "smooth_path": false,
  "transfer_id": "550e8400-e29b-41d4-a716",
  "highway_costs": { "footway": 0.0, "residential": 0.5 },
  "surface_costs": { "asphalt": 0.0, "grass": 0.5 }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `algorithm` | `"rrt"` | `"astar"` = Grid A*, `"rrt"` = RRT*, `"graph"` = on-way graph planning |
| `sub_algorithm` | `"astar"` | Used when `algorithm` is `"astar"` or `"rrt"` — selects the segment-level solver |
| `allowed_ways` | `["footway"]` | Way categories to treat as low-cost: `"footway"`, `"road"`, or both |
| `transfer_id` | `null` | Client-generated UUID; pass to `/api/cancel_replan` to abort |

**Response**

| `retrieveNum` | `newPath` | Meaning |
|---------------|-----------|---------|
| `0` | `[[lat,lon],...]` | Path changed from input |
| `-1` | `[[lat,lon],...]` | Path unchanged (already obstacle-free) |
| `1` | `null` | Cancelled or failed (check `status` field) |

---

### `POST /api/cancel_replan`

Cancel an in-progress planning request.

**Request body**

```json
{ "transfer_id": "550e8400-e29b-41d4-a716" }
```

**Response** — `{"success": true}`

---

## Wormhole (Robot File Transfer)

Wormhole endpoints use [magic-wormhole](https://magic-wormhole.readthedocs.io/) to transfer
files peer-to-peer. Requires `pip install magic-wormhole` on the server.

### `POST /api/create_wormhole`

Send a GPX file to a robot over the network. The server starts a wormhole send process and
waits up to 15 seconds for the one-time transfer code to be generated.

**Request body**

```json
{ "gpx": "<?xml version=\"1.0\"?>..." }
```

**Response 200**

```json
{ "success": true, "code": "7-guitar-academy", "transfer_id": "uuid" }
```

Give the code to the robot operator, who runs `wormhole receive <code>` to receive the file.

**Response 500** if `wormhole` is not installed or the code was not captured in time.

---

### `POST /api/cancel_wormhole`

Cancel an active wormhole transfer.

**Request body**

```json
{ "transfer_id": "uuid" }
```

**Response** — `{"success": bool, "message": "..."}`

---

## WebSocket (SocketIO)

Live robot telemetry is pushed over SocketIO rather than polled over REST. The server is a
Flask-SocketIO app, so clients connect to the same host and port as the REST API.

### `connect`

Handled server-side to enforce authentication. When `MAP_DATA_ACCESS_TOKEN` is set, the
connecting client must supply a matching token — via the `X-Access-Token` header, an
`access_token` query parameter, or the `map_data_access_token` cookie set by a prior
authenticated HTTP request — or the connection is rejected. With the token unset, all
connections are accepted. Allowed origins are governed by `MAP_DATA_CORS_ORIGINS`; see
[Deployment Security](../viewer.md#deployment-security).

### `telemetry` (server → client)

Broadcast by a background thread at the rate set by `map_data_viewer --telemetry-rate`
(default 2 Hz). Emitted only when the tracker node has new data since the last broadcast,
so a stationary robot produces no traffic. Nothing is emitted at all unless the viewer was
started inside a sourced ROS2 workspace and the tracker node initialized successfully.

```json
{
  "enabled_features": { "gps": true, "ekf": true },
  "position": {
    "gps": { "lat": 50.071, "lon": 14.401 },
    "ekf": { "lat": 50.071, "lon": 14.401 }
  },
  "mission": {
    "waypoints": [{ "lat": 50.072, "lon": 14.402 }],
    "current_waypoint_index": 0
  },
  "status": {
    "battery": { "voltage": 24.6, "current": 3.1 },
    "motors_enabled": true,
    "motor_error": 0,
    "gps_fix": 2,
    "teensy_temp": 41.2,
    "speed": 0.8,
    "speed_limit": { "value": 0.5, "percentage": false },
    "collision_action": null,
    "recovery_active": false,
    "teleop_active": false,
    "nav_state": "navigating",
    "localization_state": null,
    "last_speech": { "level": "info", "text": "..." }
  }
}
```

Field notes: `enabled_features` is a map of feature name to enabled flag, not a list.
`gps_fix` is the raw integer `NavSatStatus.status`. `motor_error` is an integer error code
(`0` = no error). `position.gps`, `position.ekf`, `speed_limit`, `collision_action`, and
`last_speech` are `null`/`{}` until the corresponding topic publishes.

See [Tracker mode](../viewer_tracker.md) for how the viewer renders these fields.

---

## Page routes

### `GET /`

Serves the viewer's single-page application shell (`templates/index.html`). Not a JSON
endpoint. The template receives `THUNDERFOREST_API_KEY` and `SEZNAM_API_KEY` from the
environment so the optional basemap layers can be enabled — see
[Tile Layer API Keys](../viewer.md#tile-layer-api-keys).
