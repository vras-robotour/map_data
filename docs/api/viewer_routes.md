# Viewer REST API

!!! note "Internal API"
    The viewer's JavaScript is the client of this API. It is documented for tooling that talks to a running viewer.

All endpoints are served by `map_data_viewer`. Handlers live in `map_data.viewer.routes`, one module per section: `files`, `annotations`, `ways` (ways and nodes), `planning`, `sharing` (goal QR codes and wormhole); shared helpers in `common`. The tracker endpoints are in `map_data.viewer.tracker_routes`.

## Conventions

- **Errors** are JSON `{"error": "<message>"}` with the HTTP status, for every endpoint.
- **409**: the file's annotation store exists but is not a readable JSON object. It is moved aside to `<store>.corrupt-<timestamp>` and never overwritten; reload to continue with an empty store.
- **`file`** is a `.mapdata` name in the data directory. A name that resolves outside it is 400, a missing file 404.
- **`way_id`** is an OSM way ID or a split segment `"<id>:<index>"`.
- Edits are written to `<stem>.annotations.json` under a per-file lock; the `.mapdata` file is never modified. See [Data formats](../dev/data_formats.md#annotationsjson-sidecar-format).

---

## Files

### `GET /api/files`

`{"mapdata": [name, ...], "gpx": [name, ...]}`, sorted. Empty lists when the data directory does not exist.

### `GET /api/mapdata?file=<name>`

The map with its way edits and tag overrides applied, as a GeoJSON `FeatureCollection`. Drawn annotations are not included (the client draws them from `/api/annotations`). The response has an `ETag`; a matching `If-None-Match` gets 304.

Way feature properties:

| Property | Type | Description |
|----------|------|-------------|
| `id` | int or str | Way ID, `"<id>:<index>"` for split segments |
| `category` | str | `road`, `footway`, `barrier` or `crossroad` |
| `tags` | object | OSM tags with overrides applied |
| `in_out` | str | `"outer"` / `"inner"` for multipolygon members, else `""` |
| `is_node` | bool | `true` for a barrier made from a single obstacle node |

Geometry is `Polygon`, `MultiPolygon` or `LineString`; roads and footways are buffered polygons. Waypoints are `Point` features with `category: "waypoint"` and `index`.

### `POST /api/fetch_area`

Starts a background download and parse of an OSM area; returns `{"task_id": "<uuid4>"}` at once.

```json
{
  "min_lat": 50.07, "min_lon": 14.40, "max_lat": 50.09, "max_lon": 14.43,
  "name": "my_area",
  "grid_margin": 150, "obstacle_radius": 2.0,
  "buffer_widths": {"road": 7, "footway": 3, "barrier": 2}
}
```

- The box plus its `grid_margin` must be at most 25 km² (`MAX_FETCH_AREA_KM2`).
- `name` is sanitized to `[A-Za-z0-9_-]`. An existing map, or a leftover `<name>.annotations.json`, is never overwritten: the map is saved as `<name>_1.mapdata`, `<name>_2.mapdata`, ...
- `grid_margin`, `obstacle_radius`, `buffer_widths` are optional; defaults from `config/planner_defaults.yaml`.

**400**: missing or invalid field, box too large, empty name.

### `GET /api/fetch_area/<task_id>`

The task record, polled by the client:

| `status` | Other fields |
|---|---|
| `pending`, `querying`, `parsing` | `detail` (progress text) |
| `done` | `result`: `{"filename", "roads", "footways", "barriers", "crossroads"}` |
| `failed` | `error` |

A finished task is kept for 60 s after it is first polled. **404**: unknown or evicted task.

### `POST /api/upload_gpx`

Builds a map around an uploaded GPX track, synchronously. `multipart/form-data`:

| Field | Required | Description |
|-------|----------|-------------|
| `file` | yes | The GPX file |
| `name` | no | Output stem (default: the upload's stem), sanitized as for `fetch_area` |
| `options` | no | JSON object with `grid_margin`, `obstacle_radius`, `buffer_widths` |

Saved under the next free name, as for `fetch_area`; the Overpass cache is written next to it. Returns `{"filename", "roads", "footways", "barriers", "crossroads"}`.

**400**: bad file, name or options, or the area (with margin) exceeds 25 km². **503**: Overpass unreachable. **500**: parsing failed.

### `POST /api/upload_mapdata`

Saves an uploaded `.mapdata` file (`multipart/form-data`, field `file`) under the next free name. A file `MapData.load` cannot read is deleted again. Returns `{"filename": "<saved name>"}`.

**400**: missing file, wrong extension or unreadable map.

### `GET /api/export?file=<name>`

Downloads `<stem>.exported.mapdata`: the map with every annotation applied, as `.mapdata` JSON. Nothing is written to the data directory.

### `GET /api/export/geojson?file=<name>`

Downloads the same merged map as `<stem>.geojson` (`application/geo+json`).

---

## Annotations

All take `?file=<name>`.

### `GET /api/annotations`

The whole annotation store (every edit, not only drawn annotations).

### `POST /api/annotations`

Adds a drawn annotation.

```json
{"type": "obstacle", "geometry": {"type": "Polygon", "coordinates": [[[14.4, 50.07], ...]]}, "properties": {}}
```

`geometry` is required: a `LineString`, `Polygon` or `MultiPolygon`. `type` is `"path"` (a drawn way) or anything else, default `"obstacle"`. Returns **201** with the annotation and its uuid4 `id`. **400**: invalid or unsupported geometry, or invalid type.

### `PUT /api/annotations/<ann_id>`

Replaces the geometry, and `type`/`properties` when given; the body is validated as for `POST`. Returns the annotation. **400**: invalid body. **404**: unknown ID.

### `DELETE /api/annotations/<ann_id>`

**204**. **404**: unknown ID.

---

## Ways

All take `?file=<name>`.

### `GET /api/ways/<way_id>`

The edited way or segment as a GeoJSON Feature (properties as in `/api/mapdata`). **404**: unknown way, way reduced to nothing by node deletions, or segment index out of range.

### `DELETE /api/ways/<way_id>`

Marks a way, or one segment, as deleted (idempotent). Optional body `{"category", "label"}` for the change list. **204**.

### `PUT /api/ways/<way_id>/restore`

Undoes the deletion of the same way or segment ID. **204**.

### `PUT /api/ways/<way_id>/tags`

Replaces the way's tag override; applies to every segment.

```json
{"tags": {"highway": "footway", "surface": "gravel"}, "category": "footway", "label": "..."}
```

**204**. **400**: `tags` is not an object of strings.

### `DELETE /api/ways/<way_id>/tags`

Removes the tag override. **204**.

### `PUT /api/ways/<way_id>/hide`, `PUT /api/ways/<way_id>/show`

Hides or shows a whole way in the viewer. Export and planning still use hidden ways. Optional body `{"category", "label"}`. **204**.

### `GET /api/ways/<way_id>/segments`

`{"segments": [Feature, ...]}`, empty for a missing or fully deleted way.

### `POST /api/ways/split`

Body `{"way_id": 123456, "node_id": 789012}` (an interior node). The segment after the split starts at a new synthetic copy of the node, so the ends move independently and the planner does not route across the split. Returns `{"success": true, "segments": [...]}`. **400**: `node_id` is not an interior node of the edited way. **404**: unknown way.

### `DELETE /api/ways/split?way_id=<id>&node_id=<id>`

Removes one split point; the detached copy and the moves of both ends are dropped. Returns `{"segments": [...]}`.

---

## Nodes

All take `?file=<name>&way_id=<id>`.

### `GET /api/way_nodes`

```json
{"way_id": "123456", "nodes": [{"id": 111, "lat": 50.071, "lon": 14.401, "tags": {}}]}
```

`nodes` is `[]` when deletions removed the whole way or segment.

### `POST /api/way_node`

Inserts a synthetic node after an existing one. Body `{"after_node_id": 111, "lat": 50.0712, "lon": 14.4013}`. Returns `{"id": <negative id>, "lat", "lon"}`. **400**: `after_node_id` is not on the edited way. **404**: unknown way.

### `DELETE /api/way_node?node_id=<id>`

Deletes a node from the way, or from one segment when `way_id` is a segment ID. A node added with `POST /api/way_node` is removed outright. **204**.

### `PUT /api/way_node/restore?node_id=<id>`

Undoes a node deletion (same `way_id`). Removed added nodes cannot be restored. **204**.

### `PUT /api/way_nodes/move`

Records node positions for the way; every segment shares them.

```json
{"nodes": [{"id": 111, "lat": 50.0715, "lon": 14.4015}], "category": "footway", "label": "..."}
```

All nodes are validated before anything is written; each must be on the edited way (a split end counts, a way without nodes takes its own ID). **204**. **400**: invalid position or a node not on the way. **404**: unknown way.

### `DELETE /api/way_nodes/move`

Discards every node move of the way. **204**.

---

## Planning

### `GET /api/planner_defaults`

`config/planner_defaults.yaml` as JSON (see [Planner configuration](../dev/planner_config.md)), plus:

| Field | Description |
|---|---|
| `source` | `"route_planner"` when the viewer runs with ROS 2 and the tracker's `planner_node` answers, else `"yaml"` |
| `node` | The node name, only with `source: "route_planner"` |

With `route_planner`, that node's `cell_size`, `inflate_obstacles`, `simplify_path` and `smooth_path` replace the file's values.

### `POST /api/create_replan`

Plans a route on the edited map with `plan_route`.

```json
{
  "file": "coords.mapdata",
  "points": [[50.071, 14.401], [50.075, 14.408]],
  "algorithm": "graph",
  "allowed_ways": ["footway"],
  "transfer_id": "0b8f4c1e-..."
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `file`, `points` | required | Map and `[[lat, lon], ...]`, at least two |
| `algorithm` | `"rrt"` | `"graph"` follows the ways; `"astar"` / `"rrt"` use the grid planner |
| `sub_algorithm` | `"astar"` | The grid search actually run (`"astar"` or `"rrt"`); RRT* falls back to A* |
| `allowed_ways` | `["footway"]` | `"footway"` and/or `"road"` |
| `cell_size` | `planner_defaults.yaml` | 0.05–10 m |
| `inflate_obstacles` | `planner_defaults.yaml` | 0–10 m |
| `simplify_path`, `smooth_path` | `planner_defaults.yaml` | Booleans |
| `grid_cost_weight` | `planner_defaults.yaml` | 0–1000 |
| `highway_costs`, `surface_costs` | `planner_defaults.yaml` | `{tag value: cost}`, for both planners |
| `traversability` | the rule file | Rule YAML text |
| `transfer_id` | none | Makes a grid plan cancellable with `/api/cancel_replan`; unique per request |

**Response**

| Body | Meaning |
|---|---|
| `{"retrieveNum": 0, "newPath": [[lat, lon], ...]}` | Route differs from the input polyline |
| `{"retrieveNum": -1, "newPath": [[lat, lon], ...]}` | Route equals the input polyline |
| `{"retrieveNum": 1, "newPath": null, "status": "cancelled", "reason": "cancelled"}` | Cancelled with `/api/cancel_replan` |
| `{"retrieveNum": 1, "newPath": null, "status": "failed", "reason": "<reason>"}` | No route |

Failure `reason`s: `snap_too_far`, `unreachable`, `start_outside_map`, `goal_outside_map` (graph); `no_path`, `start_blocked`, `goal_blocked`, `start_outside_map`, `goal_outside_map` (grid). See [Action failure reasons](../planning.md#action-failure-reasons).

**400**: missing or invalid parameter, unknown algorithm, fewer than two points, or a grid needing more than 4 million cells (`grid_too_large`). **404**: unknown file.

### `POST /api/cancel_replan`

Body `{"transfer_id": "..."}`. Returns `{"success": true}`, also for an unknown or finished ID. A cancel that arrives before its plan starts still aborts it.

### `GET /api/cost_grid`

Query: `file`, `min_lat`, `min_lon`, `max_lat`, `max_lon`, optional JSON-encoded `highway_costs` / `surface_costs`.

Renders the edited map's 1 m cost grid over the box: `{"image": "data:image/png;base64,...", "bounds": [[south, west], [north, east]]}` for `L.imageOverlay`, both `null` for an empty grid. Black = obstacle, translucent grey = off-path (≥ `default_off_path_cost`), green → red = way cost / `path_cost_cap`. **400**: the box needs more than 4 million cells.

### `GET /api/traversability`

`{"yaml": "<rule file text>", "path": "<path>"}`; empty text if the file is missing.

### `PUT /api/traversability`

Body `{"traversability": "<yaml text>"}`. Validates the rules and writes the text verbatim (comments kept). Returns `{"path": "<path>"}`. `route_planner` uses the new rules on its next goal; `osm_cloud` needs a restart. **400**: invalid rules.

### `POST /api/traversability/blocked`

Body `{"file", "traversability"?, "allowed_ways"?}`. Lists the ways the graph planner will not route over: `{"blocked": [[way_id, reason], ...]}`.

---

## Sharing

### `GET /api/qr`, `GET /api/qr.svg`

`?lat=50.11&lon=14.41[&scale=12][&download=1][&caption=...]`

A QR code of the goal as PNG or SVG; the payload is `geo:<lat>,<lon>`, also sent in the `X-Geo-URI` header. `scale` is pixels per module. `caption` is the text under the code: absent = the payload, empty = none. `download=1` adds a `Content-Disposition` file name. **400**: missing or invalid coordinates.

### `POST /api/create_wormhole`

Body `{"gpx": "<GPX text>"}`. Starts `wormhole send` (the [magic-wormhole](https://magic-wormhole.readthedocs.io/) CLI must be on `PATH`) and waits up to 15 s for the code. Returns `{"success": true, "code": "7-guitar-academy", "transfer_id": "<uuid4>"}`; the receiver runs `wormhole receive <code>`.

**400**: no `gpx`. **500**: the CLI failed to start or no code within 15 s.

### `POST /api/cancel_wormhole`

Body `{"transfer_id": "..."}`. Returns `{"success": true, "message": "Transfer cancelled"}`. **400**: no `transfer_id`. **404**: unknown transfer.

---

## Tracker

Every tracker endpoint answers **503** without a ROS 2 context.

### `GET /api/tracker/settings`

```json
{
  "path": "/.../config/tracker.yaml",
  "exists": true,
  "file_error": null,
  "heading_types": {"imu": "sensor_msgs/msg/Imu", "...": "..."},
  "topics": {"/gps/fix": ["sensor_msgs/msg/NavSatFix"]},
  "settings": [
    {
      "name": "gps_fix_topic",
      "section": "Position / heading",
      "description": "Raw GPS fix",
      "msg_type": "sensor_msgs/msg/NavSatFix",
      "choices": [],
      "default": "/gps/fix",
      "value": "/gps/fix",
      "saved": "/fixposition/odometry_llh"
    }
  ]
}
```

`value` is live, `saved` the config file's value (the default when unset). When the file cannot be read, `file_error` is set and `saved` falls back to the defaults.

### `PUT /api/tracker/settings`

```json
{"settings": {"gps_fix_topic": "/fixposition/odometry_llh", "heading_type": "yaw_vector3"}, "save": true}
```

Applies a subset of settings to the running tracker, which resubscribes. With `save`, all current settings are written to the config file, comments kept. Returns `{"changed": bool, "path": str | null}`.

**400**: unknown setting or invalid value; nothing is applied. **500**: applied but not saved.

### `DELETE /api/tracker/trail`

Clears the robot trail kept on the tracker node. Returns `{"dropped": <number of fixes>}`.
