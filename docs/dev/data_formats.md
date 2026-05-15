# Data Formats

This page documents every file format produced or consumed by the `map_data` package.

---

## `.mapdata` file format

A `.mapdata` file is a **JSON text file** written by `MapData.save()` via `json.dump`. It is not a pickle file and it is not binary. (Legacy files created before the JSON migration may be Python pickle — these are detected by a `0x80` leading byte and loaded transparently for backwards compatibility, but the format is deprecated.)

Read a `.mapdata` file with:

```python
from map_data.map_data import MapData

md = MapData.load("route.mapdata")
```

Never open `.mapdata` files with `pickle.load` directly — use `MapData.load()` so that both the current JSON format and the legacy pickle format are handled correctly.

### Top-level JSON structure

```json
{
  "metadata": { ... },
  "waypoints": [[x0, y0], [x1, y1], ...],
  "roads": [ <way object>, ... ],
  "footways": [ <way object>, ... ],
  "barriers": [ <way object>, ... ],
  "crossroads": [ <way object>, ... ],
  "nodes_cache": { "<node_id>": {"lat": 50.1, "lon": 14.5, "tags": {}}, ... }
}
```

### `metadata` object

| Field | Type | Description |
|-------|------|-------------|
| `zone_number` | int | UTM zone number |
| `zone_letter` | str | UTM zone letter |
| `min_x` / `max_x` | float | UTM easting bounding box (includes margins) |
| `min_y` / `max_y` | float | UTM northing bounding box (includes margins) |
| `min_lat` / `max_lat` | float | WGS-84 latitude bounding box |
| `min_long` / `max_long` | float | WGS-84 longitude bounding box |
| `coords_file` | str or null | Path of the source GPX/YAML file, or `null` if constructed from an array |

### Way object (within `roads`, `footways`, `barriers`, `crossroads`)

| Field | Type | Description |
|-------|------|-------------|
| `id` | int or str | OSM way/node ID, or a negative synthetic ID for merged segments |
| `is_area` | bool | `true` if the geometry is a closed polygon |
| `nodes` | list[int] | Ordered list of OSM node IDs |
| `tags` | object | OSM tag key-value pairs |
| `line` | str | Shapely WKT representation of the geometry (UTM coordinates) |
| `in_out` | str or null | `"outer"` or `"inner"` for multipolygon relation members; `null` otherwise |

### Human-readable export

The viewer's **Export** button writes `<stem>.exported.mapdata` — a JSON file with the same schema as above but with all viewer annotations (deleted ways, tag overrides, node position overrides, drawn annotations) applied. This file can be used as a clean, annotation-resolved snapshot for downstream tools.

---

## `.annotations.json` sidecar format

Annotation files are written alongside the `.mapdata` file as `<stem>.annotations.json`. They store all edits made in the viewer without modifying the original map data. The viewer merges the sidecar at load time.

### Top-level keys

| Key | Type | Description |
|-----|------|-------------|
| `version` | int | Schema version (currently `1`) |
| `annotations` | list | Drawn geometry annotations (obstacles, path segments) |
| `deleted_ways` | list | Ways removed from the map view |
| `hidden_ways` | list | Ways temporarily hidden but not deleted |
| `tag_overrides` | object | Per-way OSM tag edits |
| `split_ways` | object | Node IDs at which a way is split into segments |
| `deleted_nodes` | object | Node IDs deleted from specific ways |
| `node_position_overrides` | object | Dragged node position corrections |
| `change_log` | list | Audit log of all user-initiated changes |
| `change_log_migration` | str | Internal migration version tag |

### `annotations` list

Each entry is a GeoJSON Feature-like object:

```json
{
  "id": "ann_1715600000000",
  "type": "obstacle",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[14.5678, 50.1234], [14.5680, 50.1234],
                     [14.5680, 50.1236], [14.5678, 50.1234]]]
  },
  "properties": {}
}
```

`type` is either `"obstacle"` (drawn barrier polygon) or `"path"` (drawn navigable path segment).

### `deleted_ways` list

```json
[
  {"id": 123456789, "category": "barrier", "label": "building"}
]
```

### `hidden_ways` list

Same structure as `deleted_ways`. Hidden ways are still included in export but are not rendered in the viewer.

### `tag_overrides` object

Keys are way IDs as strings; values are dicts of tag key-value pairs that replace the original OSM tags for that way:

```json
{
  "987654321": {"highway": "footway", "surface": "asphalt"}
}
```

### `split_ways` object

Keys are way IDs as strings; values are lists of OSM node IDs at which the way is split:

```json
{
  "123456789": [9876543, 9876544]
}
```

### `deleted_nodes` object

Keys are way IDs as strings; values are lists of OSM node IDs deleted from that way:

```json
{
  "123456789": [9876540, 9876541]
}
```

### `node_position_overrides` object

Keys are way IDs as strings; values are dicts mapping node IDs (as strings) to corrected `{lat, lon}` positions:

```json
{
  "123456789": {
    "9876542": {"lat": 50.1235, "lon": 14.5679}
  }
}
```

### `change_log` list

An ordered audit log. Each entry has a `type` field and optional `ts` (ISO timestamp for user-initiated changes):

```json
[
  {"type": "way",  "id": 123456789, "category": "barrier", "label": "building", "ts": "2025-11-01T10:00:00Z"},
  {"type": "node", "way_id": 123456789, "node_id": 9876540, "ts": "2025-11-01T10:01:00Z"},
  {"type": "tag",  "id": 987654321},
  {"type": "move", "id": 123456789, "category": "footway", "label": ""},
  {"type": "split","way_id": 123456789, "node_id": 9876543}
]
```

---

## GPX waypoint format

The package reads standard **GPX 1.1** files. Waypoints must be `<wpt>` elements. Track points (`<trkpt>`) and route points (`<rtept>`) are also accepted by `MapData` (for map creation), but `parse_gpx_file` in `map_data/utils/gpx.py` reads only `<wpt>` elements for path planning.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1" creator="MapData Planner">
  <wpt lat="50.1234" lon="14.5678"></wpt>
  <wpt lat="50.1240" lon="14.5690"></wpt>
  <wpt lat="50.1250" lon="14.5700"></wpt>
</gpx>
```

Elevation is optional for path planning; when present it is preserved in the output GPX produced by `create_gpx_content`.

---

## YAML waypoint format

As an alternative to GPX, the package accepts a simple YAML file via `parse_yaml_file` (called automatically by `parse_path` when the file extension is `.yaml`).

```yaml
waypoints:
  - latitude: 50.1234
    longitude: 14.5678
    elevation: 200.0
  - latitude: 50.1240
    longitude: 14.5690
    elevation: 201.5
  - latitude: 50.1250
    longitude: 14.5700
```

Rules:

- The top-level key must be `waypoints`.
- Each entry must have `latitude` and `longitude` (decimal degrees, WGS-84).
- `elevation` is optional; it defaults to `0.0` if omitted.
- The file must use the `.yaml` extension for automatic format detection.

The parsed result is identical to the GPX path: a NumPy array of UTM coordinates plus the inferred UTM zone number and letter.
