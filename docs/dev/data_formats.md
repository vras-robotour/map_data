# Data formats

File formats the `map_data` package reads and writes.

---

## `.mapdata` file format

A `.mapdata` file is JSON, written atomically by `MapData.save()`. Pickle files are not supported; rebuild them from the source GPX with `create_mapdata`.

Read a `.mapdata` file with:

```python
from map_data.map_data import MapData

md = MapData.load("route.mapdata")
```

### Top-level JSON structure

```json
{
  "format_version": 1,
  "metadata": { ... },
  "waypoints": [[x0, y0], [x1, y1], ...],
  "roads": [ <way object>, ... ],
  "footways": [ <way object>, ... ],
  "barriers": [ <way object>, ... ],
  "crossroads": [ <way object>, ... ],
  "nodes_cache": { "<node_id>": {"lat": 50.1, "lon": 14.5, "tags": {}}, ... }
}
```

`format_version` is the schema version (`FORMAT_VERSION` in `map_data/utils/serialization.py`, `1`). A file without the key is read as version 0; a file with a newer version than the build knows raises `ValueError`. Every way key is optional on load; a missing key takes the `Way` default.

`MapData.load()` recomputes the node-based crossroads, so the saved `crossroads` list only matters for drawn-path intersections (`type=annotation_intersection`).

### `metadata` object

| Field | Type | Description |
|-------|------|-------------|
| `zone_number` | int | UTM zone number |
| `zone_letter` | str | UTM zone letter |
| `min_x` / `max_x` | float | UTM easting bounding box (includes margins) |
| `min_y` / `max_y` | float | UTM northing bounding box (includes margins) |
| `min_lat` / `max_lat` | float | WGS-84 latitude bounding box |
| `min_long` / `max_long` | float | WGS-84 longitude bounding box |
| `coords_file` | str or null | Path of the source GPX file, or `null` if built from an array |

### Way object (within `roads`, `footways`, `barriers`, `crossroads`)

| Field | Type | Description |
|-------|------|-------------|
| `id` | int or str | OSM way or node ID; negative for synthetic ways (drawn annotations, merged relation rings), `"<id>:<index>"` for a split segment in an export |
| `is_area` | bool | `true` if `line` is a polygon |
| `nodes` | list[int] | Ordered list of OSM node IDs |
| `tags` | object | OSM tag key-value pairs |
| `line` | str or null | WKT geometry in UTM, 6 decimals. Roads, footways and open barrier lines are buffered `POLYGON`s |
| `in_out` | str or null | `"outer"` or `"inner"` for multipolygon relation members; `null` otherwise |

### Export

The viewer's **Export** button downloads `<stem>.exported.mapdata`: the same schema with every annotation applied (`apply_store`). **GeoJSON** downloads the same merged map as `<stem>.geojson`, a `FeatureCollection` (`mapdata_to_geojson` in `map_data/viewer/helpers.py`) for QGIS or geojson.io. Nothing is written to the data directory.

---

## `.annotations.json` sidecar format

`<stem>.annotations.json` next to the `.mapdata` file holds every viewer edit; the `.mapdata` file is never modified. `apply_store()` in `map_data/annotations.py` merges it; the viewer, `map_data_plan`, `route_planner` and `osm_cloud` load maps with it (`load_mapdata_with_annotations`).

A store that exists but is not a readable JSON object raises `AnnotationStoreError` and is never overwritten; the viewer answers 409 and moves it aside to `<path>.corrupt-<timestamp>` (`quarantine_annotation_store`).

### Top-level keys

| Key | Type | Description |
|-----|------|-------------|
| `version` | int | Schema version (currently `1`) |
| `annotations` | list | Drawn obstacles and paths |
| `deleted_ways` | list | Deleted ways and segments |
| `hidden_ways` | list | Ways hidden in the viewer only |
| `tag_overrides` | object | Replacement tags per way |
| `tag_override_meta` | object | `{category, label}` of each overridden way, for the change list |
| `split_ways` | object | Node IDs a way is split at |
| `detached_nodes` | list | Synthetic node that starts the segment after a split |
| `added_nodes` | list | Synthetic nodes inserted into ways |
| `deleted_nodes` | list | Nodes deleted from ways or segments |
| `node_position_overrides` | object | Moved node positions |
| `change_log` | list | The viewer's undoable change list |

Every key except `version` and `annotations` is optional.

### `annotations` list

Each entry is a GeoJSON Feature-like object:

```json
{
  "id": "3f6c2a1e-8d4b-4c47-9a51-2b7e0f9d6c13",
  "type": "obstacle",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[14.5678, 50.1234], [14.5680, 50.1234],
                     [14.5680, 50.1236], [14.5678, 50.1234]]]
  },
  "properties": {}
}
```

`id` is a uuid4. `type` is `"path"` (a drawn way: a road or footway, buffered by half its `width` property; `highway` defaults to `path`) or anything else, usually `"obstacle"` (a barrier).

### `deleted_ways` list

```json
[
  {"id": 123456789, "category": "barrier", "label": "building"}
]
```

### `hidden_ways` list

Same structure as `deleted_ways`, by original way ID. Hiding only affects rendering: export and planning still use hidden ways, and hiding adds no `change_log` entry.

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

### `detached_nodes` list

A split made in the viewer detaches the segments. The segment after the split
starts at its own copy of the split node, with a negative synthetic `id`. This
means each end can be moved on its own, and the planner does not route across
the split. Splits without an entry (older stores) keep sharing the node.
Undoing the split removes the entry and the position overrides of both ends.

```json
[
  {"way_id": 123456789, "node_id": 9876543, "id": -3}
]
```

Positions of moved nodes, including detached ends, are also what the graph
planner uses.

### `added_nodes` list

Nodes inserted with the viewer's *add node* tool. `id` is negative and shared with the `detached_nodes` ID space (`next_synthetic_node_id`); the node goes after `after_node_id` in the way's node list.

```json
[
  {"id": -4, "way_id": 123456789, "after_node_id": 9876542, "lat": 50.1236, "lon": 14.5681}
]
```

### `deleted_nodes` list

Nodes deleted from a way, or from one segment when `way_id` is a `"<id>:<index>"` string:

```json
[
  {"way_id": 123456789, "node_id": 9876540},
  {"way_id": "123456789:1", "node_id": 9876541}
]
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

A move applies to that way only. When the node is also used by another way, the merged map
gives the moved way its own copy of the node under a fresh negative id (minted at merge time,
not stored) and the other ways keep the original position, so the ways no longer join there.
Ways that put the node within 0.5 m of each other keep sharing it; when all of them do, the
node simply moves. A move recorded on a way that no longer uses the node is ignored.

A moved **end** node that lands within 5 m of another way is joined to it at merge time
(`join_ways` in `map_data/annotations.py`): the junction, a node of that way or a new one on
its edge, is put after the end in the way's node list. The stored position is not changed.

### `change_log` list

The viewer's change list, in order. Each entry has a `type`, its key fields and `ts` (Unix time in seconds, float):

```json
[
  {"type": "way",      "id": 123456789, "ts": 1761900000.12},
  {"type": "node",     "way_id": 123456789, "node_id": 9876540, "ts": 1761900060.5},
  {"type": "add_node", "way_id": 123456789, "node_id": -4, "ts": 1761900070.0},
  {"type": "tag",      "id": "987654321", "ts": 1761900080.3},
  {"type": "move",     "id": 123456789, "category": "footway", "label": "", "ts": 1761900090.9},
  {"type": "split",    "way_id": 123456789, "node_id": 9876543, "ts": 1761900100.4}
]
```

---

## GPX waypoint format

`MapData` reads GPX 1.1 files (`create_mapdata`, the viewer's GPX upload, `osm_cloud`'s `gpx_file` parameter). It reads `<wpt>` elements if the file has any; otherwise it falls back to track points (`<trkpt>`) from `<trk>` elements, and then to route points (`<rtept>`) from `<rte>` elements. Loading fails if none of the three are present.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1" creator="MapData Planner">
  <wpt lat="50.1234" lon="14.5678"></wpt>
  <wpt lat="50.1240" lon="14.5690"></wpt>
  <wpt lat="50.1250" lon="14.5700"></wpt>
</gpx>
```

YAML waypoint files are not supported.
