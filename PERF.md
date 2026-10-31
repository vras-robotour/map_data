# Performance log

## Viewer API

Measured with `scripts/bench/viewer.py` (Flask test client; p50 over 3-20 requests, cold = every
viewer cache cleared before each request, warm = filled caches) and, for the last column, a real
`map_data_viewer` plus `curl -w %{time_total}`. Maps: strom (11 MB .mapdata, 169 KB store),
robotour, kralovska_obora, demo/buchlovice (store left out). Load: one or two operators on a LAN.
Run-to-run spread on this (shared, loaded) machine is ~5-15 %, so only changes beyond that count.

`uv run python scripts/bench/viewer.py --quick` (demo map, ~15 s) is the guard;
`--data-dir DIR [--maps ...] [--only reads|edit] [--profile mapdata|export|...]` for the rest.

### Baseline (strom unless noted; ms)

| Request | cold p50 | warm p50 | size |
|---|---|---|---|
| GET /api/mapdata | 720 | 5 (304: 5) | 4.2 MB, uncompressed (gzip would be 1.3 MB) |
| GET /api/annotations | 4 | 4 | 103 KB |
| GET /api/ways/<id>/segments, /api/way_nodes | ~300 (map parse) | 2-3 | 3-12 KB |
| edit: PUT move / tags / DELETE way | 8-9 each | | |
| edit: GET /api/mapdata after | 456 | | |
| edit: POST create_replan graph after | 223 | | |
| POST create_replan graph / rrt / astar (~500 m) | 553 / 1756 / 1818 | 12 / 1304 / 1501 | |
| GET /api/cost_grid | 779 | 224 | 89 KB |
| GET /api/export, /api/export/geojson | 503, 732 | 138, 351 | 5.2 MB, 4.2 MB |

Findings: everything is a few ms when caches are warm. The expensive requests are (1) the GET
`/api/mapdata` after **every** edit (456 ms strom, 58 ms robotour: the cache key contains the store
digest, so each edit is a full miss and all ~3000 geometries were re-projected UTM -> lon/lat and
the crossroads recomputed up to 3x), (2) the first request after a start (map parse ~300 ms plus
the same projection), (3) the grid planners (1.3-1.8 s on strom; `pathsolver`, not the viewer's).

### Attempts

| Idea | Baseline -> Result | Verdict | Why |
|---|---|---|---|
| Cache the GeoJSON of a geometry by value (`lru_cache` on `geom_to_geojson`; shapely hashes by coordinates, so nothing to invalidate) | GET /api/mapdata after an edit 456 -> 224 ms (strom), 58 -> 36 (robotour); export/geojson warm 351 -> 107 | **kept** (17c6a8e) | An edit touches a handful of ways; the other ~3000 geometries (and unchanged recomputed crossroads) now hit. Responses byte-identical on 4 maps x 6 edit states (216 responses). Cold unchanged. |
| Recompute crossroads once after the way-edit and tag-override passes (`apply_way_and_tag_edits`) | GET after edit 206 -> 154 ms (strom, A/B x3), 29 -> 23 (robotour) | **kept** (f632911) | `recompute_crossroads` (~50 ms strom) ran after each pass and its result was thrown away by the next. Also speeds up `apply_store` (planning, export). Identical output; test counts the calls. |
| Build `join_ways` edge LineStrings with one `shapely.linestrings` call | POST replan graph after an edit 212 -> 191 ms (strom, A/B x3) | **kept** (6ad421b) | ~6000 `LineString()` constructions took ~80 ms of the merged-map build. Small (10 %) but consistent and 6 lines. Identical output. |
| gzip `/api/mapdata` in `after_request` (level 1, > 64 KB, only when `Accept-Encoding: gzip`), live server + `curl --limit-rate` | strom 4.2 -> 1.46 MB; fast link 10 -> 153 ms; 100 Mbit 342 -> 240 ms; 24 Mbit 1396 -> 583 ms; robotour (0.4 MB) 8 -> 21 ms | **reverted** | On a gigabit LAN/loopback it is a 15x loss (compressing 4 MB costs ~150 ms); it only wins below ~100 Mbit/s, and caching the compressed body would still add ~150 ms to every post-edit GET. Not justified for the stated load; revisit if operators work over slow WiFi/VPN (the prototype is ~15 lines). |
| `json.dumps(indent=None)` for `/api/export` | warm 105 -> 29 ms for the dump (of 138) | **not applied** | Changes the downloaded file's format (diff-friendly indent=2) for an imperceptible gain on a manual download. |
| Batch the UTM projection of all geometries in one call for the cold path | not built; analysis: `utm.to_latlon` costs ~1.6 us/point (100k points = 160 ms), a per-point cost, not per-call overhead | **not tried** | Vectorising was already in place per ring; batching saves at most the ~1400 call overheads. The real lever (pyproj) is a new dependency. |
| Serve `/api/mapdata` from per-feature JSON strings instead of one `json.dumps` | not built; `json.dumps` of the 4 MB document is ~60 ms of the remaining 154 ms (strom) | **not tried** | Would need a hand-assembled body that must stay byte-identical (key order, floats); gain is below what an operator notices at this point. |
| Planner / grid / `get_points` / map parse | | **out of scope** | `pathsolver/**`, `map_data.py` belong to other agents (rrt/astar 1.3 s, cost_grid fill 190 ms, parse 300 ms, `get_points` 70 ms, `parse_intersections` 50 ms are the remaining big items). |

### After (same bench, same machine, back to back with the baseline tree; ms)

| Request | cold p50 | warm p50 |
|---|---|---|
| strom GET /api/mapdata after an edit | 456 -> **156** | |
| strom POST create_replan graph after an edit | 223 -> **180** | |
| strom GET /api/export/geojson | 732 -> 780 (noise) | 351 -> **107** |
| robotour GET /api/mapdata after an edit | 58 -> **28** | |
| kralovska_obora GET /api/mapdata after an edit | 360 -> **108** | |
| buchlovice GET /api/mapdata after an edit | 230 -> **78** | |
| everything else | unchanged within spread | unchanged |

Real server (`map_data_viewer`, curl, strom, p50): GET /api/mapdata after a tag PUT 495 -> **160** ms
(p95 588 -> 205); first request 778 -> 732; warm 200 / 304 / annotations / PUT unchanged (7-11 ms).
