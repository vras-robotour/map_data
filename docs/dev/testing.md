# Testing

## Running the suite

```bash
pytest tests/ -v
```

No ROS2 context, no network access, and no external services are required — all external dependencies are mocked.

Run a single file:

```bash
pytest tests/test_viewer_routes.py -v
```

Run tests matching a keyword:

```bash
pytest tests/ -k "overpass" -v
```

## Test files

| File | Scope | Key dependencies mocked |
|---|---|---|
| `test_core.py` | `Way` classification, serialization, `combine_ways` | — |
| `test_astar.py` | Grid A* planner | — |
| `test_rrt.py` | RRT* planner — path found, collision avoidance, edge cases | — |
| `test_graph_planner.py` | Graph-based planner | — |
| `test_overpass.py` | `OverpassClient` — retry logic, rate limiting, status polling | `requests.Session`, `time.sleep` |
| `test_parsing.py` | `parse_osm_ways`, `separate_ways`, `parse_osm_nodes` — OSM element classification and buffering | `overpy.Overpass` (fixture JSON) |
| `test_fill_grid.py` | `ReplanPath.fill_grid` — footway cost assignment, barrier cell marking | — |
| `test_viewer_helpers.py` | `helpers.py` — GeoJSON conversion, way splitting, node overrides, annotation I/O, change log migration | — |
| `test_viewer_routes.py` | Flask REST API — annotation CRUD, file listing, mapdata fetch, way operations, path-traversal security | Filesystem (temp dir) |
| `test_integration.py` | Full pipeline — GPX parse, save/reload roundtrip, OSM cache, mocked Overpass query, `parse_intersections` | `OverpassClient` |
| `test_errors.py` | Error paths — malformed GPX, corrupt files, Overpass timeouts, planning failures | `requests.Session`, `time.sleep` |

## Test design principles

**No network.** Every test that would otherwise reach the Overpass API patches `requests.Session.post` / `requests.Session.get` via `unittest.mock.patch`. `time.sleep` is patched alongside so retries complete instantly.

**No ROS2.** The Flask app is created via `create_app(data_dir=...)` which bypasses the ROS2 node initialisation (guarded by `ROS_AVAILABLE`).

**Real filesystem, isolated.** Route tests use pytest's `tmp_path` fixture, giving each test function its own directory. The `load_mapdata_cached` cache keys on `(path, mtime)` so cross-test contamination cannot occur.

**Minimal fixtures.** Shared helper functions (e.g. `_make_mapdata`) are plain functions defined in the test module, not conftest fixtures, to keep test files self-contained.

## Module-by-module notes

### `test_rrt.py`

`test_rrt_star_with_obstacle` checks obstacle avoidance at two levels:

1. **Node check** — each waypoint in the returned path must have grid cost `< 0.95`.
2. **Segment check** — 20 linearly-interpolated samples along each segment between consecutive waypoints must also have grid cost `< 0.95`.

The segment check is critical: without it a two-point path `[start, goal]` passes the node check even when the straight line between them cuts directly through the obstacle region.

The test sets `traversability_threshold=0.5` so that cells with the obstacle value `1.0` are treated as hard collisions by the planner.

The segment check excludes the single outer ring of boundary cells (rows/columns 40 and 59) to account for a known Bresenham corner artefact where a diagonal segment can graze a single boundary cell that the rasteriser does not visit.

### `test_parsing.py`

Tests the three OSM parsing functions directly using `overpy.Overpass().parse_json(fixture_json)` as input — no real network call:

- `parse_osm_ways` — footway produces a `LineString`, road is classified via `is_road()`, closed way produces a `Polygon`.
- `separate_ways` — footway goes to the footways list (buffered to Polygon); road goes to roads; barrier way (with `barrier: *` tag dict) goes to barriers. `BUFFER_WIDTHS` for road > footway.
- `parse_osm_nodes` — node with `barrier: block` produces a buffered-Point `Way`; nodes in `way_node_ids` are skipped; non-obstacle nodes are excluded.

### `test_fill_grid.py`

Tests `ReplanPath.fill_grid` with a hand-built `MapData` containing a 16 m footway in a 20 × 20 m grid (UTM zone 33U). Key assertions:

- Cells exactly on the footway centreline have cost `< DEFAULT_OFF_PATH_COST` (0.9).
- Cells at the far corner (> `max_path_dist` from any path) have cost ≥ 0.85.
- Cells inside a 2 × 2 m barrier polygon passed as an obstacle are `np.inf` in the 2-D grid cache.
- Cells clearly outside the barrier are finite.

### `test_overpass.py`

`OverpassClient` is instantiated directly and its `session` attribute is patched in place, which avoids replacing the class globally and keeps each test independent. The nine tests cover:

- Successful query returning an `overpy.Result`
- 429/406 response rotating the active endpoint
- 500 response triggering a retry on the same endpoint
- All retries exhausted → `None`
- `requests.Timeout` → `None`
- `_wait_for_slot` short-circuiting for non-`overpass-api.de` endpoints
- `_wait_for_slot` returning immediately when slots are available
- `_wait_for_slot` sleeping the correct number of seconds when no slots are available

### `test_viewer_helpers.py`

Pure functions in `helpers.py` are tested in isolation using hand-constructed `Way` objects and fixed UTM coordinates in zone 33U (Prague area, easting ≈ 458 000, northing ≈ 5 550 500). Notable cases:

- **GeoJSON roundtrip** — `geom_to_geojson` followed by `geojson_geom_to_utm` must recover the original geometry within 2 m (`equals_exact(tolerance=2.0)`).
- **Way splitting** — a five-node `LineString` way split at its middle node must yield two segments with virtual IDs `"{id}:0"` and `"{id}:1"` and the correct node subsets.
- **Node deletion** — removing a node from a four-node way must return a three-node way; removing a node from a two-node way must return `None`.
- **Change log migration** — `migrate_change_log` populates the log for untracked deletions, is idempotent once the migration version matches, drops legacy entries without a `ts` key on re-migration, and preserves entries that carry a timestamp.

### `test_viewer_routes.py`

A `_make_mapdata` helper creates a minimal `.mapdata` file (one footway `Way`, two entries in `nodes_cache`) and writes it to a `tmp_path`. Two fixtures build on this:

- `app_client` — empty data directory, used for 400/404 edge cases.
- `app_client_with_file` — data directory pre-populated with `test.mapdata`.

The annotation lifecycle (create → update → delete) is tested end-to-end: each step checks both the HTTP status code and the persisted state. Additional coverage includes:

- **Path-traversal security** — `GET /api/mapdata?file=../../../etc/passwd` must return 400.
- **Way tag overrides** — `PUT /api/ways/{id}/tags` stores tag overrides; `DELETE` removes them.
- **Hide / show / restore** — the hide, show, and restore endpoints return 204 and update the annotations file correctly.
- **Node operations** — node deletion and position-override move operations.
- **Way splitting** — a 3-node footway can be split at its middle node, returning two segments; undoing the split restores one segment.

### `test_integration.py`

`test_run_parse_with_mocked_overpass` patches `map_data.map_data.OverpassClient` at the class level, replacing `query_raw` with a return value of a minimal Overpass JSON string containing one footway way. `client.api` is set to a real `overpy.Overpass()` instance so that `parse_json` works correctly. After `run_queries(use_cache=False)` and `run_parse()`, the test asserts that at least one footway was classified.

Additional coverage:

- **OSM cache roundtrip** — `_save_osm_cache` then `_load_osm_cache` returns the same raw JSON strings.
- **OSM cache bbox mismatch** — shifting `min_lat` by 1° makes `_load_osm_cache` return `None`.
- **`parse_intersections`** — two footways sharing a middle node produce exactly one crossroad `Way` at that node; no shared nodes → no crossroads; three footways sharing one endpoint → crossroad.

### `test_errors.py`

`test_rrt_star_goal_in_isolated_obstacle` creates a grid that is entirely blocked except for a small free patch around the start. Since the goal lies within the blocked region and no path can reach it, `find_path` must return `None`.

## Adding new tests

1. Place the test file in `tests/`.
2. Import only from `map_data.*` — do not import from neighbouring test modules.
3. Mock any external I/O at the lowest sensible level (`requests.Session`, not the whole `requests` module).
4. Use `tmp_path` for any test that reads or writes files.
5. Run `pytest tests/ -v` before opening a pull request.
