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
digest, so each edit is a full miss, re-projecting all ~3000 geometries UTM -> lon/lat and
recomputing crossroads up to 3x), (2) the first request after a start (map parse ~300 ms plus
the same projection), (3) the grid planners (1.3-1.8 s on strom; `pathsolver`, not the viewer's).

### Attempts

| Idea | Baseline -> Result | Verdict | Why |
|---|---|---|---|
| Cache the GeoJSON of a geometry by value (`lru_cache` on `geom_to_geojson`; shapely hashes by coordinates, so nothing to invalidate) | GET /api/mapdata after an edit 456 -> 224 ms (strom), 58 -> 36 (robotour); export/geojson warm 351 -> 107 | **kept** | An edit touches a handful of ways; the other ~3000 geometries (and unchanged recomputed crossroads) hit. Responses byte-identical on 4 maps x 6 edit states (216 responses). Cold unchanged. |
| Recompute crossroads once after the way-edit and tag-override passes (`apply_way_and_tag_edits`) | GET after edit 206 -> 154 ms (strom, A/B x3), 29 -> 23 (robotour) | **kept** | `recompute_crossroads` (~50 ms strom) ran after each pass and its result was thrown away by the next. Also speeds up `apply_store` (planning, export). Identical output; test counts the calls. |
| Build `join_ways` edge LineStrings with one `shapely.linestrings` call | POST replan graph after an edit 212 -> 191 ms (strom, A/B x3) | **kept** | ~6000 `LineString()` constructions took ~80 ms of the merged-map build. Small (10 %) but consistent and 6 lines. Identical output. |
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

## Map loading and osm_cloud

Benchmark: `uv run python scripts/bench/mapload.py --repeat 5 --inputs <maps> --osm-cache <cache> --gpx <gpx>`. Values are median ± median absolute deviation. The real-map run used `strom`, `stromovka_full`, `kralovska_obora`, `KN`, and `robotour`, plus a cached Buchlovice Overpass response.

| Idea | Baseline -> Result (median ± spread) | Verdict | Why |
|---|---|---|---|
| Replace repeated `np.unique` sorting in `create_grid` with a boolean cell mask | `strom` grid 1.752 ± 0.014 s -> 1.137 ± 0.012 s; `stromovka_full` 1.568 ± 0.004 s -> 1.128 ± 0.035 s; peak RSS 333.0 -> 302.5 MB | Kept | cProfile put ~1.36 s of `get_cloud` in `create_grid`, mostly sorting expanded cells. The resulting arrays are identical on `strom` and `KN`. |
| Drop the initial `np.unique` on reference cells and expand all input cells directly | `strom` grid 1.137 ± 0.012 s -> 1.048 ± 0.002 s, but peak RSS 302.5 -> 773.1 MB | Reverted | Faster, but the memory increase breaks the peak-memory objective. |
| Batch all OSM way nodes into one `utm.from_latlon` call in `parse_osm_ways` | Focused cached-response parse 0.0673 ± 0.0015 s -> 0.0425 ± 0.0001 s; full benchmark 0.0894 ± 0.0019 s -> 0.0569 ± 0.0006 s | Kept | cProfile showed repeated per-way UTM calls dominating `run_parse`. Parsed way count and output lists are unchanged. |
| Additional map-load changes | `strom` load remained ~0.44–0.46 s across baseline and final runs, with run-to-run spread up to ~0.023 s | Reverted / not attempted | No repeated-work candidate beat the measured spread, so speculative changes were not kept. |

Final real-map medians: loads 0.080–0.446 s; grids 0.140–1.109 s; cached-response parse 0.0569 ± 0.0006 s; peak RSS 302.3 MB. `uv run python scripts/bench/mapload.py --quick` completed in about 1.2 s on tracked maps.

## Planner

Metric: per-goal latency of `plan_route` as `route_planner` and the viewer call it (map,
annotations and traversability loaded once; the graph planner cached, the grid planners
rebuilt per goal as both callers do). Benchmark: `scripts/bench/planner.py`, 5 footway-node
goal pairs 300 m to 2 km apart per map and algorithm (grid pairs also within
`MAX_GRID_CELLS` at the default 0.25 m cell), seed 7, RRT seeded, 3 repetitions; median ±
half the range of the per-repetition mean. `--compare` checked every route (length,
vertex count, SHA-1 of the vertices) against the baseline after each change: **no route
changed**. Machine: 16 cores, shared with other jobs (load ~5).

    uv run python scripts/bench/planner.py --quick        # tracked maps, ~25 s
    uv run python scripts/bench/planner.py --data-dir <dir with strom.mapdata ...> --json out.json [--compare base.json]

Where the time went (cProfile, baseline): the grid planners were 99% of it. `grid_astar`'s
pure-Python loop took 60–87% of an `astar` goal (~1 M expansions a leg, the search
covers 30–75% of the grid because the Euclidean heuristic is weak against the 5.5x
off-path step cost); `PathGrid.fill` most of the rest (a cKDTree query for every cell,
then `contains_xy` in `burn_obstacles`); RRT's own loop ~0.7 s a leg, and a failed RRT
leg falls back to A*. The graph planner was already 3–4 ms per goal warm and ≤ 82 ms
cold.

| Idea | Baseline -> Result (median ± spread) | Verdict | Why |
|---|---|---|---|
| `grid_astar`: column-major flat index as the heap tie-break (same order as `(ix, iy)`), precomputed heuristic and float32 diagonal-cost tables, float64 pre-check before the float32 rounding (rounding is monotonic) | astar stromovka 4682 ± 105 -> 3321 ± 77 ms; strom 3347 ± 76 -> 2437 ± 214; KN 800 ± 14 -> 557 ± 22 (all 6 maps 1.3–1.6x) | kept | Same expansions, less Python work each; routes bit-identical |
| `grid_astar`: plain lists instead of `array` for g scores / parents | captured strom grid 4.4 s -> 5.1 s (micro A/B) | reverted | Slower, and ~130 MB more per 4 M cells |
| `PathGrid.fill`: KD-query only cells within `max_path_dist` + 1 cell of a way point (`maximum_filter` mask; points off the grid clipped to its edge) | rrt stromovka 793 ± 17 -> 623 ± 3 ms; astar 3321 ± 77 -> 3107 ± 69; 11 of 12 grid cases faster beyond spread | kept | Most cells are far from every way; grid bit-identical on 30 real boxes |
| `burn_obstacles`: `shapely.prepare` each obstacle before `contains_xy` | fill 0.159 -> 0.156 s per goal (30 boxes) | reverted | No gain: `contains_xy` already prepares internally |
| RRT: vectorize the linear scans over nodes added since the last KD-tree rebuild (`_nearest_node`, `_get_near_nodes`), same first-minimum tie-break | rrt stromovka 623 ± 3 -> 540 ± 15 ms; strom 1620 ± 24 -> 1471 ± 6; all 6 maps −6…−14% | kept | Up to 49 numpy scalar ops per call became one |
| `grid_astar`: unroll the 8-neighbour loop | astar stromovka 3107 ± 69 -> 2754 ± 13 ms; buchlovice 1561 ± 22 -> 1334 ± 65; all 6 maps −6…−15% | kept | Tuple building/unpacking per neighbour was ~10% of the search |
| RRT `_segment_cost`: numpy over the Bresenham cells (`cumsum` keeps the float32 running total) | rrt stromovka 540 ± 15 -> 628 ± 11 ms; stromovka_full 598 ± 11 -> 696 ± 13 | reverted | ~10 cells per segment: array overhead beats the scalar loop |
| `GraphPlanner`: build all edge `LineString`s with one `shapely.linestrings` call | graph cold strom 82 ± 2 -> 54 ± 13 ms; kralovska_obora 65 ± 1 -> 41 ± 1; stromovka 23 ± 1 -> 16 ± 1 | kept | 20 k single-geometry constructions were ~40% of a cold build |

End to end (baseline -> all kept changes, same run conditions as above):

| Map | graph cold | graph warm | astar | rrt |
|---|---|---|---|---|
| stromovka (tracked) | 23.1 ± 0.5 -> 15.3 ± 0.4 ms | 3.1 -> 2.8 ms | 4683 ± 105 -> 2870 ± 58 ms | 805 ± 10 -> 529 ± 4 ms |
| buchlovice (tracked) | – (no connected footway pair ≥ 300 m) | – | 2751 ± 107 -> 1448 ± 40 ms | 2347 ± 11 -> 1392 ± 99 ms |
| strom | 81.7 ± 2.1 -> 56.6 ± 1.8 ms | 3.9 -> 4.0 ms | 3347 ± 76 -> 2216 ± 41 ms | 2226 ± 65 -> 1509 ± 35 ms |
| stromovka_full | 71.4 ± 1.3 -> 59.0 ± 1.2 ms | 3.7 -> 3.9 ms | 2693 ± 48 -> 1736 ± 26 ms | 819 ± 16 -> 601 ± 21 ms |
| kralovska_obora | 65.3 ± 1.3 -> 44.9 ± 1.6 ms | 3.6 -> 3.4 ms | 2960 ± 50 -> 1815 ± 41 ms | 3099 ± 51 -> 1833 ± 15 ms |
| KN | 24.0 ± 9.2 -> 16.2 ± 0.3 ms | 2.9 -> 2.8 ms | 800 ± 14 -> 456 ± 2 ms | 841 ± 40 -> 520 ± 5 ms |

Graph warm is unchanged by design (±0.2 ms is run-to-run drift between sessions).

Left alone: `grid_astar` is near its pure-Python floor (heap operations alone are
~25%); a bigger win needs a different search (fewer expansions changes which of several
equal-cost routes is returned) or native code, both out of scope here. `MapData.get_points`
(~30 ms per grid goal) belongs to map loading.

## Viewer frontend

Measured with `scripts/bench/frontend.py` (headless Chromium via Playwright, software rendering,
fresh browser context per repetition, 1 uncounted warm-up + N counted reps, **median ± half-range**,
loopback server, CDN/tiles served from a local cache).

```
uv run --with playwright python scripts/bench/frontend.py --quick        # demo map + behaviour flows, ~50 s (guard)
uv run --with playwright python scripts/bench/frontend.py --map strom.mapdata --reps 7 --out base.json
uv run --with playwright python scripts/bench/frontend.py --map strom.mapdata --compare base.json
uv run --with playwright python scripts/bench/frontend.py --map strom.mapdata --profile   # V8 sampler per step
uv run --with playwright python scripts/bench/frontend.py --map strom.mapdata --trace     # script/style/layout/paint ms per step
uv run --with playwright python scripts/bench/frontend.py --flows       # behaviour checks only
uv run --with playwright python scripts/bench/frontend.py --sweep        # canvas vs SVG on N synthetic polygons
```

Latency = input event `timeStamp` → first frame after the UI reached the expected state
(rAF + setTimeout), the usual INP approximation. Milliseconds unless noted. A change is kept only if
the median moved by more than both spreads **and** `--flows` (select, search, hide/show, layer
toggle, edit + node drag, midpoint insert, undo, split, annotation add/select/drag/vertex-drag/edit/
delete, zero console errors) still passes.

Inputs: `strom.mapdata` (11 MB file, 4.2 MB GeoJSON, 980 ways + 407 crossroads, 993-entry change log),
`demo/buchlovice.mapdata` (1.2 MB GeoJSON, 249 ways).

### Result (7 reps)

| Metric | strom baseline | strom optimized | demo baseline | demo optimized |
|---|---|---|---|---|
| Load: click Load → layers rendered | 325 ± 35 | **226 ± 20** | 175 ± 56 | **124 ± 15** |
| Load total blocking time | 128 ± 33 | **50 ± 6** | 21 ± 39 | **0 ± 1** |
| DOM nodes after load | 10 864 | **2 827** | 4 910 | **2 777** |
| JS heap after load (MB) | 24.7 | 26.4 | 9.4 | 9.9 |
| Click a way | 24 ± 12 | 23 ± 4 | 18 ± 3 | 21 ± 7 |
| Search keystroke (median / worst) | 9 / ~40 | 7 / ~25 | 5 / ~25 | 5 / ~25 |
| Layer checkbox off / on | 23 / 47 | 22 / 41 | 19 / 23 | 19 / 26 |
| Enter edit mode | 18 | 16 | 16 | 15 |
| Edit: select the longest way (node handles) | 85 ± 19 | **54 ± 15** | 77 ± 10 | **53 ± 11** |
| Node drag, per mousemove | 12.5 ± 15 | **3.9 ± 9** | 10.6 ± 10 | **3.8 ± 9** |
| Node drag release → settled | 190 ± 21 | **129 ± 15** | 123 ± 15 | 111 ± 18 |
| Undo (Ctrl+Z) | 171 ± 23 | **133 ± 20** | 111 ± 19 | 96 ± 24 |
| Empty-viewer LCP | 96 ± 14 | 108 ± 20 | 100 ± 16 | 88 ± 16 |

Everything the user triggers is under 200 ms (INP "good"); the remaining time is Leaflet canvas
re-projection/raster of ~1 400 vector layers, which is already native (`(program)` in profiles).

### Attempts

| Idea | Baseline → Result (median ± spread) | Verdict | Why |
|---|---|---|---|
| Changes panel: build rows 100 at a time, rest on scroll (`_fillPanelList`) | strom load 325 ± 35 → 281 ± 23; undo 171 ± 23 → 137 ± 23; drag release 190 → 147 ± 7; DOM 10 864 → 2 827 | **kept** | 993 change-log rows were rebuilt and re-laid-out on load and after every edit (trace: layout 97 ms, style 42 ms on load). Same rows, same count, scroll position preserved. |
| `content-visibility:auto` on `.change-item`/`.ann-item` (CSS only) | strom load 325 ± 35 → 299 ± 12 (within spread); undo 171 → 144 ± 8 | reverted | Layout only halved (boxes still exist), DOM/JS cost unchanged; the batching above removes the cost instead. |
| One shared SVG renderer for edit handles + way ghost (`handleRenderer()`), replacing one `L.svg()` per marker | strom select long way 73 ± 17 → 50 ± 10 (client part 62 → 31); drag mousemove 12 → 3.5; demo 77 → 51 | **kept** | Selecting a 109-node way created 434 `<svg>` renderers (each re-laid out on every pan/zoom). Add order inside the shared SVG keeps the ghost under the handles; node drag, way drag, midpoint insert and annotation vertex drag re-verified. |
| Fit map before adding layers (layers were projected at the old zoom and again by `fitBounds`); bounds from coordinates instead of a throw-away `L.geoJSON` | strom load 280 ± 2 → 240 ± 8; TBT 92 → 51; demo 151 → 127 | **kept** | Removes a full second `_project` pass over all layers. Heap +1.8 MB after load (layers keep projections at final zoom). |
| Bounds from coordinates alone, layers still added before fit | strom load 280 ± 2 → 287 ± 41 | reverted | Neutral; only the reordering above helps. |
| Fetch `/api/mapdata` and `/api/annotations` in parallel | strom load 240 ± 8 → 219 ± 5; demo 127 → 124 (within spread) | **kept** | Hides one round trip (more on a real LAN than loopback); no behaviour change, same error path. |
| `_reloadWay`: fetch segments and annotations in parallel | strom drag release 124 ± 14 → 136 ± 13; undo 116 ± 24 → 108 ± 13 | reverted | Within spread; loopback requests are too fast to matter. |
| SVG renderer instead of `preferCanvas` (experiment, 5 reps) | strom load 325 → 339 ± 15; search clear 45 → 54; toggle on 47 → 54; DOM +1 400, heap +1.5 MB; demo equal | not adopted | Equal at ~1 000 ways, worse from 5 000 up (sweep: 5 000 polys add 87 vs 115 ms, pan frame 33 vs 43 ms; 20 000: 312 vs 463 ms, 124 vs 152 ms). Canvas (already the default) stays. |
| Debounce the search box | not tried | – | Per keystroke 4–25 ms on strom (filter over 980 layers); the 40–50 ms spike is *clearing* the box, which re-adds ~1 000 layers one by one inside Leaflet and a debounce would not shorten it. |
| Canvas redraw reduction | not tried | – | `L.Canvas._redraw` is 1–5 ms of JS per call and called 1–2× per interaction; the rest is raster, not under our control. |

### Not changed here (backend)

- `/api/mapdata` for strom is **4.26 MB uncompressed** (1.33 MB gzipped, −69 %). Response compression
  (e.g. `flask-compress` or gzip in the route) would cut the transfer on a Wi-Fi LAN from ~0.3 s to
  ~0.1 s; on loopback it is ~40 ms and invisible. First request after server start costs 1.3 s of
  server-side parse (cached afterwards).
