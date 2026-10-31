# Performance notes

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

### Result (original → now, 7 reps)

| Metric | strom before | strom now | demo before | demo now |
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
| One shared SVG renderer for edit handles + way ghost (`handleRenderer()`), was one `L.svg()` per marker | strom select long way 73 ± 17 → 50 ± 10 (client part 62 → 31); drag mousemove 12 → 3.5; demo 77 → 51 | **kept** | Selecting a 109-node way created 434 `<svg>` renderers (each re-laid out on every pan/zoom). Add order inside the shared SVG keeps the ghost under the handles; node drag, way drag, midpoint insert and annotation vertex drag re-verified. |
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
