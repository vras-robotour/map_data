# Changelog

## [Unreleased]

### Fixed

- Crossroad detection no longer invents junctions where one corridor is mapped
  twice. A node was a crossroad when several ways used it, so a cycleway or an
  annotated path drawn along an existing footway — which reuses its node ids —
  made a junction of every node of the shared run. A node is now a crossroad
  when more than two *distinct* neighbouring nodes leave it: two ways passing
  between the same neighbours are one path, three directions are a fork. On the
  full Stromovka map this drops 64 junctions, every one of them mid-corridor
- Roads take part in crossroad detection. Only footways were considered, so a
  footway meeting a service road was not a junction at all and the same physical
  junction was found or missed depending on how the through way happened to be
  tagged — 93 of them on the full Stromovka map, 42 on one drawn road alone.
  This matters now that the graph planner routes over roads
  (``highway_types: [footway, road]``)
- The geometric detector used for annotated paths matches centre line against
  centre line. It tested the drawn line against the other way's *buffered*
  geometry, which yields the stretch of line inside a 3 m (footway) or 7 m
  (road) corridor rather than a crossing: a path merely running alongside a way
  reported a junction it never reaches, every real crossing was placed half a
  corridor — up to 3.5 m, most of the follower's 5 m enter radius — from where
  it happens, and the T-junction tolerance reached half a corridor too far.
  ``MapData.centre_line`` rebuilds the unbuffered line from a way's node ids
- ``MapData.load`` recomputes the node-based crossroads instead of trusting the
  ones in the file, so a ``.mapdata`` written by an older version is corrected
  on load; crossroads that cannot be recomputed from node ids (an annotated
  path's) are kept as saved

## [1.4.0] — 2026-09-12

### Added

- Traversability rules: `config/traversability.yaml` decides from OSM tags which
  ways the robot may drive on (first matching rule wins, each with a `reason` the
  operator sees in the node log) and what they cost. `MapData` applies the rules
  when a map is loaded for planning — `exclude_ways` is now the shortcut for
  rules that only deny `highway` values — and `GraphPlanner` weighs its edges by
  the same highway/surface cost tables as the grid planner through the shared
  `pathsolver.way_cost` helper, while the reported route length stays geometric.
  The file is selectable per run (`traversability_file` parameter,
  `traversability:=`, `--traversability`) and is part of the map and planner
  cache keys, mtime included
- Offline route planning: `map_data.pathsolver.route.plan_route` (the Planner
  screen's graph/grid planning as a library call, with `densify`, failure reasons
  and snap distances), `map_data.annotations` (annotation-store merge without the
  web app, `load_mapdata_with_annotations`), the `map_data_plan` CLI, the
  `route_planner` ROS 2 action server with the new `map_data_interfaces`
  package (`PlanRoute.action`), `route_planner.launch.py`, `GraphPlanner.snap_distance`
  and `gpx.create_gpx_track`. The viewer's `/api/create_replan` now delegates to
  `plan_route` and reports a `reason` on failure
- `config/route_planner.yaml`: every `route_planner` parameter in one documented
  file, loaded by `route_planner.launch.py` (`params_file:=` takes an absolute
  path or a name in `config/`). The launch arguments now default to empty and are
  applied on top of the file, so an argument left unset keeps the file's value;
  `highway_types:=footway,road` lets the graph planner use roads as well as
  footways. `map_data.utils.launch` holds the argument helpers
- `keep_goal` (default on) ends a route at the requested goal itself rather than
  at its projection onto the network, which can be tens of metres short; the
  final off-network leg is densified like the rest. The goal has its own snap
  limit (`goal_max_snap_distance`, 30 m, failing with `snap_too_far`) separate
  from the start's 100 m, which is the robot's own fix
- `exclude_highway` (default `steps`): stairs stay in the saved map for the
  viewer, but `MapData.exclude_ways` drops them at load time and `GraphPlanner`
  filters them for callers passing a raw map, so no route is ever planned over
  them
- Explicit annotations switch — `--annotations auto|none|FILE` and the
  `annotations` parameter — because a store may delete a large part of the map
  (Stromovka's deletes 838 ways) and planning has to be able to opt out of it
- Goal QR codes for Robotour: `map_data.utils.qr` encodes a goal as a geo URI,
  and the viewer serves one per waypoint (`/api/qr`, `/api/qr.svg`) with a
  full-screen modal and PNG download
- `osm_cloud` geodetic placement of map data via the ECEF → local TF: new
  `transform_mode` (`tf` | `auto` | `geodetic`) and `earth_frame` parameters. In
  geodetic mode UTM points are converted to lat/lon → ECEF and placed in
  `local_frame` through the `earth_frame` → `local_frame` transform (`FP_ECEF` →
  `FP_ENU0` on Helhest), which is exact where a UTM translation is off by grid
  convergence (5–8 m per km). Grid bounds use all four UTM corners and the frames
  are launch arguments
- Latched `osm_cloud` publishing (the grid and intersections are published once
  at start-up and after parameter rebuilds via transient-local publishers;
  `republish_period` restores periodic re-publishing) and
  `MapData.geometric_intersections()`, which finds crossings and endpoint
  T-junctions geometrically so that annotated viewer paths — which share no OSM
  node ids and were therefore invisible to node-based detection — reach export
  and `osm_cloud` as crossroads
- Viewer tracker for the Helhest field stack: poses in any TF frame are converted
  to lat/lon through `earth_frame` (ECEF) instead of requiring a `utm` frame; new
  inputs `BatteryState`, `Temperature`, e-stop `Bool`, `DiagnosticArray`, `Joy`,
  crl_commander state string, `road_follower` state, goal (`PoseStamped`),
  waypoint sequence (`Path`/`PoseArray`) and visual road path; heading from
  `Imu`, `Vector3Stamped` yaw or `Odometry` (`heading_type`). The sidebar gained
  E-Stop / Follower / Diagnostics rows and battery percentage; the map draws the
  sequence (blue), road path (cyan) and goal (orange). `config/helhest.yaml`
  now targets this stack (verified against the Stromovka bags)
- Tracker map: `osm_cloud` intersections, the `road_follower` active
  intersection with its enter/exit radii, the follower's waypoint window,
  a robot trail and a fix-age / stale indicator; the sidebar has a map legend
- `geodesy.ecef_to_latlon_array`
- Documentation: offline route planning, the `route_planner` config file, the
  traversability rules and the cost tables the graph planner now shares, and the
  goal QR codes

### Changed

- `route_planner` preloads the map and caches its graph planners, converts ECEF
  in a vectorised form and subscribes to static TF only, so a plan request no
  longer pays for the map on every call
- `setup.py` installs `.mapdata` files and annotation stores, so a planning node
  finds its map in an installed workspace
- `data/` is no longer tracked (the directory is kept): local working datasets do
  not belong in the package
- `config/helhest.yaml` uses the real crl_commander topic names, and the
  follower threshold now lives in `road_and_gps_follower.yaml`
- Nodes log through `warning()`; on Kilted `warn()` shares one caller id, which
  collapses distinct warnings into one throttled message
- Repo hygiene: the stale `todo_fixes_plan.md` is gone, the unused
  `python3-joblib` dependency is dropped from `package.xml`, `.mypy_cache/` is
  ignored, and a test now asserts that the version stays in step across
  `pyproject.toml`, `package.xml`, `map_data_interfaces/package.xml` and
  `CITATION.cff` and that the released version has a dated changelog section

### Fixed

- `GraphPlanner.plan()` re-inserted the raw clicked coordinate at both ends of
  every segment, so each via point became a degenerate out-and-back off the
  network. It now returns on-network points only and collapses coincident
  vertices and sub-metre spurs — a 900 m loop goes from 49 to 37 vertices —
  while `keep_start` (`start_from_robot`) still keeps waypoint 0 verbatim,
  because planning from the robot's pose has to begin where the robot is
- `annotations`: `apply_added_nodes` inserts each added node after its anchor
  instead of at a running offset, so added nodes land in the right place when a
  map is loaded for planning
- The viewer re-maps segment annotations when a way's split points change
- The viewer's sidebar mode panels scroll when their body is too long: as
  `flex:1` children `min-height:auto` let them grow past the sidebar and `#main`
  clipped the overflow, leaving the bottom of the planner unreachable
- `osm_cloud` no longer clips intersections to the auto grid bounds: those bounds
  come from the map's query bbox, but ways crossing it are downloaded whole, so
  crossroads on the route were silently dropped
- `osm_cloud` loads the same map as the planner (annotations, `exclude_highway`);
  a junction on a retagged or deleted way published a ring nothing routes over
- Two field fixes ported from the robot: `TransformListener(..., spin_thread=True)`
  so TF lookups do not depend on the node's own executor, and the `utm_to_local`
  parameter reshaped to 4×4 — a flat 16-element list otherwise breaks every
  transform
- Goal codes are served as vectors (`/api/qr.svg`) and the modal caption renders
  as HTML
- CI: the OpenCV runtime dependency is declared and the new mypy errors are fixed

## [1.3.0] — 2026-08-24

### Security

- Fixed two stored-XSS vectors in the viewer: virtual way IDs are now strictly
  validated server-side (`<int>` or `<int>:<int>` only) before being stored in
  the annotation store or change log, the changes/hidden panels build rows via
  DOM APIs instead of inline `onclick` strings, and `escHtml` escapes quotes so
  attribute-context interpolation (e.g. the tag editor's `value="..."`) cannot
  break out
- CSRF protection for cookie-based auth: with `MAP_DATA_ACCESS_TOKEN` set, the
  cookie alone now authenticates only safe methods — state-changing requests
  must carry the `X-Requested-With` header (attached automatically by the
  viewer's same-origin fetches) or the token header/query parameter; threat
  model documented in the viewer docs
- Request bodies are capped at 100 MB (`MAX_CONTENT_LENGTH`), so an oversized
  upload can no longer fill the data directory
- `cost_grid` and `create_replan` validate every client parameter up front
  (bbox sanity, cell budget of at most `MAX_GRID_CELLS`, numeric ranges for
  `cell_size`/`inflate_obstacles`/`grid_cost_weight`, cost-dict shapes, path
  points within UTM-supported ranges), closing a compute-DoS lever and turning
  deep planner 500s into clean 400s

### Fixed

- Annotation edits no longer race: a shared `annotation_store` context manager
  holds the per-file lock across the whole load → mutate → save cycle in all
  16 mutating viewer routes, fixing lost updates between concurrent editors
  and duplicate synthetic node IDs from concurrent `add_way_node` calls
- `GraphPlanner` applies annotation splits per way in descending segment order
  (junctions no longer land between the wrong nodes when one way is split in
  several places) and splices into per-planner node-list copies instead of
  mutating the shared `MapData` (a second planner on the same map no longer
  raises `KeyError`)
- `grid_astar` diagonal moves require both edge-adjacent cells to be free, so
  paths can no longer cut through the corner where two blocked cells touch;
  grid cells are sampled at their centers, removing the half-cell
  obstacle-membership bias
- Douglas–Peucker simplification in `replan`/`grid_astar` collision-checks
  every shortcut it introduces and keeps the original vertices where a chord
  would cross an obstacle
- RRT* rewiring propagates cost changes to descendants and keeps the informed
  ellipse synced with the goal's true cost, instead of sampling from a stale,
  oversized ellipse
- `GraphPlanner.plan()` returns `None` for fewer than two waypoints (per its
  documented contract) and rejects waypoints farther than a configurable
  `max_snap_distance` (default 100 m) from the network instead of silently
  snapping to an arbitrarily distant edge
- GPX/YAML parsing forces every waypoint into the first point's UTM zone, so a
  route crossing a zone boundary no longer produces a ~400 km easting
  discontinuity
- Overpass HTTP-200 error bodies (a `remark` runtime error or an HTML page
  from a busy mirror) now rotate to the next mirror and retry like server
  errors instead of crashing the fetch; unparseable bodies — including a
  corrupt cached response — degrade gracefully
- OSM relation inner rings are subtracted from the outer polygons, so
  courtyards and islands are no longer covered by the merged barrier; member
  ways consumed by a relation merge are dropped from the parse, eliminating
  duplicated barrier geometry
- Degenerate OSM ways (too few coordinates for their geometry type) are
  skipped with a warning instead of aborting the whole parse
- `.mapdata` and OSM-cache writes are atomic (temp file + `os.replace`), so a
  crash or full disk mid-save can no longer truncate a previously good file
- `map_data_info` reports the footway centerline length instead of the
  buffered-polygon perimeter (which was roughly double the walked distance)
- `MapData` accepts a 1-D `current_robot_position` and reconciles its
  elevation column against the waypoints; `_csv_to_dict` handles single-row
  CSVs; loaded instances restore the `points` attribute
- Viewer: `cancel_replan`, `create_wormhole`, and `cancel_wormhole` return a
  clean 400 on a missing/non-JSON body (previously an unhandled 500); the two
  cancel endpoints also 400 on a missing `transfer_id`
- Abandoned fetch tasks are swept from the registry after the 60 s retention
  window on every new fetch; cancelled replan transfer IDs are discarded on
  entry and exit, so a late cancel cannot poison the next replan with the
  same transfer ID

### Added

- Regression tests for every fix above plus coverage for the previously
  untested riskiest paths: Overpass body-level failures, `GraphPlanner`
  annotation splicing, the auth/CSRF gate, `cost_grid`/`create_replan`
  validation, `upload_mapdata`, native export, mocked wormhole transfers,
  `info.get_stats`, `combine_ways` backward-prepend and closed-ring merges,
  grid A* cost-weighting, and replan cancellation (~130 new tests; suite now
  311 tests in ~6 s)
- `tests/conftest.py` with a shared mocked-Overpass fixture, and pytest
  configuration centralized in `[tool.pytest.ini_options]` so a plain
  `pytest` runs with the same coverage flags locally and in CI

## [1.2.1] — 2026-07-15

### Security

- Viewer no longer leaks exception detail in HTTP 500 responses (the `upload_gpx`
  and `create_wormhole` handlers now return a generic message and log the real
  error server-side)
- SocketIO CORS now defaults to same-origin instead of a hardcoded `*`;
  `MAP_DATA_CORS_ORIGINS` opts into a specific origin list
- Optional access-token gate via `MAP_DATA_ACCESS_TOKEN` (off by default) for
  network deployments; documented the `--host 0.0.0.0` attack surface in the
  viewer docs

### Fixed

- Guarded three latent `None`-dereference paths surfaced by mypy: `buffer_line`
  raises a clear error for a way with no geometry; `osm_cloud` raises a clear error
  if the UTM-to-local transform is still unresolved at startup (rclpy shutdown race)
  instead of crashing later; `upload_gpx` rejects a file part with no filename
  (HTTP 400) instead of raising `TypeError`
- Fixed `TrackerNode.num_waypoints` never being updated in the ROS node
- Fixed polygon rasterization re-blocking holes when drawing obstacles
- Fixed inconsistent UTM zone forcing during map parsing
- Fixed `combine_ways` crash when OSM relation merging yields disconnected MultiLineString geometries
- Fixed `Way.to_pcd_points` ignoring the `density` parameter for linestring geometries
- Fixed `Way.to_pcd_points` cache returning stale results when called with different arguments
- Fixed `parse_gpx_file` / `parse_yaml_file` returning inconsistent shapes on empty input
- Fixed `parse_gpx_file` only reading waypoints — now falls back to tracks and routes
- Fixed RRT* `__main__` demo crashing due to tuple start/goal (requires `np.array`)
- Synced hardcoded fallback `sand` surface cost (`0.7` → `0.4`) with `planner_defaults.yaml`
- Wired up the dead `mapdata_path` launch argument in `osm_cloud.launch.py` (bare
  `mapdata_file`/`gpx_file` names now resolve against the data directory)

### Changed

- Overpass `run_queries` now filters server-side by the tag families the parser
  inspects instead of downloading every way/node in the bounding box; matching
  multipolygon relations and their member ways are recursed in so those obstacles
  are still classified
- Removed the no-op `joblib` threading parallelism from `ReplanPath.replan` (the
  A*/RRT* segment work is GIL-bound); segments run sequentially and the grid cache
  is warmed once up front, removing a redundant per-segment cache-build race
- Removed the now-unused `joblib` runtime dependency from `pyproject.toml`
- Deduplicated the triplicated way-resolution pipeline in `viewer/routes.py` into a
  shared `_resolve_way` helper, and documented the `MapData` shallow-vs-deep copy
  invariant
- Removed the unused multi-robot `robot_id` scaffolding from the tracker (parameter,
  telemetry field, and `helhest.yaml` entry) — the viewer targets a single robot

### Added

- Static type checking with `mypy` (`[tool.mypy]` in `pyproject.toml`), enforced
  in CI (`typecheck` job) and as a pre-commit hook, to catch bugs like
  keyword-only-argument mismatches and inconsistent Flask route return types
- Interactive viewer: `GET /api/export/geojson` endpoint and toolbar button to download the merged (annotation-resolved) map as a `.geojson` file, for use in QGIS/geojson.io
- Unit tests for `osm_cloud` pure helper functions (`create_grid`, `points_near_ref`, `transform_points`, `split_ways_to_points`)
- `pytest-cov` coverage reporting in CI
- `.pre-commit-config.yaml` for local ruff lint/format checks
- Documented the `map_data_info` CLI (including `--validate`), the viewer's `--data-dir`/`--host`/`--port` flags, and the `THUNDERFOREST_API_KEY`/`SEZNAM_API_KEY` environment variables in the README and viewer docs
- Unit test for the `osm_cloud` ROS node initialization (parameter wiring, publishers, timer)
- Tests for `parse_osm_rels` multipolygon member-way tagging, `combine_ways` disconnected members, `Way.to_pcd_points` cache invalidation, empty-GPX return shape, the off-path zero-cost regression, and launch-argument consumption
- NumPy-style docstrings for `pathsolver/grid_constructor.py`, `viewer/helpers.py`, and the `viewer/routes.py` handlers

## [1.2.0] — 2026-07-14

### Fixed

- Fixed critical positional argument crash in `osm_cloud.py`'s `run_all`
- Fixed custom planner costs being silently ignored by the viewer and path planner
- Fixed application crash when refreshing OSM data over a loaded `.mapdata` file
- Fixed GPX launch mode being unreachable in `osm_cloud.launch.py`
- Fixed phantom zero-cost cells appearing off-path due to spacing mismatches
- Fixed non-atomic annotation saves and missing locks causing corruption
- Fixed `__version__` single-sourcing (now reads dynamically via `importlib.metadata`)
- Fixed double-normalization in quadratic neighbor cost logic
- Fixed viewer routes re-raising HTTP exceptions properly and applying zero-coordinate checks


## [1.1.0] — 2026-07-07

### Added

- Interactive viewer: base-layer switcher with a satellite imagery option (Esri World Imagery); the selection persists across sessions
- Interactive viewer: copy-to-clipboard buttons for lat/lon in the node inspector
- Interactive viewer: per-annotation "revert geometry" action (↺ in the Annotations panel) to undo geometry drags back to the last loaded state
- Interactive viewer: the planner distance readout now labels whether it shows straight-line waypoint distance or actual planned path length
- `map_data_viewer --telemetry-rate` flag to configure the Tracker telemetry broadcast rate (default 2 Hz)
- `map_data_info --validate` checks a `.mapdata` file for structural issues (missing metadata or geometry, duplicate way IDs, nodes missing from the cache, disconnected footway networks) and exits non-zero when any are found

### Changed

- The package now targets ROS2 Jazzy or later and Python 3.12+
- `create_mapdata` no longer requires ROS2 — it falls back to the repo data directory when `ament_index_python` is unavailable
- Packaging metadata (version, dependencies) is single-sourced in `pyproject.toml`; `requirements.txt` was replaced by the `[dev]` extra
- CI runs ruff lint/format checks and tests on Python 3.12 and 3.13

### Fixed

- `astar_search` used syntax unavailable on the previously documented minimum Python version
- Stale version numbers in `setup.py` and the Overpass User-Agent

## [1.0.0] — 2026-05-29

### Added

- Interactive viewer: way splitting, node deletion, and node position drag editing
- Interactive viewer: nodes can be inserted into OSM ways by dragging the blue midpoint handles between existing nodes in Edit mode
- Interactive viewer: barrier ways can now be split (✂️ button on nodes), matching the existing road and footway split behaviour; closed barriers are excluded
- Interactive viewer: vertex-level editing for manual annotations in Edit mode
- Interactive viewer: tag editing with save and undo support
- Interactive viewer: audit change log stored in `.annotations.json`
- Interactive viewer: feature search by ID or name in the sidebar; search now also matches all OSM tag values and, for annotations, the annotation type and all extra property keys and values (e.g. searching "wall" or "barrier" finds matching annotations)
- Interactive viewer: category and subtype visibility toggles
- Interactive viewer: Leaflet.Snap integration for precise annotation alignment
- Interactive viewer: GPX file import for waypoint overlays
- Interactive viewer: GPX upload modal with map name input (previously a stub)
- Interactive viewer: drag-and-drop `.mapdata` file upload; `/api/upload_mapdata` validates via `MapData.load` and saves to the data directory with collision-safe naming
- Interactive viewer: Tracker mode for live robot position via ROS2
- Interactive viewer: planner mode can download and parse OSM data on demand
- Interactive viewer: planning parameters (`grid_margin`, `obstacle_radius`, `buffer_widths`, `grid_cost_weight`) configurable in all three map creation dialogs (fetch area, GPX upload, planner fetch) via collapsible advanced-options panels; server defaults populate fields on load
- Path planning module (`pathsolver`) with A* graph search and cost-grid support
- Path planning: annotated paths take priority over obstacle cells
- Path planning: paths-only planning mode (constrained to mapped ways)
- Path planning: cancellable planning requests
- New planner config parameters in `planner_defaults.yaml`: `grid_cost_weight`, `obstacle_radius`, and per-type `buffer_widths` (`road`, `footway`, `barrier`) — previously hardcoded in source
- OSM response caching: Overpass query results are persisted to a `.osm_cache.json` sidecar file and reused on subsequent loads when the bounding box matches, avoiding redundant network requests
- YAML waypoint format as an alternative to GPX
- `map_data_info` CLI tool to print statistics about a `.mapdata` file
- `osm_cloud` launch file with configurable grid topic and static transform publishing
- `osm_cloud`: dynamic reconfigure support for runtime tuning of `max_path_dist`, `neighbor_cost`, and `grid_res`
- Documentation site (MkDocs Material)
- Dedicated `Testing` documentation page (`docs/dev/testing.md`) with per-module design notes and guidance for adding new tests
- `pyproject.toml` with ruff configuration for code style enforcement, compatible with ROS2 builds
- Type hints across core and utility modules
- pytest suite covering core logic and path planning edge cases
- Expanded test suite: `test_overpass.py` (retry logic, rate limiting, status polling), `test_errors.py` (malformed GPX, corrupt files, Overpass timeouts, planning failures), `test_parsing.py` (OSM element classification and buffering), `test_fill_grid.py` (footway cost assignment, barrier cell marking), `test_viewer_helpers.py` (GeoJSON roundtrip, way splitting, change log migration), `test_viewer_routes.py` (annotation CRUD, path-traversal security, way operations), and extended `test_integration.py` (OSM cache roundtrip, bbox mismatch, `parse_intersections`)

### Changed

- Interactive viewer: GPX download now exports a `<trk><trkseg><trkpt>` track by default; a Track / Waypoints toggle in the planner panel switches back to the legacy `<wpt>` format
- Interactive viewer: annotation vertex handles in Edit mode now only appear when the annotation is clicked, not for all annotations at once
- Interactive viewer: annotation vertex handles now use the same orange circle + midpoint style as OSM node editing, replacing Leaflet.draw square handles
- Interactive viewer: transparent overlay markers provide larger click/drag hit targets for OSM nodes, annotation vertices, and planner waypoints without changing their visual appearance
- Interactive viewer: node drag in Edit mode is more responsive
- `visualize_mapdata` CLI tool removed; the browser-based viewer supersedes it
- `grid_cost_weight`, `obstacle_radius`, and `buffer_widths` are now per-run arguments to `grid_astar`, `RRTStar`, and `ReplanPath`; `osm_margin + reserve_margin` replaced with a single `grid_margin` constant (150 m default) accepted as a per-call override in `MapData.__init__`, `parse_osm_nodes`, and `separate_ways`
- `MapData.get_points()` now vectorizes UTM conversion by passing full lat/lon arrays to `utm.from_latlon` in one call instead of looping per node; all nodes are projected into the map's zone for consistency
- `RRTStar` now supports Informed RRT* sampling (`informed=True`, default): once an initial path is found, random samples are drawn from an ellipsoidal subset defined by the current best cost, accelerating convergence toward the optimum
- `RRTStar` now supports adaptive neighbor radius (`adaptive_radius=True`, default): the rewiring radius shrinks as `γ·√(log n / n)` so the number of rewiring checks stays bounded while preserving asymptotic optimality
- `smooth_path` now returns the best collision-free intermediate state instead of reverting to the original path entirely when a smoothed segment collides with an obstacle
- `ReplanPath` refactored into focused sub-modules: grid construction moved to `PathGrid` (`pathsolver/grid_constructor.py`), path smoothing to `smooth_path` (`pathsolver/smoothing.py`), and matplotlib debug visualization to `visualize_replan` (`pathsolver/visualizer.py`)
- Core architecture split into modular components: `OverpassClient`, `parsing`, `serialization`
- Config loading centralized to `map_data/utils/config.py`, eliminating duplicated YAML-loading logic from `map_data.py` and `replan.py`
- Logging configuration centralized in `setup_logging()` (`map_data/utils/config.py`); `info.py`, `create_mapdata.py`, and `viewer/app.py` now call it inside `main()` instead of invoking `logging.basicConfig` at module level with inconsistent formats
- `.mapdata` serialisation migrated from pickle to JSON + WKT; legacy pickle support was subsequently removed for security reasons
- Overpass queries parallelised for faster map data loading
- `Way` class refactored to a `@dataclass` with full type hints
- `os.path` replaced with `pathlib.Path` throughout the codebase
- Code formatting standardized via ruff across all source modules
- `pyproject.toml` and `setup.py` aligned for ROS2 build compatibility

### Fixed

- `visualizer.py`: swapped X/Y axis labels corrected — Easting is now on the X axis, Northing on Y
- Interactive viewer: node drag in Edit mode missed clicks intermittently because canvas node markers rendered below the SVG ghost layer; node markers now use the SVG renderer so events reach them directly
- Interactive viewer: adding a node to a buffered (Polygon) way drew a faint spike from the buffer outline back to the new node position because `apply_added_nodes` inserted the centerline coordinate into the exterior ring of the buffer polygon; geometry is now left unchanged for Polygon ways and only `way.nodes` is updated
- Interactive viewer: split segments of buffered footways and barriers were rendered thinner than the original after a node had been added to the way; the spike inserted into the buffer polygon corrupted its area/perimeter, causing `split_way` to calculate a smaller-than-correct buffer radius
- Interactive viewer: splitting a way at a synthetic (user-added) node failed silently because `apply_added_nodes` was not called before `split_way` in the segments and node-list endpoints; synthetic nodes are now inserted into the way geometry and an extended nodes cache is built so the split resolves correctly
- Interactive viewer: added nodes did not appear anywhere in the sidebar; the `add_node` event is now written to the change log on creation and removed on undo, and added-node entries are shown in the Annotations panel with an undo button
- Interactive viewer: deleting one segment of a split way caused the entire original way to disappear on the next file reload; segment deletions are now stored under their virtual ID (`"<id>:<index>"`) so only the deleted segment is suppressed while the remaining segments survive
- Interactive viewer: reverting any annotation edit (tag override, node deletion, node move) on a way that had also been split left the stale pre-revert geometry on the map alongside the newly restored geometry until page reload; `_reloadWay` now always uses the segments endpoint so split virtual layers (`id:n`) are atomically replaced
- Interactive viewer: Closed roads and footways (e.g. roundabouts) were rendered as filled polygons instead of an annular ring: `buffer_line` now converts a closed-loop `Polygon` to a `LineString` before buffering, unless the way carries `area=yes`
- Interactive viewer: tag change-log entries silently disappeared from the changes panel after any metadata refresh (`refreshMetadata` / `loadMapData`) because `tagMap` used numeric keys while the server's `change_log` stores tag ids as strings; both call-sites now normalise to string keys
- Interactive viewer: `_reselectFeature` and `focusFeatureById` now compare `String(_featureId) === String(wayId)`; previously strict equality failed to reselect real OSM ways (numeric `_featureId`) after a reload that converted the ID to a string, leaving the feature visually deselected
- Interactive viewer: duplicate `click` listeners on `way-edit-save` and `way-edit-add-prop-btn` caused double row insertion and double API calls on each action
- Interactive viewer: planner `mousemove`/`mouseup` handlers accumulated on the map with each `redraw()` call; handlers are now tracked in `_mapDragListeners` and removed via `map.off()` before each redraw
- Interactive viewer: `fetch_area` and OSM data parsing now run in a background thread; the route returns a task ID immediately and the client polls `/api/fetch_area/<task_id>` for completion, preventing UI hangs and WebSocket timeouts during long Overpass fetches
- Annotation deletion via the Del key in the viewer
- Path traversal vulnerability in viewer API: user-supplied `file` parameter is now validated against the resolved data directory before any file access
- `/api/fetch_area` now rejects requests where `min_lat >= max_lat` or `min_lon >= max_lon`
- `parse_yaml_file` now wraps all parse errors in a try/except and returns `[]` with a log message, matching the error contract of `parse_gpx_file`
- Path planning with split ways
- Thread-safe cancellation in the replanning module
- `create_mapdata` node: existing file load was missing the `.mapdata` suffix, causing `MapData.load` to receive an incorrect path
- `create_mapdata` node: `--download` flag was passed as a positional argument to `process_map_data`, now correctly passed as a keyword argument
- Multi-zone UTM boundary warning when loaded area spans two UTM zones

- `route_planner` preloads its map and caches one `GraphPlanner` per map / way set / snap distance (`plan_route(planner=...)`); a request plans in milliseconds instead of rebuilding the graph.

- Planner screen: Robotour goal QR codes per waypoint (`/api/qr`, `map_data.utils.qr`) shown full screen for the robot camera or downloaded as PNG.

- Goal QR codes are vector art (`/api/qr.svg`, `qr_svg`): the caption used to be rasterised into the PNG, so it went blocky when the viewer scaled the code up to fill the screen. The modal now shows an SVG with the caption as HTML text beside it, and offers an SVG download for printing.
