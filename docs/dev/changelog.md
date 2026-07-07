# Changelog

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
