# Changelog

## Unreleased

### Added

- Expanded test suite: `test_overpass.py` (retry logic, rate limiting, status polling), `test_errors.py` (malformed GPX, corrupt files, Overpass timeouts, planning failures), `test_parsing.py` (OSM element classification and buffering), `test_fill_grid.py` (footway cost assignment, barrier cell marking), `test_viewer_helpers.py` (GeoJSON roundtrip, way splitting, change log migration), `test_viewer_routes.py` (annotation CRUD, path-traversal security, way operations), and extended `test_integration.py` (OSM cache roundtrip, bbox mismatch, `parse_intersections`)
- Dedicated `Testing` documentation page (`docs/dev/testing.md`) with per-module design notes and guidance for adding new tests
- `pyproject.toml` with ruff configuration for code style enforcement, compatible with ROS2 builds
- Interactive viewer: Tracker mode for live robot position via ROS2
- Interactive viewer: way splitting, node deletion, and node position drag editing
- Interactive viewer: audit change log stored in `.annotations.json`
- Interactive viewer: tag editing with save and undo support
- Interactive viewer: feature search by ID or name in the sidebar
- Interactive viewer: category and subtype visibility toggles
- Interactive viewer: Leaflet.Snap integration for precise annotation alignment
- Interactive viewer: vertex-level editing for manual annotations in Edit mode
- Interactive viewer: GPX file import for waypoint overlays
- Interactive viewer: planner mode can download and parse OSM data on demand
- Interactive viewer: GPX upload modal with map name input (previously a stub)
- Path planning module (`pathsolver`) with A* graph search and cost-grid support
- Path planning: annotated paths take priority over obstacle cells
- Path planning: paths-only planning mode (constrained to mapped ways)
- Path planning: cancellable planning requests
- `map_data_info` CLI tool to print statistics about a `.mapdata` file
- YAML waypoint format as an alternative to GPX
- `osm_cloud` launch file with configurable grid topic and static transform publishing
- `osm_cloud`: dynamic reconfigure support for runtime tuning of `max_path_dist`, `neighbor_cost`, and `grid_res`
- Documentation site (MkDocs Material)
- Type hints across core and utility modules
- pytest suite covering core logic and path planning edge cases
- OSM response caching: Overpass query results are persisted to a `.osm_cache.json` sidecar file and reused on subsequent loads when the bounding box matches, avoiding redundant network requests
- New planner config parameters in `planner_defaults.yaml`: `grid_cost_weight`, `obstacle_radius`, and per-type `buffer_widths` (`road`, `footway`, `barrier`) — previously hardcoded in source

### Changed

- `os.path` replaced with `pathlib.Path` throughout the codebase
- Code formatting standardized via ruff across all source modules
- `pyproject.toml` and `setup.py` aligned for ROS2 build compatibility
- Core architecture split into modular components: `OverpassClient`, `parsing`, `serialization`
- `Way` class refactored to a `@dataclass` with full type hints
- `.mapdata` serialisation migrated from pickle to JSON + WKT; legacy pickle support was subsequently removed for security reasons
- Overpass queries parallelised for faster map data loading
- `ReplanPath` refactored into focused sub-modules: grid construction moved to `PathGrid` (`pathsolver/grid_constructor.py`), path smoothing to `smooth_path` (`pathsolver/smoothing.py`), and matplotlib debug visualization to `visualize_replan` (`pathsolver/visualizer.py`)
- Config loading centralized to `map_data/utils/config.py`, eliminating duplicated YAML-loading logic from `map_data.py` and `replan.py`
- `visualize_mapdata` CLI tool removed; the browser-based viewer supersedes it

### Fixed

- `create_mapdata` node: existing file load was missing the `.mapdata` suffix, causing `MapData.load` to receive an incorrect path
- `create_mapdata` node: `--download` flag was passed as a positional argument to `process_map_data`, now correctly passed as a keyword argument
- Multi-zone UTM boundary warning when loaded area spans two UTM zones
- Annotation deletion via the Del key in the viewer
- Path planning with split ways
- Thread-safe cancellation in the replanning module
- `parse_yaml_file` now wraps all parse errors in a try/except and returns `[]` with a log message, matching the error contract of `parse_gpx_file`
- Path traversal vulnerability in viewer API: user-supplied `file` parameter is now validated against the resolved data directory before any file access
- `/api/fetch_area` now rejects requests where `min_lat >= max_lat` or `min_lon >= max_lon`
