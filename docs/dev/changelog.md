# Changelog

## Unreleased

### Added

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

### Changed

- Core architecture split into modular components: `OverpassClient`, `parsing`, `serialization`
- `Way` class refactored to a `@dataclass` with full type hints
- `.mapdata` serialisation migrated from pickle to JSON + WKT; legacy pickle files are still loaded transparently
- Overpass queries parallelised for faster map data loading

### Fixed

- Multi-zone UTM boundary warning when loaded area spans two UTM zones
- Annotation deletion via the Del key in the viewer
- Path planning with split ways
- Thread-safe cancellation in the replanning module
