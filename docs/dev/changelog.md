# Changelog

## Unreleased

### Added

- Interactive viewer: Tracker mode for live robot position via ROS2
- Interactive viewer: way splitting, node deletion, and node position drag editing
- Interactive viewer: audit change log stored in `.annotations.json`
- `map_data_info` CLI tool to print statistics about a `.mapdata` file
- YAML waypoint format as an alternative to GPX
- `osm_cloud` launch file with configurable grid topic and static transform publishing
- Documentation site (MkDocs Material)

### Changed

- `.mapdata` serialisation migrated from pickle to JSON; legacy pickle files are still loaded transparently

### Fixed

- Multi-zone UTM boundary warning when loaded area spans two UTM zones
