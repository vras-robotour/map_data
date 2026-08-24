# Utilities

Supporting modules used by `MapData` and the planners. Most callers reach these indirectly,
but they are usable on their own — the Overpass client and the GPX parser in particular are
useful without constructing a `MapData` at all.

---

## Waypoint parsing

`map_data.utils.gpx` reads `.gpx` and `.yaml` waypoint files and converts between WGS84 and
UTM.

::: map_data.utils.gpx.parse_path
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.gpx.utm_path_to_latlon
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.gpx.create_gpx_content
    options:
      show_source: true
      heading_level: 3

---

## Overpass client

`map_data.utils.overpass` wraps the Overpass API with endpoint rotation, retry, and rate
limiting. Several Overpass failure modes arrive as HTTP 200 with an error body (a `remark`
runtime error, or an HTML page from an overloaded mirror), so the client validates response
bodies rather than trusting status codes alone.

::: map_data.utils.overpass.OverpassClient
    options:
      show_root_heading: false
      members: false

::: map_data.utils.overpass.OverpassClient.query_raw
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.overpass.OverpassClient.query
    options:
      show_source: true
      heading_level: 3

---

## OSM parsing

`map_data.utils.parsing` turns raw Overpass results into classified, buffered
[`Way`](way.md) objects. Buffer widths and the point-obstacle radius come from
[`planner_defaults.yaml`](../dev/planner_config.md).

::: map_data.utils.parsing.parse_osm_ways
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.parsing.parse_osm_nodes
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.parsing.separate_ways
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.parsing.buffer_line
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.parsing.ways_to_shapely
    options:
      show_source: true
      heading_level: 3

---

## Serialization

`map_data.utils.serialization` implements `.mapdata` file I/O. Writes are atomic — the file
is written to a temporary sibling and then `os.replace`d — so an interrupted save cannot
truncate a previously good file.

::: map_data.utils.serialization.save_mapdata
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.serialization.load_mapdata
    options:
      show_source: true
      heading_level: 3

See [Data Formats](../dev/data_formats.md) for the on-disk schema.

---

## Configuration

::: map_data.utils.config.load_config
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.config.setup_logging
    options:
      show_source: true
      heading_level: 3

---

## Geometry helpers

::: map_data.utils.points_to_graph_points.get_point_line
    options:
      show_source: true
      heading_level: 3
