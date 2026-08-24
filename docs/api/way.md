# Way

::: map_data.utils.way.Way
    options:
      show_root_heading: false
      members: false

## Attributes

`Way` objects are created by `MapData.run_parse()` and are not typically constructed manually.

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `Any` | OSM way or relation ID (positive `int`). Manually annotated ways use a negative `int`, and split segments use a `"<id>:<index>"` string. Defaults to `-1`. |
| `is_area` | `bool` | `True` if `line` is a polygon (area), `False` for a linestring. Set to `True` by `buffer_line()` once a way has been buffered to its corridor width, and by `parse_osm_nodes()` for point barriers. |
| `nodes` | `list` | Ordered list of OSM node IDs defining the geometry |
| `tags` | `dict[str, str]` | OSM tags (e.g. `{"highway": "footway", "surface": "asphalt"}`) |
| `line` | `shapely.Geometry \| None` | Geometry in UTM metres. `LineString` for unbuffered ways; `Polygon` after buffering and for area-type barriers. `None` if unparsed. |
| `in_out` | `str` | Direction hint set by `GraphPlanner`: `"in"`, `"out"`, or `""` (the default) for bidirectional |
| `pcd_points` | `np.ndarray \| None` | Array of equidistant 3D points along the way, populated by `to_pcd_points()` |

!!! note "Most ways are polygons by the time you see them"
    `separate_ways()` buffers every road and footway to a corridor polygon using
    [`buffer_widths`](../dev/planner_config.md#buffer_widths), so a `Way` reached through
    `md.roads_list` or `md.footways_list` normally has `is_area=True` and a `Polygon`
    `line`, not the raw centerline.

!!! note
    The `line` geometry uses the same UTM coordinate system (metres) as the parent `MapData`
    object. To convert a point to WGS84, use `utm.to_latlon(x, y, md.zone_number, md.zone_letter)`.

## Methods

::: map_data.utils.way.Way.is_road
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.way.Way.is_footway
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.way.Way.is_barrier
    options:
      show_source: true
      heading_level: 3

::: map_data.utils.way.Way.to_pcd_points
    options:
      show_source: true
      heading_level: 3
