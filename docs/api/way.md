# Way

::: map_data.utils.way.Way
    options:
      show_root_heading: false
      members: false

## Attributes

`Way` objects are created by `MapData.run_parse()` and are not typically constructed manually.

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `int` | OSM way or relation ID |
| `nodes` | `list` | Ordered list of OSM node IDs defining the geometry |
| `tags` | `dict[str, str]` | OSM tags (e.g. `{"highway": "footway", "surface": "asphalt"}`) |
| `line` | `shapely.Geometry \| None` | Geometry in UTM metres. `LineString` for roads and footways; `Polygon` for area-type barriers. `None` if unparsed. |
| `in_out` | `str \| None` | Direction hint set by `GraphPlanner`: `"in"`, `"out"`, or `None` for bidirectional |
| `pcd_points` | `np.ndarray \| None` | Array of equidistant 3D points along the way, populated by `to_pcd_points()` |

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
