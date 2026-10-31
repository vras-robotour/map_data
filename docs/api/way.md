# Way

`Way` objects are created by `MapData.run_parse()`. Geometry is in the UTM zone of the parent `MapData`; convert a point with `utm.to_latlon(x, y, md.zone_number, md.zone_letter)`.

::: map_data.utils.way.Way
    options:
      show_root_heading: false
      heading_level: 3
