# Test data

`stromovka.mapdata` is a crop of the OpenStreetMap map of Stromovka / Královská obora, Prague.
`stromovka.annotations.json` is a small viewer annotation store for it (two deleted footways and
one added node), written by hand.

Map data © OpenStreetMap contributors, available under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).
See <https://www.openstreetmap.org/copyright>.

## How it was made

Loaded a full-Stromovka `.mapdata` (made with `create_mapdata`) with `MapData.load`, kept the
ways with a node in one of three lat/lon boxes, and wrote it back with `map_data_to_dict`:

| box (S, W, N, E) | why |
| --- | --- |
| 50.1045, 14.4215, 50.1085, 14.4295 | footways, roads and stairs; the field routes |
| 50.1058, 14.4170, 50.1070, 14.4190 | two `bridge=boardwalk` ways |
| 50.1050, 14.4092, 50.1056, 14.4100 | a `smoothness=bad` way |

Way outlines were then simplified to 5 cm and rounded to 1 cm to stay under 300 KB, and the
`crossroads` list was emptied (`MapData.load` recomputes it). Two cycleways, `-1` and `-15`, were
added by copying the longest footway's nodes and outline, to reproduce a corridor mapped twice.
