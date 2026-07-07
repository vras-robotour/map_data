from shapely.geometry import LineString

from map_data.info import validate_mapdata
from map_data.map_data import MapData
from map_data.utils.way import Way


def _make_md(roads=None, footways=None, barriers=None, nodes_cache=None):
    md = MapData.__new__(MapData)
    md.zone_number = 33
    md.zone_letter = "U"
    md.min_x = 0.0
    md.max_x = 100.0
    md.min_y = 0.0
    md.max_y = 100.0
    md.roads_list = roads or []
    md.footways_list = footways or []
    md.barriers_list = barriers or []
    md.crossroads_list = []
    md.nodes_cache = nodes_cache or {}
    return md


def _way(wid, nodes, coords):
    return Way(id=wid, nodes=nodes, line=LineString(coords), tags={"highway": "footway"})


def test_valid_mapdata_has_no_issues():
    fw1 = _way(1, [10, 11], [(0, 0), (1, 1)])
    fw2 = _way(2, [11, 12], [(1, 1), (2, 2)])
    md = _make_md(footways=[fw1, fw2])
    assert validate_mapdata(md) == []


def test_missing_metadata_reported():
    md = _make_md(footways=[_way(1, [10, 11], [(0, 0), (1, 1)])])
    md.zone_number = None
    md.min_x = None
    issues = validate_mapdata(md)
    assert any("zone_number" in i for i in issues)
    assert any("min_x" in i for i in issues)


def test_empty_map_reported():
    issues = validate_mapdata(_make_md())
    assert any("no roads" in i for i in issues)


def test_missing_geometry_reported():
    broken = Way(id=5, nodes=[1, 2], line=None, tags={"highway": "footway"})
    issues = validate_mapdata(_make_md(footways=[broken]))
    assert any("footway 5" in i and "missing geometry" in i for i in issues)


def test_duplicate_id_across_categories_reported():
    fw = _way(7, [1, 2], [(0, 0), (1, 1)])
    road = Way(id=7, nodes=[3, 4], line=LineString([(2, 2), (3, 3)]), tags={"highway": "primary"})
    issues = validate_mapdata(_make_md(roads=[road], footways=[fw]))
    assert any("duplicate id" in i for i in issues)


def test_nodes_missing_from_cache_reported():
    fw = _way(1, [10, 11], [(0, 0), (1, 1)])
    nodes_cache = {10: {"lat": 50.0, "lon": 14.0}}  # 11 missing
    issues = validate_mapdata(_make_md(footways=[fw], nodes_cache=nodes_cache))
    assert any("missing from nodes_cache" in i for i in issues)


def test_disconnected_footways_reported():
    fw1 = _way(1, [10, 11], [(0, 0), (1, 1)])
    fw2 = _way(2, [20, 21], [(5, 5), (6, 6)])  # shares no nodes with fw1
    issues = validate_mapdata(_make_md(footways=[fw1, fw2]))
    assert any("2 disconnected components" in i for i in issues)


def test_connected_footways_not_reported():
    fw1 = _way(1, [10, 11], [(0, 0), (1, 1)])
    fw2 = _way(2, [11, 12], [(1, 1), (2, 2)])
    fw3 = _way(3, [12, 13], [(2, 2), (3, 3)])
    issues = validate_mapdata(_make_md(footways=[fw1, fw2, fw3]))
    assert not any("disconnected" in i for i in issues)
