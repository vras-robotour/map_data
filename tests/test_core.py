import pytest
from shapely.geometry import LineString, Point, Polygon

from map_data.utils.parsing import combine_ways
from map_data.utils.serialization import way_from_dict, way_to_dict
from map_data.utils.way import Way


def test_way_classification():
    road = Way(tags={"highway": "primary"})
    footway = Way(tags={"highway": "footway"})
    barrier = Way(tags={"barrier": "wall"})

    assert road.is_road()
    assert not road.is_footway()
    assert footway.is_footway()
    assert not footway.is_road()

    yes_tags = {"barrier": ["wall"]}
    assert barrier.is_barrier(yes_tags, {}, {})


def test_way_serialization():
    line = LineString([(0, 0), (1, 1)])
    way = Way(id=123, tags={"highway": "path"}, line=line)

    data = way_to_dict(way)
    assert data["id"] == 123
    assert data["line"].startswith("LINESTRING")

    way2 = way_from_dict(data)
    assert way2.id == way.id
    assert way2.tags == way.tags
    assert way2.line.equals(way.line)


def test_combine_ways():
    w1 = Way(id=1, nodes=[10, 11], line=LineString([(0, 0), (1, 1)]))
    w2 = Way(id=2, nodes=[11, 12], line=LineString([(1, 1), (2, 2)]))

    ways = {1: w1, 2: w2}
    merged_ids = combine_ways([1, 2], ways)

    assert len(merged_ids) == 1
    new_way = ways[merged_ids[0]]
    assert len(new_way.nodes) == 3
    assert new_way.nodes == [10, 11, 12]
    assert new_way.line.length == pytest.approx(LineString([(0, 0), (2, 2)]).length)


def test_way_pcd_points():
    line = LineString([(0, 0), (10, 0)])
    way = Way(line=line)
    pcd = way.to_pcd_points(density=1.0, filled=False)
    # 10m line with 1m density should have roughly 11 points (0 to 10)
    assert len(pcd) >= 11
    assert pcd[0][0] == 0
    assert pcd[-1][0] == 10


def test_way_is_barrier_wildcard():
    """Wildcard '*' in yes_tags matches any tag value."""
    way = Way(tags={"barrier": "anything"})
    assert way.is_barrier({"barrier": ["*"]}, {}, {})


def test_way_is_barrier_anti_tag():
    """Anti-tag blocks a way that would otherwise qualify as a barrier."""
    way = Way(tags={"barrier": "wall", "access": "yes"})
    yes_tags = {"barrier": ["*"]}
    anti_tags = {"access": ["yes"]}
    assert not way.is_barrier(yes_tags, {}, anti_tags)


def test_combine_ways_reversed():
    """combine_ways handles a segment whose endpoint order is reversed."""
    # w1: 10→11, w2: 12→11 — they share node 11 but w2's end is first
    w1 = Way(id=1, nodes=[10, 11], line=LineString([(0, 0), (1, 0)]))
    w2 = Way(id=2, nodes=[12, 11], line=LineString([(2, 0), (1, 0)]))
    ways = {1: w1, 2: w2}
    merged_ids = combine_ways([1, 2], ways)

    assert len(merged_ids) == 1
    new_way = ways[merged_ids[0]]
    assert len(new_way.nodes) == 3
    assert new_way.nodes == [10, 11, 12]


def test_way_pcd_points_filled_polygon():
    """to_pcd_points with filled=True returns interior grid points for a polygon."""
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    way = Way(line=poly, is_area=True)
    pts = way.to_pcd_points(density=1.0, filled=True)

    assert len(pts) > 0
    for p in pts:
        assert poly.contains(Point(float(p[0]), float(p[1])))
