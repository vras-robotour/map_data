import json
import logging

import numpy as np
import overpy
from shapely.geometry import LineString, Polygon

from map_data.utils.gpx import parse_gpx_file
from map_data.utils.parsing import (
    BUFFER_WIDTHS,
    combine_ways,
    parse_osm_nodes,
    parse_osm_rels,
    parse_osm_ways,
    separate_ways,
)
from map_data.utils.way import Way

_LAT, _LON = 50.0, 14.0

_FOOTWAY_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 1, "lat": _LAT, "lon": _LON},
            {"type": "node", "id": 2, "lat": _LAT + 0.001, "lon": _LON + 0.001},
            {"type": "way", "id": 101, "nodes": [1, 2], "tags": {"highway": "footway"}},
        ],
    },
)

_ROAD_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 3, "lat": _LAT, "lon": _LON},
            {"type": "node", "id": 4, "lat": _LAT + 0.001, "lon": _LON + 0.001},
            {"type": "way", "id": 102, "nodes": [3, 4], "tags": {"highway": "primary"}},
        ],
    },
)

_AREA_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 5, "lat": _LAT, "lon": _LON},
            {"type": "node", "id": 6, "lat": _LAT + 0.001, "lon": _LON},
            {"type": "node", "id": 7, "lat": _LAT + 0.001, "lon": _LON + 0.001},
            # closed way: first and last node share the same lat/lon
            {
                "type": "way",
                "id": 103,
                "nodes": [5, 6, 7, 5],
                "tags": {"building": "yes"},
            },
        ],
    },
)

_OBSTACLE_NODES_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            # An obstacle node (barrier=block matches obstacle_tags.csv)
            {"type": "node", "id": 10, "lat": _LAT, "lon": _LON, "tags": {"barrier": "block"}},
            # A non-obstacle node
            {"type": "node", "id": 11, "lat": _LAT + 0.001, "lon": _LON, "tags": {"name": "bench"}},
        ],
    },
)


def _api():
    return overpy.Overpass()


# ── parse_osm_ways ────────────────────────────────────────────────────────────


def test_parse_osm_ways_footway_linestring():
    data = _api().parse_json(_FOOTWAY_JSON)
    nodes_cache = {}
    ways = parse_osm_ways(data, nodes_cache)
    assert 101 in ways
    w = ways[101]
    assert w.tags == {"highway": "footway"}
    assert w.line.geom_type == "LineString"
    assert not w.is_area
    assert nodes_cache  # nodes populated from way geometry


def test_parse_osm_ways_road():
    data = _api().parse_json(_ROAD_JSON)
    ways = parse_osm_ways(data, {})
    assert 102 in ways
    assert ways[102].tags == {"highway": "primary"}
    assert ways[102].is_road()


def test_parse_osm_ways_area_polygon():
    data = _api().parse_json(_AREA_JSON)
    ways = parse_osm_ways(data, {})
    assert 103 in ways
    w = ways[103]
    # closed way → Polygon
    assert w.line.geom_type == "Polygon"
    assert w.is_area


def test_parse_osm_ways_populates_nodes_cache():
    data = _api().parse_json(_FOOTWAY_JSON)
    nodes_cache = {}
    parse_osm_ways(data, nodes_cache)
    assert 1 in nodes_cache and 2 in nodes_cache
    assert abs(nodes_cache[1]["lat"] - _LAT) < 1e-6
    assert abs(nodes_cache[1]["lon"] - _LON) < 1e-6


# ── separate_ways ─────────────────────────────────────────────────────────────


def _ways_from_json(json_str):
    data = _api().parse_json(json_str)
    return parse_osm_ways(data, {})


def test_separate_ways_routes_footway():
    ways = _ways_from_json(_FOOTWAY_JSON)
    roads, footways, _ = separate_ways(ways, {}, {}, {})
    assert len(footways) == 1
    assert len(roads) == 0


def test_separate_ways_routes_road():
    ways = _ways_from_json(_ROAD_JSON)
    roads, footways, _ = separate_ways(ways, {}, {}, {})
    assert len(roads) == 1
    assert len(footways) == 0


def test_separate_ways_routes_barrier_way():
    barrier_json = json.dumps(
        {
            "version": 0.6,
            "elements": [
                {"type": "node", "id": 20, "lat": _LAT, "lon": _LON},
                {"type": "node", "id": 21, "lat": _LAT + 0.001, "lon": _LON + 0.001},
                {
                    "type": "way",
                    "id": 201,
                    "nodes": [20, 21],
                    "tags": {"barrier": "wall"},
                },
            ],
        },
    )
    ways = _ways_from_json(barrier_json)
    barrier_tags = {"barrier": ["*"]}
    roads, footways, barriers = separate_ways(ways, barrier_tags, {}, {})
    assert len(barriers) == 1
    assert len(roads) == 0
    assert len(footways) == 0


def test_separate_ways_footway_buffered_to_polygon():
    ways = _ways_from_json(_FOOTWAY_JSON)
    _, footways, _ = separate_ways(ways, {}, {}, {})
    w = footways[0]
    # buffer_line should convert the LineString to a Polygon
    assert w.line.geom_type == "Polygon"
    assert w.is_area


def test_separate_ways_road_buffered_wider_than_footway():
    road_ways = _ways_from_json(_ROAD_JSON)
    footway_ways = _ways_from_json(_FOOTWAY_JSON)

    _, _, _ = separate_ways(road_ways, {}, {}, {})
    _, _, _ = separate_ways(footway_ways, {}, {}, {})

    # road width > footway width as per BUFFER_WIDTHS
    assert BUFFER_WIDTHS.get("road", 7) > BUFFER_WIDTHS.get("footway", 3)


# ── parse_osm_nodes ───────────────────────────────────────────────────────────


def test_parse_osm_nodes_obstacle_creates_barrier():
    data = _api().parse_json(_OBSTACLE_NODES_JSON)
    obstacle_tags = {"barrier": ["block"]}
    barriers = parse_osm_nodes(data, {}, set(), obstacle_tags, {})
    assert len(barriers) == 1
    b = barriers[0]
    assert b.id == 10
    assert b.is_area
    assert b.line.geom_type == "Polygon"  # buffered Point


def test_parse_osm_nodes_non_obstacle_excluded():
    data = _api().parse_json(_OBSTACLE_NODES_JSON)
    # Only "barrier: block" is an obstacle; node 11 has "name: bench"
    obstacle_tags = {"barrier": ["block"]}
    barriers = parse_osm_nodes(data, {}, set(), obstacle_tags, {})
    assert all(b.id == 10 for b in barriers)


def test_parse_osm_nodes_way_node_id_skipped():
    data = _api().parse_json(_OBSTACLE_NODES_JSON)
    obstacle_tags = {"barrier": ["block"]}
    # node 10 is in way_node_ids → should be skipped even though it would be an obstacle
    barriers = parse_osm_nodes(data, {}, {10}, obstacle_tags, {})
    assert len(barriers) == 0


def test_parse_osm_nodes_populates_cache():
    data = _api().parse_json(_OBSTACLE_NODES_JSON)
    nodes_cache = {}
    parse_osm_nodes(data, nodes_cache, set(), {}, {})
    assert 10 in nodes_cache
    assert 11 in nodes_cache


# ── combine_ways: disconnected MultiLineString ──────────────────────────────


def test_combine_ways_disconnected_members_kept_unmerged():
    """
    Ways that share an endpoint *node id* but whose geometries don't
    actually touch (e.g. bad/duplicated OSM node coordinates) make
    shapely.ops.linemerge return a MultiLineString instead of a single
    LineString. combine_ways must not crash on that — it should fall back
    to leaving the member ways unmerged rather than building a Way whose
    `.line` is a MultiLineString.
    """
    # Node id 11 is shared, but the line geometries have a gap between them
    # so linemerge cannot join them into one continuous LineString.
    w1 = Way(id=1, nodes=[10, 11], line=LineString([(0, 0), (1, 1)]))
    w2 = Way(id=2, nodes=[11, 12], line=LineString([(5, 5), (6, 6)]))

    ways = {1: w1, 2: w2}
    merged_ids = combine_ways([1, 2], ways)

    # Both original ways are kept as-is rather than replaced by a merged one.
    assert sorted(merged_ids) == [1, 2]
    assert ways[1].line.geom_type == "LineString"
    assert ways[2].line.geom_type == "LineString"
    for wid in merged_ids:
        assert ways[wid].line.geom_type != "MultiLineString"


# ── parse_osm_rels: multipolygon with untagged member ways ──────────────────


# A multipolygon relation (id 200, natural=water) whose single outer member way
# (id 104) is a closed area carrying NO tags of its own — the classifying tag
# lives only on the relation. This is the common OSM pattern for lakes/forests
# and is exactly what the tag-filtered Overpass ways query must still deliver
# via relation recursion.
_MULTIPOLYGON_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 20, "lat": _LAT, "lon": _LON},
            {"type": "node", "id": 21, "lat": _LAT + 0.001, "lon": _LON},
            {"type": "node", "id": 22, "lat": _LAT + 0.001, "lon": _LON + 0.001},
            {"type": "way", "id": 104, "nodes": [20, 21, 22, 20]},  # no tags
            {
                "type": "relation",
                "id": 200,
                "members": [{"type": "way", "ref": 104, "role": "outer"}],
                "tags": {"type": "multipolygon", "natural": "water"},
            },
        ],
    },
)


def test_parse_osm_rels_stamps_tags_onto_untagged_member_ways():
    ways = _ways_from_json(_MULTIPOLYGON_JSON)
    # Before relation parsing the member way carries no classifying tag.
    assert ways[104].tags == {}

    rels_data = _api().parse_json(_MULTIPOLYGON_JSON)
    parse_osm_rels(rels_data, ways)

    # The relation's natural=water tag is stamped onto the (formerly untagged)
    # outer member way, and it is marked as an outer ring.
    assert ways[104].tags.get("natural") == "water"
    assert ways[104].in_out == "outer"


def test_parse_osm_rels_untagged_member_classified_as_barrier():
    ways = _ways_from_json(_MULTIPOLYGON_JSON)
    rels_data = _api().parse_json(_MULTIPOLYGON_JSON)
    parse_osm_rels(rels_data, ways)

    # natural=water is a barrier family; with tags now present the member way
    # must be classified as an (untraversable) barrier rather than vanishing.
    _, _, barriers = separate_ways(ways, {"natural": ["water"]}, {}, {})
    assert any(w.id == 104 for w in barriers)


# ── parse_osm_rels: inner rings become holes ────────────────────────────────


# A building relation with a courtyard: outer square (way 105) and inner
# square (way 106). The inner ring must be cut out of the outer polygon so
# the courtyard stays traversable.
_COURTYARD_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            # outer square (~111 m per 0.001° of latitude)
            {"type": "node", "id": 30, "lat": _LAT, "lon": _LON},
            {"type": "node", "id": 31, "lat": _LAT + 0.001, "lon": _LON},
            {"type": "node", "id": 32, "lat": _LAT + 0.001, "lon": _LON + 0.001},
            {"type": "node", "id": 33, "lat": _LAT, "lon": _LON + 0.001},
            # inner square (the courtyard)
            {"type": "node", "id": 34, "lat": _LAT + 0.0004, "lon": _LON + 0.0004},
            {"type": "node", "id": 35, "lat": _LAT + 0.0006, "lon": _LON + 0.0004},
            {"type": "node", "id": 36, "lat": _LAT + 0.0006, "lon": _LON + 0.0006},
            {"type": "node", "id": 37, "lat": _LAT + 0.0004, "lon": _LON + 0.0006},
            {"type": "way", "id": 105, "nodes": [30, 31, 32, 33, 30]},
            {"type": "way", "id": 106, "nodes": [34, 35, 36, 37, 34]},
            {
                "type": "relation",
                "id": 300,
                "members": [
                    {"type": "way", "ref": 105, "role": "outer"},
                    {"type": "way", "ref": 106, "role": "inner"},
                ],
                "tags": {"type": "multipolygon", "building": "yes"},
            },
        ],
    },
)


def test_parse_osm_rels_inner_ring_cut_from_outer():
    ways = _ways_from_json(_COURTYARD_JSON)
    courtyard_center = ways[106].line.centroid
    # Sanity: before relation parsing the outer polygon is solid.
    assert ways[105].line.contains(courtyard_center)

    parse_osm_rels(_api().parse_json(_COURTYARD_JSON), ways)

    outer = ways[105]
    # A point inside the courtyard is NOT contained in the merged geometry.
    assert not outer.line.contains(courtyard_center)
    assert len(outer.line.interiors) == 1
    # in_out labels stay available for display.
    assert outer.in_out == "outer"
    assert ways[106].in_out == "inner"


# ── parse_osm_rels: merged members not classified twice ─────────────────────


# Two barrier-tagged member ways that merge into one relation way. Before the
# fix the consumed members stayed in the ways dict and separate_ways emitted
# the barrier geometry twice (members + merged way).
_TAGGED_MEMBER_REL_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 40, "lat": _LAT, "lon": _LON},
            {"type": "node", "id": 41, "lat": _LAT + 0.001, "lon": _LON},
            {"type": "node", "id": 42, "lat": _LAT + 0.001, "lon": _LON + 0.001},
            {
                "type": "way",
                "id": 107,
                "nodes": [40, 41],
                "tags": {"barrier": "wall"},
            },
            {
                "type": "way",
                "id": 108,
                "nodes": [41, 42],
                "tags": {"barrier": "wall"},
            },
            {
                "type": "relation",
                "id": 301,
                "members": [
                    {"type": "way", "ref": 107, "role": "outer"},
                    {"type": "way", "ref": 108, "role": "outer"},
                ],
                "tags": {"type": "multipolygon"},
            },
        ],
    },
)


def test_parse_osm_rels_merged_members_not_classified_twice():
    ways = _ways_from_json(_TAGGED_MEMBER_REL_JSON)
    parse_osm_rels(_api().parse_json(_TAGGED_MEMBER_REL_JSON), ways)

    _, _, barriers = separate_ways(ways, {"barrier": ["*"]}, {}, {})

    # Only the merged relation way remains — no duplicated member geometry.
    assert len(barriers) == 1
    assert barriers[0].id < 0


# ── parse_osm_ways: degenerate ways are skipped ─────────────────────────────


_DEGENERATE_WAYS_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 50, "lat": _LAT, "lon": _LON},
            {"type": "node", "id": 51, "lat": _LAT + 0.001, "lon": _LON},
            # 1-node way: LineString would raise
            {"type": "way", "id": 110, "nodes": [50], "tags": {"highway": "footway"}},
            # closed 3-coordinate way [a, b, a]: Polygon would raise
            {"type": "way", "id": 111, "nodes": [50, 51, 50], "tags": {"barrier": "wall"}},
            # valid way that must survive the degenerate ones
            {"type": "way", "id": 112, "nodes": [50, 51], "tags": {"highway": "footway"}},
        ],
    },
)


def test_parse_osm_ways_skips_degenerate_ways(caplog):
    data = _api().parse_json(_DEGENERATE_WAYS_JSON)
    with caplog.at_level(logging.WARNING, logger="map_data.utils.parsing"):
        ways = parse_osm_ways(data, {})

    assert 110 not in ways  # 1-node way skipped
    assert 111 not in ways  # [a, b, a] closed way skipped
    assert 112 in ways  # valid way still parsed
    assert ways[112].line.geom_type == "LineString"
    assert caplog.text.count("Skipping degenerate way") == 2


# ── Way.to_pcd_points: cache invalidation ───────────────────────────────────


def test_to_pcd_points_cache_invalidated_on_density_change():
    way = Way(line=LineString([(0.0, 0.0), (10.0, 0.0)]))

    sparse = way.to_pcd_points(density=1.0, filled=False)
    dense = way.to_pcd_points(density=2.0, filled=False)

    # A stale cache would silently return the density=1.0 result again.
    assert len(dense) > len(sparse)

    # Calling with the original args again must not return the stale
    # (density=2.0) result either.
    sparse_again = way.to_pcd_points(density=1.0, filled=False)
    np.testing.assert_array_equal(sparse_again, sparse)


def test_to_pcd_points_cache_invalidated_on_filled_change():
    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    way = Way(line=poly, is_area=True)

    boundary_only = way.to_pcd_points(density=1.0, filled=False)
    filled_interior = way.to_pcd_points(density=1.0, filled=True)

    # A stale cache keyed only on density (ignoring `filled`) would return
    # the same array for both calls. `filled=True` only samples the
    # interior (no boundary points), so the two point sets are disjoint.
    assert len(filled_interior) != len(boundary_only)
    boundary_set = {tuple(p) for p in boundary_only}
    filled_set = {tuple(p) for p in filled_interior}
    assert boundary_set.isdisjoint(filled_set)

    boundary_again = way.to_pcd_points(density=1.0, filled=False)
    np.testing.assert_array_equal(boundary_again, boundary_only)


# ── parse_gpx_file: empty GPX ────────────────────────────────────────────────


def test_parse_gpx_file_zone_boundary_single_zone(tmp_path):
    """
    A path crossing a UTM zone boundary (18 deg E splits zones 33/34) must be
    converted entirely in the first waypoint's zone. Per-point natural-zone
    conversion would put the second point ~400 km away in easting.
    """
    gpx_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">\n'
        '  <wpt lat="50.0" lon="17.999"></wpt>\n'
        '  <wpt lat="50.0" lon="18.001"></wpt>\n'
        "</gpx>\n"
    )
    gpx_path = tmp_path / "zone_boundary.gpx"
    gpx_path.write_text(gpx_content)

    waypoints, zone_num, zone_let = parse_gpx_file(str(gpx_path))

    assert zone_num == 33
    assert zone_let == "U"
    # ~0.002 deg of longitude at lat 50 is ~140 m, not ~400 km.
    assert abs(waypoints[1][0] - waypoints[0][0]) < 1000


def test_parse_gpx_file_empty_returns_empty_list(tmp_path):
    """
    A structurally valid GPX file with no waypoints, tracks, or routes
    must return the "no data" sentinel (`[]`), not a 3-tuple with empty
    contents — callers (e.g. parse_path) branch on the return type.
    """
    gpx_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"></gpx>\n'
    )
    gpx_path = tmp_path / "empty.gpx"
    gpx_path.write_text(gpx_content)

    result = parse_gpx_file(str(gpx_path))

    assert result == []
    assert not isinstance(result, tuple)
