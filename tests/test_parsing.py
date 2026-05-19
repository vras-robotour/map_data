import json

import overpy

from map_data.utils.parsing import BUFFER_WIDTHS, parse_osm_nodes, parse_osm_ways, separate_ways

_LAT, _LON = 50.0, 14.0

_FOOTWAY_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 1, "lat": _LAT, "lon": _LON},
            {"type": "node", "id": 2, "lat": _LAT + 0.001, "lon": _LON + 0.001},
            {"type": "way", "id": 101, "nodes": [1, 2], "tags": {"highway": "footway"}},
        ],
    }
)

_ROAD_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 3, "lat": _LAT, "lon": _LON},
            {"type": "node", "id": 4, "lat": _LAT + 0.001, "lon": _LON + 0.001},
            {"type": "way", "id": 102, "nodes": [3, 4], "tags": {"highway": "primary"}},
        ],
    }
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
    }
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
    }
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
    roads, footways, barriers = separate_ways(ways, {}, {}, {})
    assert len(footways) == 1
    assert len(roads) == 0
    assert len(barriers) == 0


def test_separate_ways_routes_road():
    ways = _ways_from_json(_ROAD_JSON)
    roads, footways, barriers = separate_ways(ways, {}, {}, {})
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
        }
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
