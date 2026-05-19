import json
from unittest.mock import MagicMock, patch

import numpy as np
import overpy
import pytest
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.utils.way import Way

_LAT, _LON = 50.0, 14.0

# One footway way between two nodes — used as the mocked Overpass response
_WAYS_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 1, "lat": 50.001, "lon": 14.001},
            {"type": "node", "id": 2, "lat": 50.002, "lon": 14.002},
            {"type": "way", "id": 101, "nodes": [1, 2], "tags": {"highway": "footway"}},
        ],
    },
)
_EMPTY_JSON = json.dumps({"version": 0.6, "elements": []})


def _make_md():
    e, n, zn, zl = utm.from_latlon(_LAT, _LON)
    waypoints = np.array([[e, n], [e + 200, n + 200]])
    return MapData([waypoints, int(zn), zl], coords_type="array")


def test_mapdata_save_reload_preserves_metadata(tmp_path):
    md = _make_md()
    path = str(tmp_path / "test.mapdata")
    md.save(path)

    loaded = MapData.load(path)
    assert loaded.zone_number == md.zone_number
    assert loaded.zone_letter == md.zone_letter
    assert np.allclose(loaded.min_x, md.min_x)
    assert np.allclose(loaded.waypoints, md.waypoints)


def test_mapdata_save_reload_preserves_ways(tmp_path):
    md = _make_md()
    e, n, _, _ = utm.from_latlon(_LAT, _LON)
    way = Way(
        id=42,
        is_area=False,
        nodes=[1, 2],
        tags={"highway": "footway"},
        line=LineString([(e, n), (e + 50, n + 50)]),
        in_out="",
    )
    md.footways_list.append(way)
    path = str(tmp_path / "test.mapdata")
    md.save(path)

    loaded = MapData.load(path)
    assert len(loaded.footways_list) == 1
    w = loaded.footways_list[0]
    assert w.id == 42
    assert w.tags == {"highway": "footway"}
    assert w.line is not None


def test_run_parse_with_mocked_overpass(tmp_path):
    md = _make_md()

    with patch("map_data.map_data.OverpassClient") as MockClient:
        instance = MagicMock()
        instance.query_raw.return_value = _WAYS_JSON
        instance.api = overpy.Overpass()
        MockClient.return_value = instance
        md.run_queries(use_cache=False)

    assert md.osm_ways_data is not None
    result = md.run_parse()
    assert result == 0
    assert len(md.footways_list) >= 1


def test_gpx_parse_creates_waypoints(tmp_path):
    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="50.0" lon="14.0"/>
  <wpt lat="50.001" lon="14.001"/>
  <wpt lat="50.002" lon="14.002"/>
</gpx>"""
    gpx_path = str(tmp_path / "test.gpx")
    with open(gpx_path, "w") as f:
        f.write(gpx_content)

    md = MapData(gpx_path, coords_type="file")
    assert md.waypoints.shape[1] == 2
    assert len(md.waypoints) == 3


def test_mapdata_nodes_cache_survives_roundtrip(tmp_path):
    md = _make_md()
    md.nodes_cache = {99: {"lat": 50.001, "lon": 14.001, "tags": {"name": "test"}}}
    path = str(tmp_path / "test.mapdata")
    md.save(path)

    loaded = MapData.load(path)
    assert 99 in loaded.nodes_cache
    assert loaded.nodes_cache[99]["lat"] == pytest.approx(50.001)


# ── OSM cache ─────────────────────────────────────────────────────────────────


def test_osm_cache_roundtrip(tmp_path):
    md = _make_md()
    # Give the instance a file path so _get_osm_cache_path can derive its location
    md.coords_file = str(tmp_path / "test.gpx")

    ways_raw = _EMPTY_JSON
    rels_raw = _EMPTY_JSON
    nodes_raw = _EMPTY_JSON

    md._save_osm_cache(ways_raw, rels_raw, nodes_raw)
    loaded = md._load_osm_cache()

    assert loaded is not None
    assert loaded["ways"] == ways_raw
    assert loaded["rels"] == rels_raw
    assert loaded["nodes"] == nodes_raw


def test_osm_cache_bbox_mismatch_returns_none(tmp_path):
    md = _make_md()
    md.coords_file = str(tmp_path / "test.gpx")

    md._save_osm_cache(_EMPTY_JSON, _EMPTY_JSON, _EMPTY_JSON)

    # Shift the stored bounding box by more than the 1e-6 tolerance
    md.min_lat += 1.0
    result = md._load_osm_cache()
    assert result is None


def test_osm_cache_missing_file_returns_none(tmp_path):
    md = _make_md()
    md.coords_file = str(tmp_path / "nonexistent.gpx")
    # No cache file written → should return None without raising
    assert md._load_osm_cache() is None


# ── parse_intersections ───────────────────────────────────────────────────────


def test_parse_intersections_finds_shared_node():
    """
    Node shared by two footways at a non-endpoint position → crossroad.
    """
    import utm as _utm

    md = _make_md()
    e, n, _, _ = _utm.from_latlon(_LAT, _LON)

    # Way 1 has 3 nodes; node 102 is its middle node
    way1 = Way(
        id=10,
        is_area=False,
        nodes=[101, 102, 103],
        tags={"highway": "footway"},
        line=LineString([(e, n), (e + 50, n), (e + 100, n)]),
        in_out="",
    )
    # Way 2 starts at node 102 (endpoint of way2 but middle of way1)
    way2 = Way(
        id=20,
        is_area=False,
        nodes=[102, 104],
        tags={"highway": "footway"},
        line=LineString([(e + 50, n), (e + 50, n + 50)]),
        in_out="",
    )

    lat102, lon102 = _utm.to_latlon(e + 50, n, 33, "U")
    md.nodes_cache = {
        101: {"lat": _LAT, "lon": _LON, "tags": {}},
        102: {"lat": lat102, "lon": lon102, "tags": {}},
        103: {"lat": _LAT, "lon": _LON + 0.001, "tags": {}},
        104: {"lat": _LAT + 0.0005, "lon": _LON + 0.0005, "tags": {}},
    }

    crossroads = md.parse_intersections({10: way1, 20: way2})
    assert len(crossroads) == 1
    assert crossroads[0].id == 102
    assert crossroads[0].line.geom_type == "Polygon"  # buffered Point


def test_parse_intersections_no_shared_nodes():
    """
    Two footways with no shared nodes → no crossroads.
    """
    import utm as _utm

    md = _make_md()
    e, n, _, _ = _utm.from_latlon(_LAT, _LON)

    way1 = Way(
        id=1,
        is_area=False,
        nodes=[1, 2],
        tags={"highway": "footway"},
        line=LineString([(e, n), (e + 50, n)]),
        in_out="",
    )
    way2 = Way(
        id=2,
        is_area=False,
        nodes=[3, 4],
        tags={"highway": "footway"},
        line=LineString([(e + 100, n), (e + 150, n)]),
        in_out="",
    )

    lat2, lon2 = _utm.to_latlon(e + 50, n, 33, "U")
    lat3, lon3 = _utm.to_latlon(e + 100, n, 33, "U")
    lat4, lon4 = _utm.to_latlon(e + 150, n, 33, "U")
    md.nodes_cache = {
        1: {"lat": _LAT, "lon": _LON, "tags": {}},
        2: {"lat": lat2, "lon": lon2, "tags": {}},
        3: {"lat": lat3, "lon": lon3, "tags": {}},
        4: {"lat": lat4, "lon": lon4, "tags": {}},
    }

    crossroads = md.parse_intersections({1: way1, 2: way2})
    assert len(crossroads) == 0


def test_parse_intersections_three_way_junction():
    """
    Node appearing in three footways → crossroad with count=3.
    """
    import utm as _utm

    md = _make_md()
    e, n, _, _ = _utm.from_latlon(_LAT, _LON)
    shared_lat, shared_lon = _utm.to_latlon(e + 50, n, 33, "U")

    ways = {}
    for wid, dx, dy in ((1, 0, 0), (2, 50, 50), (3, -50, 50)):
        ways[wid] = Way(
            id=wid,
            is_area=False,
            nodes=[100 + wid, 200],  # all share end node 200
            tags={"highway": "footway"},
            line=LineString([(e + dx, n + dy), (e + 50, n)]),
            in_out="",
        )

    md.nodes_cache = {
        101: {"lat": _LAT, "lon": _LON, "tags": {}},
        102: {"lat": _LAT + 0.0005, "lon": _LON + 0.0005, "tags": {}},
        103: {"lat": _LAT + 0.0005, "lon": _LON - 0.0005, "tags": {}},
        200: {"lat": shared_lat, "lon": shared_lon, "tags": {}},
    }

    crossroads = md.parse_intersections(ways)
    assert any(c.id == 200 for c in crossroads)
