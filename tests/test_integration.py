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
    }
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
