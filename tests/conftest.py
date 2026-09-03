"""Shared fixtures and canned data for the test suite."""

import json
from unittest.mock import MagicMock, patch

import overpy
import pytest

# One footway way between two nodes — the canonical mocked Overpass response.
# The nodes sit just inside a 50.000-50.001 / 14.000-14.001 bbox so the same
# payload works for both MapData integration tests and the viewer's
# fetch_area tests (whose bbox is that tight).
FOOTWAY_WAYS_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 1, "lat": 50.0005, "lon": 14.0005},
            {"type": "node", "id": 2, "lat": 50.0006, "lon": 14.0006},
            {"type": "way", "id": 101, "nodes": [1, 2], "tags": {"highway": "footway"}},
        ],
    },
)

EMPTY_OSM_JSON = json.dumps({"version": 0.6, "elements": []})


@pytest.fixture
def mock_overpass_client():
    """
    Patch ``map_data.map_data.OverpassClient`` with a canned footway response.

    Yields the mock instance so tests can override ``query_raw.return_value``
    or inspect calls. ``api`` is a real ``overpy.Overpass`` so ``parse_json``
    behaves exactly as in production.
    """
    with patch("map_data.map_data.OverpassClient") as mock_client:
        instance = MagicMock()
        instance.query_raw.return_value = FOOTWAY_WAYS_JSON
        instance.api = overpy.Overpass()
        mock_client.return_value = instance
        yield instance


def build_footway_network_mapdata(path):
    """
    Write a small planning-capable ``.mapdata`` to *path* and return
    ``(lat0, lon0)`` of its origin node.

    Network (UTM metres from the origin, all ``highway=footway``):

    - way 1: (0,0) -> (100,0) -> (200,0)         nodes 101, 102, 103
    - way 2: (100,0) -> (100,100)                 nodes 102, 104  (T-junction at 102)
    - way 3: (400,0) -> (500,0)                   nodes 201, 202  (disconnected)
    """
    import numpy as np
    import utm
    from shapely.geometry import LineString

    from map_data.map_data import MapData
    from map_data.utils.way import Way

    lat0, lon0 = 50.0, 14.0
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    coords = {
        101: (0.0, 0.0),
        102: (100.0, 0.0),
        103: (200.0, 0.0),
        104: (100.0, 100.0),
        201: (400.0, 0.0),
        202: (500.0, 0.0),
    }
    waypoints = np.array([[e0 - 50, n0 - 50], [e0 + 550, n0 + 150]])
    md = MapData([waypoints, int(zn), zl], coords_type="array")
    for way_id, node_ids in ((1, [101, 102, 103]), (2, [102, 104]), (3, [201, 202])):
        line = LineString([(e0 + coords[n][0], n0 + coords[n][1]) for n in node_ids])
        md.footways_list.append(
            Way(
                id=way_id,
                is_area=False,
                nodes=list(node_ids),
                tags={"highway": "footway"},
                line=line.buffer(1.0),
                in_out="",
            )
        )
    md.nodes_cache = {}
    for nid, (dx, dy) in coords.items():
        lat, lon = utm.to_latlon(e0 + dx, n0 + dy, zn, zl)
        md.nodes_cache[nid] = {"lat": lat, "lon": lon, "tags": {}}
    md.crossroads_list = md.parse_intersections({w.id: w for w in md.footways_list})
    md.save(str(path))
    return lat0, lon0


@pytest.fixture
def footway_network_mapdata(tmp_path):
    """``(path, lat0, lon0)`` of :func:`build_footway_network_mapdata` in a temp dir."""
    path = tmp_path / "network.mapdata"
    lat0, lon0 = build_footway_network_mapdata(path)
    return path, lat0, lon0
