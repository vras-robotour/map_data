"""Viewer endpoints for the traversability rules: read, save, blocked ways, replan."""

import numpy as np
import pytest
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.utils.way import Way
from map_data.viewer import routes as viewer_routes
from map_data.viewer.app import create_app

GRASS_RULES = "rules:\n  - match: {surface: grass}\n    traversable: false\n    reason: soft\n"


@pytest.fixture
def client(tmp_path, monkeypatch):
    rules_file = tmp_path / "traversability.yaml"
    rules_file.write_text("rules: []\n")
    monkeypatch.setattr(viewer_routes, "_traversability_path", lambda: rules_file)

    e, n, zn, zl = utm.from_latlon(50.0, 14.0)
    md = MapData([np.array([[e, n], [e + 100, n + 100]]), int(zn), zl], coords_type="array")
    ways = {
        1: {"highway": "footway"},
        2: {"highway": "footway", "surface": "grass"},
        3: {"highway": "steps"},
        4: {"highway": "residential"},
    }
    md.nodes_cache = {}
    for wid, tags in ways.items():
        for nid, x in ((wid * 100 + 1, e), (wid * 100 + 2, e + 50)):
            lat, lon = utm.to_latlon(x, n + wid, zn, zl)
            md.nodes_cache[nid] = {"lat": lat, "lon": lon, "tags": {}}
        way = Way(
            id=wid,
            is_area=False,
            nodes=[wid * 100 + 1, wid * 100 + 2],
            tags=tags,
            line=LineString([(e, n + wid), (e + 50, n + wid)]),
            in_out="",
        )
        (md.roads_list if wid == 4 else md.footways_list).append(way)
    md.save(str(tmp_path / "ways.mapdata"))

    app = create_app(data_dir=str(tmp_path))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, rules_file


def _blocked(c, **body):
    resp = c.post("/api/traversability/blocked", json={"file": "ways.mapdata", **body})
    assert resp.status_code == 200, resp.data
    return dict(map(tuple, resp.get_json()["blocked"]))


def test_get_returns_the_file_text(client):
    c, _ = client
    assert c.get("/api/traversability").get_json()["yaml"] == "rules: []\n"


def test_blocked_follows_file_rules_stairs_and_allowed_ways(client):
    c, _ = client
    assert set(_blocked(c)) == {3, 4}
    assert set(_blocked(c, allowed_ways=["footway", "road"])) == {3}


def test_blocked_uses_unsaved_rules_from_the_body(client):
    c, _ = client
    assert _blocked(c, traversability=GRASS_RULES)[2] == "soft"


def test_save_validates_before_writing(client):
    c, rules_file = client
    bad = c.put("/api/traversability", json={"traversability": "rules: [{match: {}}]"})
    assert bad.status_code == 400
    assert rules_file.read_text() == "rules: []\n"

    assert c.put("/api/traversability", json={"traversability": GRASS_RULES}).status_code == 200
    assert rules_file.read_text() == GRASS_RULES
    assert 2 in _blocked(c)


def test_replan_uses_rules_from_the_body(client):
    c, _ = client
    e, n, zn, zl = utm.from_latlon(50.0, 14.0)
    points = [list(utm.to_latlon(e + x, n + 1, zn, zl)) for x in (5, 45)]
    body = {"file": "ways.mapdata", "points": points, "algorithm": "graph"}

    assert c.post("/api/create_replan", json=body).get_json()["newPath"]
    closed = {**body, "traversability": "default: {traversable: false}"}
    assert c.post("/api/create_replan", json=closed).get_json()["retrieveNum"] == 1
    malformed = {**body, "traversability": "rules: 5"}
    assert c.post("/api/create_replan", json=malformed).status_code == 400
