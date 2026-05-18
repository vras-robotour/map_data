import json

import numpy as np
import pytest
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.utils.way import Way
from map_data.viewer.app import create_app


def _make_mapdata(path):
    """Write a minimal valid .mapdata file with one footway to `path`."""
    lat, lon = 50.0, 14.0
    e, n, zn, zl = utm.from_latlon(lat, lon)
    waypoints = np.array([[e, n], [e + 100, n + 100]])
    md = MapData([waypoints, int(zn), zl], coords_type="array")
    way = Way(
        id=1,
        is_area=False,
        nodes=[101, 102],
        tags={"highway": "footway"},
        line=LineString([(e, n), (e + 50, n + 50)]),
        in_out="",
    )
    md.footways_list.append(way)
    md.nodes_cache = {
        101: {"lat": lat, "lon": lon, "tags": {}},
        102: {"lat": lat + 0.0005, "lon": lon + 0.0005, "tags": {}},
    }
    md.save(str(path))


@pytest.fixture
def app_client(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, tmp_path


@pytest.fixture
def app_client_with_file(tmp_path):
    _make_mapdata(tmp_path / "test.mapdata")
    app = create_app(data_dir=str(tmp_path))
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, tmp_path, "test.mapdata"


def test_list_files_empty(app_client):
    client, _ = app_client
    resp = client.get("/api/files")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mapdata"] == []
    assert data["gpx"] == []


def test_list_files_with_mapdata(app_client):
    client, tmp_path = app_client
    _make_mapdata(tmp_path / "mymap.mapdata")
    resp = client.get("/api/files")
    assert resp.status_code == 200
    assert "mymap.mapdata" in resp.get_json()["mapdata"]


def test_get_mapdata_missing_param(app_client):
    client, _ = app_client
    resp = client.get("/api/mapdata")
    assert resp.status_code == 400


def test_get_mapdata_not_found(app_client):
    client, _ = app_client
    resp = client.get("/api/mapdata?file=missing.mapdata")
    assert resp.status_code == 404


def test_get_mapdata_success(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.get(f"/api/mapdata?file={filename}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data


def test_get_annotations_empty(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.get(f"/api/annotations?file={filename}")
    assert resp.status_code == 200
    assert resp.get_json()["annotations"] == []


def test_add_annotation(app_client_with_file):
    client, _, filename = app_client_with_file
    body = {
        "type": "obstacle",
        "geometry": {"type": "Point", "coordinates": [14.0, 50.0]},
        "properties": {},
    }
    resp = client.post(
        f"/api/annotations?file={filename}",
        data=json.dumps(body),
        content_type="application/json",
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert "id" in data
    assert data["type"] == "obstacle"


def test_update_annotation(app_client_with_file):
    client, _, filename = app_client_with_file
    body = {
        "type": "obstacle",
        "geometry": {"type": "Point", "coordinates": [14.0, 50.0]},
        "properties": {},
    }
    ann_id = client.post(
        f"/api/annotations?file={filename}",
        data=json.dumps(body),
        content_type="application/json",
    ).get_json()["id"]

    update_body = {"geometry": {"type": "Point", "coordinates": [14.001, 50.001]}}
    resp = client.put(
        f"/api/annotations/{ann_id}?file={filename}",
        data=json.dumps(update_body),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json()["geometry"]["coordinates"] == [14.001, 50.001]


def test_delete_annotation(app_client_with_file):
    client, _, filename = app_client_with_file
    body = {
        "type": "obstacle",
        "geometry": {"type": "Point", "coordinates": [14.0, 50.0]},
        "properties": {},
    }
    ann_id = client.post(
        f"/api/annotations?file={filename}",
        data=json.dumps(body),
        content_type="application/json",
    ).get_json()["id"]

    resp = client.delete(f"/api/annotations/{ann_id}?file={filename}")
    assert resp.status_code == 204

    remaining = client.get(f"/api/annotations?file={filename}").get_json()["annotations"]
    assert all(a["id"] != ann_id for a in remaining)


def test_delete_annotation_not_found(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.delete(f"/api/annotations/nonexistent-uuid?file={filename}")
    assert resp.status_code == 404


def test_delete_way(app_client_with_file):
    client, tmp_path, filename = app_client_with_file
    resp = client.delete(
        f"/api/ways/1?file={filename}",
        data=json.dumps({"category": "footway", "label": ""}),
        content_type="application/json",
    )
    assert resp.status_code == 204

    ann_path = tmp_path / "test.annotations.json"
    with open(ann_path) as f:
        store = json.load(f)
    deleted_ids = {
        (d["id"] if isinstance(d, dict) else d)
        for d in store.get("deleted_ways", [])
    }
    assert 1 in deleted_ids


def test_get_way_not_found(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.get(f"/api/ways/9999?file={filename}")
    assert resp.status_code == 404
