import json

import numpy as np
import pytest
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.utils.way import Way
from map_data.viewer.app import create_app


def _make_mapdata(path):
    """
    Write a minimal valid .mapdata file with one footway to `path`.
    """
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
    deleted_ids = {(d["id"] if isinstance(d, dict) else d) for d in store.get("deleted_ways", [])}
    assert 1 in deleted_ids


def test_get_way_not_found(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.get(f"/api/ways/9999?file={filename}")
    assert resp.status_code == 404


# ── security ──────────────────────────────────────────────────────────────────


def test_path_traversal_rejected(app_client):
    client, _ = app_client
    resp = client.get("/api/mapdata?file=../../../etc/passwd")
    assert resp.status_code == 400


def test_path_traversal_nested_rejected(app_client):
    client, _ = app_client
    resp = client.get("/api/mapdata?file=sub/../../etc/passwd")
    assert resp.status_code == 400


# ── way tags ─────────────────────────────────────────────────────────────────


def test_update_way_tags(app_client_with_file):
    client, _, filename = app_client_with_file
    body = {"tags": {"highway": "path"}, "category": "footway", "label": "test"}
    resp = client.put(
        f"/api/ways/1/tags?file={filename}",
        data=json.dumps(body),
        content_type="application/json",
    )
    assert resp.status_code == 204


def test_delete_way_tags(app_client_with_file):
    client, _, filename = app_client_with_file
    # Set tags first
    client.put(
        f"/api/ways/1/tags?file={filename}",
        data=json.dumps({"tags": {"highway": "path"}}),
        content_type="application/json",
    )
    resp = client.delete(f"/api/ways/1/tags?file={filename}")
    assert resp.status_code == 204


# ── hide / show / restore ─────────────────────────────────────────────────────


def test_hide_way(app_client_with_file):
    client, _, filename = app_client_with_file
    body = {"category": "footway", "label": "test"}
    resp = client.put(
        f"/api/ways/1/hide?file={filename}",
        data=json.dumps(body),
        content_type="application/json",
    )
    assert resp.status_code == 204


def test_show_way(app_client_with_file):
    client, _, filename = app_client_with_file
    body = {"category": "footway", "label": "test"}
    client.put(
        f"/api/ways/1/hide?file={filename}",
        data=json.dumps(body),
        content_type="application/json",
    )
    resp = client.put(
        f"/api/ways/1/show?file={filename}",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 204


def test_restore_way(app_client_with_file):
    client, tmp_path, filename = app_client_with_file
    # Delete first
    client.delete(
        f"/api/ways/1?file={filename}",
        data=json.dumps({"category": "footway", "label": ""}),
        content_type="application/json",
    )
    # Now restore
    resp = client.put(
        f"/api/ways/1/restore?file={filename}",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 204

    ann_path = tmp_path / "test.annotations.json"
    with open(ann_path) as f:
        store = json.load(f)
    deleted_ids = {(d["id"] if isinstance(d, dict) else d) for d in store.get("deleted_ways", [])}
    assert 1 not in deleted_ids


# ── node operations ───────────────────────────────────────────────────────────


def test_delete_way_node(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.delete(f"/api/way_node?file={filename}&way_id=1&node_id=101")
    assert resp.status_code == 204


def test_move_way_nodes(app_client_with_file):
    client, _, filename = app_client_with_file
    lat, lon = 50.0, 14.0
    body = {"nodes": [{"id": 101, "lat": lat + 0.0001, "lon": lon + 0.0001}]}
    resp = client.put(
        f"/api/way_nodes/move?file={filename}&way_id=1",
        data=json.dumps(body),
        content_type="application/json",
    )
    assert resp.status_code == 204


# ── way split ─────────────────────────────────────────────────────────────────


def _make_mapdata_3node(path):
    """
    Write a .mapdata file with a 3-node footway (so a middle split is possible).
    """
    lat, lon = 50.0, 14.0
    e, n, zn, zl = utm.from_latlon(lat, lon)
    waypoints = np.array([[e, n], [e + 100, n + 100]])
    md = MapData([waypoints, int(zn), zl], coords_type="array")
    way = Way(
        id=2,
        is_area=False,
        nodes=[201, 202, 203],
        tags={"highway": "footway"},
        line=LineString([(e, n), (e + 50, n + 50), (e + 100, n + 100)]),
        in_out="",
    )
    md.footways_list.append(way)
    lat2, lon2 = utm.to_latlon(e + 50, n + 50, int(zn), zl)
    lat3, lon3 = utm.to_latlon(e + 100, n + 100, int(zn), zl)
    md.nodes_cache = {
        201: {"lat": lat, "lon": lon, "tags": {}},
        202: {"lat": lat2, "lon": lon2, "tags": {}},
        203: {"lat": lat3, "lon": lon3, "tags": {}},
    }
    md.save(str(path))


@pytest.fixture
def app_client_3node(tmp_path):
    _make_mapdata_3node(tmp_path / "three.mapdata")
    app = create_app(data_dir=str(tmp_path))
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, tmp_path, "three.mapdata"


def test_split_way_endpoint_saves_split(app_client_3node):
    client, tmp_path, filename = app_client_3node
    body = {"way_id": 2, "node_id": 202}
    resp = client.post(
        f"/api/ways/split?file={filename}",
        data=json.dumps(body),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    # The split should result in two segments
    assert len(data["segments"]) == 2


def test_split_way_undo(app_client_3node):
    client, _, filename = app_client_3node
    # First split
    client.post(
        f"/api/ways/split?file={filename}",
        data=json.dumps({"way_id": 2, "node_id": 202}),
        content_type="application/json",
    )
    # Undo the split
    resp = client.delete(f"/api/ways/split?file={filename}&way_id=2&node_id=202")
    assert resp.status_code == 200
    data = resp.get_json()
    # After undo, only one segment (original way)
    assert len(data["segments"]) == 1
