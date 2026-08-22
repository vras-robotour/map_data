import io
import json
import threading
import time
from unittest.mock import MagicMock, patch
from urllib.parse import quote

import numpy as np
import overpy
import pytest
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.utils.way import Way
from map_data.viewer.app import ACCESS_TOKEN_COOKIE, MAX_CONTENT_LENGTH, create_app
from map_data.viewer.routes import MAX_FETCH_AREA_KM2, _bbox_area_km2


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


def test_export_geojson_missing_param(app_client):
    client, _ = app_client
    resp = client.get("/api/export/geojson")
    assert resp.status_code == 400


def test_export_geojson_not_found(app_client):
    client, _ = app_client
    resp = client.get("/api/export/geojson?file=missing.mapdata")
    assert resp.status_code == 404


def test_export_geojson_success(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.get(f"/api/export/geojson?file={filename}")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("application/geo+json")
    disposition = resp.headers["Content-Disposition"]
    assert "attachment" in disposition
    assert "test.geojson" in disposition
    data = json.loads(resp.get_data(as_text=True))
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
    with ann_path.open() as f:
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
    with ann_path.open() as f:
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
    client, _, filename = app_client_3node
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


# ── fetch_area / upload_gpx area limit ──────────────────────────────────────


def test_bbox_area_km2_matches_known_scale():
    # ~1 deg lat is ~111km; 1x1 deg box near the equator is ~111x111 km
    assert _bbox_area_km2(0.0, 0.0, 1.0, 1.0) == pytest.approx(111.32 * 111.32, rel=0.01)


def test_fetch_area_rejects_oversized_bbox(app_client):
    client, _ = app_client
    # 1x1 deg box near Prague is ~7700 km^2, well over the limit
    assert _bbox_area_km2(50.0, 14.0, 51.0, 15.0) > MAX_FETCH_AREA_KM2
    resp = client.post(
        "/api/fetch_area",
        data=json.dumps(
            {
                "min_lat": 50.0,
                "max_lat": 51.0,
                "min_lon": 14.0,
                "max_lon": 15.0,
                "name": "too_big",
            },
        ),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "km" in resp.get_data(as_text=True)


def test_fetch_area_accepts_small_bbox_and_completes(app_client):
    client, _ = app_client
    ways_raw = json.dumps(
        {
            "version": 0.6,
            "elements": [
                {"type": "node", "id": 1, "lat": 50.0005, "lon": 14.0005},
                {"type": "node", "id": 2, "lat": 50.0006, "lon": 14.0006},
                {"type": "way", "id": 101, "nodes": [1, 2], "tags": {"highway": "footway"}},
            ],
        },
    )
    with patch("map_data.map_data.OverpassClient") as MockClient:
        instance = MagicMock()
        instance.query_raw.return_value = ways_raw
        instance.api = overpy.Overpass()
        MockClient.return_value = instance

        resp = client.post(
            "/api/fetch_area",
            data=json.dumps(
                {
                    "min_lat": 50.000,
                    "max_lat": 50.001,
                    "min_lon": 14.000,
                    "max_lon": 14.001,
                    "name": "small",
                },
            ),
            content_type="application/json",
        )
        assert resp.status_code == 200
        task_id = resp.get_json()["task_id"]

        status = None
        for _ in range(100):
            poll = client.get(f"/api/fetch_area/{task_id}")
            status = poll.get_json()["status"]
            if status in ("done", "failed"):
                break
            time.sleep(0.05)
        assert status == "done"


def test_upload_gpx_rejects_oversized_track(app_client):
    client, _ = app_client
    gpx_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">\n'
        '  <wpt lat="50.0" lon="14.0"/>\n'
        '  <wpt lat="51.0" lon="15.0"/>\n'
        "</gpx>"
    )
    data = {"file": (io.BytesIO(gpx_content.encode()), "big_track.gpx")}
    resp = client.post("/api/upload_gpx", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "km" in resp.get_data(as_text=True)


# ── virtual way IDs ───────────────────────────────────────────────────────────


def test_delete_way_valid_virtual_id(app_client_with_file):
    client, tmp_path, filename = app_client_with_file
    resp = client.delete(
        f"/api/ways/1:0?file={filename}",
        data=json.dumps({"category": "footway", "label": ""}),
        content_type="application/json",
    )
    assert resp.status_code == 204

    with (tmp_path / "test.annotations.json").open() as f:
        store = json.load(f)
    deleted_ids = {(d["id"] if isinstance(d, dict) else d) for d in store["deleted_ways"]}
    assert "1:0" in deleted_ids


@pytest.mark.parametrize(
    "bad_id",
    [
        "1:evil",
        "1:0');alert(1);('",
        "1:0:1",
        "abc",
        "1:",
        "1:<img src=x onerror=alert(1)>",
    ],
)
def test_delete_way_invalid_virtual_id_rejected(app_client_with_file, bad_id):
    client, tmp_path, filename = app_client_with_file
    resp = client.delete(
        f"/api/ways/{quote(bad_id, safe='')}?file={filename}",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    # Nothing may have been written to the store
    assert not (tmp_path / "test.annotations.json").exists()


def test_restore_way_invalid_virtual_id_rejected(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.put(
        f"/api/ways/{quote('1:evil', safe='')}/restore?file={filename}",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_delete_way_node_valid_virtual_way_id(app_client_with_file):
    client, tmp_path, filename = app_client_with_file
    resp = client.delete(f"/api/way_node?file={filename}&way_id=1:0&node_id=101")
    assert resp.status_code == 204

    with (tmp_path / "test.annotations.json").open() as f:
        store = json.load(f)
    assert {"way_id": "1:0", "node_id": 101} in store["deleted_nodes"]


def test_delete_way_node_invalid_virtual_way_id_rejected(app_client_with_file):
    client, _, filename = app_client_with_file
    bad = quote("1:0');alert(1);('", safe="")
    resp = client.delete(f"/api/way_node?file={filename}&way_id={bad}&node_id=101")
    assert resp.status_code == 400


# ── request size limit ────────────────────────────────────────────────────────


def test_max_content_length_configured(app_client):
    client, _ = app_client
    assert client.application.config["MAX_CONTENT_LENGTH"] == MAX_CONTENT_LENGTH
    assert MAX_CONTENT_LENGTH == 100 * 1024 * 1024


# ── access token auth / CSRF ──────────────────────────────────────────────────


def test_auth_get_with_cookie_accepted(app_client, monkeypatch):
    client, _ = app_client
    monkeypatch.setenv("MAP_DATA_ACCESS_TOKEN", "sekret")
    assert client.get("/api/files").status_code == 401
    client.set_cookie(ACCESS_TOKEN_COOKIE, "sekret")
    assert client.get("/api/files").status_code == 200


def test_auth_mutating_cookie_only_rejected(app_client_with_file, monkeypatch):
    client, _, filename = app_client_with_file
    monkeypatch.setenv("MAP_DATA_ACCESS_TOKEN", "sekret")
    client.set_cookie(ACCESS_TOKEN_COOKIE, "sekret")
    resp = client.delete(
        f"/api/ways/1?file={filename}",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 401


def test_auth_mutating_cookie_with_csrf_header_accepted(app_client_with_file, monkeypatch):
    client, _, filename = app_client_with_file
    monkeypatch.setenv("MAP_DATA_ACCESS_TOKEN", "sekret")
    client.set_cookie(ACCESS_TOKEN_COOKIE, "sekret")
    resp = client.delete(
        f"/api/ways/1?file={filename}",
        data=json.dumps({}),
        content_type="application/json",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 204


def test_auth_mutating_token_header_accepted(app_client_with_file, monkeypatch):
    client, _, filename = app_client_with_file
    monkeypatch.setenv("MAP_DATA_ACCESS_TOKEN", "sekret")
    resp = client.delete(
        f"/api/ways/1?file={filename}",
        data=json.dumps({}),
        content_type="application/json",
        headers={"X-Access-Token": "sekret"},
    )
    assert resp.status_code == 204


def test_auth_wrong_token_header_rejected(app_client_with_file, monkeypatch):
    client, _, filename = app_client_with_file
    monkeypatch.setenv("MAP_DATA_ACCESS_TOKEN", "sekret")
    resp = client.delete(
        f"/api/ways/1?file={filename}",
        data=json.dumps({}),
        content_type="application/json",
        headers={"X-Access-Token": "wrong"},
    )
    assert resp.status_code == 401


# ── annotation store concurrency ──────────────────────────────────────────────


def test_add_way_node_concurrent_ids_unique(app_client_with_file):
    # Two concurrent adds must not mint the same min(existing_ids)-1 synthetic
    # ID or lose each other's entries.
    client, tmp_path, filename = app_client_with_file
    app = client.application
    n = 4
    barrier = threading.Barrier(n)
    results = []
    results_lock = threading.Lock()

    def worker():
        c = app.test_client()
        barrier.wait()
        resp = c.post(
            f"/api/way_node?file={filename}&way_id=1",
            data=json.dumps({"after_node_id": 101, "lat": 50.0, "lon": 14.0}),
            content_type="application/json",
        )
        with results_lock:
            results.append(resp.get_json()["id"])

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [-4, -3, -2, -1]
    with (tmp_path / "test.annotations.json").open() as f:
        store = json.load(f)
    assert len(store["added_nodes"]) == n
