import io
import json
import math
import threading
import time
from unittest.mock import patch
from urllib.parse import quote

import numpy as np
import pytest
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
from map_data.utils.way import Way
from map_data.viewer import routes as viewer_routes
from map_data.viewer.app import ACCESS_TOKEN_COOKIE, MAX_CONTENT_LENGTH, create_app
from map_data.viewer.routes import MAX_FETCH_AREA_KM2, MAX_GRID_CELLS, _bbox_area_km2


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


def test_fetch_area_accepts_small_bbox_and_completes(app_client, mock_overpass_client):
    client, _ = app_client
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


def test_fetch_area_sweeps_stale_terminal_tasks(app_client, mock_overpass_client):
    client, _ = app_client
    now = time.time()
    stale = now - viewer_routes.FETCH_TASK_RETENTION_S - 1
    injected = {
        "stale-done": {"status": "done", "result": {}, "completed_at": stale},
        "stale-failed": {"status": "failed", "error": "boom", "completed_at": stale},
        "fresh-done": {"status": "done", "result": {}, "completed_at": now},
        "unstamped-done": {"status": "done", "result": {}},
        "still-running": {"status": "querying", "detail": "…"},
    }
    viewer_routes._fetch_tasks.update(injected)
    try:
        resp = client.post(
            "/api/fetch_area",
            data=json.dumps(
                {
                    "min_lat": 50.000,
                    "max_lat": 50.001,
                    "min_lon": 14.000,
                    "max_lon": 14.001,
                    "name": "sweep",
                },
            ),
            content_type="application/json",
        )
        assert resp.status_code == 200
        task_id = resp.get_json()["task_id"]
        # Abandoned terminal tasks past the retention window are swept.
        assert "stale-done" not in viewer_routes._fetch_tasks
        assert "stale-failed" not in viewer_routes._fetch_tasks
        # Recent terminal, non-terminal, and the new task all survive.
        assert "fresh-done" in viewer_routes._fetch_tasks
        assert "still-running" in viewer_routes._fetch_tasks
        assert task_id in viewer_routes._fetch_tasks
        # A terminal task never polled gets its retention clock started.
        assert "unstamped-done" in viewer_routes._fetch_tasks
        assert viewer_routes._fetch_tasks["unstamped-done"]["completed_at"] >= now
        # The sweep never stamps non-terminal tasks.
        assert "completed_at" not in viewer_routes._fetch_tasks["still-running"]
    finally:
        for key in injected:
            viewer_routes._fetch_tasks.pop(key, None)


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


# ── added node resolution round-trip ─────────────────────────────────────────


def test_add_way_node_and_resolution_round_trip(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.post(
        f"/api/way_node?file={filename}&way_id=1",
        data=json.dumps({"after_node_id": 101, "lat": 50.00025, "lon": 14.00025}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    new_id = resp.get_json()["id"]
    assert new_id == -1

    nodes = client.get(f"/api/way_nodes?file={filename}&way_id=1").get_json()["nodes"]
    assert [n["id"] for n in nodes] == [101, new_id, 102]
    added = next(n for n in nodes if n["id"] == new_id)
    assert added["lat"] == pytest.approx(50.00025)
    assert added["lon"] == pytest.approx(14.00025)


# ── cost grid ────────────────────────────────────────────────────────────────


def test_cost_grid_small_bbox_returns_points(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.get(
        f"/api/cost_grid?file={filename}&min_lat=50.0&min_lon=14.0&max_lat=50.0005&max_lon=14.0005",
    )
    assert resp.status_code == 200
    points = resp.get_json()
    assert isinstance(points, list)
    assert points
    for point in points:
        lat, lon, cost = point
        assert 49.99 < lat < 50.01
        assert 13.99 < lon < 14.01
        assert math.isfinite(cost)


def test_cost_grid_oversized_bbox_rejected(app_client_with_file):
    client, _, filename = app_client_with_file
    # 0.1 x 0.1 deg near Prague is ~80 km^2 -> ~80M cells at 1 m, over the cap
    assert _bbox_area_km2(50.0, 14.0, 50.1, 14.1) * 1e6 > MAX_GRID_CELLS
    resp = client.get(
        f"/api/cost_grid?file={filename}&min_lat=50.0&min_lon=14.0&max_lat=50.1&max_lon=14.1",
    )
    assert resp.status_code == 400
    assert "cell" in resp.get_data(as_text=True)


def test_cost_grid_inverted_bbox_rejected(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.get(
        f"/api/cost_grid?file={filename}&min_lat=50.001&min_lon=14.0&max_lat=50.0&max_lon=14.001",
    )
    assert resp.status_code == 400


@pytest.mark.parametrize("bad_value", ["abc", "nan", "inf", "1e999"])
def test_cost_grid_non_numeric_bbox_rejected(app_client_with_file, bad_value):
    client, _, filename = app_client_with_file
    resp = client.get(
        f"/api/cost_grid?file={filename}"
        f"&min_lat={bad_value}&min_lon=14.0&max_lat=50.001&max_lon=14.001",
    )
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "bad_costs",
    [
        "not-json",
        '["footway"]',
        '{"footway": "cheap"}',
        '{"footway": -1}',
        '{"footway": NaN}',
    ],
)
def test_cost_grid_malformed_cost_dict_rejected(app_client_with_file, bad_costs):
    client, _, filename = app_client_with_file
    resp = client.get(
        f"/api/cost_grid?file={filename}"
        "&min_lat=50.0&min_lon=14.0&max_lat=50.0005&max_lon=14.0005"
        f"&highway_costs={quote(bad_costs, safe='')}",
    )
    assert resp.status_code == 400


# ── create_replan ────────────────────────────────────────────────────────────


def _replan_body(filename, **overrides):
    body = {
        "points": [[50.0, 14.0], [50.0005, 14.0005]],
        "file": filename,
        "algorithm": "grid",
        "sub_algorithm": "astar",
        "cell_size": 1.0,
    }
    body.update(overrides)
    return body


def test_create_replan_minimal_grid_request(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.post(
        "/api/create_replan",
        data=json.dumps(_replan_body(filename)),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["retrieveNum"] in (-1, 0)
    assert data["newPath"] is not None
    for lat, lon in data["newPath"]:
        assert 49.9 < lat < 50.1
        assert 13.9 < lon < 14.1


@pytest.mark.parametrize(
    "overrides",
    [
        {"cell_size": 0.001},
        {"cell_size": 100},
        {"cell_size": "0.25"},
        {"cell_size": True},
        {"inflate_obstacles": -1},
        {"inflate_obstacles": 100},
        {"highway_costs": "footway"},
        {"highway_costs": {"footway": "cheap"}},
        {"highway_costs": {"footway": -0.5}},
        {"surface_costs": ["asphalt"]},
        {"points": [[50.0, 14.0], ["x", 14.0]]},
        {"points": [[50.0, 14.0], [91.0, 14.0]]},
        {"points": "not-a-list"},
        {"simplify_path": "yes"},
        {"allowed_ways": "footway"},
        {"grid_cost_weight": "heavy"},
        {"transfer_id": 42},
    ],
)
def test_create_replan_invalid_params_rejected(app_client_with_file, overrides):
    client, _, filename = app_client_with_file
    resp = client.post(
        "/api/create_replan",
        data=json.dumps(_replan_body(filename, **overrides)),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_create_replan_missing_points_rejected(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.post(
        "/api/create_replan",
        data=json.dumps({"file": filename}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_create_replan_grid_cell_budget_rejected(app_client_with_file):
    client, _, filename = app_client_with_file
    # The tiny cell size makes even this small map's clipped planning bbox
    # (~400 m x 400 m incl. grid margin) exceed the cell budget.
    body = _replan_body(
        filename,
        cell_size=0.05,
        points=[[49.999, 13.999], [50.002, 14.003]],
    )
    resp = client.post(
        "/api/create_replan",
        data=json.dumps(body),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "cell" in resp.get_data(as_text=True)


# ── upload_mapdata ───────────────────────────────────────────────────────────


def _mapdata_bytes(tmp_path):
    src_dir = tmp_path / "upload_src"
    src_dir.mkdir(exist_ok=True)
    src = src_dir / "orig.mapdata"
    if not src.exists():
        _make_mapdata(src)
    return src.read_bytes()


def test_upload_mapdata_valid(app_client):
    client, tmp_path = app_client
    payload = _mapdata_bytes(tmp_path)
    resp = client.post(
        "/api/upload_mapdata",
        data={"file": (io.BytesIO(payload), "uploaded.mapdata")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["filename"] == "uploaded.mapdata"
    assert (tmp_path / "uploaded.mapdata").is_file()


def test_upload_mapdata_invalid_extension_rejected(app_client):
    client, tmp_path = app_client
    resp = client.post(
        "/api/upload_mapdata",
        data={"file": (io.BytesIO(b"whatever"), "notmap.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert not (tmp_path / "notmap.txt").exists()


def test_upload_mapdata_invalid_content_deleted(app_client):
    client, tmp_path = app_client
    resp = client.post(
        "/api/upload_mapdata",
        data={"file": (io.BytesIO(b"this is not a mapdata file"), "bad.mapdata")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    # The failed upload must not be left behind in the data dir
    assert not (tmp_path / "bad.mapdata").exists()


def test_upload_mapdata_duplicate_name_disambiguated(app_client):
    client, tmp_path = app_client
    payload = _mapdata_bytes(tmp_path)
    first = client.post(
        "/api/upload_mapdata",
        data={"file": (io.BytesIO(payload), "dup.mapdata")},
        content_type="multipart/form-data",
    )
    assert first.get_json()["filename"] == "dup.mapdata"
    second = client.post(
        "/api/upload_mapdata",
        data={"file": (io.BytesIO(payload), "dup.mapdata")},
        content_type="multipart/form-data",
    )
    assert second.status_code == 200
    assert second.get_json()["filename"] == "dup_1.mapdata"
    assert (tmp_path / "dup.mapdata").is_file()
    assert (tmp_path / "dup_1.mapdata").is_file()


# ── native export ────────────────────────────────────────────────────────────


def test_export_native_missing_param(app_client):
    client, _ = app_client
    assert client.get("/api/export").status_code == 400


def test_export_native_not_found(app_client):
    client, _ = app_client
    assert client.get("/api/export?file=missing.mapdata").status_code == 404


def test_export_native_success(app_client_with_file):
    client, _, filename = app_client_with_file
    resp = client.get(f"/api/export?file={filename}")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("application/json")
    disposition = resp.headers["Content-Disposition"]
    assert "attachment" in disposition
    assert "test.exported.mapdata" in disposition
    data = json.loads(resp.get_data(as_text=True))
    assert data["metadata"]["zone_number"] == 33
    assert len(data["footways"]) == 1
    assert data["footways"][0]["id"] == 1
    assert data["footways"][0]["tags"] == {"highway": "footway"}


# ── wormhole ─────────────────────────────────────────────────────────────────


def test_create_wormhole_returns_code_and_transfer_id(app_client):
    client, _ = app_client
    with (
        patch.object(
            viewer_routes.wormhole_manager,
            "create_transfer",
            return_value="tid-1",
        ) as mock_create,
        patch.object(
            viewer_routes.wormhole_manager,
            "get_transfer_code",
            return_value="7-crossover-clockwork",
        ),
    ):
        resp = client.post("/api/create_wormhole", json={"gpx": "<gpx></gpx>"})
    assert resp.status_code == 200
    assert resp.get_json() == {
        "success": True,
        "code": "7-crossover-clockwork",
        "transfer_id": "tid-1",
    }
    mock_create.assert_called_once_with("<gpx></gpx>")


def test_create_wormhole_missing_gpx_rejected(app_client):
    client, _ = app_client
    with patch.object(viewer_routes.wormhole_manager, "create_transfer") as mock_create:
        resp = client.post("/api/create_wormhole", json={})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False
    mock_create.assert_not_called()


def test_create_wormhole_code_timeout_cancels_transfer(app_client):
    client, _ = app_client
    with (
        patch.object(viewer_routes.wormhole_manager, "create_transfer", return_value="tid-2"),
        patch.object(viewer_routes.wormhole_manager, "get_transfer_code", return_value=None),
        patch.object(
            viewer_routes.wormhole_manager,
            "cancel_transfer",
            return_value=(True, "Transfer cancelled"),
        ) as mock_cancel,
    ):
        resp = client.post("/api/create_wormhole", json={"gpx": "<gpx></gpx>"})
    assert resp.status_code == 500
    assert resp.get_json()["success"] is False
    mock_cancel.assert_called_once_with("tid-2")


def test_cancel_wormhole_unknown_transfer(app_client):
    client, _ = app_client
    resp = client.post("/api/cancel_wormhole", json={"transfer_id": "does-not-exist"})
    assert resp.status_code == 200
    assert resp.get_json() == {"success": False, "message": "Invalid or unknown transfer ID"}


# ── non-JSON / empty bodies on JSON endpoints ────────────────────────────────


@pytest.mark.parametrize(
    "endpoint",
    ["/api/cancel_replan", "/api/create_wormhole", "/api/cancel_wormhole"],
)
def test_json_endpoints_reject_missing_content_type(app_client, endpoint):
    client, _ = app_client
    # No Content-Type header at all.
    resp = client.post(endpoint, data="transfer_id=x")
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


@pytest.mark.parametrize(
    "endpoint",
    ["/api/cancel_replan", "/api/create_wormhole", "/api/cancel_wormhole"],
)
def test_json_endpoints_reject_empty_json_body(app_client, endpoint):
    client, _ = app_client
    # Correct content type but an empty body.
    resp = client.post(endpoint, data="", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_cancel_replan_with_transfer_id_succeeds(app_client):
    client, _ = app_client
    with patch.object(viewer_routes, "cancel_replan_backend") as mock_cancel:
        resp = client.post("/api/cancel_replan", json={"transfer_id": "tid-9"})
    assert resp.status_code == 200
    assert resp.get_json() == {"success": True}
    mock_cancel.assert_called_once_with("tid-9")


def test_cancel_replan_missing_transfer_id_rejected(app_client):
    client, _ = app_client
    with patch.object(viewer_routes, "cancel_replan_backend") as mock_cancel:
        resp = client.post("/api/cancel_replan", json={})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False
    mock_cancel.assert_not_called()


# ── annotated paths create crossroads ─────────────────────────────────────────


def _post_path_annotation(client, filename, coords_lonlat):
    body = {
        "type": "path",
        "geometry": {"type": "LineString", "coordinates": coords_lonlat},
        "properties": {"highway": "footway", "width": 1.5},
    }
    resp = client.post(
        f"/api/annotations?file={filename}",
        data=json.dumps(body),
        content_type="application/json",
    )
    assert resp.status_code == 201


def test_annotated_path_crossing_footway_creates_crossroad(app_client_with_file):
    client, _, filename = app_client_with_file
    # The fixture footway runs from (e, n) to (e+50, n+50); draw a path across it.
    e, n, zn, zl = utm.from_latlon(50.0, 14.0)
    a = utm.to_latlon(e + 50, n, zn, zl)
    b = utm.to_latlon(e, n + 50, zn, zl)
    _post_path_annotation(client, filename, [[a[1], a[0]], [b[1], b[0]]])

    resp = client.get(f"/api/export?file={filename}")
    assert resp.status_code == 200
    crossroads = resp.get_json()["crossroads"]
    assert len(crossroads) == 1
    assert crossroads[0]["tags"]["type"] == "annotation_intersection"


def test_annotated_path_touching_footway_creates_crossroad(app_client_with_file):
    client, _, filename = app_client_with_file
    # T-junction: the annotated path ends 0.5 m from the footway's midpoint.
    e, n, zn, zl = utm.from_latlon(50.0, 14.0)
    end = utm.to_latlon(e + 25 + 0.35, n + 25 - 0.35, zn, zl)
    far = utm.to_latlon(e + 60, n - 10, zn, zl)
    _post_path_annotation(client, filename, [[far[1], far[0]], [end[1], end[0]]])

    resp = client.get(f"/api/export?file={filename}")
    assert len(resp.get_json()["crossroads"]) == 1


def test_annotated_path_far_from_footway_creates_no_crossroad(app_client_with_file):
    client, _, filename = app_client_with_file
    e, n, zn, zl = utm.from_latlon(50.0, 14.0)
    a = utm.to_latlon(e + 200, n, zn, zl)
    b = utm.to_latlon(e + 250, n + 50, zn, zl)
    _post_path_annotation(client, filename, [[a[1], a[0]], [b[1], b[0]]])

    resp = client.get(f"/api/export?file={filename}")
    assert resp.get_json()["crossroads"] == []
