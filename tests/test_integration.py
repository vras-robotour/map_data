import json
from pathlib import Path

import numpy as np
import pytest
import utm
from conftest import EMPTY_OSM_JSON as _EMPTY_JSON
from conftest import FOOTWAY_WAYS_JSON as _WAYS_JSON
from shapely.geometry import LineString

from map_data.create_mapdata import process_map_data
from map_data.map_data import MapData
from map_data.utils.serialization import FORMAT_VERSION
from map_data.utils.way import Way

_LAT, _LON = 50.0, 14.0


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


def test_run_parse_with_mocked_overpass(tmp_path, mock_overpass_client):
    md = _make_md()
    md.run_queries(use_cache=False)

    assert md.osm_data is not None
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
    gpx_path = tmp_path / "test.gpx"
    with gpx_path.open("w") as f:
        f.write(gpx_content)

    md = MapData(str(gpx_path), coords_type="file")
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


# ── .mapdata format version ───────────────────────────────────────────────────


def _rewrite_version(path, version):
    """Rewrite the saved file with ``version`` as its marker, or drop the key for ``None``."""
    data = json.loads(Path(path).read_text())
    if version is None:
        del data["format_version"]
    else:
        data["format_version"] = version
    Path(path).write_text(json.dumps(data))


def test_saved_mapdata_carries_format_version(tmp_path):
    md = _make_md()
    path = tmp_path / "test.mapdata"
    md.save(str(path))

    assert json.loads(path.read_text())["format_version"] == FORMAT_VERSION
    MapData.load(str(path))  # and the marker does not get in the loader's way


def test_legacy_mapdata_without_version_loads(tmp_path):
    """A file without a format_version marker still loads."""
    md = _make_md()
    path = tmp_path / "test.mapdata"
    md.save(str(path))
    _rewrite_version(path, None)

    loaded = MapData.load(str(path))
    assert loaded.zone_number == md.zone_number
    assert np.allclose(loaded.waypoints, md.waypoints)


def test_future_version_mapdata_raises(tmp_path):
    md = _make_md()
    path = tmp_path / "test.mapdata"
    md.save(str(path))
    _rewrite_version(path, FORMAT_VERSION + 1)

    with pytest.raises(ValueError, match="format version"):
        MapData.load(str(path))


def test_non_integer_version_mapdata_raises(tmp_path):
    md = _make_md()
    path = tmp_path / "test.mapdata"
    md.save(str(path))
    _rewrite_version(path, "1.0")

    with pytest.raises(ValueError, match="format version"):
        MapData.load(str(path))


# ── atomic .mapdata writes ────────────────────────────────────────────────────


def _broken_json_dumps(obj, **kwargs):
    """Simulate a crash/full disk mid-dump, after the temp file is already open."""
    raise OSError("disk full")


def test_save_mapdata_failure_preserves_existing_file(tmp_path, monkeypatch):
    md = _make_md()
    path = tmp_path / "test.mapdata"
    md.save(str(path))
    good_content = path.read_text()

    monkeypatch.setattr("map_data.utils.serialization.json.dumps", _broken_json_dumps)
    with pytest.raises(OSError, match="disk full"):
        md.save(str(path))

    # The existing file is untouched and no temp files are left behind.
    assert path.read_text() == good_content
    assert not list(tmp_path.glob("*.tmp"))


def test_save_mapdata_success_leaves_no_temp_files(tmp_path):
    md = _make_md()
    path = tmp_path / "test.mapdata"
    md.save(str(path))

    assert [p.name for p in tmp_path.iterdir()] == ["test.mapdata"]
    MapData.load(str(path))  # and the file is loadable


def test_csv_to_dict_single_row(tmp_path):
    csv_path = tmp_path / "single.csv"
    csv_path.write_text("natural,water\n")

    # A single-row CSV makes np.genfromtxt return a 1-D array; iterating its
    # "rows" then yields characters. Must parse as one key/value pair instead.
    assert MapData._csv_to_dict(csv_path) == {"natural": ["water"]}


def test_csv_to_dict_multiple_rows(tmp_path):
    csv_path = tmp_path / "multi.csv"
    csv_path.write_text("natural,water\nnatural,wood\nbarrier,wall\n")

    assert MapData._csv_to_dict(csv_path) == {
        "natural": ["water", "wood"],
        "barrier": ["wall"],
    }


# ── OSM cache ─────────────────────────────────────────────────────────────────


def test_osm_cache_roundtrip(tmp_path):
    md = _make_md()
    # Give the instance a file path so _get_osm_cache_path can derive its location
    md.coords_file = str(tmp_path / "test.gpx")

    md._save_osm_cache(_WAYS_JSON)
    loaded = md._load_osm_cache()

    assert loaded == _WAYS_JSON
    # The bbox is part of the hashed query, so it is not stored separately.
    assert "bbox" not in json.loads((tmp_path / "test.osm_cache.json").read_text())


def test_osm_cache_bbox_mismatch_returns_none(tmp_path):
    md = _make_md()
    md.coords_file = str(tmp_path / "test.gpx")

    md._save_osm_cache(_EMPTY_JSON)

    # A moved bounding box changes the query, hence its hash.
    md.min_lat += 1.0
    result = md._load_osm_cache()
    assert result is None


def test_osm_cache_query_hash_mismatch_returns_none(tmp_path, monkeypatch):
    md = _make_md()
    md.coords_file = str(tmp_path / "test.gpx")
    md._save_osm_cache(_WAYS_JSON)
    monkeypatch.setattr(md, "_osm_query_hash", lambda: "changed")

    assert md._load_osm_cache() is None


def test_osm_cache_stale_returns_none(tmp_path, monkeypatch):
    md = _make_md()
    md.coords_file = str(tmp_path / "test.gpx")
    md._save_osm_cache(_WAYS_JSON)
    path = tmp_path / "test.osm_cache.json"
    data = json.loads(path.read_text())
    data["created"] -= 31 * 24 * 60 * 60
    path.write_text(json.dumps(data))

    assert md._load_osm_cache() is None


def test_mapdata_bbox_covers_all_utm_corners():
    e, n, zn, zl = utm.from_latlon(_LAT, _LON)
    md = MapData(
        [np.array([[e, n], [e + 100_000, n + 100_000]]), int(zn), zl],
        coords_type="array",
    )

    corners = [
        utm.to_latlon(x, y, md.zone_number, md.zone_letter)
        for x in (md.min_x, md.max_x)
        for y in (md.min_y, md.max_y)
    ]

    assert md.min_lat == pytest.approx(min(lat for lat, _ in corners), abs=1e-8)
    assert md.min_long == pytest.approx(min(lon for _, lon in corners), abs=1e-8)
    assert md.max_lat == pytest.approx(max(lat for lat, _ in corners), abs=1e-8)
    assert md.max_long == pytest.approx(max(lon for _, lon in corners), abs=1e-8)


def test_mapdata_zone_check_ignores_band_letters():
    e, n, zn, _ = utm.from_latlon(_LAT, _LON)
    md = MapData([np.array([[e, n]]), int(zn), "X"], coords_type="array")

    md._check_utm_zone_boundary()


def test_create_mapdata_no_cache_bypasses_the_cache(tmp_path, monkeypatch, mock_overpass_client):
    gpx = tmp_path / "map.gpx"
    gpx.write_text('<gpx><wpt lat="50.0" lon="14.0"/><wpt lat="50.001" lon="14.001"/></gpx>')
    (tmp_path / "map.osm_cache.json").write_text("{}")

    process_map_data(str(gpx), download=True, use_cache=False)

    mock_overpass_client.query_raw.assert_called_once()


def test_create_mapdata_resaves_to_the_loaded_file(tmp_path):
    path = tmp_path / "map.mapdata"
    _make_md().save(str(path))  # no source GPX: coords_file is None
    data = json.loads(path.read_text())
    data["stale"] = True
    path.write_text(json.dumps(data))

    process_map_data(str(path), download=False)

    assert "stale" not in json.loads(path.read_text())


def test_create_mapdata_finds_a_bare_name_in_the_demo_dir(
    tmp_path, monkeypatch, mock_overpass_client
):
    (tmp_path / "data").mkdir()
    (tmp_path / "demo").mkdir()
    gpx = tmp_path / "demo" / "map.gpx"
    gpx.write_text('<gpx><wpt lat="50.0" lon="14.0"/><wpt lat="50.001" lon="14.001"/></gpx>')
    monkeypatch.setattr("map_data.create_mapdata.package_share", lambda d: tmp_path / d)
    monkeypatch.chdir(tmp_path)

    process_map_data("map.gpx", download=True)

    assert (tmp_path / "demo" / "map.mapdata").is_file()


def test_save_osm_cache_failure_preserves_existing_cache(tmp_path, monkeypatch):
    md = _make_md()
    md.coords_file = str(tmp_path / "test.gpx")
    md._save_osm_cache(_WAYS_JSON)

    monkeypatch.setattr("map_data.utils.serialization.json.dumps", _broken_json_dumps)
    # _save_osm_cache logs a warning on OSError instead of raising.
    md._save_osm_cache(_EMPTY_JSON)
    monkeypatch.undo()

    # The existing cache is untouched and no temp files remain.
    assert md._load_osm_cache() == _WAYS_JSON
    assert not list(tmp_path.glob("*.tmp"))


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
