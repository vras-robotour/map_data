import types

import numpy as np
import pytest
import utm
from shapely.geometry import LineString, MultiPolygon, Polygon

from map_data.utils.way import Way
from map_data.viewer.helpers import (
    apply_node_position_overrides,
    geojson_geom_to_utm,
    geom_to_geojson,
    get_deleted_way_ids,
    load_annotations,
    mapdata_to_geojson,
    rebuild_way_without_nodes,
    save_annotations,
    split_way,
)

# Prague area — UTM zone 33U
_ZN, _ZL = 33, "U"
_E0, _N0 = 458000.0, 5550500.0


def _make_way(id_, nodes, coords):
    return Way(id=id_, nodes=list(nodes), line=LineString(coords), tags={}, in_out="")


# ── GeoJSON conversion ────────────────────────────────────────────────────────


def test_geom_to_geojson_polygon():
    poly = Polygon([(_E0, _N0), (_E0 + 10, _N0), (_E0 + 10, _N0 + 10), (_E0, _N0 + 10), (_E0, _N0)])
    result = geom_to_geojson(poly, _ZN, _ZL)
    assert result["type"] == "Polygon"
    ring = result["coordinates"][0]
    assert all(len(pt) == 2 for pt in ring)
    # [lon, lat] — longitudes near 14°E, latitudes near 50°N
    assert all(13.0 < pt[0] < 16.0 for pt in ring)
    assert all(49.0 < pt[1] < 51.0 for pt in ring)


def test_geom_to_geojson_multipolygon():
    p1 = Polygon([(_E0, _N0), (_E0 + 5, _N0), (_E0 + 5, _N0 + 5), (_E0, _N0 + 5), (_E0, _N0)])
    p2 = Polygon([(_E0 + 20, _N0), (_E0 + 25, _N0), (_E0 + 25, _N0 + 5), (_E0 + 20, _N0 + 5), (_E0 + 20, _N0)])
    mp = MultiPolygon([p1, p2])
    result = geom_to_geojson(mp, _ZN, _ZL)
    assert result["type"] == "MultiPolygon"
    assert len(result["coordinates"]) == 2


def test_geom_to_geojson_linestring():
    ls = LineString([(_E0, _N0), (_E0 + 50, _N0 + 50)])
    result = geom_to_geojson(ls, _ZN, _ZL)
    assert result["type"] == "LineString"
    assert len(result["coordinates"]) == 2


def test_geojson_geom_to_utm_roundtrip():
    poly = Polygon([(_E0, _N0), (_E0 + 10, _N0), (_E0 + 10, _N0 + 10), (_E0, _N0 + 10), (_E0, _N0)])
    geojson = geom_to_geojson(poly, _ZN, _ZL)
    reconstructed = geojson_geom_to_utm(geojson, _ZN, _ZL)
    assert reconstructed is not None
    assert reconstructed.geom_type == "Polygon"
    assert poly.equals_exact(reconstructed, tolerance=2.0)


# ── Annotation I/O ────────────────────────────────────────────────────────────


def test_load_annotations_missing_file(tmp_path):
    result = load_annotations(str(tmp_path / "nonexistent.json"))
    assert result == {"version": 1, "annotations": []}


def test_save_and_load_annotations_roundtrip(tmp_path):
    path = str(tmp_path / "ann.json")
    store = {
        "version": 1,
        "annotations": [{"id": "abc", "type": "obstacle", "geometry": {}, "properties": {}}],
    }
    save_annotations(path, store)
    loaded = load_annotations(path)
    assert loaded == store


# ── get_deleted_way_ids ───────────────────────────────────────────────────────


def test_get_deleted_way_ids_dict_format():
    store = {"deleted_ways": [{"id": 5, "category": "barrier", "label": ""}]}
    assert get_deleted_way_ids(store) == {5}


def test_get_deleted_way_ids_int_format():
    store = {"deleted_ways": [5]}
    assert get_deleted_way_ids(store) == {5}


# ── split_way ─────────────────────────────────────────────────────────────────


def test_split_way_linestring_basic():
    coords = [
        (_E0, _N0), (_E0 + 25, _N0 + 25), (_E0 + 50, _N0 + 50),
        (_E0 + 75, _N0 + 75), (_E0 + 100, _N0 + 100),
    ]
    way = Way(id=42, nodes=[1, 2, 3, 4, 5], line=LineString(coords), tags={}, in_out="")
    segments = split_way(way, [3])
    assert len(segments) == 2
    assert segments[0].id == "42:0"
    assert segments[1].id == "42:1"
    seg0_nids = [getattr(n, "id", n) for n in segments[0].nodes]
    seg1_nids = [getattr(n, "id", n) for n in segments[1].nodes]
    assert 1 in seg0_nids and 3 in seg0_nids
    assert 3 in seg1_nids and 5 in seg1_nids


def test_split_way_returns_original_if_no_split_nids():
    way = _make_way(10, [1, 2, 3], [(_E0, _N0), (_E0 + 25, _N0), (_E0 + 50, _N0)])
    result = split_way(way, [])
    assert result == [way]


# ── rebuild_way_without_nodes ─────────────────────────────────────────────────


def test_rebuild_way_without_nodes_basic():
    coords = [(_E0, _N0), (_E0 + 25, _N0 + 25), (_E0 + 50, _N0 + 50), (_E0 + 75, _N0 + 75)]
    way = _make_way(99, [1, 2, 3, 4], coords)
    result = rebuild_way_without_nodes(way, {2})
    assert result is not None
    result_nids = [getattr(n, "id", n) for n in result.nodes]
    assert 2 not in result_nids
    assert len(result.nodes) == 3


def test_rebuild_way_without_nodes_too_few_nodes():
    way = _make_way(7, [1, 2], [(_E0, _N0), (_E0 + 10, _N0 + 10)])
    result = rebuild_way_without_nodes(way, {2})
    assert result is None


# ── apply_node_position_overrides ─────────────────────────────────────────────


def test_apply_node_position_overrides_linestring():
    p1 = (_E0, _N0)
    p2 = (_E0 + 100, _N0 + 100)
    way = _make_way(55, [1, 2], [p1, p2])

    new_lat, new_lon = utm.to_latlon(_E0 + 50, _N0 + 50, _ZN, _ZL)
    lat1, lon1 = utm.to_latlon(_E0, _N0, _ZN, _ZL)
    lat2, lon2 = utm.to_latlon(_E0 + 100, _N0 + 100, _ZN, _ZL)
    nodes_cache = {
        1: {"lat": lat1, "lon": lon1, "tags": {}},
        2: {"lat": lat2, "lon": lon2, "tags": {}},
    }
    overrides = {1: {"lat": new_lat, "lon": new_lon}}

    result = apply_node_position_overrides(way, overrides, _ZN, _ZL, nodes_cache)
    assert result is not None
    new_coords = list(result.line.coords)
    assert abs(new_coords[0][0] - (_E0 + 50)) < 1.0
    assert abs(new_coords[0][1] - (_N0 + 50)) < 1.0


# ── mapdata_to_geojson ────────────────────────────────────────────────────────


def test_mapdata_to_geojson_structure():
    road = Way(
        id=1, nodes=[10, 11], tags={"highway": "primary"},
        line=LineString([(_E0, _N0), (_E0 + 100, _N0)]).buffer(3.5),
        in_out="",
    )
    footway = Way(
        id=2, nodes=[12, 13], tags={"highway": "footway"},
        line=LineString([(_E0, _N0 + 50), (_E0 + 100, _N0 + 50)]).buffer(1.5),
        in_out="",
    )
    md = types.SimpleNamespace(
        zone_number=_ZN,
        zone_letter=_ZL,
        roads_list=[road],
        footways_list=[footway],
        barriers_list=[],
        crossroads_list=[],
        waypoints=np.array([[_E0, _N0]]),
    )
    fc = mapdata_to_geojson(md)
    assert fc["type"] == "FeatureCollection"
    categories = {f["properties"]["category"] for f in fc["features"]}
    assert "road" in categories
    assert "footway" in categories
    assert "waypoint" in categories
