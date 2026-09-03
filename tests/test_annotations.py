"""Tests for the headless annotation merge (``map_data.annotations``)."""

import json

import pytest
import utm

from map_data.annotations import (
    annotation_path_for,
    load_mapdata_with_annotations,
)
from map_data.map_data import MapData
from map_data.pathsolver.route import RoutePlanningError, plan_route


def _latlon(lat0, lon0, dx, dy):
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    return utm.to_latlon(e0 + dx, n0 + dy, zn, zl)


def test_annotation_path_for():
    assert annotation_path_for("/x/y/stromovka.mapdata").name == "stromovka.annotations.json"


def test_load_without_store_is_plain_map(footway_network_mapdata):
    path, _, _ = footway_network_mapdata
    md, store = load_mapdata_with_annotations(path)
    assert isinstance(md, MapData)
    assert store == {"version": 1, "annotations": []}
    assert len(md.footways_list) == 3


def test_annotated_path_bridges_a_gap(footway_network_mapdata):
    """
    A drawn path from node 103 (200,0) to node 201 (400,0) connects the two
    components, so a graph route that was unreachable becomes plannable and
    a crossroad appears where the annotation meets the network.
    """
    path, lat0, lon0 = footway_network_mapdata
    a = _latlon(lat0, lon0, 200.0, 0.0)
    b = _latlon(lat0, lon0, 400.0, 0.0)
    store = {
        "version": 1,
        "annotations": [
            {
                "id": "ann-1",
                "type": "path",
                "geometry": {"type": "LineString", "coordinates": [[a[1], a[0]], [b[1], b[0]]]},
                "properties": {"highway": "footway", "width": 2.0},
            }
        ],
    }
    annotation_path_for(path).write_text(json.dumps(store))

    md, loaded = load_mapdata_with_annotations(path)
    assert loaded["annotations"][0]["id"] == "ann-1"
    assert len(md.footways_list) == 4
    ann = [w for w in md.footways_list if isinstance(w.id, int) and w.id < 0]
    assert len(ann) == 1 and all(n < 0 for n in ann[0].nodes)
    assert any(w.tags.get("type") == "annotation_intersection" for w in md.crossroads_list), (
        "the annotated path should create crossroads where it touches the network"
    )

    start, goal = _latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 450.0, 0.0)
    res = plan_route(md, [start, goal])
    assert res.length_m == pytest.approx(450.0, abs=10.0)


def test_deleted_way_is_removed(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    annotation_path_for(path).write_text(
        json.dumps({"version": 1, "annotations": [], "deleted_ways": [2]})
    )
    md, _ = load_mapdata_with_annotations(path)
    assert [w.id for w in md.footways_list] == [1, 3]
    with pytest.raises(RoutePlanningError) as e:
        plan_route(
            md,
            [_latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 100.0, 95.0)],
            max_snap_distance=50.0,
        )
    assert e.value.reason == "snap_too_far"  # the branch to node 104 is gone


def test_tag_override_moves_a_footway_to_roads(footway_network_mapdata):
    path, _, _ = footway_network_mapdata
    annotation_path_for(path).write_text(
        json.dumps(
            {"version": 1, "annotations": [], "tag_overrides": {"3": {"highway": "residential"}}}
        )
    )
    md, _ = load_mapdata_with_annotations(path)
    assert [w.id for w in md.footways_list] == [1, 2]
    assert [w.id for w in md.roads_list] == [3]
