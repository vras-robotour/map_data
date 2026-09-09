"""Tests for the headless annotation merge (``map_data.annotations``)."""

import json
from pathlib import Path

import pytest
import utm
from shapely.geometry import LineString

from map_data.annotations import (
    annotation_path_for,
    load_mapdata_with_annotations,
)
from map_data.map_data import MapData
from map_data.pathsolver.route import RoutePlanningError, plan_route
from map_data.utils.way import Way


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


# ── excluded way types ─────────────────────────────────────────────────────

#: The Stromovka map used on the robot; not committed (data/*.mapdata is ignored).
KRALOVSKA = Path(__file__).resolve().parents[1] / "data" / "kralovska_obora.mapdata"


def _add_stairway(md, lat0, lon0):
    """
    A ``highway=steps`` way (105 -> 101 -> 106) crossing way 1 at its end node
    101, which turns that node into a crossroad.
    """
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    for nid, dy in ((105, -50.0), (106, 50.0)):
        lat, lon = utm.to_latlon(e0, n0 + dy, zn, zl)
        md.nodes_cache[nid] = {"lat": lat, "lon": lon, "tags": {}}
    md.footways_list.append(
        Way(
            id=4,
            nodes=[105, 101, 106],
            tags={"highway": "steps"},
            line=LineString([(e0, n0 - 50.0), (e0, n0), (e0, n0 + 50.0)]).buffer(1.0),
        )
    )
    md.crossroads_list = md.parse_intersections({w.id: w for w in md.footways_list})


def test_exclude_ways_removes_stairs_and_their_crossroads(footway_network_mapdata, tmp_path):
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    _add_stairway(md, lat0, lon0)
    n_crossroads = len(md.crossroads_list)

    removed = md.exclude_ways({"steps"})

    assert removed == 1
    assert [w.id for w in md.footways_list] == [1, 2, 3]
    # node 103 was a junction only because the stairway ended on it
    assert len(md.crossroads_list) < n_crossroads


def test_exclude_ways_empty_set_is_a_noop(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    _add_stairway(md, lat0, lon0)

    assert md.exclude_ways(()) == 0
    assert len(md.footways_list) == 4


def test_loader_excludes_stairs_by_default(footway_network_mapdata):
    """
    ``MapData.load`` stays raw (the viewer must still show the stairs); the
    planner's loader drops them.
    """
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    _add_stairway(md, lat0, lon0)
    md.save(str(path))

    assert len(MapData.load(str(path)).footways_list) == 4
    assert len(load_mapdata_with_annotations(path)[0].footways_list) == 3
    assert len(load_mapdata_with_annotations(path, exclude_highway=())[0].footways_list) == 4


@pytest.mark.skipif(not KRALOVSKA.is_file(), reason="kralovska_obora.mapdata is not in the repo")
def test_exclude_ways_on_the_stromovka_map():
    md = MapData.load(str(KRALOVSKA))
    n_crossroads = len(md.crossroads_list)

    removed = md.exclude_ways({"steps"})

    assert removed == 15  # the 15 stairways of the Královská obora map
    assert not any(w.tags.get("highway") == "steps" for w in md.footways_list)
    assert len(md.crossroads_list) < n_crossroads
