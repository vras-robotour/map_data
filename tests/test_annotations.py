"""Tests for the headless annotation merge (``map_data.annotations``)."""

import copy
import json
from pathlib import Path

import pytest
import utm
from shapely.geometry import LineString

from map_data.annotations import (
    NO_ANNOTATIONS,
    annotation_path_for,
    apply_store,
    load_mapdata_with_annotations,
)
from map_data.map_data import MapData
from map_data.pathsolver.graph_planner import GraphPlanner
from map_data.pathsolver.route import RoutePlanningError, plan_route
from map_data.traversability import TraversabilityRules, load_traversability
from map_data.utils.way import Way
from map_data.viewer.helpers import load_annotations


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
    # Its own synthetic nodes, joined to the network by the nodes it ends on.
    assert len(ann) == 1 and ann[0].nodes[0] == 103 and ann[0].nodes[-1] == 201
    assert all(n < 0 for n in ann[0].nodes[1:-1])
    assert any(w.tags.get("type") == "annotation_intersection" for w in md.crossroads_list), (
        "the annotated path should create crossroads where it touches the network"
    )

    start, goal = _latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 450.0, 0.0)
    res = plan_route(md, [start, goal])
    assert res.length_m == pytest.approx(450.0, abs=10.0)


def test_deleted_way_is_removed(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    annotation_path_for(path).write_text(
        json.dumps({"version": 1, "annotations": [], "deleted_ways": [{"id": 2}]})
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

    removed = md.apply_traversability(TraversabilityRules().extend({"steps"}))

    assert removed == 1
    assert [w.id for w in md.footways_list] == [1, 2, 3]
    # node 103 was a junction only because the stairway ended on it
    assert len(md.crossroads_list) < n_crossroads


def test_exclude_ways_empty_set_is_a_noop(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    _add_stairway(md, lat0, lon0)

    assert md.apply_traversability(TraversabilityRules().extend(())) == 0
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
    # The default rule file also refuses stairs, so both switches have to be off.
    kept = load_mapdata_with_annotations(
        path, exclude_highway=(), traversability=TraversabilityRules()
    )[0]
    assert len(kept.footways_list) == 4


@pytest.mark.skipif(not KRALOVSKA.is_file(), reason="kralovska_obora.mapdata is not in the repo")
def test_exclude_ways_on_the_stromovka_map():
    md = MapData.load(str(KRALOVSKA))
    n_crossroads = len(md.crossroads_list)

    removed = md.apply_traversability(TraversabilityRules().extend({"steps"}))

    assert removed == 15  # the 15 stairways of the Královská obora map
    assert not any(w.tags.get("highway") == "steps" for w in md.footways_list)
    assert len(md.crossroads_list) < n_crossroads


# ── traversability rules ───────────────────────────────────────────────────


def _add_grass_way(md, lat0, lon0):
    """
    A ``surface=grass`` footway (107 -> 103 -> 108) crossing way 1 at its end
    node 103, which turns that node into a crossroad.
    """
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    for nid, dy in ((107, -60.0), (108, 60.0)):
        lat, lon = utm.to_latlon(e0 + 200.0, n0 + dy, zn, zl)
        md.nodes_cache[nid] = {"lat": lat, "lon": lon, "tags": {}}
    md.footways_list.append(
        Way(
            id=5,
            nodes=[107, 103, 108],
            tags={"highway": "footway", "surface": "grass"},
            line=LineString(
                [(e0 + 200.0, n0 - 60.0), (e0 + 200.0, n0), (e0 + 200.0, n0 + 60.0)]
            ).buffer(1.0),
        )
    )
    md.crossroads_list = md.parse_intersections({w.id: w for w in md.footways_list})


def _grass_rules():
    return TraversabilityRules.from_dict(
        {"rules": [{"match": {"surface": "grass"}, "traversable": False, "reason": "soft"}]},
        source="test",
    )


def test_apply_traversability_removes_a_grass_way(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    _add_grass_way(md, lat0, lon0)

    removed = md.apply_traversability(_grass_rules())

    assert removed == 1
    assert [w.id for w in md.footways_list] == [1, 2, 3]
    assert md.traversability_removed == {"soft": 1}


def test_apply_traversability_recomputes_the_crossroads(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    plain_crossroads = len(md.crossroads_list)
    _add_grass_way(md, lat0, lon0)
    assert len(md.crossroads_list) > plain_crossroads  # node 103 became a junction

    md.apply_traversability(_grass_rules())

    assert len(md.crossroads_list) == plain_crossroads


def test_apply_traversability_without_rules_is_a_noop(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    _add_grass_way(md, lat0, lon0)

    assert md.apply_traversability(TraversabilityRules()) == 0
    assert len(md.footways_list) == 4


def test_loader_applies_the_rules_it_is_given(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    _add_grass_way(md, lat0, lon0)
    md.save(str(path))

    kept = load_mapdata_with_annotations(path, traversability=TraversabilityRules())[0]
    filtered = load_mapdata_with_annotations(path, traversability=_grass_rules())[0]

    assert len(kept.footways_list) == 4
    assert [w.id for w in filtered.footways_list] == [1, 2, 3]


def test_loader_reads_a_rule_file(footway_network_mapdata, tmp_path):
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    _add_grass_way(md, lat0, lon0)
    md.save(str(path))
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        "rules:\n  - match: {surface: grass}\n    traversable: false\n    reason: soft\n"
    )

    md, _ = load_mapdata_with_annotations(path, traversability=rules_file)

    assert [w.id for w in md.footways_list] == [1, 2, 3]


def test_loader_folds_exclude_highway_into_the_rules(footway_network_mapdata):
    """Both switches are applied at once, whichever way the caller uses."""
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    _add_stairway(md, lat0, lon0)
    _add_grass_way(md, lat0, lon0)
    md.save(str(path))

    md, _ = load_mapdata_with_annotations(
        path, exclude_highway=("steps",), traversability=_grass_rules()
    )

    assert [w.id for w in md.footways_list] == [1, 2, 3]


def _plan_with_store(path, lat0, lon0, store, start, goal):
    annotation_path_for(path).write_text(json.dumps({"version": 1, "annotations": [], **store}))
    md, _ = load_mapdata_with_annotations(path)
    return plan_route(md, [_latlon(lat0, lon0, *start), _latlon(lat0, lon0, *goal)])


def test_moved_node_is_where_the_planner_routes(footway_network_mapdata):
    """Node 103 dragged from (200,0) to (200,60): the route has to reach it there."""
    path, lat0, lon0 = footway_network_mapdata
    lat, lon = _latlon(lat0, lon0, 200.0, 60.0)
    store = {"node_position_overrides": {"1": {"103": {"lat": lat, "lon": lon}}}}
    res = _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (200.0, 60.0))
    assert res.length_m == pytest.approx(100.0 + (100.0**2 + 60.0**2) ** 0.5, abs=5.0)


def test_added_node_is_routable(footway_network_mapdata):
    """A node inserted after 101 at (50,30) bends way 1 through it on the way to 102."""
    path, lat0, lon0 = footway_network_mapdata
    lat, lon = _latlon(lat0, lon0, 50.0, 30.0)
    store = {"added_nodes": [{"id": -1, "way_id": 1, "after_node_id": 101, "lat": lat, "lon": lon}]}
    res = _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (200.0, 0.0))
    assert res.length_m == pytest.approx(2 * (50.0**2 + 30.0**2) ** 0.5 + 100.0, abs=5.0)


def test_detached_split_disconnects_the_segments(footway_network_mapdata):
    """Way 1 split at 102 with the far end detached: 103 is no longer reachable from 101."""
    path, lat0, lon0 = footway_network_mapdata
    split = {"split_ways": {"1": [102]}}
    detached = {**split, "detached_nodes": [{"way_id": 1, "node_id": 102, "id": -1}]}

    res = _plan_with_store(path, lat0, lon0, split, (0.0, 0.0), (200.0, 0.0))
    assert res.length_m == pytest.approx(200.0, abs=5.0), "a plain split stays connected"

    with pytest.raises(RoutePlanningError):
        _plan_with_store(path, lat0, lon0, detached, (0.0, 0.0), (200.0, 0.0))


def test_detached_end_moves_on_its_own(footway_network_mapdata):
    """Dragging the detached copy of 102 to (130,0) leaves 102 itself where it was."""
    path, lat0, lon0 = footway_network_mapdata
    lat, lon = _latlon(lat0, lon0, 130.0, 0.0)
    store = {
        "split_ways": {"1": [102]},
        "detached_nodes": [{"way_id": 1, "node_id": 102, "id": -1}],
        "node_position_overrides": {"1": {"-1": {"lat": lat, "lon": lon}}},
    }
    res = _plan_with_store(path, lat0, lon0, store, (200.0, 0.0), (130.0, 0.0))
    assert res.length_m == pytest.approx(70.0, abs=5.0)
    res = _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (100.0, 100.0))
    assert res.length_m == pytest.approx(200.0, abs=5.0)


def test_moving_a_shared_node_in_one_way_detaches_that_way(footway_network_mapdata):
    """Junction 102 dragged to (100,40) in way 2 only: way 1 stays put, the junction is gone."""
    path, lat0, lon0 = footway_network_mapdata
    lat, lon = _latlon(lat0, lon0, 100.0, 40.0)
    store = {"node_position_overrides": {"2": {"102": {"lat": lat, "lon": lon}}}}
    res = _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (200.0, 0.0))
    assert res.length_m == pytest.approx(200.0, abs=5.0), "way 1 must not follow way 2's move"

    md, _ = load_mapdata_with_annotations(path)
    way1, way2 = md.footways_list[:2]
    assert way1.nodes == [101, 102, 103]
    assert way2.nodes[0] < 0 and way2.nodes[1] == 104
    assert md.nodes_cache[way2.nodes[0]]["lat"] == pytest.approx(lat)
    assert md.crossroads_list == []
    with pytest.raises(RoutePlanningError):
        plan_route(
            md,
            [_latlon(lat0, lon0, 0.0, 0.0), _latlon(lat0, lon0, 100.0, 100.0)],
            max_snap_distance=5.0,
        )


def test_shared_node_moved_in_both_ways_honours_both(footway_network_mapdata):
    """102 moved to (100,-30) in way 1 and (100,40) in way 2: neither override is dropped."""
    path, lat0, lon0 = footway_network_mapdata
    p1, p2 = _latlon(lat0, lon0, 100.0, -30.0), _latlon(lat0, lon0, 100.0, 40.0)
    store = {
        "node_position_overrides": {
            "1": {"102": {"lat": p1[0], "lon": p1[1]}},
            "2": {"102": {"lat": p2[0], "lon": p2[1]}},
        }
    }
    res = _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (200.0, 0.0))
    assert res.length_m == pytest.approx(2 * (100.0**2 + 30.0**2) ** 0.5, abs=5.0)
    res = _plan_with_store(path, lat0, lon0, store, (100.0, 40.0), (100.0, 100.0))
    assert res.length_m == pytest.approx(60.0, abs=5.0)


def test_shared_node_moved_to_one_spot_in_every_way_stays_a_junction(footway_network_mapdata):
    """102 moved to (100,30) in way 1 and in way 2: the junction moved, it was not pulled apart."""
    path, lat0, lon0 = footway_network_mapdata
    lat, lon = _latlon(lat0, lon0, 100.0, 30.0)
    pos = {"102": {"lat": lat, "lon": lon}}
    store = {"node_position_overrides": {"1": pos, "2": pos}}
    res = _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (100.0, 100.0))
    assert res.length_m == pytest.approx((100.0**2 + 30.0**2) ** 0.5 + 70.0, abs=5.0)
    md, _ = load_mapdata_with_annotations(path)
    assert md.footways_list[0].nodes == [101, 102, 103] and md.footways_list[1].nodes == [102, 104]


def test_a_move_in_one_way_does_not_redraw_the_others(footway_network_mapdata):
    """
    102 moved in way 2 while way 1 loses node 103: way 1 is rebuilt, and must be rebuilt
    through 102 where way 1 has it, not where way 2 put it.
    """
    path, lat0, lon0 = footway_network_mapdata
    lat, lon = _latlon(lat0, lon0, 100.0, 40.0)
    store = {
        "node_position_overrides": {"2": {"102": {"lat": lat, "lon": lon}}},
        "deleted_nodes": [{"way_id": 1, "node_id": 103}],
    }
    _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (100.0, 0.0))
    md, _ = load_mapdata_with_annotations(path)
    e0, n0, _, _ = utm.from_latlon(lat0, lon0)
    assert md.footways_list[0].line.bounds[3] - n0 < 2.0, "way 1 bent towards way 2's move"


def test_a_move_recorded_on_a_deleted_way_moves_nothing(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    lat, lon = _latlon(lat0, lon0, 100.0, 40.0)
    store = {
        "node_position_overrides": {"2": {"102": {"lat": lat, "lon": lon}}},
        "deleted_ways": [{"id": 2}],
    }
    res = _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (200.0, 0.0))
    assert res.length_m == pytest.approx(200.0, abs=1.0)


@pytest.mark.parametrize(
    ("way", "node", "to", "start", "goal", "length", "joined_at"),
    [
        # free end of way 2 dropped on the middle of way 3's edge: a new node on that edge
        ("2", 104, (450.0, 0.0), (100.0, 0.0), (500.0, 0.0), 400.0, None),
        # free end of way 1 dropped 1 m from node 201: joined at that node
        ("1", 103, (399.0, 0.0), (0.0, 0.0), (500.0, 0.0), 500.0, 201),
        # junction end of way 2 slid along way 1: detached from 102, joined again at (150,0)
        ("2", 102, (150.0, 0.0), (0.0, 0.0), (100.0, 100.0), 150.0 + 111.8, None),
    ],
)
def test_moved_end_node_joins_the_way_it_is_dropped_on(
    footway_network_mapdata, way, node, to, start, goal, length, joined_at
):
    path, lat0, lon0 = footway_network_mapdata
    lat, lon = _latlon(lat0, lon0, *to)
    store = {"node_position_overrides": {way: {str(node): {"lat": lat, "lon": lon}}}}
    res = _plan_with_store(path, lat0, lon0, store, start, goal)
    assert res.length_m == pytest.approx(length, abs=5.0)

    md, _ = load_mapdata_with_annotations(path)
    moved = next(w for w in md.footways_list if str(w.id) == way)
    junction = moved.nodes[0] if node == 102 else moved.nodes[-1]
    assert junction == joined_at if joined_at else junction < 0
    assert sum(junction in w.nodes for w in md.footways_list) == 2, "a node of both ways"
    # a real node: the junction is still there after an export
    exported = path.with_name("exported.mapdata")
    md.save(str(exported))
    md2, _ = load_mapdata_with_annotations(exported, NO_ANNOTATIONS)
    res = plan_route(md2, [_latlon(lat0, lon0, *start), _latlon(lat0, lon0, *goal)])
    assert res.length_m == pytest.approx(length, abs=5.0)


def test_moved_end_node_out_of_reach_stays_loose(footway_network_mapdata):
    """6 m from way 3 is not on it (JOIN_DISTANCE_M is 5)."""
    path, lat0, lon0 = footway_network_mapdata
    lat, lon = _latlon(lat0, lon0, 450.0, 6.0)
    store = {"node_position_overrides": {"2": {"104": {"lat": lat, "lon": lon}}}}
    with pytest.raises(RoutePlanningError):
        _plan_with_store(path, lat0, lon0, store, (100.0, 0.0), (500.0, 0.0))


def _path_annotation(lat0, lon0, points, ann_id="ann-1"):
    coords = [list(_latlon(lat0, lon0, *p))[::-1] for p in points]
    return {
        "id": ann_id,
        "type": "path",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"highway": "footway"},
    }


def test_drawn_path_joins_the_way_it_crosses(footway_network_mapdata):
    """Drawn across way 1 at (150,0), both ends 50 m away: the crossing is the junction."""
    path, lat0, lon0 = footway_network_mapdata
    store = {"annotations": [_path_annotation(lat0, lon0, [(150.0, -50.0), (150.0, 50.0)])]}
    res = _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (150.0, 50.0))
    assert res.length_m == pytest.approx(200.0, abs=5.0)
    md, _ = load_mapdata_with_annotations(path)
    assert len(md.crossroads_list) == 2, "the crossing is listed once, not per detector"
    planner = GraphPlanner(md)
    assert not [n for n in planner.graph if n <= -2000000], "nothing left for the planner to stitch"


def test_way_ending_on_a_drawn_path_joins_it(footway_network_mapdata):
    """Way 1 ends at (200,0), in the middle of a path drawn from (200,-50) to (200,50)."""
    path, lat0, lon0 = footway_network_mapdata
    store = {"annotations": [_path_annotation(lat0, lon0, [(200.0, -50.0), (200.0, 50.0)])]}
    res = _plan_with_store(path, lat0, lon0, store, (0.0, 0.0), (200.0, 50.0))
    assert res.length_m == pytest.approx(250.0, abs=5.0)


def test_drawn_path_baked_into_an_export_survives_a_split_and_a_new_path(footway_network_mapdata):
    """
    On an exported map the drawn path is a way of its own: splitting it must not cut it
    off the network, and a path drawn afterwards must not reuse its id.
    """
    path, lat0, lon0 = footway_network_mapdata
    bridge = _path_annotation(lat0, lon0, [(200.0, 0.0), (300.0, 0.0), (400.0, 0.0)])
    _plan_with_store(path, lat0, lon0, {"annotations": [bridge]}, (0.0, 0.0), (500.0, 0.0))
    md, _ = load_mapdata_with_annotations(path)
    exported = path.with_name("exported.mapdata")
    md.save(str(exported))
    baked = next(w for w in md.footways_list if w.id == -1)

    store = {
        "split_ways": {"-1": [baked.nodes[2]]},
        "annotations": [_path_annotation(lat0, lon0, [(500.0, 0.0), (500.0, 50.0)], "ann-2")],
    }
    res = _plan_with_store(exported, lat0, lon0, store, (0.0, 0.0), (500.0, 50.0))
    assert res.length_m == pytest.approx(550.0, abs=5.0)
    md2, _ = load_mapdata_with_annotations(exported)
    assert sorted(str(w.id) for w in md2.footways_list if str(w.id).startswith("-")) == [
        "-1:0",
        "-1:1",
        "-2",
    ]


@pytest.mark.skipif(not KRALOVSKA.is_file(), reason="kralovska_obora.mapdata is not in the repo")
def test_shipped_rules_on_the_stromovka_map():
    """
    The counts the comments in ``config/traversability.yaml`` quote. They are
    what the operator sees in the node log, so a rule that silently stops
    matching (an OSM retag, a typo) shows up here.
    """
    md = MapData.load(str(KRALOVSKA))
    rules = load_traversability()

    counts: dict[str, int] = {}
    for way in md.footways_list + md.roads_list:
        verdict = rules.evaluate(way.tags)
        if not verdict.traversable:
            counts[verdict.reason] = counts.get(verdict.reason, 0) + 1

    assert counts == {"stairs": 15, "soft surface": 6, "bridge": 13, "rough surface": 1}


#: Start/goal of the two routes the robot drove in Stromovka on 2026-09-08
#: (first and last point of field_sync/2026-09-08/missions/route_*.gpx).
FIELD_ROUTES = {
    "route_20260908-103357": (
        (50.104521772138874, 14.428430322399514),
        (50.10675190261408, 14.425881099968743),
    ),
    "route_20260908-105547": (
        (50.10674062206266, 14.42576494185599),
        (50.109685402615185, 14.417962599968284),
    ),
}
#: A rule that lengthens a driven route by more than this is not shipped enabled.
MAX_DETOUR_FACTOR = 1.3


def _plan_field_routes(rules, **kwargs):
    """``{name: length_m}`` for :data:`FIELD_ROUTES` on the Stromovka map."""
    md, _ = load_mapdata_with_annotations(KRALOVSKA, NO_ANNOTATIONS, traversability=rules)
    return {
        name: plan_route(
            md,
            [start, goal],
            keep_start=True,
            keep_goal=True,
            spacing=3.0,
            traversability=rules,
            **kwargs,
        ).length_m
        for name, (start, goal) in FIELD_ROUTES.items()
    }


@pytest.mark.skipif(not KRALOVSKA.is_file(), reason="kralovska_obora.mapdata is not in the repo")
def test_shipped_rules_keep_the_driven_routes_plannable():
    """
    The defaults must not cut Stromovka in two. Both 2026-09-08 routes have to
    plan, and neither may grow by more than a third against the plain shortest
    path — the reason the ``tunnel`` rule is shipped commented out.
    """
    plain = _plan_field_routes(TraversabilityRules(), highway_costs={}, surface_costs={})
    shipped = _plan_field_routes(load_traversability())

    for name, length in shipped.items():
        assert length > 0
        assert length <= MAX_DETOUR_FACTOR * plain[name], (
            f"{name}: {length:.0f} m with the shipped rules, {plain[name]:.0f} m without"
        )


# ── apply_store copies, it does not deep-copy ──────────────────────────────

#: The full Stromovka map with the robot's real annotation store next to it.
STROMOVKA = Path(__file__).resolve().parents[1] / "data" / "stromovka.mapdata"


def _snapshot(md):
    """Everything about *md* the annotation passes could plausibly clobber."""
    return (
        [str(w.id) for w in md.roads_list],
        [str(w.id) for w in md.footways_list],
        [str(w.id) for w in md.barriers_list],
        [dict(w.tags or {}) for w in md.footways_list + md.roads_list + md.barriers_list],
        [len(w.nodes) for w in md.footways_list + md.roads_list],
        sorted(md.nodes_cache),
        len(md.crossroads_list),
    )


@pytest.mark.skipif(not STROMOVKA.is_file(), reason="stromovka.mapdata is not in the repo")
def test_apply_store_leaves_the_original_map_untouched():
    """
    ``apply_store`` shallow-copies (deep-copying a whole map cost ~370 ms of
    every load), so the caller's ``MapData`` and its ``Way`` objects must come
    out of a merge exactly as they went in.
    """
    md = MapData.load(str(STROMOVKA))
    store = load_annotations(str(annotation_path_for(STROMOVKA)))
    assert store["deleted_ways"], "expected the shipped store to delete ways"
    # The shipped store has no tag overrides; retag a surviving way so that pass runs too.
    deleted = {str(d["id"]) for d in store["deleted_ways"]}
    victim = next(w for w in md.footways_list if str(w.id) not in deleted)
    store["tag_overrides"] = {str(victim.id): {"highway": "steps", "surface": "gravel"}}

    before = copy.deepcopy(_snapshot(md))
    victim_tags = copy.deepcopy(victim.tags)

    merged = apply_store(md, store)

    assert _snapshot(md) == before
    assert any(w is victim for w in md.footways_list)
    assert victim.tags == victim_tags

    # ... and the merge did happen.
    assert len(merged.footways_list) < len(md.footways_list)
    overridden = [
        w
        for w in merged.footways_list + merged.roads_list
        if str(w.id).split(":")[0] == str(victim.id)
    ]
    assert overridden and all(w.tags["surface"] == "gravel" for w in overridden)
    assert sorted(merged.nodes_cache) != before[5]  # annotation/added nodes

    # No state leaks into the original, so a second merge gives the same map.
    assert _snapshot(apply_store(md, store)) == _snapshot(merged)
