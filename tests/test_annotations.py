"""Tests for the headless annotation merge (``map_data.annotations``)."""

import json
from pathlib import Path

import pytest
import utm
from shapely.geometry import LineString

from map_data.annotations import (
    NO_ANNOTATIONS,
    annotation_path_for,
    load_mapdata_with_annotations,
)
from map_data.map_data import MapData
from map_data.pathsolver.route import RoutePlanningError, plan_route
from map_data.traversability import TraversabilityRules, load_traversability
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
    # The default rule file also refuses stairs, so both switches have to be off.
    kept = load_mapdata_with_annotations(
        path, exclude_highway=(), traversability=TraversabilityRules()
    )[0]
    assert len(kept.footways_list) == 4


@pytest.mark.skipif(not KRALOVSKA.is_file(), reason="kralovska_obora.mapdata is not in the repo")
def test_exclude_ways_on_the_stromovka_map():
    md = MapData.load(str(KRALOVSKA))
    n_crossroads = len(md.crossroads_list)

    removed = md.exclude_ways({"steps"})

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


@pytest.mark.skipif(not KRALOVSKA.is_file(), reason="kralovska_obora.mapdata is not in the repo")
def test_shipped_rules_on_the_stromovka_map():
    """
    The counts the comments in ``config/traversability.yaml`` quote. They are
    what the operator sees in the node log, so a rule that silently stops
    matching (an OSM retag, a typo) shows up here.
    """
    md = MapData.load(str(KRALOVSKA))
    rules = load_traversability()

    summary = rules.summary(md.footways_list + md.roads_list)

    assert summary == {"stairs": 15, "soft surface": 6, "bridge": 13, "rough surface": 1}


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
