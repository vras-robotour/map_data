"""
Crossroad detection: which nodes are junctions, and where a junction actually is.

Two defects are covered here. The node-based detector used to count the *ways*
using a node, which reports a junction at every node of a corridor that happens
to be mapped twice, and it looked at footways only, so a footway meeting a road
was not a junction at all. The geometric detector used to intersect an annotated
centre line with the other way's *buffered* geometry, which finds the stretch of
line inside the corridor instead of a crossing point.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import utm
from shapely.geometry import LineString

from map_data.annotations import annotation_path_for, load_mapdata_with_annotations
from map_data.map_data import MapData
from map_data.traversability import TraversabilityRules
from map_data.utils.way import Way

#: The full Stromovka map used on the robot; not committed (data/* is ignored).
STROMOVKA = Path(__file__).resolve().parents[1] / "data" / "stromovka_full.mapdata"

needs_stromovka = pytest.mark.skipif(
    not STROMOVKA.is_file(),
    reason="stromovka_full.mapdata is not in the repo",
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _map(nodes: dict[int, tuple[float, float]]) -> MapData:
    """An empty map whose ``nodes_cache`` holds *nodes* as (east, north) offsets."""
    e0, n0, zn, zl = utm.from_latlon(50.0, 14.0)
    md = MapData(
        [np.array([[e0 - 500, n0 - 500], [e0 + 500, n0 + 500]]), int(zn), zl], coords_type="array"
    )
    md.nodes_cache = {}
    for nid, (de, dn) in nodes.items():
        lat, lon = utm.to_latlon(e0 + de, n0 + dn, zn, zl)
        md.nodes_cache[nid] = {"lat": lat, "lon": lon, "tags": {}}
    md._origin = (e0, n0)  # type: ignore[attr-defined]
    return md


def _way(way_id, node_ids, highway, md, width=3.0):
    """A way as the pipeline stores one: node ids plus *buffered* geometry."""
    e0, n0 = md._origin
    coords = []
    for nid in node_ids:
        nd = md.nodes_cache[nid]
        e, n, _, _ = utm.from_latlon(
            nd["lat"], nd["lon"], force_zone_number=md.zone_number, force_zone_letter=md.zone_letter
        )
        coords.append((e, n))
    return Way(
        id=way_id,
        is_area=True,
        nodes=list(node_ids),
        tags={"highway": highway},
        line=LineString(coords).buffer(width / 2),
    )


def _crossroads(md, ways):
    md.footways_list = [w for w in ways if w.is_footway()]
    md.roads_list = [w for w in ways if w.is_road()]
    return md.parse_intersections({str(w.id): w for w in ways})


# ── node-based detection ──────────────────────────────────────────────────────


def test_a_footway_meeting_a_road_is_a_crossroad():
    """Roads used to be left out of the detector entirely, so this found nothing."""
    md = _map({1: (0, 0), 2: (0, 100), 3: (0, 50), 4: (70, 50)})
    road = _way(10, [1, 3, 2], "residential", md, width=7.0)
    foot = _way(20, [3, 4], "footway", md)
    assert [c.id for c in _crossroads(md, [road, foot])] == [3]


def test_the_same_junction_is_found_whatever_the_through_way_is_tagged():
    """The tag on the through way must not decide whether a junction exists."""
    md = _map({1: (0, 0), 2: (0, 100), 3: (0, 50), 4: (70, 50)})
    branch = _way(20, [3, 4], "footway", md)
    for highway in ("residential", "service", "footway", "cycleway", "path"):
        through = _way(10, [1, 3, 2], highway, md)
        assert [c.id for c in _crossroads(md, [through, branch])] == [3], highway


def test_a_corridor_mapped_twice_is_not_a_junction_at_every_node():
    """
    A cycleway drawn along an existing footway reuses its node ids. Counting the
    ways through each node made every one of them a crossroad.
    """
    md = _map({i: (i * 10.0, 0.0) for i in range(1, 7)})
    foot = _way(1, [1, 2, 3, 4, 5, 6], "footway", md)
    cycle = _way(2, [1, 2, 3, 4, 5, 6], "cycleway", md)
    assert _crossroads(md, [foot, cycle]) == []


def test_a_corridor_mapped_twice_still_reports_where_it_forks():
    """Only the node where the duplicate leaves the corridor is a junction."""
    md = _map({i: (i * 10.0, 0.0) for i in range(1, 7)} | {99: (35.0, 40.0)})
    foot = _way(1, [1, 2, 3, 4, 5, 6], "footway", md)
    cycle = _way(2, [1, 2, 3, 99], "cycleway", md)  # leaves the corridor at node 3
    assert [c.id for c in _crossroads(md, [foot, cycle])] == [3]


def test_two_ways_joined_end_to_end_are_not_a_junction():
    md = _map({1: (0, 0), 2: (50, 0), 3: (100, 0)})
    a = _way(1, [1, 2], "footway", md)
    b = _way(2, [2, 3], "footway", md)
    assert _crossroads(md, [a, b]) == []


def test_a_way_ending_on_another_is_a_junction():
    md = _map({1: (0, 0), 2: (50, 0), 3: (100, 0), 4: (50, 60)})
    through = _way(1, [1, 2, 3], "footway", md)
    stub = _way(2, [4, 2], "footway", md)
    assert [c.id for c in _crossroads(md, [through, stub])] == [2]


def test_two_ways_crossing_at_a_node_are_a_junction():
    md = _map({1: (-50, 0), 2: (50, 0), 3: (0, -50), 4: (0, 50), 5: (0, 0)})
    a = _way(1, [1, 5, 2], "footway", md)
    b = _way(2, [3, 5, 4], "footway", md)
    assert [c.id for c in _crossroads(md, [a, b])] == [5]


def test_a_dead_end_is_not_a_junction():
    md = _map({1: (0, 0), 2: (50, 0)})
    assert _crossroads(md, [_way(1, [1, 2], "footway", md)]) == []


def test_a_closed_loop_is_not_a_junction_at_its_seam():
    md = _map({1: (0, 0), 2: (50, 0), 3: (50, 50), 4: (0, 50)})
    loop = _way(1, [1, 2, 3, 4, 1], "footway", md)
    assert _crossroads(md, [loop]) == []


def test_barriers_are_not_considered():
    """The whole ways_dict of a fresh parse is passed in; only routable ways count."""
    md = _map({1: (0, 0), 2: (50, 0), 3: (100, 0), 4: (50, 60)})
    through = _way(1, [1, 2, 3], "footway", md)
    fence = Way(
        id=2,
        is_area=False,
        nodes=[4, 2],
        tags={"barrier": "fence"},
        line=LineString([(0, 0), (1, 1)]),
    )
    assert _crossroads(md, [through, fence]) == []


# ── geometric detection (annotated paths) ─────────────────────────────────────


def _buffered(way_id, coords, highway, width):
    """A stored way (buffered) plus the centre line it was built from."""
    line = LineString(coords)
    w = Way(id=way_id, is_area=True, tags={"highway": highway}, line=line.buffer(width / 2))
    return w, line


def test_a_path_alongside_a_footway_is_not_a_junction():
    """
    1.2 m from the centre line is inside the footway's 3 m corridor, so matching
    against the buffered geometry reported a junction the path never reaches.
    """
    foot, foot_line = _buffered(1, [(0, 0), (100, 0)], "footway", 3.0)
    ann, ann_line = _buffered(-1, [(0, 1.2), (100, 1.2)], "cycleway", 2.0)
    assert MapData.geometric_intersections([(ann, ann_line)], [(foot, foot_line)]) == []


def test_a_crossing_is_reported_where_it_happens():
    foot, foot_line = _buffered(1, [(0, 0), (100, 0)], "footway", 3.0)
    ann, ann_line = _buffered(-1, [(50, -20), (50, 20)], "path", 2.0)
    found = MapData.geometric_intersections([(ann, ann_line)], [(foot, foot_line)])
    assert len(found) == 1
    centre = found[0].line.centroid
    assert centre.distance(LineString([(50, 0), (50, 0.001)])) < 1e-6


def test_a_crossing_is_not_displaced_by_the_width_of_a_road():
    """A 7 m road used to push its crossings 3.5 m off, half the enter radius."""
    road, road_line = _buffered(1, [(0, 0), (100, 0)], "residential", 7.0)
    ann, ann_line = _buffered(-1, [(50, -20), (50, 20)], "path", 2.0)
    found = MapData.geometric_intersections([(ann, ann_line)], [(road, road_line)])
    assert len(found) == 1
    assert abs(found[0].line.centroid.y) < 1e-6


def test_every_crossing_of_a_weaving_path_is_placed_on_the_way():
    foot, foot_line = _buffered(1, [(0, 0), (200, 0)], "footway", 3.0)
    weave = LineString([(0, -10), (25, 10), (50, -10), (75, 10), (100, -10)])
    ann = Way(id=-1, is_area=True, tags={"highway": "path"}, line=weave.buffer(1.0))
    found = MapData.geometric_intersections([(ann, weave)], [(foot, foot_line)])
    assert len(found) == 4
    assert sorted(round(c.line.centroid.x, 2) for c in found) == [12.5, 37.5, 62.5, 87.5]
    assert all(abs(c.line.centroid.y) < 1e-6 for c in found)


def test_a_path_ending_on_a_way_is_still_a_t_junction():
    foot, foot_line = _buffered(1, [(0, 0), (100, 0)], "footway", 3.0)
    ann, ann_line = _buffered(-1, [(50, 20), (50, 0.5)], "path", 2.0)
    found = MapData.geometric_intersections([(ann, ann_line)], [(foot, foot_line)])
    assert len(found) == 1
    assert found[0].line.centroid.distance(LineString([(50, 0.5), (50, 0.501)])) < 1e-6


def test_a_path_far_from_every_way_makes_no_junction():
    foot, foot_line = _buffered(1, [(0, 0), (100, 0)], "footway", 3.0)
    ann, ann_line = _buffered(-1, [(0, 40), (100, 40)], "path", 2.0)
    assert MapData.geometric_intersections([(ann, ann_line)], [(foot, foot_line)]) == []


# ── through the annotation pipeline ───────────────────────────────────────────


def _map_with_a_wide_footway(path):
    """
    A saved map holding one 3 m wide footway, stored the way the parser leaves
    it: node ids in ``nodes_cache`` and the *buffered* polygon as the geometry.
    """
    lat0, lon0 = 50.0, 14.0
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    md = MapData(
        [np.array([[e0 - 100, n0 - 100], [e0 + 300, n0 + 100]]), int(zn), zl],
        coords_type="array",
    )
    coords = {101: (0.0, 0.0), 102: (100.0, 0.0), 103: (200.0, 0.0)}
    line = LineString([(e0 + de, n0 + dn) for de, dn in coords.values()])
    md.footways_list.append(
        Way(
            id=1,
            is_area=True,
            nodes=[101, 102, 103],
            tags={"highway": "footway"},
            line=line.buffer(1.5),  # 3 m wide, as config/buffer_widths has it
        ),
    )
    md.nodes_cache = {}
    for nid, (de, dn) in coords.items():
        lat, lon = utm.to_latlon(e0 + de, n0 + dn, zn, zl)
        md.nodes_cache[nid] = {"lat": lat, "lon": lon, "tags": {}}
    md.crossroads_list = []
    md.save(str(path))
    return lat0, lon0


def _store_with_path(lat0, lon0, offsets, width=2.0):
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    coords = []
    for de, dn in offsets:
        lat, lon = utm.to_latlon(e0 + de, n0 + dn, zn, zl)
        coords.append([lon, lat])
    return {
        "version": 1,
        "annotations": [
            {
                "id": "ann-1",
                "type": "path",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"highway": "cycleway", "width": width},
            },
        ],
    }


def _annotation_crossroads(md):
    return [c for c in md.crossroads_list if c.tags.get("type") == "annotation_intersection"]


def test_an_annotated_path_alongside_a_footway_makes_no_junction(tmp_path):
    """
    The pipeline has to hand centre lines to the geometric detector. The ways it
    loads carry their buffered geometry, so a path drawn 1.2 m off the footway -
    inside its 3 m corridor, but further than the 1 m touch tolerance - used to
    be reported as a junction it never reaches.
    """
    path = tmp_path / "alongside.mapdata"
    lat0, lon0 = _map_with_a_wide_footway(path)
    annotation_path_for(path).write_text(
        json.dumps(_store_with_path(lat0, lon0, [(10.0, 1.2), (190.0, 1.2)])),
    )

    md, _ = load_mapdata_with_annotations(
        path,
        exclude_highway=(),
        traversability=TraversabilityRules(),
    )
    assert _annotation_crossroads(md) == []


def test_an_annotated_path_crossing_a_footway_lands_on_it(tmp_path):
    """The same pipeline still finds a real crossing, and puts it on the way."""
    path = tmp_path / "crossing.mapdata"
    lat0, lon0 = _map_with_a_wide_footway(path)
    annotation_path_for(path).write_text(
        json.dumps(_store_with_path(lat0, lon0, [(100.0, -30.0), (100.0, 30.0)])),
    )

    md, _ = load_mapdata_with_annotations(
        path,
        exclude_highway=(),
        traversability=TraversabilityRules(),
    )
    found = _annotation_crossroads(md)
    assert len(found) == 1

    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    centre = found[0].line.centroid
    # The crossing is at (100, 0); the buggy match against the 3 m corridor put
    # it 1.5 m away, on the edge of the buffer.
    assert abs(centre.x - (e0 + 100.0)) < 0.01
    assert abs(centre.y - n0) < 0.01


def test_two_ways_running_together_meet_where_the_shared_stretch_ends():
    """
    A collinear overlap is not one junction in the middle of the shared stretch:
    the ways join at one end of it and separate at the other.
    """
    foot_line = LineString([(0, 0), (100, 0)])
    foot = Way(id=1, is_area=True, tags={"highway": "footway"}, line=foot_line.buffer(1.5))
    ann_line = LineString([(20, 0), (60, 0), (60, 40)])
    ann = Way(id=-1, is_area=True, tags={"highway": "cycleway"}, line=ann_line.buffer(1.0))

    found = MapData.geometric_intersections([(ann, ann_line)], [(foot, foot_line)])
    xs = sorted(round(c.line.centroid.x, 2) for c in found)
    assert xs == [20.0, 60.0], f"expected the ends of the shared stretch, got {xs}"


def test_loading_refreshes_crossroads_saved_by_an_older_detector(tmp_path):
    """
    Crossroads are derived from the ways, and every .mapdata already on disk
    carries the ones the old detector found. Loading has to recompute them, or
    the fix never reaches a map the robot already has.
    """
    path = tmp_path / "stale.mapdata"
    lat0, lon0 = _map_with_a_wide_footway(path)

    md = MapData.load(str(path))
    # A junction the old detector could not see: a road ending on the footway.
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    lat, lon = utm.to_latlon(e0 + 100.0, n0 + 60.0, zn, zl)
    md.nodes_cache[104] = {"lat": lat, "lon": lon, "tags": {}}
    md.roads_list.append(
        Way(
            id=2,
            is_area=True,
            nodes=[104, 102],
            tags={"highway": "service"},
            line=LineString([(e0 + 100.0, n0 + 60.0), (e0 + 100.0, n0)]).buffer(3.5),
        ),
    )
    md.crossroads_list = []  # as an older file would have it
    md.save(str(path))

    reloaded = MapData.load(str(path))
    assert [c.id for c in reloaded.crossroads_list] == [102]


def test_loading_keeps_crossroads_that_cannot_be_recomputed(tmp_path):
    """An annotated path's crossroads come from geometry, not node ids."""
    path = tmp_path / "withann.mapdata"
    _map_with_a_wide_footway(path)

    md = MapData.load(str(path))
    md.crossroads_list = [
        Way(
            id=-1_000_000,
            is_area=True,
            tags={"type": "annotation_intersection", "count": "2"},
            line=LineString([(0, 0), (1, 1)]).centroid.buffer(1.5),
        ),
    ]
    md.save(str(path))

    reloaded = MapData.load(str(path))
    kept = [c for c in reloaded.crossroads_list if c.tags.get("type") == "annotation_intersection"]
    assert len(kept) == 1


# ── the real map the robot drives ─────────────────────────────────────────────


@needs_stromovka
def test_stromovka_road_junctions_are_found():
    """
    ``-18:1`` is "U Výstaviště", drawn in the viewer and tagged
    ``highway=residential``: as a road it was invisible to the detector, so none
    of the footways meeting it became a junction.
    """
    md = MapData.load(str(STROMOVKA))
    road = next(w for w in md.roads_list if str(w.id) == "-18:1")
    crossroads = {
        c.id
        for c in md.parse_intersections({str(w.id): w for w in md.footways_list + md.roads_list})
    }

    on_the_road = crossroads & set(road.nodes)
    assert len(on_the_road) > 20, f"only {len(on_the_road)} junctions on U Výstaviště"


@needs_stromovka
def test_stromovka_has_no_junction_without_a_fork():
    """
    Every crossroad must have more than two directions leaving it. The map has
    three annotated ways (two of them cycleways) drawn along existing footways
    and reusing their node ids, which used to make a junction of every node they
    share.
    """
    md = MapData.load(str(STROMOVKA))
    ways = md.footways_list + md.roads_list
    crossroads = md.parse_intersections({str(w.id): w for w in ways})

    neighbours: dict[int, set[int]] = {}
    for w in ways:
        for i, nid in enumerate(w.nodes):
            seen = neighbours.setdefault(nid, set())
            if i:
                seen.add(w.nodes[i - 1])
            if i < len(w.nodes) - 1:
                seen.add(w.nodes[i + 1])

    through = [c.id for c in crossroads if len(neighbours[c.id] - {c.id}) <= 2]
    assert through == [], f"{len(through)} crossroads sit mid-corridor"


@needs_stromovka
def test_stromovka_duplicate_cycleways_make_no_junctions_of_their_own():
    """
    ``-1`` and ``-15`` are two annotated cycleways tracing the same corridor
    along existing footways, sharing 70 node ids. Two ways through a node
    between the same neighbours are one path, so they may not make a junction
    between themselves: every crossroad on that corridor has to be owed to a
    third way that actually branches there.
    """
    md = MapData.load(str(STROMOVKA))
    ways = md.footways_list + md.roads_list
    crossroads = md.parse_intersections({str(w.id): w for w in ways})

    duplicates = {"-1", "-15"}
    for c in crossroads:
        users = {str(w.id) for w in ways if c.id in w.nodes}
        assert not users <= duplicates, (
            f"node {c.id} is a junction only because one corridor is mapped twice"
        )
