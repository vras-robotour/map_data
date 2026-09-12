"""
Crossroad detection: which nodes are junctions, and where a junction actually is.

The detector used to count the *ways* using a node, which reports a junction at
every node of a corridor that happens to be mapped twice, and it looked at
footways only, so a footway meeting a road was not a junction at all.
"""

from pathlib import Path

import numpy as np
import pytest
import utm
from shapely.geometry import LineString

from map_data.map_data import MapData
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
