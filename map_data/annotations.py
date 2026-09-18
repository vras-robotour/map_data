"""
Apply a viewer annotation store to a :class:`~map_data.map_data.MapData`.

The viewer keeps every user edit of a map (deleted ways/nodes, splits, moved
nodes, tag overrides, freehand paths and obstacles) in ``<stem>.annotations.json``
next to the ``.mapdata`` file, and only merges them into the map when it plans
or exports. This module is that merge without the web app, so a script, the
``map_data_plan`` CLI or the ``route_planner`` node plan on the same map the
viewer shows.

:func:`load_mapdata_with_annotations` is the one-call entry point; the viewer's
``get_merged_mapdata`` uses :func:`apply_way_edits` and
:func:`merge_annotations` on its cached copy.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import utm
from shapely import STRtree
from shapely.geometry import LineString, Point

from map_data.map_data import MapData
from map_data.traversability import TraversabilityRules, load_traversability
from map_data.utils.way import NON_ROUTABLE_HIGHWAY_VALUES, Way
from map_data.viewer.helpers import (
    apply_added_nodes,
    apply_node_position_overrides,
    edited_nodes_cache,
    geojson_geom_to_utm,
    get_deleted_node_ids,
    get_deleted_way_ids,
    get_detached_node_ids,
    get_node_position_overrides,
    get_split_node_ids,
    load_annotations,
    rebuild_way_without_nodes,
    split_way,
)

logger = logging.getLogger(__name__)

_CAT_FOR_LIST = {
    "roads_list": "road",
    "footways_list": "footway",
    "barriers_list": "barrier",
}

#: Default width (m) of an annotated path without a ``width`` property.
DEFAULT_ANNOTATION_WIDTH_M = 1.5
#: Ways that moved a shared node to within this many degrees (~0.5 m) of each other keep sharing it.
SAME_SPOT_DEG = 5e-6
#: An end of a drawn path, or a moved end of any way, this close (m) to another way is
#: joined to it (:func:`join_ways`). The viewer does not snap a dragged node, so this is
#: how exactly the user has to aim; the same 5 m the graph planner's own stitching uses.
JOIN_DISTANCE_M = 5.0
#: The junction is an existing node of that way when one is this close (m) to the nearest
#: point, else a new node on the edge.
JOIN_NODE_M = 1.0
#: Pass as ``annotations_path`` to load the map without any annotation store.
NO_ANNOTATIONS = "none"


def _same_spot(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) < SAME_SPOT_DEG and abs(a[1] - b[1]) < SAME_SPOT_DEG


def annotation_path_for(mapdata_path: str | Path) -> Path:
    """``<dir>/<stem>.annotations.json`` for a ``.mapdata`` file."""
    p = Path(mapdata_path)
    return p.with_name(f"{p.stem}.annotations.json")


def apply_way_edits(md: MapData, store: dict[str, Any]) -> None:
    """
    Apply way/node deletions, splits and node position overrides to *md*.

    Reassigns ``roads_list``/``footways_list``/``barriers_list`` (and
    ``crossroads_list`` when a way list changed); individual ``Way`` objects
    of the input lists are never mutated, so this is safe on a shallow copy.

    ``md.nodes_cache`` is replaced (not mutated) by one that also holds added
    nodes, detached split ends and moved positions, since the graph planner
    takes node positions from it.
    """
    zn, zl = md.zone_number, md.zone_letter
    # Without the moves: they are per way, so the second pass places them, and the
    # first one must not rebuild a way from a position another way moved its node to.
    nodes_cache = md.nodes_cache = edited_nodes_cache(
        {**store, "node_position_overrides": {}}, getattr(md, "nodes_cache", None)
    )

    deleted_way_ids = get_deleted_way_ids(store)
    has_node_dels = bool(store.get("deleted_nodes"))
    has_splits = bool(store.get("split_ways"))
    has_added_nodes = bool(store.get("added_nodes"))

    if deleted_way_ids or has_node_dels or has_splits or has_added_nodes:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            cat = _CAT_FOR_LIST[lst_name]
            new_lst = []
            for w in getattr(md, lst_name):
                if w.id in deleted_way_ids:
                    continue
                del_nids = get_deleted_node_ids(store, w.id)
                if del_nids:
                    w = rebuild_way_without_nodes(w, del_nids, zn, zl, nodes_cache, category=cat)  # noqa: PLW2901
                    if w is None:
                        continue

                w = apply_added_nodes(w, store, zn, zl)  # noqa: PLW2901

                split_nids = get_split_node_ids(store, w.id)
                if split_nids:
                    segments = split_way(
                        w, split_nids, zn, zl, nodes_cache, get_detached_node_ids(store, w.id)
                    )
                    for i, seg in enumerate(segments):
                        virtual_id = f"{w.id}:{i}"
                        if seg is w:  # split_way declined; don't rename the caller's way
                            seg = copy.copy(seg)  # noqa: PLW2901
                        seg.id = virtual_id
                        if virtual_id in deleted_way_ids:
                            continue
                        seg_del_nids = get_deleted_node_ids(store, virtual_id)
                        if seg_del_nids:
                            seg = rebuild_way_without_nodes(  # noqa: PLW2901
                                seg,
                                seg_del_nids,
                                zn,
                                zl,
                                nodes_cache,
                                category=cat,
                            )  # type: ignore[assignment] # narrowed by the `is None` check below
                            if seg is None:
                                continue
                        new_lst.append(seg)
                else:
                    new_lst.append(w)
            setattr(md, lst_name, new_lst)
        # str() keys because virtual split-way ids are strings; only .values() is read.
        md.crossroads_list = md.parse_intersections(
            {str(w.id): w for w in md.footways_list + md.roads_list},
        )

    node_pos_store = store.get("node_position_overrides", {})
    if node_pos_store:
        # Moves are recorded per way, node ids are global: a node moved in one way
        # but used by another becomes that way's own copy (fresh negative id), so
        # the planner sees the geometry the viewer draws and the other ways stay put.
        # Segments of one split way share their overrides, hence also the copy.
        lists = ("roads_list", "footways_list", "barriers_list")
        users: dict[int, set[str]] = {}
        for lst_name in lists:
            for w in getattr(md, lst_name):
                for n in w.nodes:
                    users.setdefault(getattr(n, "id", n), set()).add(str(w.id).split(":")[0])
        nodes_cache = md.nodes_cache = dict(nodes_cache)
        next_id = min([0, *nodes_cache]) - 1
        copies: dict[str, dict[int, int]] = {}  # original way id -> {node id: copy id}
        moves: dict[int, dict[str, tuple[float, float]]] = {}  # node id -> {way id: (lat, lon)}
        for wid, way_ov in node_pos_store.items():
            for nid_str, pos in way_ov.items():
                # An override of a way that no longer uses the node (deleted way/node) is stale.
                if wid in users.get(int(nid_str), ()) and int(nid_str) in nodes_cache:
                    moves.setdefault(int(nid_str), {})[wid] = (float(pos["lat"]), float(pos["lon"]))
        for nid, by_way in moves.items():
            spots: list[
                tuple[tuple[float, float], list[str]]
            ] = []  # ways grouped by where they put it
            for wid, pos in by_way.items():
                for spot, wids in spots:
                    if _same_spot(spot, pos):
                        wids.append(wid)
                        break
                else:
                    spots.append((pos, [wid]))
            for (lat, lon), wids in spots:
                if len(spots) == 1 and len(wids) == len(users[nid]):
                    nodes_cache[nid] = {
                        **nodes_cache[nid],
                        "lat": lat,
                        "lon": lon,
                    }  # moved as a whole
                    continue
                nodes_cache[next_id] = {**nodes_cache[nid], "lat": lat, "lon": lon}
                for wid in wids:
                    copies.setdefault(wid, {})[nid] = next_id
                next_id -= 1

        for lst_name in lists:
            new_lst = []
            for w in getattr(md, lst_name):
                ov = get_node_position_overrides(store, w.id)
                if ov:
                    result = apply_node_position_overrides(
                        w,
                        ov,
                        zn,
                        zl,
                        nodes_cache,
                        category=_CAT_FOR_LIST[lst_name],
                    )
                    w = result or w  # noqa: PLW2901
                    remap = copies.get(str(w.id).split(":")[0])
                    if remap:
                        w = copy.copy(w)  # noqa: PLW2901
                        w.nodes = [remap.get(getattr(n, "id", n), n) for n in w.nodes]
                new_lst.append(w)
            setattr(md, lst_name, new_lst)
        if moves:
            md.crossroads_list = md.parse_intersections(
                {str(w.id): w for w in md.footways_list + md.roads_list},
            )


def apply_tag_overrides(md: MapData, store: dict[str, Any]) -> None:
    """
    Merge ``store["tag_overrides"]`` into the ways, re-sort roads/footways in
    case a ``highway`` tag changed category, and recompute the crossroads.

    An overridden way is replaced by a shallow copy carrying the merged tags,
    so the input ``Way`` objects are never mutated and the three way lists are
    reassigned; this is safe on a shallow copy of the map.
    """
    tag_overrides = store.get("tag_overrides", {})
    if not tag_overrides:
        return
    for lst_name in ("roads_list", "footways_list", "barriers_list"):
        new_lst = []
        for w in getattr(md, lst_name):
            original_id = str(w.id).split(":")[0]
            ov = tag_overrides.get(original_id)
            if ov:
                w = copy.copy(w)  # noqa: PLW2901
                w.tags = {**(w.tags or {}), **ov}
            new_lst.append(w)
        setattr(md, lst_name, new_lst)
    new_roads: list[Way] = []
    new_footways: list[Way] = []
    for w in md.roads_list:
        (new_footways if w.is_footway() else new_roads).append(w)
    for w in md.footways_list:
        (new_roads if w.is_road() else new_footways).append(w)
    md.roads_list, md.footways_list = new_roads, new_footways
    md.crossroads_list = md.parse_intersections(
        {str(w.id): w for w in md.footways_list + md.roads_list},
    )


def merge_annotations(md: MapData, store: dict[str, Any]) -> None:
    """
    Synthesize a ``Way`` for every freehand annotation in ``store["annotations"]``.

    A ``"path"`` annotation becomes a road or footway (its centre line buffered
    by half its ``width``), with synthetic negative node ids registered in
    ``md.nodes_cache`` so the graph planner can route over it; anything else
    becomes a barrier. Crossroads where annotated paths meet other ways are
    appended to ``md.crossroads_list``. Ends with :func:`join_ways`, which gives the
    drawn paths (and moved end nodes) real nodes to meet the network at. Mutates *md* in place.
    """
    zn, zl = md.zone_number, md.zone_letter
    # Below the drawn paths an exported map already carries: an id names one way.
    all_ways = md.footways_list + md.roads_list + md.barriers_list
    ann_id = min([0, *(int(str(w.id).split(":")[0]) for w in all_ways)]) - 1
    ann_lines: list[tuple[Way, Any]] = []  # annotated path ways with their centre lines
    if not hasattr(md, "nodes_cache") or md.nodes_cache is None:
        md.nodes_cache = {}
    # Below the synthetic ids apply_way_edits may already have put in the cache.
    node_id = min([0, *md.nodes_cache]) - 1
    for ann in store.get("annotations", []):
        geom = geojson_geom_to_utm(ann["geometry"], zn, zl)
        if geom is None:
            continue
        props = ann.get("properties", {})
        ann_type = ann.get("type", "obstacle")

        w = Way()
        w.id = ann_id
        ann_id -= 1
        w.line = geom
        w.nodes = []
        w.in_out = ""

        if ann_type == "path":
            hw = props.get("highway", "path")
            w.tags = {"highway": hw}
            if "width" in props:
                w.tags["width"] = str(props["width"])
            for k, v in props.items():
                if k not in ("highway", "width"):
                    w.tags[k] = str(v)
            if geom.geom_type == "LineString":
                width_m = float(props.get("width", DEFAULT_ANNOTATION_WIDTH_M))
                for e_coord, n_coord in geom.coords:
                    lat, lon = utm.to_latlon(e_coord, n_coord, zn, zl)
                    md.nodes_cache[node_id] = {"lat": lat, "lon": lon, "tags": {}}
                    w.nodes.append(node_id)
                    node_id -= 1
                w.line = geom.buffer(width_m / 2)
                w.is_area = True
                ann_lines.append((w, geom))
            (md.roads_list if w.is_road() else md.footways_list).append(w)
        else:
            w.tags = {"barrier": props.get("barrier", "wall")}
            for k, v in props.items():
                if k != "barrier":
                    w.tags[k] = str(v)
            md.barriers_list.append(w)

    # Annotated paths share no OSM node ids with the map, so node-based crossroad
    # detection cannot see them; add crossroads where they cross or touch other ways.
    # Both sides have to be centre lines - the ways carry their buffered geometry,
    # against which a path running alongside one reports a junction it never reaches.
    if ann_lines:
        others = [
            (o, md.centre_line(o) or o.line) for o in list(md.footways_list) + list(md.roads_list)
        ]
        md.crossroads_list = list(md.crossroads_list) + MapData.geometric_intersections(
            ann_lines, others
        )
    join_ways(md, store)


def join_ways(md: MapData, store: dict[str, Any]) -> None:
    """
    Give ways that meet on the map, but share no node id, a node to meet at.

    The graph planner connects ways by node id only, which an edit never produces by
    itself: a drawn path has synthetic nodes of its own, and a dragged node keeps (or, if
    it was shared, loses) the junctions it had. Joined here are

    - an end of a drawn path (negative way id: drawn in *store*, baked into an exported
      map, or a split segment of one) and an end node of any way that *store* moved, when
      it is within :data:`JOIN_DISTANCE_M` of another way — preferably one of its own kind,
      as a footway-only plan never sees the roads — and not already shared with one;
    - a drawn path and every way its centre line crosses or that ends on it.

    The junction is a node of the other way (:data:`JOIN_NODE_M`), or a new one inserted on
    its edge. It is added to the node lists only — put before/after the end, inserted at a
    crossing — so no geometry changes and nothing the user placed is moved. Segments of one
    original way are never joined: a detached split stays detached. Being real shared
    nodes, the junctions survive an export. Way lists are reassigned with copies of the
    changed ways; ``md.nodes_cache`` gains the new nodes in place.
    """
    ways = md.footways_list + md.roads_list
    n_foot = len(md.footways_list)
    origin = [str(w.id).split(":")[0] for w in ways]
    drawn = [o.startswith("-") for o in origin]
    moved_to = {
        wid: {(float(p["lat"]), float(p["lon"])) for p in way_ov.values()}
        for wid, way_ov in store.get("node_position_overrides", {}).items()
    }
    users: dict[int, set[str]] = {}
    for o, w in zip(origin, ways, strict=True):
        for n in w.nodes:
            users.setdefault(n, set()).add(o)

    def looks_for_a_way(wi: int, n: int) -> bool:
        c = md.nodes_cache.get(n)
        if c is None or len(users[n]) > 1:
            return False
        return drawn[wi] or (c["lat"], c["lon"]) in moved_to.get(origin[wi], ())

    ends = [
        (wi, k)
        for wi, w in enumerate(ways)
        if len(w.nodes) >= 2 and w.nodes[0] != w.nodes[-1]
        for k in (0, -1)
        if looks_for_a_way(wi, w.nodes[k])
    ]
    if not ends and not any(drawn):
        return

    xy = {n: p.ravel()[:2] for n, p in md.get_points().items()}
    segments, owner = [], []  # every edge of every way, and its (way index, edge index)
    for wi, w in enumerate(ways):
        for i in range(len(w.nodes) - 1):
            if w.nodes[i] in xy and w.nodes[i + 1] in xy:
                segments.append(LineString([xy[w.nodes[i]], xy[w.nodes[i + 1]]]))
                owner.append((wi, i))
    if not segments:
        return
    tree = STRtree(segments)
    inserts: dict[tuple[int, int], list[tuple[float, int]]] = {}  # (way, edge) -> [(along, id)]
    next_id = min([0, *md.nodes_cache]) - 1

    def node_on(j: int, along: float) -> int:
        """The node *along* metres into edge *j*: one of its own when close, else a new one."""
        nonlocal next_id
        vi, i = owner[j]
        if along <= JOIN_NODE_M:
            return ways[vi].nodes[i]
        if segments[j].length - along <= JOIN_NODE_M:
            return ways[vi].nodes[i + 1]
        nid, next_id = next_id, next_id - 1
        e, n = segments[j].interpolate(along).coords[0]
        lat, lon = utm.to_latlon(e, n, md.zone_number, md.zone_letter)
        md.nodes_cache[nid] = {"lat": lat, "lon": lon, "tags": {}}
        inserts.setdefault((vi, i), []).append((along, nid))
        return nid

    joined: list[tuple[int, int, int]] = []  # (way, which end, junction)
    for wi, k in ends:
        end = Point(xy[ways[wi].nodes[k]])
        near = [
            ((owner[j][0] >= n_foot) != (wi >= n_foot), segments[j].distance(end), j)
            for j in tree.query(end.buffer(JOIN_DISTANCE_M), predicate="intersects")
            if origin[owner[j][0]] != origin[wi]
        ]
        if near:
            j = min(near)[2]
            joined.append((wi, k, node_on(j, segments[j].project(end))))

    end_nodes = {ways[wi].nodes[k] for wi, k in ends}  # those are joined above, not as crossings
    for j, (wi, i) in enumerate(owner):
        if not drawn[wi]:
            continue
        for h in tree.query(segments[j], predicate="intersects"):
            vi = owner[h][0]
            if origin[vi] == origin[wi] or (drawn[vi] and h < j):  # a drawn pair comes up twice
                continue
            mine, theirs = ways[wi].nodes[i : i + 2], ways[vi].nodes[owner[h][1] : owner[h][1] + 2]
            if set(mine) & set(theirs):
                continue
            pt = segments[j].intersection(segments[h])
            if pt.geom_type != "Point":  # they run together: not a crossing
                continue
            along = segments[j].project(pt)
            if along <= JOIN_NODE_M and mine[0] in end_nodes:
                continue
            if segments[j].length - along <= JOIN_NODE_M and mine[1] in end_nodes:
                continue
            inserts.setdefault((wi, i), []).append((along, node_on(h, segments[h].project(pt))))

    chains: dict[int, list[int]] = {}  # way index -> its new node list
    # Last edge first, so an insert does not shift the edges still to come.
    for (wi, i), new in sorted(inserts.items(), reverse=True):
        chain = chains.setdefault(wi, list(ways[wi].nodes))
        ids = dict.fromkeys(n for _, n in sorted(new))  # in order along the edge, each once
        chain[i + 1 : i + 1] = [n for n in ids if n not in chain]
    for wi, k, junction in joined:
        chain = chains.setdefault(wi, list(ways[wi].nodes))
        chain.insert(0 if k == 0 else len(chain), junction)
    if not chains:
        return
    for wi, chain in chains.items():
        ways[wi] = copy.copy(ways[wi])
        ways[wi].nodes = chain
    md.footways_list, md.roads_list = ways[:n_foot], ways[n_foot:]
    md.recompute_crossroads()


def apply_store(md: MapData, store: dict[str, Any]) -> MapData:
    """
    Apply every kind of edit in *store* to *md* (copied first) and return it.

    The copy is shallow: no pass mutates a ``Way``, a node entry or the
    waypoints, they only rebind the map's attributes — except
    :func:`merge_annotations`, which appends to the way lists and writes the
    annotated paths' synthetic nodes into ``nodes_cache``. Giving the copy its
    own containers is therefore enough to leave *md* untouched, and costs ~1 ms
    against the ~280 ms of deep-copying every geometry (Stromovka).
    """
    md = copy.copy(md)
    md.nodes_cache = dict(getattr(md, "nodes_cache", None) or {})
    md.roads_list = list(md.roads_list)
    md.footways_list = list(md.footways_list)
    md.barriers_list = list(md.barriers_list)
    md.crossroads_list = list(md.crossroads_list)
    apply_way_edits(md, store)
    apply_tag_overrides(md, store)
    merge_annotations(md, store)
    return md


def load_mapdata_with_annotations(
    mapdata_path: str | Path,
    annotations_path: str | Path | None = None,
    exclude_highway: Iterable[str] = NON_ROUTABLE_HIGHWAY_VALUES,
    traversability: TraversabilityRules | str | Path | None = None,
) -> tuple[MapData, dict[str, Any]]:
    """
    Load a ``.mapdata`` file and merge its annotation store.

    ``annotations_path`` defaults to :func:`annotation_path_for`; a missing
    store simply yields the unedited map, and :data:`NO_ANNOTATIONS` (``"none"``)
    skips the store on purpose.

    ``traversability`` decides from the OSM tags which ways the robot may drive
    on (stairs, grass, tunnels, ...): a
    :class:`~map_data.traversability.TraversabilityRules`, a path to a rule
    file, or ``None`` for the package's ``config/traversability.yaml``. The
    non-traversable ways are removed through
    :meth:`~map_data.map_data.MapData.apply_traversability` before the store is
    applied: drawn annotations may append geometric crossroads, which the
    crossroad recompute inside it would otherwise drop.

    ``exclude_highway`` is the older, tag-free shortcut for the same thing
    (stairs by default, see
    :data:`~map_data.utils.way.NON_ROUTABLE_HIGHWAY_VALUES`); its values are
    added to the rules as one leading deny rule. Pass an empty iterable *and*
    empty rules (``TraversabilityRules()``) to keep every way, as
    :meth:`MapData.load` itself does — the viewer must still show what the
    planner refuses.

    Returns ``(map_data, store)``.
    """
    mapdata_path = Path(mapdata_path)
    store: dict[str, Any]
    if annotations_path == NO_ANNOTATIONS:
        store = {"version": 1, "annotations": []}
    else:
        store = load_annotations(str(annotations_path or annotation_path_for(mapdata_path)))
    md = MapData.load(str(mapdata_path))
    md.apply_traversability(load_traversability(traversability).extend(exclude_highway))
    n_ann = len(store.get("annotations", []))
    if n_ann or store.get("deleted_ways") or store.get("split_ways") or store.get("tag_overrides"):
        logger.info("Applying annotation store to %s (%d annotations)", mapdata_path.name, n_ann)
    return apply_store(md, store), store
