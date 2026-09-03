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
from pathlib import Path
from typing import Any

import utm

from map_data.map_data import MapData
from map_data.utils.way import Way
from map_data.viewer.helpers import (
    apply_node_position_overrides,
    geojson_geom_to_utm,
    get_deleted_node_ids,
    get_deleted_way_ids,
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
#: Pass as ``annotations_path`` to load the map without any annotation store.
NO_ANNOTATIONS = "none"


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
    """
    zn, zl = md.zone_number, md.zone_letter
    nodes_cache = getattr(md, "nodes_cache", {})

    deleted_way_ids = get_deleted_way_ids(store)
    has_node_dels = bool(store.get("deleted_nodes"))
    has_splits = bool(store.get("split_ways"))

    if deleted_way_ids or has_node_dels or has_splits:
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
                split_nids = get_split_node_ids(store, w.id)
                if split_nids:
                    segments = split_way(w, split_nids, zn, zl, nodes_cache)
                    for i, seg in enumerate(segments):
                        virtual_id = f"{w.id}:{i}"
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
        # parse_intersections() only reads .values(); the str() keys here (needed
        # because virtual split-way ids are strings) don't match its dict[int, Way]
        # signature, but that mismatch is harmless.
        md.crossroads_list = md.parse_intersections(
            {str(w.id): w for w in md.footways_list},  # type: ignore[misc]
        )

    node_pos_store = store.get("node_position_overrides", {})
    if node_pos_store:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
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
                    new_lst.append(result or w)
                else:
                    new_lst.append(w)
            setattr(md, lst_name, new_lst)


def apply_tag_overrides(md: MapData, store: dict[str, Any]) -> None:
    """
    Merge ``store["tag_overrides"]`` into the ways (mutating them in place, so
    call this on a deep copy), re-sort roads/footways in case a ``highway``
    tag changed category, and recompute the crossroads.
    """
    tag_overrides = store.get("tag_overrides", {})
    if not tag_overrides:
        return
    for lst_name in ("roads_list", "footways_list", "barriers_list"):
        for w in getattr(md, lst_name):
            original_id = str(w.id).split(":")[0]
            ov = tag_overrides.get(original_id)
            if ov:
                w.tags = {**(w.tags or {}), **ov}
    new_roads: list[Way] = []
    new_footways: list[Way] = []
    for w in md.roads_list:
        (new_footways if w.is_footway() else new_roads).append(w)
    for w in md.footways_list:
        (new_roads if w.is_road() else new_footways).append(w)
    md.roads_list, md.footways_list = new_roads, new_footways
    md.crossroads_list = md.parse_intersections(
        {str(w.id): w for w in md.footways_list},  # type: ignore[misc]
    )


def merge_annotations(md: MapData, store: dict[str, Any]) -> None:
    """
    Synthesize a ``Way`` for every freehand annotation in ``store["annotations"]``.

    A ``"path"`` annotation becomes a road or footway (its centre line buffered
    by half its ``width``), with synthetic negative node ids registered in
    ``md.nodes_cache`` so the graph planner can route over it; anything else
    becomes a barrier. Crossroads where annotated paths meet other ways are
    appended to ``md.crossroads_list``. Mutates *md* in place.
    """
    zn, zl = md.zone_number, md.zone_letter
    ann_id = -1
    node_id = -1
    ann_lines: list[tuple[Way, Any]] = []  # annotated path ways with their centre lines
    if not hasattr(md, "nodes_cache") or md.nodes_cache is None:
        md.nodes_cache = {}
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
    if ann_lines:
        md.crossroads_list = list(md.crossroads_list) + MapData.geometric_intersections(
            ann_lines, list(md.footways_list) + list(md.roads_list)
        )


def apply_store(md: MapData, store: dict[str, Any]) -> MapData:
    """Apply every kind of edit in *store* to *md* (deep-copied first) and return it."""
    md = copy.deepcopy(md)
    apply_way_edits(md, store)
    apply_tag_overrides(md, store)
    merge_annotations(md, store)
    return md


def load_mapdata_with_annotations(
    mapdata_path: str | Path,
    annotations_path: str | Path | None = None,
) -> tuple[MapData, dict[str, Any]]:
    """
    Load a ``.mapdata`` file and merge its annotation store.

    ``annotations_path`` defaults to :func:`annotation_path_for`; a missing
    store simply yields the unedited map, and :data:`NO_ANNOTATIONS` (``"none"``)
    skips the store on purpose. Returns ``(map_data, store)``.
    """
    mapdata_path = Path(mapdata_path)
    store: dict[str, Any]
    if annotations_path == NO_ANNOTATIONS:
        store = {"version": 1, "annotations": []}
    else:
        store = load_annotations(str(annotations_path or annotation_path_for(mapdata_path)))
    md = MapData.load(str(mapdata_path))
    n_ann = len(store.get("annotations", []))
    if n_ann or store.get("deleted_ways") or store.get("split_ways") or store.get("tag_overrides"):
        logger.info("Applying annotation store to %s (%d annotations)", mapdata_path.name, n_ann)
    return apply_store(md, store), store
