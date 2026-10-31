"""Way-editing routes: tag overrides, delete/hide/restore, splits, node add/delete/move."""

import copy
from dataclasses import dataclass
from typing import Any

import utm
from flask import (
    Response,
    abort,
    jsonify,
)

from map_data.annotations import _CAT_FOR_LIST
from map_data.edits import (
    annotation_store,
    apply_added_nodes,
    apply_node_position_overrides,
    edited_nodes_cache,
    get_deleted_node_ids,
    get_deleted_way_ids,
    get_detached_node_ids,
    get_node_position_overrides,
    get_split_node_ids,
    load_annotations,
    next_synthetic_node_id,
    rebuild_way_without_nodes,
    split_way,
    update_segment_annotations_for_split_change,
)
from map_data.map_data import MapData
from map_data.utils.way import FOOTWAY_VALUES

from ..cache import load_mapdata_cached
from ..helpers import geom_to_geojson, way_feature
from .common import (
    _MAX_LAT,
    _MAX_LON,
    _MIN_LAT,
    _MIN_LON,
    _annotation_path,
    _int_param,
    _json_object,
    _log_add,
    _log_remove,
    _mapdata_path,
    _parse_way_id,
    _require_args,
    _split_way_id,
    _validated_number,
    bp,
)


@dataclass
class _ResolvedWay:
    """
    Result of :func:`_resolve_way`.

    Attributes
    ----------
    way : Way or None
        The edited way; ``None`` if not found or deleted down to nothing.
    category : str or None
        ``"road"``, ``"footway"`` or ``"barrier"``; ``None`` only when no way
        has the ID, which tells "not found" from "deleted down to nothing".
    effective_nodes_cache : dict
        ``md``'s nodes cache plus this way's added nodes and detached split
        ends, with its node moves applied (see
        :func:`~map_data.edits.edited_nodes_cache`); the raw cache
        when ``way`` is ``None``.

    """

    way: Any
    category: str | None
    effective_nodes_cache: dict[int, dict[str, Any]]


def _resolve_way(
    md: MapData,
    store: dict[str, Any],
    search_id: int,
    *,
    added_nodes_before_overrides: bool = False,
) -> _ResolvedWay:
    """
    Find a way by its original OSM ID and apply its node deletions, added nodes and moves.

    The way is a ``copy.copy`` of the cached one, which is never mutated.
    Splitting is left to the callers, which pass the returned
    ``effective_nodes_cache`` to :func:`~map_data.edits.split_way`.
    A way deleted down to nothing skips the added-node and move steps.

    Parameters
    ----------
    added_nodes_before_overrides : bool, default False
        Apply added nodes before node moves, so that a move of an added node
        is applied to the geometry (:func:`~map_data.viewer.routes.ways.get_way`);
        otherwise moves come first.

    """
    nodes_cache = getattr(md, "nodes_cache", {})

    way, category = next(
        (
            (copy.copy(w), cat)
            for lst_name, cat in _CAT_FOR_LIST.items()
            for w in getattr(md, lst_name)
            if w.id == search_id
        ),
        (None, None),
    )

    if way is None:
        return _ResolvedWay(way=None, category=None, effective_nodes_cache=nodes_cache)

    zn, zl = md.zone_number, md.zone_letter

    del_nids = get_deleted_node_ids(store, search_id)
    if del_nids:
        way = rebuild_way_without_nodes(way, del_nids, zn, zl, nodes_cache, category=category)

    effective_nc = nodes_cache
    if way is not None:
        pos_overrides = get_node_position_overrides(store, search_id)

        def _apply_overrides(w: Any) -> Any:
            if not pos_overrides:
                return w
            return (
                apply_node_position_overrides(
                    w, pos_overrides, zn, zl, nodes_cache, category=category
                )
                or w
            )

        if added_nodes_before_overrides:
            way = _apply_overrides(apply_added_nodes(way, store, zn, zl))
        else:
            way = apply_added_nodes(_apply_overrides(way), store, zn, zl)

        effective_nc = edited_nodes_cache(store, nodes_cache, search_id)

    return _ResolvedWay(way=way, category=category, effective_nodes_cache=effective_nc)


def _way_node_ids(md: MapData, store: dict[str, Any], way_id: int) -> list[int]:
    """
    Return the node IDs of way *way_id* with its node deletions and added nodes applied.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        404 if the map has no way *way_id*.

    """
    resolved = _resolve_way(md, store, way_id)
    if resolved.category is None:
        abort(404, f"Way {way_id} not found")
    return list(resolved.way.nodes) if resolved.way is not None else []


def _apply_segment_deletions(
    seg: Any,
    seg_id: str,
    store: dict[str, Any],
    zn: int,
    zl: str,
    nodes_cache: dict[int, dict[str, Any]],
    category: str | None,
) -> Any:
    """Rebuild *seg* without its segment-specific deleted nodes (keyed by *seg_id*), if any."""
    seg_del_nids = get_deleted_node_ids(store, seg_id)
    if not seg_del_nids:
        return seg
    return rebuild_way_without_nodes(seg, seg_del_nids, zn, zl, nodes_cache, category=category)


def _select_segment(
    way: Any,
    way_id: str,
    search_id: int,
    store: dict[str, Any],
    zn: int,
    zl: str,
    nodes_cache: dict[int, dict[str, Any]],
    category: str | None,
) -> Any:
    """
    Narrow *way* down to the single segment named by a virtual *way_id*.

    Shared by :func:`~map_data.viewer.routes.ways.get_way_nodes` and
    :func:`~map_data.viewer.routes.ways.get_way`. A no-op (returns *way* unchanged) if *way_id*
    carries no ``":<index>"`` suffix or the way has no recorded splits.

    Returns
    -------
    Any or None
        The resolved segment (with its segment-specific node deletions
        applied), or ``None`` if those deletions reduced it to nothing --
        callers differ on how to report that, so it's left to them.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 for an invalid segment suffix. 404 for an out-of-range segment
        index.

    """
    _, segment_idx = _split_way_id(way_id)
    if segment_idx is None:
        return way

    split_nids = get_split_node_ids(store, search_id)
    if not split_nids:
        return way

    segments = split_way(
        way, split_nids, zn, zl, nodes_cache, get_detached_node_ids(store, search_id)
    )
    if segment_idx >= len(segments):
        abort(404, "Segment not found")

    return _apply_segment_deletions(
        segments[segment_idx], way_id, store, zn, zl, nodes_cache, category
    )


def _apply_tag_override(
    tags: dict[str, Any],
    ov: dict[str, Any] | None,
    category: str | None,
) -> tuple[dict[str, Any], str | None]:
    """
    Merge a tag override into *tags* and re-derive a road/footway *category*.

    Returns *tags*/*category* unchanged if *ov* is falsy. Otherwise merges
    *ov* over *tags* and, if *category* is ``"road"`` or ``"footway"``,
    recomputes it from the merged ``highway`` tag (any other category is
    left alone).
    """
    if not ov:
        return tags, category
    merged = {**tags, **ov}
    if category in ("road", "footway"):
        hw = merged.get("highway", "")
        category = "footway" if hw in FOOTWAY_VALUES else "road"
    return merged, category


@bp.route("/api/way_nodes")
def get_way_nodes() -> Response:
    """
    Return the edited node list of a way or (``"<id>:<index>"``) way segment.

    Query: ``file``, ``way_id``. Positions come from the edited nodes cache,
    else the way's geometry; a way without nodes yields its centroid.

    Returns
    -------
    Response
        JSON ``{"way_id": ..., "nodes": [{"id", "lat", "lon", "tags"}, ...]}``;
        ``nodes`` is ``[]`` (200) if deletions reduced the way or segment
        to nothing.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file``/``way_id`` is missing or ``way_id`` is invalid. 404
        if the file or way doesn't exist or the segment index is out of range.

    """
    filename, way_id = _require_args("file", "way_id")
    path = _mapdata_path(filename)

    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))

    search_id, _ = _split_way_id(way_id)

    resolved = _resolve_way(md, store, search_id)
    if resolved.category is None:
        abort(404, f"Way {search_id} not found")
    if resolved.way is None:
        return jsonify({"way_id": way_id, "nodes": []})

    way = resolved.way
    category = resolved.category
    zn, zl = md.zone_number, md.zone_letter
    effective_nc = resolved.effective_nodes_cache
    pos_overrides = get_node_position_overrides(store, search_id)

    way = _select_segment(way, way_id, search_id, store, zn, zl, effective_nc, category)
    if way is None:
        return jsonify({"way_id": way_id, "nodes": []})

    nodes = []
    geom_latlon = None
    for i, nid in enumerate(way.nodes):
        if nid in effective_nc:
            nd = effective_nc[nid]
            nodes.append(
                {"id": nid, "lat": nd["lat"], "lon": nd["lon"], "tags": nd.get("tags", {})}
            )
        else:
            if geom_latlon is None:
                geom = way.line
                raw = list(geom.exterior.coords if hasattr(geom, "exterior") else geom.coords)
                geom_latlon = [utm.to_latlon(e, n, zn, zl) for e, n in raw]
            if i < len(geom_latlon):
                lat, lon = geom_latlon[i]
                nodes.append({"id": nid, "lat": lat, "lon": lon, "tags": {}})

    if not nodes and way.line:
        centroid = way.line.centroid
        lat, lon = utm.to_latlon(centroid.x, centroid.y, zn, zl)
        nodes = [{"id": search_id, "lat": lat, "lon": lon, "tags": {}}]

    if pos_overrides:
        for n in nodes:
            if n["id"] in pos_overrides:
                n["lat"] = pos_overrides[n["id"]]["lat"]
                n["lon"] = pos_overrides[n["id"]]["lon"]

    return jsonify({"way_id": way_id, "nodes": nodes})


@bp.route("/api/ways/<way_id>")
def get_way(way_id: str) -> Response:
    """
    Return an edited way or (``"<id>:<index>"``) way segment as a GeoJSON Feature.

    Query: ``file``. Tags and category include the way's tag override; see
    :func:`~map_data.viewer.helpers.way_feature` for the properties.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or ``way_id`` is invalid. 404 if the file
        or way doesn't exist, deletions reduced it (or the segment) to
        nothing, or the segment index is out of range. 500 if the geometry
        can't be converted to GeoJSON.

    """
    filename = _require_args("file")
    path = _mapdata_path(filename)

    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))

    search_id, _ = _split_way_id(way_id)

    resolved = _resolve_way(md, store, search_id, added_nodes_before_overrides=True)
    if resolved.category is None:
        abort(404, f"Way {way_id} not found")
    if resolved.way is None:
        abort(404, f"Way {way_id} reduced to nothing by node deletions")

    way = resolved.way
    category = resolved.category
    nodes_cache = resolved.effective_nodes_cache
    zn, zl = md.zone_number, md.zone_letter

    way = _select_segment(way, way_id, search_id, store, zn, zl, nodes_cache, category)
    if way is None:
        abort(404, "Segment reduced to nothing by segment-specific deletions")

    geom = geom_to_geojson(way.line, zn, zl)
    if geom is None:
        abort(500, "Could not convert geometry")

    feature = way_feature(way, way_id, category, way.tags or {}, geom)

    ov = store.get("tag_overrides", {}).get(str(search_id))
    feature["properties"]["tags"], feature["properties"]["category"] = _apply_tag_override(
        feature["properties"]["tags"],
        ov,
        category,
    )

    return jsonify(feature)


@bp.route("/api/ways/<way_id>", methods=["DELETE"])
def delete_way(way_id: str) -> Response:
    """
    Mark a way, or one ``"<id>:<index>"`` segment of it, as deleted (idempotent).

    Query: ``file``; optional JSON body ``category``/``label`` for the
    change log. A segment ID is stored as a string, so only that segment is
    deleted; a plain ID as an ``int``, deleting the whole way. Returns 204;
    400 if ``file`` is missing or ``way_id`` is invalid.
    """
    filename = _require_args("file")

    stored_id = _parse_way_id(way_id)

    body = _json_object()
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        if stored_id not in get_deleted_way_ids(store):
            store.setdefault("deleted_ways", []).append(
                {
                    "id": stored_id,
                    "category": body.get("category", "unknown"),
                    "label": body.get("label", ""),
                },
            )
            _log_add(store, "way", {"id": stored_id})
    return Response("", 204)


@bp.route("/api/ways/<way_id>/tags", methods=["PUT"])
def update_way_tags(way_id: str) -> Response:
    """
    Replace the tag override of a way with the body's ``tags`` (str -> str).

    Query: ``file``; JSON body ``tags`` and optional ``category``/``label``.
    Keyed by the original way ID, so it applies to every segment. Returns
    204; 400 if ``file`` is missing or ``tags`` is not a dict of strings.
    """
    filename = _require_args("file")

    original_way_id_str = str(_split_way_id(way_id)[0])

    body = _json_object()
    tags = body.get("tags")
    # OSM tags are str -> str; anything else breaks the category lookup on every load.
    if not isinstance(tags, dict) or not all(isinstance(v, str) for v in tags.values()):
        abort(400, "Request body must include a 'tags' dict of strings")
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        store.setdefault("tag_overrides", {})[original_way_id_str] = tags
        store.setdefault("tag_override_meta", {})[original_way_id_str] = {
            "category": body.get("category", "unknown"),
            "label": body.get("label", ""),
        }
        _log_add(store, "tag", {"id": original_way_id_str})
    return Response("", 204)


@bp.route("/api/ways/<way_id>/tags", methods=["DELETE"])
def delete_way_tags(way_id: str) -> Response:
    """
    Remove the tag override of a way's original ID (no-op if none).

    Query: ``file``. Returns 204; 400 if ``file`` is missing or ``way_id``
    is invalid.
    """
    filename = _require_args("file")
    original_way_id_str = str(_split_way_id(way_id)[0])
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        store.get("tag_overrides", {}).pop(original_way_id_str, None)
        store.get("tag_override_meta", {}).pop(original_way_id_str, None)
        _log_remove(store, "tag", id=original_way_id_str)
    return Response("", 204)


@bp.route("/api/ways/<way_id>/segments")
def get_way_segments(way_id: str) -> Response:
    """
    Return every segment of a (possibly split) way as GeoJSON Features.

    Query: ``file``; a ``":<index>"`` suffix on ``way_id`` is ignored.
    Returns ``{"segments": [Feature, ...]}``, ``[]`` for a missing or fully
    deleted way. 400 if ``file`` is missing or invalid, 404 if the file
    doesn't exist.
    """
    filename = _require_args("file")
    original_way_id = str(_split_way_id(way_id)[0])
    segments = _get_way_segments_geojson(filename, original_way_id)
    return jsonify({"segments": segments})


def _get_way_segments_geojson(filename: str, original_way_id: str) -> list[dict[str, Any]]:
    """
    Return the current segments of a way, with all its edits, as GeoJSON Features.

    An unsplit way is one Feature. ``[]`` if the way doesn't exist or was
    deleted down to nothing. *original_way_id* must already be validated:
    a non-integer raises ``ValueError``.
    """
    path = _mapdata_path(filename)
    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))

    search_id = int(original_way_id)
    resolved = _resolve_way(md, store, search_id)
    if resolved.category is None or resolved.way is None:
        return []

    way = resolved.way
    category = resolved.category
    zn, zl = md.zone_number, md.zone_letter
    effective_nc = resolved.effective_nodes_cache

    split_nids = get_split_node_ids(store, search_id)
    segments = split_way(
        way, split_nids, zn, zl, effective_nc, get_detached_node_ids(store, search_id)
    )

    features = []
    tag_overrides = store.get("tag_overrides", {})
    ov = tag_overrides.get(str(original_way_id))

    for i, seg in enumerate(segments):
        virtual_id = f"{original_way_id}:{i}"

        seg = _apply_segment_deletions(seg, virtual_id, store, zn, zl, effective_nc, category)
        if seg is None:
            continue

        tags, feat_cat = _apply_tag_override(seg.tags or {}, ov, category)

        # An unsplit way keeps its plain id, the one /api/mapdata and its deletions use.
        seg_id = virtual_id if len(segments) > 1 else int(original_way_id)
        features.append(way_feature(seg, seg_id, feat_cat, tags, geom_to_geojson(seg.line, zn, zl)))
    return features


@bp.route("/api/ways/split", methods=["POST"])
def split_way_endpoint() -> Response:
    """
    Record a split point on a way and return its segments.

    Query: ``file``; JSON body ``way_id`` (a segment suffix is ignored) and
    ``node_id`` (an interior node). The segment after the split starts at a
    new synthetic copy of the node (``store["detached_nodes"]``, inheriting
    its move), so both ends move independently and the planner does not
    route across the split.

    Returns
    -------
    Response
        JSON ``{"success": true, "segments": [Feature, ...]}``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing, ``way_id``/``node_id`` is invalid or
        ``node_id`` is not an interior node of the way. 404 if the file or
        way doesn't exist.

    """
    filename = _require_args("file")
    body = _json_object()
    # A virtual way_id (e.g. 123:0) splits the original way.
    original_way_id = str(_split_way_id(body.get("way_id"))[0])
    node_id_int = _int_param(body.get("node_id"), "node_id")

    md = load_mapdata_cached(str(_mapdata_path(filename)))
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        if node_id_int not in _way_node_ids(md, store, int(original_way_id))[1:-1]:
            abort(400, f"Node {node_id_int} is not an interior node of way {original_way_id}")
        splits = store.setdefault("split_ways", {})
        way_splits = splits.setdefault(original_way_id, [])

        if node_id_int not in way_splits:
            old_splits = list(way_splits)
            way_splits.append(node_id_int)
            new_splits = list(way_splits)

            detached_id = next_synthetic_node_id(store)
            store.setdefault("detached_nodes", []).append(
                {"way_id": int(original_way_id), "node_id": node_id_int, "id": detached_id}
            )
            way_ov = store.get("node_position_overrides", {}).get(original_way_id, {})
            if str(node_id_int) in way_ov:
                way_ov[str(detached_id)] = dict(way_ov[str(node_id_int)])

            resolved = _resolve_way(md, store, int(original_way_id))
            if resolved.way is not None:
                update_segment_annotations_for_split_change(
                    store, int(original_way_id), resolved.way, old_splits, new_splits
                )

            _log_add(store, "split", {"way_id": int(original_way_id), "node_id": node_id_int})

    segments = _get_way_segments_geojson(filename, original_way_id)
    return jsonify({"success": True, "segments": segments})


@bp.route("/api/ways/split", methods=["DELETE"])
def undo_way_split() -> Response:
    """
    Remove one split point from a way and return its segments.

    Query: ``file``, ``way_id``, ``node_id``. The split node is restored as
    in the map: its detached copy and any deletion of that copy are dropped
    and the moves of both ends discarded (with the way's ``move`` change-log
    entry once no moved node is left).

    Returns
    -------
    Response
        JSON ``{"segments": [Feature, ...]}``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if a parameter is missing or not an integer. 404 if the file
        doesn't exist.

    """
    filename, way_id, node_id = _require_args("file", "way_id", "node_id")
    way_id_int = _int_param(way_id, "way_id")
    node_id_int = _int_param(node_id, "node_id")
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        splits = store.get("split_ways", {})
        if str(way_id_int) in splits:
            old_splits = list(splits[str(way_id_int)])
            new_splits = [nid for nid in old_splits if nid != node_id_int]

            if len(new_splits) != len(old_splits):
                splits[str(way_id_int)] = new_splits
                if not new_splits:
                    del splits[str(way_id_int)]

                detached_id = get_detached_node_ids(store, way_id_int).get(node_id_int)
                if detached_id is not None:
                    store["detached_nodes"] = [
                        d for d in store["detached_nodes"] if d["id"] != detached_id
                    ]
                    store["deleted_nodes"] = [
                        d for d in store.get("deleted_nodes", []) if d["node_id"] != detached_id
                    ]
                overrides = store.get("node_position_overrides", {})
                way_ov = overrides.get(str(way_id_int), {})
                way_ov.pop(str(node_id_int), None)
                way_ov.pop(str(detached_id), None)
                if str(way_id_int) in overrides and not way_ov:
                    del overrides[str(way_id_int)]
                    _log_remove(store, "move", id=way_id_int)

                path = _mapdata_path(filename)
                md = load_mapdata_cached(str(path))
                resolved = _resolve_way(md, store, way_id_int)
                if resolved.way is not None:
                    update_segment_annotations_for_split_change(
                        store, way_id_int, resolved.way, old_splits, new_splits
                    )

        _log_remove(store, "split", way_id=way_id_int, node_id=node_id_int)

    segments = _get_way_segments_geojson(filename, str(way_id_int))
    return jsonify({"segments": segments})


@bp.route("/api/ways/<way_id>/hide", methods=["PUT"])
def hide_way(way_id: str) -> Response:
    """
    Hide a whole way in the viewer (idempotent); export and planning ignore it.

    Query: ``file``; optional JSON body ``category``/``label``. Recorded in
    ``store["hidden_ways"]`` by original ID, with no ``change_log`` entry:
    hiding is not an edit. Returns 204; 400 if ``file`` is missing or
    ``way_id`` is invalid.
    """
    filename = _require_args("file")

    way_id_int, _ = _split_way_id(way_id)

    body = _json_object()
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        hw = store.setdefault("hidden_ways", [])
        existing_ids = {d["id"] for d in hw}
        if way_id_int not in existing_ids:
            hw.append(
                {
                    "id": way_id_int,
                    "category": body.get("category", "unknown"),
                    "label": body.get("label", ""),
                },
            )
    return Response("", 204)


@bp.route("/api/ways/<way_id>/show", methods=["PUT"])
def show_way(way_id: str) -> Response:
    """
    Undo :func:`hide_way` (no-op if not hidden).

    Query: ``file``. Returns 204; 400 if ``file`` is missing or ``way_id``
    is invalid.
    """
    filename = _require_args("file")

    way_id_int, _ = _split_way_id(way_id)

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        hw = store.get("hidden_ways", [])
        store["hidden_ways"] = [d for d in hw if d["id"] != way_id_int]
    return Response("", 204)


@bp.route("/api/ways/<way_id>/restore", methods=["PUT"])
def restore_way(way_id: str) -> Response:
    """
    Undo :func:`delete_way` for the same way or segment ID (no-op if not deleted).

    Query: ``file``. Returns 204; 400 if ``file`` is missing or ``way_id``
    is invalid.
    """
    filename = _require_args("file")

    stored_id = _parse_way_id(way_id)

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        dw = store.get("deleted_ways", [])
        store["deleted_ways"] = [d for d in dw if d["id"] != stored_id]
        _log_remove(store, "way", id=stored_id)
    return Response("", 204)


@bp.route("/api/way_node", methods=["POST"])
def add_way_node() -> Response:
    """
    Insert a new synthetic node into a way after an existing node.

    Query: ``file``, ``way_id``; JSON body ``after_node_id`` (may itself be
    synthetic), ``lat``, ``lon``. The node gets the next free negative ID
    (:func:`~map_data.edits.next_synthetic_node_id`; OSM IDs are
    positive) and is recorded in ``store["added_nodes"]``.

    Returns
    -------
    Response
        JSON ``{"id": <new synthetic node id>, "lat": ..., "lon": ...}``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if a parameter is missing or invalid, ``lat``/``lon`` is out
        of range, or ``after_node_id`` is not on the way. 404 if the file
        or way doesn't exist.

    """
    filename, way_id = _require_args("file", "way_id")
    way_id_int, _ = _split_way_id(way_id)

    body = _json_object()
    if any(body.get(k) is None for k in ("after_node_id", "lat", "lon")):
        abort(400, "Request body must include after_node_id, lat, lon")
    after_node_id = _int_param(body["after_node_id"], "after_node_id")
    lat = _validated_number(body["lat"], "lat", _MIN_LAT, _MAX_LAT)
    lon = _validated_number(body["lon"], "lon", _MIN_LON, _MAX_LON)

    md = load_mapdata_cached(str(_mapdata_path(filename)))
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        if after_node_id not in _way_node_ids(md, store, way_id_int):
            abort(400, f"Node {after_node_id} is not on way {way_id_int}")
        synth_id = next_synthetic_node_id(store)

        store.setdefault("added_nodes", []).append(
            {
                "id": synth_id,
                "way_id": way_id_int,
                "after_node_id": after_node_id,
                "lat": lat,
                "lon": lon,
            }
        )
        _log_add(store, "add_node", {"way_id": way_id_int, "node_id": synth_id})
    return jsonify({"id": synth_id, "lat": lat, "lon": lon})


@bp.route("/api/way_node", methods=["DELETE"])
def delete_way_node() -> Response:
    """
    Delete a node from a way, or from one ``"<id>:<index>"`` segment of it.

    Query: ``file``, ``way_id``, ``node_id``. A node added by
    :func:`add_way_node` is removed from ``added_nodes`` together with its
    move. Any other node, including a detached split end, is recorded in
    ``store["deleted_nodes"]`` under the given ``way_id``, so a segment ID
    deletes it from that segment only. Returns 204; 400 if a parameter is
    missing or invalid.
    """
    filename, way_id, node_id_arg = _require_args("file", "way_id", "node_id")

    # The full (possibly segment) way_id keys the deletion.
    target_id = _parse_way_id(way_id)
    way_id_int, _ = _split_way_id(way_id)
    node_id = _int_param(node_id_arg, "node_id")

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        # Added nodes (negative IDs) live in added_nodes, not in the OSM node list;
        # a detached split end is deleted per segment like a real node.
        if node_id < 0 and node_id not in get_detached_node_ids(store, way_id_int).values():
            added = store.get("added_nodes", [])
            removed = next(
                (a for a in added if a.get("way_id") == way_id_int and a.get("id") == node_id), None
            )
            if removed is not None:
                added.remove(removed)
                # Nodes added after this one would lose their anchor and vanish.
                for a in added:
                    if a.get("way_id") == way_id_int and a.get("after_node_id") == node_id:
                        a["after_node_id"] = removed["after_node_id"]
            pos_ov = store.get("node_position_overrides", {}).get(str(way_id_int), {})
            pos_ov.pop(str(node_id), None)
            _log_remove(store, "add_node", way_id=way_id_int, node_id=node_id)
            return Response("", 204)

        dn = store.setdefault("deleted_nodes", [])
        if node_id not in get_deleted_node_ids(store, target_id):
            dn.append({"way_id": target_id, "node_id": node_id})
            _log_add(store, "node", {"way_id": target_id, "node_id": node_id})
    return Response("", 204)


@bp.route("/api/way_node/restore", methods=["PUT"])
def restore_way_node() -> Response:
    """
    Undo :func:`delete_way_node` for a node recorded in ``deleted_nodes`` (no-op if absent).

    Query: ``file``, ``way_id`` (the same, possibly segment, ID), ``node_id``.
    A removed added node cannot be restored. Returns 204; 400 if a parameter
    is missing or invalid.
    """
    filename, way_id, node_id_arg = _require_args("file", "way_id", "node_id")

    target_id = _parse_way_id(way_id)
    node_id = _int_param(node_id_arg, "node_id")

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        dn = store.get("deleted_nodes", [])
        store["deleted_nodes"] = [
            d
            for d in dn
            if not (str(d.get("way_id")) == str(target_id) and d.get("node_id") == node_id)
        ]
        _log_remove(store, "node", way_id=target_id, node_id=node_id)
    return Response("", 204)


@bp.route("/api/way_nodes/move", methods=["PUT"])
def move_way_nodes() -> Response:
    """
    Record new positions for nodes of a way.

    Query: ``file``, ``way_id``; JSON body ``nodes``
    (``[{"id", "lat", "lon"}, ...]``) and optional ``category``/``label``.
    Merged per node into ``store["node_position_overrides"]`` under the
    original way ID, so every segment shares them. Returns 204; 400 if a
    parameter is missing or invalid, a position is out of range or a node
    is not on the way (its split ends included); 404 if the file or way
    doesn't exist.
    """
    filename, way_id = _require_args("file", "way_id")
    way_id_int, _ = _split_way_id(way_id)

    body = _json_object()
    nodes = body.get("nodes")
    if not isinstance(nodes, list) or not all(isinstance(n, dict) for n in nodes):
        abort(400, "Request body must include a 'nodes' list of objects")
    # Validate every node before touching the store: one bad entry would break the file.
    moves = {
        str(_int_param(n.get("id"), "node id")): {
            "lat": _validated_number(n.get("lat"), "lat", _MIN_LAT, _MAX_LAT),
            "lon": _validated_number(n.get("lon"), "lon", _MIN_LON, _MAX_LON),
        }
        for n in nodes
    }
    md = load_mapdata_cached(str(_mapdata_path(filename)))
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        node_ids = _way_node_ids(md, store, way_id_int)
        # A way without nodes (a point barrier) is moved under its own ID, see get_way_nodes.
        on_way = (
            {*node_ids, *get_detached_node_ids(store, way_id_int).values()}
            if node_ids
            else {way_id_int}
        )
        off_way = sorted(int(nid) for nid in moves if int(nid) not in on_way)
        if off_way:
            abort(400, f"Nodes {off_way} are not on way {way_id_int}")
        overrides = store.setdefault("node_position_overrides", {})
        way_key = str(way_id_int)
        if way_key not in overrides:
            overrides[way_key] = {}
        overrides[way_key].update(moves)
        _log_add(
            store,
            "move",
            {"id": way_id_int},
            category=body.get("category", "unknown"),
            label=body.get("label", ""),
        )
    return Response("", 204)


@bp.route("/api/way_nodes/move", methods=["DELETE"])
def undo_move_way_nodes() -> Response:
    """
    Undo :func:`move_way_nodes`: discard every node move of a way (no per-node undo).

    Query: ``file``, ``way_id``. Returns 204; 400 if a parameter is missing
    or invalid.
    """
    filename, way_id = _require_args("file", "way_id")
    way_id_int, _ = _split_way_id(way_id)

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        store.get("node_position_overrides", {}).pop(str(way_id_int), None)
        _log_remove(store, "move", id=way_id_int)
    return Response("", 204)
