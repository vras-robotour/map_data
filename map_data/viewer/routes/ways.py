"""Way-editing routes: tag overrides, delete/hide/restore, splits, node add/delete/move."""

from typing import Any

import utm
from flask import (
    Response,
    abort,
    jsonify,
    request,
)

from ..cache import load_mapdata_cached
from ..helpers import (
    annotation_store,
    geom_to_geojson,
    get_deleted_node_ids,
    get_deleted_way_ids,
    get_detached_node_ids,
    get_node_position_overrides,
    get_split_node_ids,
    load_annotations,
    next_synthetic_node_id,
    split_way,
    update_segment_annotations_for_split_change,
    way_feature,
)
from .common import (
    _annotation_path,
    _apply_segment_deletions,
    _apply_tag_override,
    _log_add,
    _log_remove,
    _mapdata_path,
    _original_way_id,
    _parse_way_id,
    _require_args,
    _resolve_way,
    _select_segment,
    bp,
)


@bp.route("/api/way_nodes")
def get_way_nodes() -> Response:
    """
    Return the resolved node list (id/lat/lon/tags) for a way or way segment.

    Resolves ``way_id`` (an original OSM way ID, or a virtual
    ``"<id>:<segment_index>"`` produced by :func:`split_way_endpoint`)
    through :func:`~map_data.viewer.routes.common._resolve_way`, then, if virtual, narrows down to
    the requested segment. Node positions prefer the effective nodes cache
    (including any user-added synthetic nodes), falling back to the way's
    own geometry coordinates and, if the way has no nodes at all, to a
    single centroid point.

    Returns
    -------
    Response
        JSON ``{"way_id": ..., "nodes": [{"id", "lat", "lon", "tags"}, ...]}``.
        ``nodes`` is ``[]`` (with a 200 status, not a 404) if node
        deletions reduced the way -- or the requested segment -- to
        nothing.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file``/``way_id`` is missing, ``way_id`` isn't a valid
        (possibly virtual) integer ID, or the virtual ID's segment suffix
        isn't a valid integer. 404 if the file or the original way
        doesn't exist, or the requested segment index is out of range.

    """
    filename, way_id = _require_args("file", "way_id")
    path = _mapdata_path(filename)

    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))

    search_id = _original_way_id(way_id)

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
    for i, nid_obj in enumerate(way.nodes):
        nid = getattr(nid_obj, "id", nid_obj)
        if nid in effective_nc:
            nd = effective_nc[nid]
            nodes.append(
                {"id": nid, "lat": nd["lat"], "lon": nd["lon"], "tags": nd.get("tags", {})}
            )
        else:
            # Fallback to geometry
            if geom_latlon is None:
                geom = way.line
                raw = list(geom.exterior.coords if hasattr(geom, "exterior") else geom.coords)
                geom_latlon = [utm.to_latlon(e, n, zn, zl) for e, n in raw]
            if i < len(geom_latlon):
                lat, lon = geom_latlon[i]
                nodes.append({"id": nid, "lat": lat, "lon": lon, "tags": {}})

    # centroid fallback
    if not nodes and way.line:
        centroid = way.line.centroid
        lat, lon = utm.to_latlon(centroid.x, centroid.y, zn, zl)
        nodes = [{"id": search_id, "lat": lat, "lon": lon, "tags": {}}]

    # Ensure position overrides are applied to the resulting nodes list
    if pos_overrides:
        for n in nodes:
            if n["id"] in pos_overrides:
                n["lat"] = pos_overrides[n["id"]]["lat"]
                n["lon"] = pos_overrides[n["id"]]["lon"]

    return jsonify({"way_id": way_id, "nodes": nodes})


@bp.route("/api/ways/<way_id>")
def get_way(way_id: str) -> Response:
    """
    Return a single way (or way segment) as a GeoJSON Feature.

    Resolves ``way_id`` (an original OSM way ID, or a virtual
    ``"<id>:<segment_index>"`` produced by :func:`split_way_endpoint`)
    through :func:`~map_data.viewer.routes.common._resolve_way` -- passing
    ``added_nodes_before_overrides=True``, unlike
    :func:`get_way_nodes`/:func:`_get_way_segments_geojson`, so that a recorded position override
    for a user-added node is picked up by ``apply_node_position_overrides`` -- then, if virtual,
    narrows down to the requested segment and applies any tag overrides recorded for the
    original way.

    Returns
    -------
    Response
        A GeoJSON ``Feature`` with ``properties`` including ``id``,
        ``category``, ``is_node`` (true for a barrier with no OSM nodes,
        i.e. a synthesized point obstacle), ``tags`` (merged with any tag
        override), and ``in_out``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing, ``way_id`` isn't a valid (possibly
        virtual) integer ID, or its segment suffix isn't a valid integer.
        404 if the file or the original way doesn't exist, node deletions
        reduced the way (or requested segment) to nothing, or the
        requested segment index is out of range. 500 if the resolved
        geometry can't be converted to GeoJSON.

    """
    filename = _require_args("file")
    path = _mapdata_path(filename)

    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))

    search_id = _original_way_id(way_id)

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
    Mark a way (or a single split segment) as deleted.

    Records ``way_id`` in ``store["deleted_ways"]`` (idempotent -- a
    second delete of the same ID is a no-op) and appends a
    ``change_log`` entry. A virtual segment ID (``"<id>:<index>"``) is
    stored as-is (a string), so only that segment is suppressed on
    reload; a plain numeric ID is stored as an ``int`` and suppresses the
    whole way.

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or ``way_id`` isn't a valid plain or
        virtual way ID (see :func:`~map_data.viewer.routes.common._parse_way_id`).

    """
    filename = _require_args("file")

    # Segments (virtual IDs like "123:0") are stored as strings so that only the
    # specific segment is suppressed on reload, not the whole original way.
    stored_id = _parse_way_id(way_id)

    body = request.get_json(force=True) or {}
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
    Replace the tag-override dict recorded for a way's original OSM ID.

    Overrides are keyed by the *original* way ID (a virtual segment ID's
    ``":<index>"`` suffix is stripped), so overrides apply to every
    segment of a split way uniformly. Overwrites any previous override
    for this way wholesale (not merged with the new *tags*).

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or the request body has no ``tags``
        dict.

    """
    filename = _require_args("file")

    original_way_id_str = str(way_id).split(":")[0]

    body = request.get_json(force=True) or {}
    tags = body.get("tags")
    if not isinstance(tags, dict):
        abort(400, "Request body must include 'tags' dict")
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
    Remove the tag override recorded for a way.

    Unlike :func:`update_way_tags`, this looks the override up by
    ``way_id`` verbatim rather than stripping a ``":<index>"`` segment
    suffix first -- passing a virtual segment ID here will not remove an
    override that was recorded under the original way ID. A missing
    override is not an error (no-op removal).

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing.

    """
    filename = _require_args("file")
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        store.get("tag_overrides", {}).pop(str(way_id), None)
        store.get("tag_override_meta", {}).pop(str(way_id), None)
        _log_remove(store, "tag", id=way_id)
    return Response("", 204)


@bp.route("/api/ways/<way_id>/segments")
def get_way_segments(way_id: str) -> Response:
    """
    Return every segment of a (possibly split) way as GeoJSON Features.

    Thin wrapper around :func:`_get_way_segments_geojson`; unlike
    :func:`get_way`, this always computes *all* segments regardless of
    whether ``way_id`` carries a ``":<index>"`` suffix (any suffix is
    stripped and ignored).

    Returns
    -------
    Response
        JSON ``{"segments": [Feature, ...]}``. ``segments`` is ``[]`` if
        the file, the way, or every remaining segment doesn't exist --
        this endpoint never 404s on a missing way.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or invalid. 404 if the file doesn't exist.

    """
    filename = _require_args("file")
    original_way_id = str(way_id).split(":")[0]
    segments = _get_way_segments_geojson(filename, original_way_id)
    return jsonify({"segments": segments})


def _get_way_segments_geojson(filename: str, original_way_id: str) -> list[dict[str, Any]]:
    """
    Resolve a way's edits and split it into GeoJSON Feature segments.

    Shared by :func:`get_way_segments`, :func:`split_way_endpoint`, and
    :func:`undo_way_split` -- all three want the *current* full set of
    segments for a way after a split-point change. Resolves
    ``original_way_id`` via :func:`~map_data.viewer.routes.common._resolve_way` (default
    ``added_nodes_before_overrides=False``, matching :func:`get_way_nodes`),
    then always calls :func:`~map_data.viewer.helpers.split_way` (even if
    the way has no recorded splits, in which case it returns the way
    unchanged as the sole "segment"), and applies segment-specific node
    deletions and the way's tag overrides to each resulting segment.

    Parameters
    ----------
    filename : str
        Mapdata filename, resolved and validated via
        :func:`~map_data.viewer.routes.common._mapdata_path`.
    original_way_id : str
        Original (non-virtual) OSM way ID, as a string.

    Returns
    -------
    list of dict
        One GeoJSON ``Feature`` per segment (``[]`` if the way doesn't
        exist, was deleted down to nothing, or ``original_way_id`` can't
        be parsed as an int -- the latter raises ``ValueError`` instead,
        uncaught, since callers are expected to have already validated it).

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

        # An unsplit way keeps its plain id, the one /api/mapdata uses; a
        # "<id>:0" here gets deleted as a segment the full load never applies.
        seg_id = virtual_id if len(segments) > 1 else int(original_way_id)
        features.append(way_feature(seg, seg_id, feat_cat, tags, geom_to_geojson(seg.line, zn, zl)))
    return features


@bp.route("/api/ways/split", methods=["POST"])
def split_way_endpoint() -> Response:
    """
    Record a split point on a way and return its resulting segments.

    Adds *node_id* to ``store["split_ways"][original_way_id]`` (a way may
    have several split points, producing more than two segments) and
    then recomputes every segment via :func:`_get_way_segments_geojson`
    so the client can immediately render the result.

    The split detaches the segments: the one after the split starts at a new
    synthetic copy of *node_id* (``store["detached_nodes"]``, inheriting any
    move of the node), so both ends can be moved independently and the
    planner does not route across the split.

    Parameters (JSON body)
    -----------------------
    way_id : int or str
        Way ID to split (a virtual segment ID's suffix is stripped, so
        splitting always applies to the original way).
    node_id : int
        Node ID to split at (must be interior to the way; see
        :func:`~map_data.viewer.helpers.split_way`).

    Returns
    -------
    Response
        JSON ``{"success": true, "segments": [Feature, ...]}``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or ``way_id``/``node_id`` aren't
        valid.

    """
    filename = _require_args("file")
    body = request.get_json(force=True) or {}
    way_id = body.get("way_id")
    node_id = body.get("node_id")

    try:
        way_id_val = str(way_id)
        # node_id may be None (missing from body); int(None) raises TypeError,
        # which is caught below -- not a genuine unguarded None deref.
        node_id_int = int(node_id)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        abort(400, "Invalid way_id or node_id")

    # If way_id is virtual (e.g. 123:0), get original ID
    original_way_id = way_id_val.split(":", maxsplit=1)[0]

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
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

            # Re-map segment references
            path = _mapdata_path(filename)
            md = load_mapdata_cached(str(path))
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
    Remove a single recorded split point from a way and return its (new) segments.

    The way still ends up split if other split points remain on it;
    removing the last one collapses ``split_ways`` back to a single
    unsplit way.

    The split node is restored as it was in the map: its detached copy is
    dropped along with any deletion of it, and the moves of both ends are
    discarded (the way's ``move`` change-log entry goes too once no moved
    node is left).

    Returns
    -------
    Response
        JSON ``{"segments": [Feature, ...]}`` for the way's current
        segments (a one-element list, containing the whole way, if no
        splits remain).

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file``/``way_id``/``node_id`` is missing or not a valid
        integer.

    """
    filename, way_id, node_id = _require_args("file", "way_id", "node_id")
    try:
        way_id_int = int(way_id)
        node_id_int = int(node_id)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers")
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
    Mark a way as hidden (a lighter-weight, purely-client-side visibility toggle than deletion).

    Recorded in ``store["hidden_ways"]`` by the way's original ID (any
    segment suffix is stripped, so hiding applies to the whole way);
    idempotent. Unlike :func:`delete_way`, this has no ``change_log``
    entry -- hiding is not treated as an edit for undo/history purposes.

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or ``way_id``'s non-segment part isn't
        a valid integer.

    """
    filename = _require_args("file")

    way_id_int = _original_way_id(way_id)

    body = request.get_json(force=True) or {}
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
    Undo :func:`hide_way`: remove a way from ``store["hidden_ways"]``.

    Not being hidden is not an error (no-op removal).

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or ``way_id``'s non-segment part isn't
        a valid integer.

    """
    filename = _require_args("file")

    way_id_int = _original_way_id(way_id)

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        hw = store.get("hidden_ways", [])
        store["hidden_ways"] = [d for d in hw if d["id"] != way_id_int]
    return Response("", 204)


@bp.route("/api/ways/<way_id>/restore", methods=["PUT"])
def restore_way(way_id: str) -> Response:
    """
    Undo :func:`delete_way`: remove a way (or segment) from ``deleted_ways``.

    Mirrors :func:`delete_way`'s stored-ID convention -- a virtual
    segment ID restores only that segment, a plain way ID restores the
    whole way. Not being deleted is not an error (no-op removal).

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or ``way_id`` isn't a valid plain or
        virtual way ID (see :func:`~map_data.viewer.routes.common._parse_way_id`).

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
    Insert a new synthetic node into a way, immediately after an existing node.

    Assigns the new node a fresh negative ID (one less than the smallest
    existing synthetic ID recorded for this file, or ``-1`` if none),
    since real OSM node IDs are always positive; recorded in
    ``store["added_nodes"]`` and consumed by
    :func:`~map_data.viewer.helpers.apply_added_nodes`.

    Parameters (JSON body)
    -----------------------
    after_node_id : int
        Existing node ID (may itself be a previously-added synthetic
        node) the new node should be spliced in after.
    lat, lon : float
        Position of the new node.

    Returns
    -------
    Response
        JSON ``{"id": <new synthetic node id>, "lat": ..., "lon": ...}``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file``/``way_id`` is missing, ``way_id``'s non-segment
        part isn't a valid integer, or the body is missing
        ``after_node_id``/``lat``/``lon`` or ``after_node_id`` isn't an integer.

    """
    filename, way_id = _require_args("file", "way_id")
    way_id_int = _original_way_id(way_id)

    body = request.get_json(force=True) or {}
    after_node_id = body.get("after_node_id")
    lat = body.get("lat")
    lon = body.get("lon")
    if after_node_id is None or lat is None or lon is None:
        abort(400, "Request body must include after_node_id, lat, lon")
    try:
        after_node_id = int(after_node_id)
    except (ValueError, TypeError):
        abort(400, "after_node_id must be an integer")

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        synth_id = next_synthetic_node_id(store)

        store.setdefault("added_nodes", []).append(
            {
                "id": synth_id,
                "way_id": way_id_int,
                "after_node_id": after_node_id,
                "lat": float(lat),
                "lon": float(lon),
            }
        )
        _log_add(store, "add_node", {"way_id": way_id_int, "node_id": synth_id})
    return jsonify({"id": synth_id, "lat": float(lat), "lon": float(lon)})


@bp.route("/api/way_node", methods=["DELETE"])
def delete_way_node() -> Response:
    """
    Delete a node from a way (or a specific split segment).

    Synthetic nodes (negative ``node_id``, previously created by
    :func:`add_way_node`) are removed outright from ``added_nodes`` (plus
    any position override for them) rather than recorded as a deletion,
    since they don't exist independently of that record. Real (positive
    ID) OSM nodes are instead recorded in ``store["deleted_nodes"]``,
    keyed by the *given* ``way_id`` (which may be a virtual segment ID,
    for a deletion that should apply to only that segment, or a plain
    way ID for a deletion applying to every segment).

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if a required parameter is missing, or ``way_id``'s
        non-segment part / ``node_id`` isn't a valid integer.

    """
    filename, way_id, node_id_arg = _require_args("file", "way_id", "node_id")

    # Use the full way_id (could be virtual like "123:0") to allow segment-specific
    # deletion; _parse_way_id validates the segment suffix so raw strings never
    # reach the store.
    target_id = _parse_way_id(way_id)
    way_id_int = _original_way_id(target_id)
    try:
        node_id = int(node_id_arg)
    except (ValueError, TypeError):
        abort(400, "node_id must be an integer")

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        # Added nodes (negative IDs) live in added_nodes, not in the OSM node list;
        # a detached split end is deleted per segment like a real node.
        if node_id < 0 and node_id not in get_detached_node_ids(store, way_id_int).values():
            store["added_nodes"] = [
                a
                for a in store.get("added_nodes", [])
                if not (a.get("way_id") == way_id_int and a.get("id") == node_id)
            ]
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
    Undo :func:`delete_way_node` for a real (non-synthetic) node.

    Only handles the ``deleted_nodes`` record path -- there is no
    counterpart here for restoring a synthetic node that was outright
    removed by ``delete_way_node`` (that node is gone for good; the user
    would re-add it via :func:`add_way_node`). Not being deleted is not
    an error (no-op removal).

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if a required parameter is missing, or ``way_id``'s
        non-segment part / ``node_id`` isn't a valid integer.

    """
    filename, way_id, node_id_arg = _require_args("file", "way_id", "node_id")

    # Use the full way_id to match the deletion record; validated the same way
    # as in delete_way_node.
    target_id = _parse_way_id(way_id)
    try:
        node_id = int(node_id_arg)
    except (ValueError, TypeError):
        abort(400, "node_id must be an integer")

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
    Record new positions for one or more nodes of a way.

    Overrides are merged into ``store["node_position_overrides"][original_way_id]``
    (existing overrides for other nodes on the same way are preserved,
    not replaced wholesale); each entry in *nodes* overwrites any
    previous override for that specific node ID. Applies uniformly to
    every segment of a split way, since it's keyed by the original way
    ID.

    Parameters (JSON body)
    -----------------------
    nodes : list of dict
        ``[{"id": node_id, "lat": ..., "lon": ...}, ...]`` new positions.

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file``/``way_id`` is missing, ``way_id``'s non-segment
        part isn't a valid integer, or the body has no ``nodes`` list.

    """
    filename, way_id = _require_args("file", "way_id")
    way_id_int = _original_way_id(way_id)

    body = request.get_json(force=True) or {}
    nodes = body.get("nodes")
    if not isinstance(nodes, list):
        abort(400, "Request body must include 'nodes' list")
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        overrides = store.setdefault("node_position_overrides", {})
        way_key = str(way_id_int)
        if way_key not in overrides:
            overrides[way_key] = {}
        for n in nodes:
            overrides[way_key][str(n["id"])] = {
                "lat": float(n["lat"]),
                "lon": float(n["lon"]),
            }
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
    Undo :func:`move_way_nodes`: discard *all* node position overrides for a way.

    Unlike deletion/split undo, this clears every overridden node on the
    way at once (there is no per-node undo). Not having any overrides is
    not an error (no-op removal).

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file``/``way_id`` is missing or ``way_id``'s non-segment
        part isn't a valid integer.

    """
    filename, way_id = _require_args("file", "way_id")
    way_id_int = _original_way_id(way_id)

    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        store.get("node_position_overrides", {}).pop(str(way_id_int), None)
        _log_remove(store, "move", id=way_id_int)
    return Response("", 204)
