"""CRUD routes for freehand annotations (obstacle polygons and drawn paths)."""

import uuid

from flask import (
    Response,
    abort,
    jsonify,
    request,
)
from flask.typing import ResponseReturnValue

from ..helpers import (
    annotation_store,
    load_annotations,
)
from .common import (
    _annotation_path,
    _require_args,
    bp,
)


@bp.route("/api/annotations")
def get_annotations() -> Response:
    """
    Return the raw annotation store (all recorded edits) for a mapdata file.

    Note this is the *whole* store, not just the freehand ``annotations``
    list -- also includes deletions, splits, moves, tag overrides, etc.

    Returns
    -------
    Response
        JSON annotation store, as returned by
        :func:`~map_data.viewer.helpers.load_annotations`.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing.

    """
    filename = _require_args("file")
    return jsonify(load_annotations(str(_annotation_path(filename))))


@bp.route("/api/annotations", methods=["POST"])
def add_annotation() -> ResponseReturnValue:
    """
    Add a freehand annotation (a user-drawn obstacle or path) to a mapdata file.

    Appends to ``store["annotations"]`` (distinct from way/node edits --
    these are geometries with no corresponding OSM way, consumed by
    :func:`~map_data.viewer.routes.common.get_merged_mapdata` to synthesize new ``Way`` objects) and
    persists the store.

    Returns
    -------
    Response
        The newly created annotation (with a generated ``id``) as JSON,
        with status 201.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or the request body has no ``geometry``.

    """
    filename = _require_args("file")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = str(_annotation_path(filename))
    ann = {
        "id": str(uuid.uuid4()),
        "type": body.get("type", "obstacle"),
        "geometry": body["geometry"],
        "properties": body.get("properties", {}),
    }
    with annotation_store(ann_path) as store:
        store["annotations"].append(ann)
    return jsonify(ann), 201


@bp.route("/api/annotations/<ann_id>", methods=["PUT"])
def update_annotation(ann_id: str) -> Response:
    """
    Replace the geometry (and optionally type/properties) of a freehand annotation.

    Returns
    -------
    Response
        The updated annotation as JSON.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing or the request body has no ``geometry``.
        404 if no annotation with ``ann_id`` exists for this file.

    """
    filename = _require_args("file")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        ann = next((a for a in store["annotations"] if a["id"] == ann_id), None)
        if ann is None:
            abort(404, "Annotation not found")
        ann["geometry"] = body["geometry"]
        if "type" in body:
            ann["type"] = body["type"]
        if "properties" in body:
            ann["properties"] = body["properties"]
    return jsonify(ann)


@bp.route("/api/annotations/<ann_id>", methods=["DELETE"])
def delete_annotation(ann_id: str) -> ResponseReturnValue:
    """
    Delete a freehand annotation from a mapdata file's annotation store.

    Returns
    -------
    Response
        Empty body, status 204.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing. 404 if no annotation with ``ann_id``
        exists for this file.

    """
    filename = _require_args("file")
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        before = len(store["annotations"])
        store["annotations"] = [a for a in store["annotations"] if a["id"] != ann_id]
        if len(store["annotations"]) == before:
            abort(404, "Annotation not found")
    return "", 204
