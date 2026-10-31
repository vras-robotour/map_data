"""CRUD routes for freehand annotations (obstacle polygons and drawn paths)."""

import math
import uuid
from typing import Any

from flask import (
    Response,
    abort,
    jsonify,
)
from flask.typing import ResponseReturnValue

from ..cache import load_mapdata_cached
from ..helpers import (
    annotation_store,
    geojson_geom_to_utm,
    load_annotations,
)
from .common import (
    _annotation_path,
    _json_object,
    _mapdata_path,
    _require_args,
    bp,
)


def _validated_annotation_body(filename: str) -> dict[str, Any]:
    """
    Return the request body of an annotation create/update, validated.

    The body needs a ``geometry``; ``type`` must be a string and
    ``properties`` an object whose ``width``, if any, is a positive number.
    The geometry is trial-converted to the map's UTM zone, as
    :func:`~map_data.annotations.merge_annotations` does on every load, so
    one that cannot be converted never reaches the store.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if any of the above does not hold.

    """
    body = _json_object()
    if "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    if not isinstance(body.get("type", ""), str):
        abort(400, "'type' must be a string")
    props = body.get("properties", {})
    if not isinstance(props, dict):
        abort(400, "'properties' must be an object")
    if "width" in props:
        try:
            width = float(props["width"])
        except (TypeError, ValueError):
            width = math.nan
        if isinstance(props["width"], bool) or not (math.isfinite(width) and width > 0):
            abort(400, "'width' must be a positive number")
    md = load_mapdata_cached(str(_mapdata_path(filename)))
    try:
        geojson_geom_to_utm(body["geometry"], md.zone_number, md.zone_letter)
    # Any failure here (wrong shape, bad coordinates, too few points) is bad input.
    except Exception:
        abort(400, "Invalid 'geometry'")
    return body


@bp.route("/api/annotations")
def get_annotations() -> Response:
    """
    Return a mapdata file's whole annotation store (every edit, not only ``annotations``).

    Query: ``file``. 400 if it is missing, 404 if the file doesn't exist.
    """
    filename = _require_args("file")
    return jsonify(load_annotations(str(_annotation_path(filename))))


@bp.route("/api/annotations", methods=["POST"])
def add_annotation() -> ResponseReturnValue:
    """
    Add a freehand annotation (a drawn obstacle or path) to ``store["annotations"]``.

    Query: ``file``; JSON body ``geometry``, ``type`` and ``properties``
    (see :func:`_validated_annotation_body`). Returns the annotation, with a
    new ``id``, and 201; 400 if ``file`` or the body is invalid.
    :func:`~map_data.annotations.merge_annotations` turns it into a ``Way``.
    """
    filename = _require_args("file")
    body = _validated_annotation_body(filename)
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
    Replace the geometry (and the ``type``/``properties`` given) of a freehand annotation.

    Returns the updated annotation; 400 if ``file`` or the body is invalid,
    404 if there is no annotation *ann_id*.
    """
    filename = _require_args("file")
    body = _validated_annotation_body(filename)
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
    Delete a freehand annotation.

    Returns 204; 400 if ``file`` is missing, 404 if there is no annotation *ann_id*.
    """
    filename = _require_args("file")
    ann_path = str(_annotation_path(filename))
    with annotation_store(ann_path) as store:
        before = len(store["annotations"])
        store["annotations"] = [a for a in store["annotations"] if a["id"] != ann_id]
        if len(store["annotations"]) == before:
            abort(404, "Annotation not found")
    return "", 204
