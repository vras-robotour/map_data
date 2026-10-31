"""
Shared helpers for the viewer routes: the ``viewer`` blueprint, data-dir path
helpers, way resolution, request validators, and the merged ``MapData`` builder.
"""

import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    request,
)
from flask.typing import ResponseReturnValue
from werkzeug.exceptions import HTTPException

from map_data.annotations import (
    _CAT_FOR_LIST,
    apply_store,
)
from map_data.map_data import MapData
from map_data.pathsolver.route import MAX_GRID_CELLS as MAX_GRID_CELLS  # re-exported
from map_data.utils.config import package_share
from map_data.utils.way import FOOTWAY_VALUES

from ..cache import load_mapdata_cached, merged_mapdata_cached
from ..helpers import (
    AnnotationStoreError,
    apply_added_nodes,
    apply_node_position_overrides,
    edited_nodes_cache,
    get_deleted_node_ids,
    get_detached_node_ids,
    get_node_position_overrides,
    get_split_node_ids,
    load_annotations,
    quarantine_annotation_store,
    rebuild_way_without_nodes,
    split_way,
)

bp = Blueprint("viewer", __name__)


@bp.app_errorhandler(HTTPException)
def _http_error(e: HTTPException) -> ResponseReturnValue:
    """Answer every HTTP error as JSON ``{"error": description}``, the API's one error shape."""
    resp = e.get_response()  # keeps the status and headers such as Allow
    resp.set_data(current_app.json.dumps({"error": e.description}))
    resp.content_type = "application/json"
    return resp


@bp.app_errorhandler(AnnotationStoreError)
def _unreadable_store(e: AnnotationStoreError) -> tuple[Response, int]:
    """Move an unreadable annotation store aside and answer 409, keeping the bad file for repair."""
    backup = quarantine_annotation_store(e.path)
    kept = f"moved it to {backup.name}" if backup else "it was already moved aside"
    msg = f"Annotation store unreadable ({e}); {kept}. Reload to continue."
    return jsonify({"error": msg}), 409


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
        :func:`~map_data.viewer.helpers.edited_nodes_cache`); the raw cache
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
    ``effective_nodes_cache`` to :func:`~map_data.viewer.helpers.split_way`.
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


def _get_data_dir() -> Path:
    """
    Return the directory of ``.mapdata``/``.gpx``/annotation files (may not exist).

    The app's ``DATA_DIR`` config if set, else
    :func:`~map_data.utils.config.package_share`'s ``data``.
    """
    if current_app.config.get("DATA_DIR"):
        return Path(current_app.config["DATA_DIR"])
    return package_share("data")


def _safe_data_path(filename: str) -> Path:
    """
    Resolve a user-supplied filename within the data directory.

    Aborts with 400 if the resolved path would escape the data directory
    (e.g. via '../' traversal sequences or an absolute path). Symlinks inside
    the data directory are followed, wherever they point.
    """
    data_dir = _get_data_dir().resolve()
    # Normalise lexically, not with resolve(): a colcon --symlink-install
    # links share/map_data/data/*.mapdata to build/, outside data_dir, so
    # resolving the symlink would reject every installed map. The normalised
    # path is the one returned and opened, so '..' cannot slip past the check.
    resolved = Path(os.path.normpath(data_dir / filename))
    if not (resolved == data_dir or data_dir in resolved.parents):
        abort(400, "Invalid file path")
    return resolved


def _mapdata_path(filename: str) -> Path:
    """
    Resolve a user-supplied filename to an existing ``.mapdata`` file.

    The one validated resolver every endpoint taking a mapdata ``file``
    goes through: :func:`_safe_data_path` containment, then a ``.mapdata``
    extension check, then an existence check.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if the path escapes the data directory or isn't a ``.mapdata``
        file. 404 if the file doesn't exist.

    """
    path = _safe_data_path(filename)
    if path.suffix != ".mapdata":
        abort(400, "Invalid file path")
    if not path.is_file():
        abort(404, f"File not found: {filename}")
    return path


def _annotation_path(filename: str) -> Path:
    """
    Return the annotation-store path paired with a mapdata *filename*.

    *filename* is validated via :func:`_mapdata_path` (so a missing or
    out-of-tree ``.mapdata`` aborts 404/400 before any annotation file can
    be read or written).

    Returns
    -------
    Path
        ``<data_dir>/<stem>.annotations.json``.

    """
    return _get_data_dir().resolve() / f"{_mapdata_path(filename).stem}.annotations.json"


def _split_way_id(way_id: Any) -> tuple[int, int | None]:
    """
    Parse a plain (``"<id>"``) or virtual (``"<id>:<segment_index>"``) way ID.

    The one way-ID parser of the routes: raw client strings that fail it
    never reach the annotation store or the change log (where the frontend
    would later render them).

    Returns
    -------
    tuple of (int, int or None)
        The original OSM way ID and the segment index (``None`` for a plain ID).

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if the ID part is not an integer, or the segment suffix is not a
        non-negative integer.

    """
    # str() first so a JSON float or bool is rejected rather than truncated by int().
    base_str, sep, suffix_str = str(way_id).partition(":")
    try:
        base_int = int(base_str)
    except ValueError:
        abort(400, "Invalid way ID")
    if not sep:
        return base_int, None
    try:
        suffix_int = int(suffix_str)
    except ValueError:
        abort(400, "Invalid virtual way ID")
    if suffix_int < 0:
        abort(400, "Invalid virtual way ID")
    return base_int, suffix_int


def _parse_way_id(way_id: Any) -> int | str:
    """
    Return the canonical stored form of a way ID parsed by :func:`_split_way_id`.

    ``int`` for a plain way ID, ``"<int>:<int>"`` for a virtual segment ID.
    """
    original, segment = _split_way_id(way_id)
    return original if segment is None else f"{original}:{segment}"


def _int_param(value: Any, name: str) -> int:
    """
    Parse an integer from a query-string or JSON value.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if *value* is not an integer (a float or bool is rejected, not truncated).

    """
    try:
        return int(str(value))
    except ValueError:
        abort(400, f"{name} must be an integer")


def _json_object() -> dict[str, Any]:
    """
    Return the request's JSON body as a dict (``{}`` for a JSON ``null``).

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if the body is not valid JSON or not a JSON object.

    """
    body = request.get_json(force=True)
    if body is None:
        return {}
    if not isinstance(body, dict):
        abort(400, "Request body must be a JSON object")
    return body


def _require_args(*names: str) -> Any:
    """
    Return each of *names* from the query string, aborting 400 if any is missing.

    A single name returns its value directly; more than one returns a
    tuple in the same order as *names*.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if any named query parameter is missing or empty.

    """
    values = tuple(request.args.get(name) for name in names)
    if not all(values):
        if len(names) == 1:
            abort(400, f"Missing '{names[0]}' query parameter")
        abort(400, "Missing required query parameters")
    return values[0] if len(names) == 1 else values


def _log_add(store: dict[str, Any], entry_type: str, key: dict[str, Any], **extra: Any) -> None:
    """
    Append a timestamped ``change_log`` entry unless one with *entry_type* and *key* exists.

    *key* fields are compared via ``str()`` (so ``123`` matches ``"123"``);
    **extra** fields are stored but not compared.
    """
    cl = store.setdefault("change_log", [])
    if not any(
        e.get("type") == entry_type and all(str(e.get(k)) == str(v) for k, v in key.items())
        for e in cl
    ):
        cl.append({"type": entry_type, **key, **extra, "ts": time.time()})


def _log_remove(store: dict[str, Any], entry_type: str, **key: Any) -> None:
    """Drop ``change_log`` entries matching *entry_type* and *key* fields (via ``str()``)."""
    cl = store.get("change_log", [])
    store["change_log"] = [
        e
        for e in cl
        if not (
            e.get("type") == entry_type and all(str(e.get(k)) == str(v) for k, v in key.items())
        )
    ]


# A robotour course is a few km at most; larger "whole city" boxes make
# Overpass queries time out on every mirror.
MAX_FETCH_AREA_KM2 = 25.0


def _bbox_area_km2(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> float:
    mean_lat_rad = math.radians((min_lat + max_lat) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(mean_lat_rad)
    return abs(max_lat - min_lat) * km_per_deg_lat * abs(max_lon - min_lon) * km_per_deg_lon


# Sane grid-planner parameter ranges: cell sizes below 5 cm explode the cell
# count (see MAX_GRID_CELLS) while sizes above 10 m are useless for a footpath
# planner; obstacle inflation beyond 10 m would swallow whole maps.
MIN_CELL_SIZE_M = 0.05
MAX_CELL_SIZE_M = 10.0
MAX_INFLATE_OBSTACLES_M = 10.0

# utm.from_latlon() only supports latitudes in [-80, 84]; reject anything
# outside so bad input becomes a 400 instead of an exception deep in planning.
_MIN_LAT, _MAX_LAT = -80.0, 84.0
_MIN_LON, _MAX_LON = -180.0, 180.0


def _validated_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    """
    Validate that *value* is a finite JSON number within ``[minimum, maximum]``.

    Booleans are rejected (JSON ``true``/``false`` are not numbers even though
    ``bool`` subclasses ``int`` in Python), as are numeric strings, NaN, and
    infinities.

    Returns
    -------
    float
        The validated value as a ``float``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if *value* is not a number or is outside the allowed range.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        abort(400, f"{name} must be a number")
    num = float(value)
    if not math.isfinite(num) or not (minimum <= num <= maximum):
        abort(400, f"{name} must be a finite number between {minimum} and {maximum}")
    return num


def _validated_bbox(
    min_lat: Any,
    min_lon: Any,
    max_lat: Any,
    max_lon: Any,
) -> tuple[float, float, float, float]:
    """
    Validate a client-supplied WGS84 bounding box.

    Each corner must be a finite number within the UTM-supported lat/lon
    range, and min must be strictly less than max on each axis.

    Returns
    -------
    tuple of float
        ``(min_lat, min_lon, max_lat, max_lon)`` as validated floats.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if any corner is not a finite in-range number or the box is
        inverted/degenerate.

    """
    min_lat_f = _validated_number(min_lat, "min_lat", _MIN_LAT, _MAX_LAT)
    max_lat_f = _validated_number(max_lat, "max_lat", _MIN_LAT, _MAX_LAT)
    min_lon_f = _validated_number(min_lon, "min_lon", _MIN_LON, _MAX_LON)
    max_lon_f = _validated_number(max_lon, "max_lon", _MIN_LON, _MAX_LON)
    if min_lat_f >= max_lat_f or min_lon_f >= max_lon_f:
        abort(400, "min_lat/min_lon must be strictly less than max_lat/max_lon")
    return min_lat_f, min_lon_f, max_lat_f, max_lon_f


def _validated_cost_dict(value: Any, name: str) -> dict[str, float] | None:
    """
    Validate a client-supplied per-tag cost override dict.

    ``None`` passes through unchanged (meaning "use the defaults"); anything
    else must be a dict mapping strings to finite non-negative numbers.

    Returns
    -------
    dict of str to float, or None
        The validated cost dict (values coerced to ``float``), or ``None``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if *value* is not a dict of str -> finite non-negative number.

    """
    if value is None:
        return None
    if not isinstance(value, dict):
        abort(400, f"{name} must be an object mapping tag values to costs")
    out: dict[str, float] = {}
    for key, cost in value.items():
        if not isinstance(key, str):
            abort(400, f"{name} keys must be strings")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            abort(400, f"{name}[{key!r}] must be a number")
        cost_f = float(cost)
        if not math.isfinite(cost_f) or cost_f < 0:
            abort(400, f"{name}[{key!r}] must be a finite non-negative number")
        out[key] = cost_f
    return out


def get_merged_mapdata(filename: str) -> tuple[MapData | None, dict[str, Any] | None]:
    """
    Return the file's ``MapData`` with every edit in its store applied, plus the store.

    Built by :func:`~map_data.annotations.apply_store` (way edits, tag
    overrides, freehand annotations) for export and planning. The result is
    cached by :func:`~map_data.viewer.cache.merged_mapdata_cached` and shared
    between requests: treat it as read-only.

    Returns
    -------
    tuple of (MapData or None, dict or None)
        ``(md, store)``, or ``(None, None)`` if the file does not exist.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if *filename* escapes the data directory or names an existing
        non-``.mapdata`` file.

    """
    path = _safe_data_path(filename)
    if not path.is_file():
        return None, None

    store = load_annotations(str(_annotation_path(filename)))

    def build() -> MapData:
        return apply_store(load_mapdata_cached(str(path)), store)

    return merged_mapdata_cached(str(path), store, build), store
