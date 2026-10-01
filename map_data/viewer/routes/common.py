"""
Shared helpers for the viewer routes: the ``viewer`` blueprint, data-dir path
helpers, way resolution, request validators, and the merged ``MapData`` builder.

MapData copy semantics (read before touching any way-editing code)
--------------------------------------------------------------------
:func:`~map_data.viewer.cache.load_mapdata_cached` caches parsed
:class:`~map_data.map_data.MapData` objects process-wide, keyed by
``(path, mtime)``. Every request that touches a given mapdata file gets
back the *same* cached object, so code must never mutate it (or anything
it references) in place -- doing so would corrupt the cache for every
subsequent request. Two copy strategies are used, chosen per call site for
a performance/safety tradeoff:

- **Shallow copy** (``copy.copy``) is used for the hot, frequently-hit
  paths: :func:`~map_data.viewer.routes.files._merged_mapdata_geojson` (behind
  :func:`~map_data.viewer.routes.files.get_mapdata`) copies the top-level ``MapData``, and
  :func:`_resolve_way` (used by :func:`~map_data.viewer.routes.ways.get_way`,
  :func:`~map_data.viewer.routes.ways.get_way_nodes`, and
  :func:`~map_data.viewer.routes.ways._get_way_segments_geojson`) copies an individual ``Way`` out
  of the cached lists. A shallow copy only duplicates the wrapper object -- its
  ``roads_list``/``footways_list``/``barriers_list`` (for ``MapData``) or its
  attributes-still-pointing-at-shared-subobjects (for a ``Way``) remain *shared* with the cached
  instance. Consequently, **every editing helper that touches a way must itself return a fresh
  ``copy.copy`` rather than mutating its argument** -- ``rebuild_way_without_nodes``,
  ``apply_node_position_overrides``, ``apply_added_nodes``, and
  ``split_way`` (all in :mod:`map_data.viewer.helpers`) follow this
  convention; any new edit helper must too. Where a whole *list* needs to
  change (see :func:`apply_way_edits`), it is replaced via
  ``setattr(md, lst_name, new_lst)`` on the already-copied top-level
  object, not mutated in place, so the cached list itself is left alone.
- **Deep copy** (``copy.deepcopy``) is used once, in
  :func:`get_merged_mapdata`, which backs export and path planning. That
  path synthesizes new ``Way`` objects from annotations, reassigns
  ``roads_list``/``footways_list`` wholesale, and reassigns ``w.tags`` for
  tag overrides directly on ways pulled out of the copied lists; deepcopy
  sidesteps having to audit every one of those sites for the
  copy-before-mutate discipline above, at the cost of being noticeably
  slower for large mapdata files. Because that cost is paid only on
  export/planning (not on every pan/zoom/click), it is not worth applying
  everywhere; :func:`~map_data.viewer.routes.files._merged_mapdata_geojson`/:func:`_resolve_way`
  intentionally keep the cheaper shallow copy plus the manual-copy
  discipline instead.

Functions relying on this invariant: :func:`~map_data.viewer.routes.files._merged_mapdata_geojson`,
:func:`apply_way_edits`, :func:`_resolve_way`, :func:`~map_data.viewer.routes.ways.get_way`,
:func:`~map_data.viewer.routes.ways.get_way_nodes`,
:func:`~map_data.viewer.routes.ways._get_way_segments_geojson`, :func:`get_merged_mapdata`.
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
    abort,
    current_app,
    request,
)

from map_data.annotations import (
    _CAT_FOR_LIST,
    apply_tag_overrides,
    apply_way_edits,
    merge_annotations,
)
from map_data.map_data import MapData
from map_data.utils.config import package_share
from map_data.utils.way import FOOTWAY_VALUES

from ..cache import load_mapdata_cached
from ..helpers import (
    apply_added_nodes,
    apply_node_position_overrides,
    edited_nodes_cache,
    get_deleted_node_ids,
    get_detached_node_ids,
    get_node_position_overrides,
    get_split_node_ids,
    load_annotations,
    rebuild_way_without_nodes,
    split_way,
)

bp = Blueprint("viewer", __name__)


@dataclass
class _ResolvedWay:
    """
    Result of running a single way ID through :func:`_resolve_way`.

    Attributes
    ----------
    way : Way or None
        The resolved, edited way, or ``None`` if either no way with the
        requested ID exists (see ``category``) or it was reduced to
        nothing by node deletions.
    category : str or None
        ``"road"``, ``"footway"``, or ``"barrier"`` -- whichever list the
        way was found in. ``None`` means no way with the requested ID
        exists in *any* list; this is the only reliable way to distinguish
        "not found" from "found but deleted down to nothing" once ``way``
        is ``None``.
    effective_nodes_cache : dict
        ``md``'s raw ``{node_id: {"lat", "lon", "tags"}}`` cache, merged
        with synthetic entries for any user-added nodes on this way (keyed
        by their negative synthetic IDs, with any recorded position
        override already applied) -- the cache to pass to
        :func:`~map_data.viewer.helpers.split_way` when splitting must
        also work at a synthetic node. Equal to ``md``'s raw cache when
        the way was not found or has no added nodes.

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
    Look up a way by its original OSM ID and apply its recorded edits.

    This is the shared core of the way-resolution pipeline used by
    :func:`~map_data.viewer.routes.ways.get_way`,
    :func:`~map_data.viewer.routes.ways.get_way_nodes`, and
    :func:`~map_data.viewer.routes.ways._get_way_segments_geojson`: find the way in ``md``'s
    ``roads_list``/``footways_list``/``barriers_list`` (making a ``copy.copy`` so the cached
    ``MapData`` is never mutated -- see the module docstring), then apply node deletions, added
    nodes, and node position overrides recorded in *store* for ``search_id``, in that
    order except for added-nodes vs. overrides, whose relative order is
    controlled by *added_nodes_before_overrides* because callers
    genuinely disagree on it (see below). Splitting into segments is
    *not* handled here, since callers consume split segments differently
    (picking a single segment vs. building GeoJSON for every segment);
    callers call :func:`~map_data.viewer.helpers.split_way` themselves
    with the ``effective_nodes_cache`` returned here (or, like
    :func:`~map_data.viewer.routes.ways.get_way`, with ``md.nodes_cache`` directly).

    If node deletions reduce the way to nothing, ``rebuild_way_without_nodes``
    returns ``None`` and the added-nodes/overrides steps are skipped
    entirely, matching the original per-caller behaviour where each
    function bailed out immediately in that case.

    Parameters
    ----------
    md : MapData
        Loaded map data (as returned by
        :func:`~map_data.viewer.cache.load_mapdata_cached`) to search.
    store : dict
        Annotation store, as returned by
        :func:`~map_data.viewer.helpers.load_annotations`.
    search_id : int
        Original (non-virtual) OSM way ID to resolve.
    added_nodes_before_overrides : bool, default False
        If ``True``, apply user-added nodes before node position
        overrides (as :func:`~map_data.viewer.routes.ways.get_way` requires, so that a position
        override recorded for a synthetic node is picked up by
        :func:`~map_data.viewer.helpers.apply_node_position_overrides`);
        if ``False`` (the default), apply overrides first and added
        nodes last (as :func:`~map_data.viewer.routes.ways.get_way_nodes` and
        :func:`~map_data.viewer.routes.ways._get_way_segments_geojson` require).

    Returns
    -------
    _ResolvedWay
        See :class:`_ResolvedWay`.

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
    if ":" not in str(way_id):
        return way
    try:
        segment_idx = int(str(way_id).split(":")[1])
    except ValueError:
        abort(400, "Invalid virtual ID")

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
    Return the directory holding ``.mapdata``/``.gpx``/annotation files.

    Resolution order: the Flask app's ``DATA_DIR`` config value if set
    (used by tests and non-ROS2 deployments); else
    :func:`~map_data.utils.config.package_share`'s installed
    ``share/map_data/data`` directory, falling back to a ``data``
    directory next to the source tree when no ROS2 environment is
    available.

    Returns
    -------
    Path
        Directory path (not guaranteed to exist).

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


def _parse_way_id(way_id: str) -> int | str:
    """
    Validate a possibly-virtual way ID and return its canonical stored form.

    A plain integer ID is returned as an ``int``. A virtual segment ID
    (``"<original_id>:<segment_index>"``, produced by
    :func:`~map_data.viewer.helpers.split_way`) is returned as a
    canonicalized ``str`` only if *both* parts parse as integers. Anything
    else -- a non-numeric base, a non-numeric segment suffix, or extra
    colons -- is rejected, so raw attacker-controlled strings never reach
    the annotation store or the change log (where the frontend would later
    render them).

    Parameters
    ----------
    way_id : str
        Way ID as supplied by the client.

    Returns
    -------
    int or str
        ``int`` for a plain way ID, ``"<int>:<int>"`` for a virtual one.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if *way_id* is not a valid plain or virtual way ID.

    """
    way_id_str = str(way_id)
    base_str, sep, suffix_str = way_id_str.partition(":")
    try:
        base_int = int(base_str)
    except ValueError:
        abort(400, "Invalid way ID")
    if not sep:
        return base_int
    try:
        suffix_int = int(suffix_str)
    except ValueError:
        abort(400, "Invalid virtual way ID")
    return f"{base_int}:{suffix_int}"


def _original_way_id(way_id: str | int) -> int:
    """
    Return the original (non-virtual) integer way ID for *way_id*.

    Strips any ``":<segment_index>"`` virtual-segment suffix first, so a
    plain or virtual way ID both resolve to the same underlying OSM way ID.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if the non-segment part isn't a valid integer.

    """
    try:
        return int(str(way_id).split(":")[0])
    except (ValueError, TypeError):
        abort(400, "Invalid way ID")


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
    Append a ``change_log`` entry unless one matching *entry_type* and *key* already exists.

    *key* fields identify the edit (e.g. ``{"id": way_id}`` or
    ``{"way_id": ..., "node_id": ...}``) and are compared via ``str()`` so
    an int/str type mismatch between calls doesn't spuriously fail to
    match; they're included in the appended entry along with **extra**
    fields (e.g. ``category``/``label``), which are not compared for the
    dedup check. The entry is stamped with the current time.
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


# A robotour course is a few km at most; this comfortably covers that with
# room to spare while still rejecting the "whole city" rectangles that make
# Overpass queries time out no matter how many mirrors are available.
MAX_FETCH_AREA_KM2 = 25.0


def _bbox_area_km2(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> float:
    mean_lat_rad = math.radians((min_lat + max_lat) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(mean_lat_rad)
    return abs(max_lat - min_lat) * km_per_deg_lat * abs(max_lon - min_lon) * km_per_deg_lon


# Upper bound on the number of grid cells a single cost-grid/replan request may
# allocate. Each cell costs ~32 bytes in the planner's [N, 4] float64 grid plus
# per-cell Python-loop time, so a few million is already generous: 4 million
# cells is a 2 km x 2 km box at the 1 m visualization resolution, or
# 500 m x 500 m at the 0.25 m default replan cell size.
MAX_GRID_CELLS = 4_000_000

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
    Build a fully-edited, planning/export-ready ``MapData`` for a file.

    Used by :func:`~map_data.viewer.routes.files.export_mapdata`,
    :func:`~map_data.viewer.routes.planning.get_cost_grid`, and
    :func:`~map_data.viewer.routes.planning.create_replan`, all of which need a genuinely
    self-consistent ``MapData`` (not just a GeoJSON view of one) to feed to
    :mod:`map_data.utils.serialization` or the path planner. Deep-copies the cached ``MapData`` (see
    the module docstring's "MapData copy semantics" section for why deepcopy is used here rather
    than the shallow-copy-plus-manual-Way-copy convention used elsewhere) before:

    1. Applying way/node deletions, splits, and node moves via
       :func:`apply_way_edits`.
    2. Merging tag overrides directly into each way's ``tags`` (mutating
       the way in place -- safe only because of the deepcopy), then
       re-sorting every way between ``roads_list``/``footways_list`` in
       case an overridden ``highway`` tag changed its category, and
       recomputing ``crossroads_list``.
    3. Synthesizing a new ``Way`` for each freehand annotation
       (``store["annotations"]``, from :func:`~map_data.viewer.routes.annotations.add_annotation`):
       a ``"path"`` annotation becomes a road/footway (buffered by its
       ``width`` tag if the geometry is a ``LineString``, with synthetic
       negative node IDs registered in ``md.nodes_cache``), anything else
       becomes a barrier.

    Parameters
    ----------
    filename : str
        Mapdata filename to resolve, load, and edit.

    Returns
    -------
    tuple of (MapData or None, dict or None)
        ``(md, store)``, or ``(None, None)`` if *filename* does not
        resolve to an existing file.

    """
    path = _safe_data_path(filename)
    if not path.is_file():
        return None, None

    store = load_annotations(str(_annotation_path(filename)))
    md = copy.deepcopy(load_mapdata_cached(str(path)))

    apply_way_edits(md, store)
    apply_tag_overrides(md, store)
    merge_annotations(md, store)

    return md, store
