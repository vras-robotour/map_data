"""
Flask routes for the interactive OSM map-data viewer/editor.

Exposes a JSON/GeoJSON API (registered on ``bp``, mounted by the Flask app
factory) that the browser-side viewer uses to: list and upload ``.mapdata``/
``.gpx`` files, read map geometry as GeoJSON, record user edits (way/node
deletion, node moves, node insertion, way splitting, tag overrides) as a
JSON "annotation store" alongside each mapdata file, export an edited map,
compute cost grids, and drive path (re)planning, including a wormhole-based
GPX handoff to a companion mobile app.

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
  paths: :func:`get_mapdata` copies the top-level ``MapData``, and
  :func:`_resolve_way` (used by :func:`get_way`, :func:`get_way_nodes`, and
  :func:`_get_way_segments_geojson`) copies an individual ``Way`` out of
  the cached lists. A shallow copy only duplicates the wrapper object --
  its ``roads_list``/``footways_list``/``barriers_list`` (for ``MapData``)
  or its attributes-still-pointing-at-shared-subobjects (for a ``Way``)
  remain *shared* with the cached instance. Consequently, **every editing
  helper that touches a way must itself return a fresh ``copy.copy`` rather
  than mutating its argument** -- ``rebuild_way_without_nodes``,
  ``apply_node_position_overrides``, ``apply_added_nodes``, and
  ``split_way`` (all in :mod:`map_data.viewer.helpers`) follow this
  convention; any new edit helper must too. Where a whole *list* needs to
  change (see :func:`_apply_way_edits`), it is replaced via
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
  everywhere; :func:`get_mapdata`/:func:`_resolve_way` intentionally keep
  the cheaper shallow copy plus the manual-copy discipline instead.

Functions relying on this invariant: :func:`get_mapdata`,
:func:`_apply_way_edits`, :func:`_resolve_way`, :func:`get_way`,
:func:`get_way_nodes`, :func:`_get_way_segments_geojson`,
:func:`get_merged_mapdata`.
"""

import copy
import io
import json
import logging
import os
import re
import select
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import utm
from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)
from shapely import geometry

from map_data.map_data import MapData
from map_data.pathsolver.graph_planner import GraphPlanner
from map_data.pathsolver.replan import (
    ReplanPath,
    cancel_replan_backend,
    load_planner_defaults,
    parse_args,
)
from map_data.utils.parsing import ways_to_shapely
from map_data.utils.serialization import map_data_to_dict
from map_data.utils.way import FOOTWAY_VALUES, Way

from .cache import load_mapdata_cached
from .helpers import (
    apply_added_nodes,
    apply_node_position_overrides,
    geojson_geom_to_utm,
    geom_to_geojson,
    get_deleted_node_ids,
    get_deleted_way_ids,
    get_node_position_overrides,
    get_split_node_ids,
    load_annotations,
    mapdata_to_geojson,
    migrate_change_log,
    rebuild_way_without_nodes,
    save_annotations,
    split_way,
)

SIGNIFICANT_CHANGE_TOLERANCE = 0.1

logger = logging.getLogger(__name__)
bp = Blueprint("viewer", __name__)


@bp.route("/api/planner_defaults")
def get_planner_defaults() -> Response:
    """
    Return the default path-planning parameters as JSON.

    Returns
    -------
    Response
        JSON dict of planner defaults, as loaded by
        :func:`~map_data.pathsolver.replan.load_planner_defaults`.

    """
    return jsonify(load_planner_defaults())


_CAT_FOR_LIST = {
    "roads_list": "road",
    "footways_list": "footway",
    "barriers_list": "barrier",
}


def _apply_way_edits(md: MapData, store: dict[str, Any]) -> None:
    """
    Apply way/node deletions, splits, and node position overrides to *md*.

    Used by both :func:`get_mapdata` (on a shallow copy) and
    :func:`get_merged_mapdata` (on a deep copy) to turn the raw parsed
    map data into what the user currently sees after their edits. Mutates
    *md* by reassigning its ``roads_list``/``footways_list``/``barriers_list``
    (and, if any way list changed, ``crossroads_list``) attributes -- see
    the module docstring's "MapData copy semantics" section for why this
    is safe on a shallow copy: whole lists are replaced via ``setattr``,
    and every individual ``Way`` mutation goes through
    :func:`~map_data.viewer.helpers.rebuild_way_without_nodes` /
    :func:`~map_data.viewer.helpers.split_way` /
    :func:`~map_data.viewer.helpers.apply_node_position_overrides`, all of
    which return a fresh copy rather than mutating their argument.

    Processing order: whole-way deletions are filtered out first; then,
    for each surviving way, node deletions are applied, then splits (each
    split segment gets its own virtual ID ``"<way_id>:<index>"`` and is
    itself subject to deletion and segment-specific node deletion); finally,
    in a second pass over the (possibly now-split) ways, node position
    overrides are applied. If any way list changed, ``crossroads_list`` is
    recomputed from the new ``footways_list``.

    Parameters
    ----------
    md : MapData
        Map data to edit in place (its list attributes are reassigned;
        the ``Way`` objects referenced by the *original* lists are never
        mutated).
    store : dict
        Annotation store, as returned by
        :func:`~map_data.viewer.helpers.load_annotations`.

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
                            )
                            if seg is None:
                                continue
                        new_lst.append(seg)
                else:
                    new_lst.append(w)
            setattr(md, lst_name, new_lst)
        md.crossroads_list = md.parse_intersections({str(w.id): w for w in md.footways_list})

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


_WAY_LISTS = (
    ("roads_list", "road"),
    ("footways_list", "footway"),
    ("barriers_list", "barrier"),
)


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
    nodes_cache : dict
        ``md``'s raw ``{node_id: {"lat", "lon", "tags"}}`` cache (or
        ``{}`` if the loaded ``MapData`` has none), unmodified.
    effective_nodes_cache : dict
        ``nodes_cache`` merged with synthetic entries for any user-added
        nodes on this way (keyed by their negative synthetic IDs, with any
        recorded position override already applied) -- the cache to pass
        to :func:`~map_data.viewer.helpers.split_way` when splitting must
        also work at a synthetic node. Equal to ``nodes_cache`` when the
        way was not found or has no added nodes.

    """

    way: Any
    category: str | None
    nodes_cache: dict[int, dict[str, Any]]
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
    :func:`get_way`, :func:`get_way_nodes`, and
    :func:`_get_way_segments_geojson`: find the way in ``md``'s
    ``roads_list``/``footways_list``/``barriers_list`` (making a
    ``copy.copy`` so the cached ``MapData`` is never mutated -- see the
    module docstring), then apply node deletions, added nodes, and node
    position overrides recorded in *store* for ``search_id``, in that
    order except for added-nodes vs. overrides, whose relative order is
    controlled by *added_nodes_before_overrides* because callers
    genuinely disagree on it (see below). Splitting into segments is
    *not* handled here, since callers consume split segments differently
    (picking a single segment vs. building GeoJSON for every segment);
    callers call :func:`~map_data.viewer.helpers.split_way` themselves
    with the ``nodes_cache`` or ``effective_nodes_cache`` returned here.

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
        overrides (as :func:`get_way` requires, so that a position
        override recorded for a synthetic node is picked up by
        :func:`~map_data.viewer.helpers.apply_node_position_overrides`);
        if ``False`` (the default), apply overrides first and added
        nodes last (as :func:`get_way_nodes` and
        :func:`_get_way_segments_geojson` require).

    Returns
    -------
    _ResolvedWay
        See :class:`_ResolvedWay`.

    """
    nodes_cache = getattr(md, "nodes_cache", {})

    way = None
    category = None
    for lst_name, cat in _WAY_LISTS:
        for w in getattr(md, lst_name):
            if w.id == search_id:
                way = copy.copy(w)
                category = cat
                break
        if way:
            break

    if way is None:
        return _ResolvedWay(
            way=None,
            category=None,
            nodes_cache=nodes_cache,
            effective_nodes_cache=nodes_cache,
        )

    zn, zl = md.zone_number, md.zone_letter

    del_nids = get_deleted_node_ids(store, search_id)
    if del_nids:
        way = rebuild_way_without_nodes(way, del_nids, zn, zl, nodes_cache, category=category)

    effective_nc = nodes_cache
    if way is not None:

        def _apply_overrides() -> None:
            """Apply recorded node position overrides to the enclosing ``way``."""
            nonlocal way
            pos_overrides = get_node_position_overrides(store, search_id)
            if pos_overrides:
                way = (
                    apply_node_position_overrides(
                        way,
                        pos_overrides,
                        zn,
                        zl,
                        nodes_cache,
                        category=category,
                    )
                    or way
                )

        def _apply_added() -> None:
            """Apply recorded user-added synthetic nodes to the enclosing ``way``."""
            nonlocal way
            way = apply_added_nodes(way, store, zn, zl)

        if added_nodes_before_overrides:
            _apply_added()
            _apply_overrides()
        else:
            _apply_overrides()
            _apply_added()

        synth_nc: dict[int, dict[str, Any]] = {}
        for a in store.get("added_nodes", []):
            if a.get("way_id") == search_id:
                pos_ov = (
                    store.get("node_position_overrides", {})
                    .get(str(search_id), {})
                    .get(str(a["id"]))
                )
                synth_nc[a["id"]] = {
                    "lat": float(pos_ov["lat"] if pos_ov else a["lat"]),
                    "lon": float(pos_ov["lon"] if pos_ov else a["lon"]),
                    "tags": {},
                }
        effective_nc = {**nodes_cache, **synth_nc}

    return _ResolvedWay(
        way=way,
        category=category,
        nodes_cache=nodes_cache,
        effective_nodes_cache=effective_nc,
    )


def _get_data_dir() -> Path:
    """
    Return the directory holding ``.mapdata``/``.gpx``/annotation files.

    Resolution order: the Flask app's ``DATA_DIR`` config value if set
    (used by tests and non-ROS2 deployments); else the ``map_data`` ROS2
    package's installed ``share/map_data/data`` directory, via
    ``ament_index_python``; else (no ROS2 environment available) a
    ``data`` directory next to the installed package source, as a
    filesystem fallback for running the viewer standalone.

    Returns
    -------
    Path
        Directory path (not guaranteed to exist).

    """
    if current_app.config.get("DATA_DIR"):
        return Path(current_app.config["DATA_DIR"])
    try:
        from ament_index_python.resources import get_resource

        _, pkg = get_resource("packages", "map_data")
        return Path(pkg) / "share" / "map_data" / "data"
    except (ImportError, LookupError):
        # Fallback to the local data directory relative to this file
        return (Path(__file__).parent / ".." / ".." / "data").resolve()


def _safe_data_path(filename: str) -> Path:
    """
    Resolve a user-supplied filename within the data directory.

    Aborts with 400 if the resolved path would escape the data directory
    (e.g. via '../' traversal sequences).
    """
    data_dir = _get_data_dir().resolve()
    resolved = (data_dir / filename).resolve()
    if not (resolved == data_dir or data_dir in resolved.parents):
        abort(400, "Invalid file path")
    return resolved


def _annotation_path(filename: str) -> Path:
    """
    Return the annotation-store path paired with a mapdata *filename*.

    Not itself traversal-safe -- callers pass the result to functions
    like :func:`~map_data.viewer.helpers.load_annotations` that only read
    or atomically rewrite this exact path, and *filename* has typically
    already been validated via :func:`_safe_data_path` for the
    corresponding ``.mapdata`` file.

    Parameters
    ----------
    filename : str
        Mapdata filename (with or without directory components; only the
        stem is used).

    Returns
    -------
    Path
        ``<data_dir>/<stem>.annotations.json``.

    """
    base = Path(filename).stem
    return _get_data_dir().resolve() / f"{base}.annotations.json"


@bp.route("/")
def index() -> str:
    """Render the viewer's single-page HTML shell, injecting tile-provider API keys."""
    api_key_thunderforest = os.getenv("THUNDERFOREST_API_KEY")
    api_key_seznam = os.getenv("SEZNAM_API_KEY")
    return render_template(
        "index.html",
        apikey_thunderforest=api_key_thunderforest,
        apikey_seznam=api_key_seznam,
    )


@bp.route("/api/files")
def list_files() -> Response:
    """
    List available data files in the data directory.

    Returns
    -------
    Response
        JSON ``{"mapdata": [name, ...], "gpx": [name, ...]}``, sorted by
        filename. Both lists are empty (not an error) if the data
        directory doesn't exist yet.

    """
    result = {"mapdata": [], "gpx": []}
    try:
        for p in sorted(_get_data_dir().iterdir()):
            if p.suffix == ".mapdata":
                result["mapdata"].append(p.name)
            elif p.suffix == ".gpx":
                result["gpx"].append(p.name)
    except FileNotFoundError:
        pass
    return jsonify(result)


@bp.route("/api/mapdata")
def get_mapdata() -> Response:
    """
    Return a mapdata file, with the user's recorded edits applied, as GeoJSON.

    Loads the cached ``MapData`` for ``file``, takes a shallow ``copy.copy``
    (see the module docstring's "MapData copy semantics" section -- this
    is safe because :func:`_apply_way_edits` never mutates a ``Way`` object
    shared with the cache), applies deletions/splits/moves via
    :func:`_apply_way_edits`, converts to GeoJSON, and finally merges in
    any tag overrides (re-deriving each affected feature's ``road``/
    ``footway`` category from the merged ``highway`` tag).

    Returns
    -------
    Response
        A GeoJSON ``FeatureCollection`` (see
        :func:`~map_data.viewer.helpers.mapdata_to_geojson`).

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing. 404 if the file doesn't exist.

    """
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = _safe_data_path(filename)
    if not path.is_file():
        abort(404, f"File not found: {filename}")
    map_data = copy.copy(load_mapdata_cached(str(path)))
    store = load_annotations(str(_annotation_path(filename)))

    _apply_way_edits(map_data, store)

    geojson = mapdata_to_geojson(map_data)
    tag_overrides = store.get("tag_overrides", {})
    if tag_overrides:
        for f in geojson["features"]:
            ov = tag_overrides.get(str(f["properties"].get("id", "")))
            if ov:
                merged = {**(f["properties"].get("tags") or {}), **ov}
                f["properties"]["tags"] = merged
                cat = f["properties"].get("category")
                if cat in ("road", "footway"):
                    hw = merged.get("highway", "")
                    f["properties"]["category"] = "footway" if hw in FOOTWAY_VALUES else "road"
    return jsonify(geojson)


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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    return jsonify(load_annotations(_annotation_path(filename)))


@bp.route("/api/annotations", methods=["POST"])
def add_annotation() -> Response:
    """
    Add a freehand annotation (a user-drawn obstacle or path) to a mapdata file.

    Appends to ``store["annotations"]`` (distinct from way/node edits --
    these are geometries with no corresponding OSM way, consumed by
    :func:`get_merged_mapdata` to synthesize new ``Way`` objects) and
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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    ann = {
        "id": str(uuid.uuid4()),
        "type": body.get("type", "obstacle"),
        "geometry": body["geometry"],
        "properties": body.get("properties", {}),
    }
    store["annotations"].append(ann)
    save_annotations(ann_path, store)
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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    ann = next((a for a in store["annotations"] if a["id"] == ann_id), None)
    if ann is None:
        abort(404, "Annotation not found")
    ann["geometry"] = body["geometry"]
    if "type" in body:
        ann["type"] = body["type"]
    if "properties" in body:
        ann["properties"] = body["properties"]
    save_annotations(ann_path, store)
    return jsonify(ann)


@bp.route("/api/annotations/<ann_id>", methods=["DELETE"])
def delete_annotation(ann_id: str) -> Response:
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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    before = len(store["annotations"])
    store["annotations"] = [a for a in store["annotations"] if a["id"] != ann_id]
    if len(store["annotations"]) == before:
        abort(404, "Annotation not found")
    save_annotations(ann_path, store)
    return "", 204


_fetch_tasks: dict[str, dict[str, Any]] = {}


def _run_fetch_task(
    task_id: str,
    waypoints: Any,
    zone_number: Any,
    zone_letter: Any,
    out_path: Path,
    grid_margin: Any,
    obstacle_radius: Any,
    buffer_widths: Any,
) -> None:
    """
    Download, parse, and save OSM data for a bounding box, in a background thread.

    Run via a daemon ``threading.Thread`` started by :func:`fetch_area`
    (which returns immediately with a task ID); progress and the outcome
    are recorded in the module-level ``_fetch_tasks`` dict, keyed by
    *task_id*, for :func:`fetch_area_status` to poll. Any exception is
    caught, logged, and reported as a generic "Internal server error"
    task failure rather than propagating (there is no request context to
    propagate it to).

    Parameters
    ----------
    task_id : str
        Key into ``_fetch_tasks`` to update with progress/results.
    waypoints : np.ndarray
        ``(4, 2)`` array of UTM easting/northing corner coordinates
        defining the query bounding box.
    zone_number : int
        UTM zone number of *waypoints*.
    zone_letter : str
        UTM zone letter of *waypoints*.
    out_path : Path
        Destination ``.mapdata`` file path.
    grid_margin : float or None
        Passed through to :class:`~map_data.map_data.MapData`.
    obstacle_radius : float or None
        Passed through to :class:`~map_data.map_data.MapData`.
    buffer_widths : dict or None
        Passed through to :class:`~map_data.map_data.MapData`.

    Side Effects
    ------------
    Writes *out_path* on success. Updates ``_fetch_tasks[task_id]`` to a
    dict with ``"status"`` of ``"failed"`` (plus an ``"error"`` message)
    or ``"done"`` (plus a ``"result"`` summary).

    """
    try:
        md = MapData(
            [waypoints, zone_number, zone_letter],
            coords_type="array",
            grid_margin=grid_margin,
            obstacle_radius=obstacle_radius,
            buffer_widths=buffer_widths,
        )
        md.run_queries()
        if any(d is None for d in (md.osm_ways_data, md.osm_rels_data, md.osm_nodes_data)):
            _fetch_tasks[task_id] = {
                "status": "failed",
                "error": "Overpass API unavailable — try again later",
            }
            return
        if md.run_parse() != 0:
            _fetch_tasks[task_id] = {"status": "failed", "error": "Parsing failed"}
            return
        md.save(out_path)
        _fetch_tasks[task_id] = {
            "status": "done",
            "result": {
                "filename": out_path.name,
                "roads": len(md.roads_list),
                "footways": len(md.footways_list),
                "barriers": len(md.barriers_list),
                "crossroads": len(md.crossroads_list),
            },
        }
    except Exception:
        logger.exception("fetch task %s failed", task_id)
        _fetch_tasks[task_id] = {"status": "failed", "error": "Internal server error"}


@bp.route("/api/fetch_area", methods=["POST"])
def fetch_area() -> Response:
    """
    Start an async OSM fetch-and-parse job for a lat/lon bounding box.

    Validates and sanitizes the request, computes the UTM bounding box,
    and starts :func:`_run_fetch_task` in a background thread; the
    response returns immediately with a task ID for
    :func:`fetch_area_status` to poll.

    Parameters (JSON body)
    -----------------------
    min_lat, min_lon, max_lat, max_lon : float
        Bounding box corners (WGS84 degrees); min must be strictly less
        than max on each axis.
    name : str
        Output filename stem; sanitized to ``[A-Za-z0-9_-]`` (aborts if
        that leaves nothing). The mapdata is saved to
        ``<data_dir>/<name>.mapdata``.
    grid_margin, obstacle_radius, buffer_widths : optional
        Passed through to :class:`~map_data.map_data.MapData`.

    Returns
    -------
    Response
        JSON ``{"task_id": ...}``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if a required field is missing, the bounding box is
        degenerate/inverted, or ``name`` sanitizes to empty.

    """
    body = request.get_json(force=True) or {}
    for field in ("min_lat", "min_lon", "max_lat", "max_lon", "name"):
        if field not in body:
            abort(400, f"Missing field: {field}")

    if body["min_lat"] >= body["max_lat"] or body["min_lon"] >= body["max_lon"]:
        abort(400, "min_lat/min_lon must be strictly less than max_lat/max_lon")

    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(body["name"]).strip())
    if not name:
        abort(400, "name is empty after sanitizing")

    grid_margin = body.get("grid_margin")
    obstacle_radius = body.get("obstacle_radius")
    buffer_widths = body.get("buffer_widths")

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{name}.mapdata"

    corners = np.array(
        [
            [body["min_lat"], body["min_lon"]],
            [body["min_lat"], body["max_lon"]],
            [body["max_lat"], body["min_lon"]],
            [body["max_lat"], body["max_lon"]],
        ],
    )
    easting, northing, zone_number, zone_letter = utm.from_latlon(corners[:, 0], corners[:, 1])
    waypoints = np.column_stack([easting, northing])

    task_id = str(uuid.uuid4())
    _fetch_tasks[task_id] = {"status": "pending"}
    threading.Thread(
        target=_run_fetch_task,
        args=(
            task_id,
            waypoints,
            zone_number,
            zone_letter,
            out_path,
            grid_margin,
            obstacle_radius,
            buffer_widths,
        ),
        daemon=True,
    ).start()
    return jsonify({"task_id": task_id})


@bp.route("/api/fetch_area/<task_id>", methods=["GET"])
def fetch_area_status(task_id: str) -> Response:
    """
    Poll the status of an async fetch job started by :func:`fetch_area`.

    Once a task reaches a terminal state (``"done"``/``"failed"``), it is
    kept around for 60 seconds after first being observed as terminal
    (to give the client a chance to see the final status), then evicted
    from ``_fetch_tasks`` on the next poll.

    Returns
    -------
    Response
        JSON task record (``{"status": ..., "result": ...}`` or
        ``{"status": ..., "error": ...}``).

    Raises
    ------
    werkzeug.exceptions.HTTPException
        404 if *task_id* is unknown (never existed, or was already
        evicted after completing).

    """
    task = _fetch_tasks.get(task_id)
    if task is None:
        abort(404, "Unknown task ID")
    if task["status"] in ("done", "failed"):
        if "completed_at" not in task:
            task["completed_at"] = time.time()
        elif time.time() - task["completed_at"] > 60:
            _fetch_tasks.pop(task_id, None)
    return jsonify(task)


@bp.route("/api/upload_gpx", methods=["POST"])
def upload_gpx() -> Response:
    """
    Upload a GPX track, fetch its surrounding OSM data, parse, and save it.

    Synchronous (unlike :func:`fetch_area`'s background-thread version):
    the GPX is saved to a temp file (never persisted in the data
    directory), a :class:`~map_data.map_data.MapData` is built from it,
    Overpass is queried and parsed, and the result is saved as
    ``<name>.mapdata``. The temp GPX file is always removed afterward.

    Parameters (multipart form)
    -----------------------------
    file : file
        The GPX track.
    name : str, optional
        Output filename stem; sanitized to ``[A-Za-z0-9_-]``. Defaults to
        the uploaded file's stem if omitted.
    options : str, optional
        JSON-encoded dict with optional ``grid_margin``,
        ``obstacle_radius``, ``buffer_widths`` keys, passed through to
        :class:`~map_data.map_data.MapData`.

    Returns
    -------
    Response
        JSON summary: ``{"filename", "roads", "footways", "barriers",
        "crossroads"}`` (counts of parsed ways/crossroads).

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if no file/empty filename, or ``name`` sanitizes to empty.
        503 if the Overpass API is unreachable. 500 if parsing fails or
        an unexpected error occurs.

    """
    if "file" not in request.files:
        abort(400, "No file part")
    file = request.files["file"]
    if file.filename == "":
        abort(400, "No selected file")

    name = request.form.get("name")
    if not name:
        name = Path(file.filename).stem

    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
    if not name:
        abort(400, "name is empty after sanitizing")

    import json as _json

    _parse_opts = _json.loads(request.form.get("options", "{}"))
    grid_margin = _parse_opts.get("grid_margin")
    obstacle_radius = _parse_opts.get("obstacle_radius")
    buffer_widths = _parse_opts.get("buffer_widths")

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary file to avoid saving the GPX to the data directory
    with tempfile.NamedTemporaryFile(suffix=".gpx", delete=False) as tmp:
        file.save(tmp.name)
        gpx_tmp_path = Path(tmp.name)

    try:
        md = MapData(
            str(gpx_tmp_path),
            coords_type="file",
            grid_margin=grid_margin,
            obstacle_radius=obstacle_radius,
            buffer_widths=buffer_widths,
        )
        # Restore the original filename for metadata purposes
        md.coords_file = file.filename

        md.run_queries()
        if any(d is None for d in (md.osm_ways_data, md.osm_rels_data, md.osm_nodes_data)):
            abort(503, "Overpass API unavailable — try again later")

        if md.run_parse() != 0:
            abort(500, "Parsing failed")

        out_path = data_dir / f"{name}.mapdata"
        md.save(str(out_path))

        return jsonify(
            {
                "filename": f"{name}.mapdata",
                "roads": len(md.roads_list),
                "footways": len(md.footways_list),
                "barriers": len(md.barriers_list),
                "crossroads": len(md.crossroads_list),
            },
        )
    except Exception as e:
        from werkzeug.exceptions import HTTPException

        if isinstance(e, HTTPException):
            raise
        logger.exception("Error processing GPX upload")
        abort(500, "Internal server error")
    finally:
        if gpx_tmp_path.exists():
            gpx_tmp_path.unlink()


@bp.route("/api/upload_mapdata", methods=["POST"])
def upload_mapdata() -> Response:
    """
    Upload a pre-built ``.mapdata`` file to the data directory.

    Saves the upload under its original basename, disambiguating with a
    ``_1``, ``_2``, ... suffix if a file of that name already exists, then
    validates it by attempting to load it; an unloadable file is deleted
    again rather than left behind.

    Returns
    -------
    Response
        JSON ``{"filename": ...}`` (the name actually saved under, which
        may differ from the upload's name if disambiguated).

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if no file was given or it doesn't have a ``.mapdata``
        extension, or if it fails to load as a valid ``MapData`` file.

    """
    if "file" not in request.files:
        abort(400, "No file part")
    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".mapdata"):
        abort(400, "File must have a .mapdata extension")

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    dest = data_dir / safe_name
    stem, suffix = dest.stem, dest.suffix
    counter = 1
    while dest.exists():
        dest = data_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    file.save(dest)

    try:
        MapData.load(str(dest))
    except Exception:
        dest.unlink(missing_ok=True)
        abort(400, "Invalid .mapdata file")

    return jsonify({"filename": dest.name})


@bp.route("/api/way_nodes")
def get_way_nodes() -> Response:
    """
    Return the resolved node list (id/lat/lon/tags) for a way or way segment.

    Resolves ``way_id`` (an original OSM way ID, or a virtual
    ``"<id>:<segment_index>"`` produced by :func:`split_way_endpoint`)
    through :func:`_resolve_way`, then, if virtual, narrows down to the
    requested segment. Node positions prefer the effective nodes cache
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
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing 'file' or 'way_id' query parameter")
    path = _safe_data_path(filename)
    if not path.is_file():
        abort(404, f"File not found: {filename}")

    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))

    original_way_id_str = str(way_id).split(":")[0]
    try:
        search_id = int(original_way_id_str)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer or virtual ID")

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

    # Handle split segments if it's a virtual ID
    if ":" in str(way_id):
        try:
            segment_idx = int(str(way_id).split(":")[1])
        except ValueError:
            abort(400, "Invalid virtual ID")

        split_nids = get_split_node_ids(store, search_id)
        if split_nids:
            segments = split_way(way, split_nids, zn, zl, effective_nc)
            if segment_idx < len(segments):
                way = segments[segment_idx]
                # Apply segment-specific deletions
                seg_del_nids = get_deleted_node_ids(store, way_id)
                if seg_del_nids:
                    way = rebuild_way_without_nodes(
                        way,
                        seg_del_nids,
                        zn,
                        zl,
                        effective_nc,
                        category=category,
                    )
                    if way is None:
                        return jsonify({"way_id": way_id, "nodes": []})
            else:
                abort(404, "Segment not found")

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
    through :func:`_resolve_way` -- passing ``added_nodes_before_overrides=True``,
    unlike :func:`get_way_nodes`/:func:`_get_way_segments_geojson`, so that
    a recorded position override for a user-added node is picked up by
    ``apply_node_position_overrides`` -- then, if virtual, narrows down to
    the requested segment and applies any tag overrides recorded for the
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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = _safe_data_path(filename)
    if not path.is_file():
        abort(404, f"File not found: {filename}")

    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))

    # Virtual ID handling: split by colon
    original_way_id_str = str(way_id).split(":")[0]
    try:
        search_id = int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    resolved = _resolve_way(md, store, search_id, added_nodes_before_overrides=True)
    if resolved.category is None:
        abort(404, f"Way {way_id} not found")
    if resolved.way is None:
        abort(404, f"Way {way_id} reduced to nothing by node deletions")

    way = resolved.way
    category = resolved.category
    nodes_cache = resolved.nodes_cache
    zn, zl = md.zone_number, md.zone_letter

    # Handle split segments if it's a virtual ID
    if ":" in str(way_id):
        try:
            segment_idx = int(str(way_id).split(":")[1])
        except ValueError:
            abort(400, "Invalid virtual ID")

        split_nids = get_split_node_ids(store, search_id)
        if split_nids:
            segments = split_way(way, split_nids, zn, zl, nodes_cache)
            if segment_idx < len(segments):
                way = segments[segment_idx]
                # Apply segment-specific deletions
                seg_del_nids = get_deleted_node_ids(store, way_id)
                if seg_del_nids:
                    way = rebuild_way_without_nodes(
                        way,
                        seg_del_nids,
                        zn,
                        zl,
                        nodes_cache,
                        category=category,
                    )
                    if way is None:
                        abort(
                            404,
                            "Segment reduced to nothing by segment-specific deletions",
                        )
            else:
                abort(404, "Segment not found")

    geom = geom_to_geojson(way.line, zn, zl)
    if geom is None:
        abort(500, "Could not convert geometry")

    feature = {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "id": way_id,
            "category": category,
            "is_node": category == "barrier" and not bool(way.nodes),
            "tags": way.tags or {},
            "in_out": way.in_out,
        },
    }

    ov = store.get("tag_overrides", {}).get(original_way_id_str)
    if ov:
        merged = {**(feature["properties"]["tags"]), **ov}
        feature["properties"]["tags"] = merged
        if category in ("road", "footway"):
            hw = merged.get("highway", "")
            feature["properties"]["category"] = "footway" if hw in FOOTWAY_VALUES else "road"

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
        400 if ``file`` is missing or ``way_id``'s non-segment part isn't
        a valid integer.

    """
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    way_id_str = str(way_id)
    original_way_id_str = way_id_str.split(":")[0]
    try:
        int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    # Segments (virtual IDs like "123:0") are stored as strings so that only the
    # specific segment is suppressed on reload, not the whole original way.
    stored_id: int | str = way_id_str if ":" in way_id_str else int(original_way_id_str)

    body = request.get_json(force=True) or {}
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    migrate_change_log(store)
    if stored_id not in get_deleted_way_ids(store):
        store.setdefault("deleted_ways", []).append(
            {
                "id": stored_id,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
            },
        )
        cl = store.setdefault("change_log", [])
        if not any(e.get("type") == "way" and e.get("id") == stored_id for e in cl):
            cl.append({"type": "way", "id": stored_id, "ts": time.time()})
    save_annotations(ann_path, store)
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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    original_way_id_str = str(way_id).split(":")[0]

    body = request.get_json(force=True) or {}
    tags = body.get("tags")
    if not isinstance(tags, dict):
        abort(400, "Request body must include 'tags' dict")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    migrate_change_log(store)
    store.setdefault("tag_overrides", {})[original_way_id_str] = tags
    store.setdefault("tag_override_meta", {})[original_way_id_str] = {
        "category": body.get("category", "unknown"),
        "label": body.get("label", ""),
    }
    cl = store.setdefault("change_log", [])
    if not any(e.get("type") == "tag" and e.get("id") == original_way_id_str for e in cl):
        cl.append({"type": "tag", "id": original_way_id_str, "ts": time.time()})
    save_annotations(ann_path, store)
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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    store.get("tag_overrides", {}).pop(str(way_id), None)
    store.get("tag_override_meta", {}).pop(str(way_id), None)
    cl = store.get("change_log", [])
    store["change_log"] = [e for e in cl if not (e.get("type") == "tag" and e.get("id") == way_id)]
    save_annotations(ann_path, store)
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
        400 if ``file`` is missing.

    """
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    original_way_id = str(way_id).split(":")[0]
    segments = _get_way_segments_geojson(filename, original_way_id)
    return jsonify({"segments": segments})


def _get_way_segments_geojson(filename: str, original_way_id: str) -> list[dict[str, Any]]:
    """
    Resolve a way's edits and split it into GeoJSON Feature segments.

    Shared by :func:`get_way_segments`, :func:`split_way_endpoint`, and
    :func:`undo_way_split` -- all three want the *current* full set of
    segments for a way after a split-point change. Resolves
    ``original_way_id`` via :func:`_resolve_way` (default
    ``added_nodes_before_overrides=False``, matching :func:`get_way_nodes`),
    then always calls :func:`~map_data.viewer.helpers.split_way` (even if
    the way has no recorded splits, in which case it returns the way
    unchanged as the sole "segment"), and applies segment-specific node
    deletions and the way's tag overrides to each resulting segment.

    Parameters
    ----------
    filename : str
        Mapdata filename (resolved relative to the data directory; not
        validated for path traversal here since callers already do that
        via :func:`_safe_data_path` before invoking a route, or pass an
        already-int-parsed ID).
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
    path = _safe_data_path(filename)
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
    segments = split_way(way, split_nids, zn, zl, effective_nc)

    features = []
    tag_overrides = store.get("tag_overrides", {})
    ov = tag_overrides.get(str(original_way_id))

    for i, seg in enumerate(segments):
        virtual_id = f"{original_way_id}:{i}"

        # Apply segment-specific deletions to segment geometry
        seg_del_nids = get_deleted_node_ids(store, virtual_id)
        if seg_del_nids:
            seg = rebuild_way_without_nodes(  # noqa: PLW2901
                seg,
                seg_del_nids,
                zn,
                zl,
                effective_nc,
                category=category,
            )
            if seg is None:
                continue

        feat_cat = category
        tags = seg.tags or {}
        if ov:
            tags = {**tags, **ov}
            if feat_cat in ("road", "footway"):
                hw = tags.get("highway", "")
                feat_cat = "footway" if hw in FOOTWAY_VALUES else "road"

        features.append(
            {
                "type": "Feature",
                "geometry": geom_to_geojson(seg.line, zn, zl),
                "properties": {
                    "id": virtual_id,
                    "category": feat_cat,
                    "is_node": feat_cat == "barrier" and not bool(seg.nodes),
                    "tags": tags,
                    "in_out": seg.in_out,
                },
            },
        )
    return features


@bp.route("/api/ways/split", methods=["POST"])
def split_way_endpoint() -> Response:
    """
    Record a split point on a way and return its resulting segments.

    Adds *node_id* to ``store["split_ways"][original_way_id]`` (a way may
    have several split points, producing more than two segments) and
    then recomputes every segment via :func:`_get_way_segments_geojson`
    so the client can immediately render the result.

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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True) or {}
    way_id = body.get("way_id")
    node_id = body.get("node_id")

    try:
        way_id_val = str(way_id)
        node_id_int = int(node_id)
    except (ValueError, TypeError):
        abort(400, "Invalid way_id or node_id")

    # If way_id is virtual (e.g. 123:0), get original ID
    original_way_id = way_id_val.split(":", maxsplit=1)[0]

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    migrate_change_log(store)
    splits = store.setdefault("split_ways", {})
    way_splits = splits.setdefault(original_way_id, [])

    if node_id_int not in way_splits:
        way_splits.append(node_id_int)
        cl = store.setdefault("change_log", [])
        if not any(
            e.get("type") == "split"
            and e.get("way_id") == int(original_way_id)
            and e.get("node_id") == node_id_int
            for e in cl
        ):
            cl.append(
                {
                    "type": "split",
                    "way_id": int(original_way_id),
                    "node_id": node_id_int,
                    "ts": time.time(),
                },
            )

    save_annotations(ann_path, store)

    segments = _get_way_segments_geojson(filename, original_way_id)
    return jsonify({"success": True, "segments": segments})


@bp.route("/api/ways/split", methods=["DELETE"])
def undo_way_split() -> Response:
    """
    Remove a single recorded split point from a way and return its (new) segments.

    The way still ends up split if other split points remain on it;
    removing the last one collapses ``split_ways`` back to a single
    unsplit way.

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
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    node_id = request.args.get("node_id")
    if not filename or way_id is None or node_id is None:
        abort(400, "Missing required query parameters")
    try:
        way_id_int = int(way_id)
        node_id_int = int(node_id)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    splits = store.get("split_ways", {})
    if str(way_id_int) in splits:
        splits[str(way_id_int)] = [nid for nid in splits[str(way_id_int)] if nid != node_id_int]
        if not splits[str(way_id_int)]:
            del splits[str(way_id_int)]

    cl = store.get("change_log", [])
    store["change_log"] = [
        e
        for e in cl
        if not (
            e.get("type") == "split"
            and e.get("way_id") == way_id_int
            and e.get("node_id") == node_id_int
        )
    ]
    save_annotations(ann_path, store)

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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    body = request.get_json(force=True) or {}
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    hw = store.setdefault("hidden_ways", [])
    existing_ids = {(d["id"] if isinstance(d, dict) else d) for d in hw}
    if way_id_int not in existing_ids:
        hw.append(
            {
                "id": way_id_int,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
            },
        )
    save_annotations(ann_path, store)
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
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    hw = store.get("hidden_ways", [])
    store["hidden_ways"] = [d for d in hw if (d["id"] if isinstance(d, dict) else d) != way_id_int]
    save_annotations(ann_path, store)
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
        400 if ``file`` is missing or ``way_id``'s non-segment part isn't
        a valid integer.

    """
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    way_id_str = str(way_id)
    original_way_id_str = way_id_str.split(":")[0]
    try:
        int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    stored_id: int | str = way_id_str if ":" in way_id_str else int(original_way_id_str)

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    dw = store.get("deleted_ways", [])
    store["deleted_ways"] = [d for d in dw if (d["id"] if isinstance(d, dict) else d) != stored_id]
    cl = store.get("change_log", [])
    store["change_log"] = [
        e for e in cl if not (e.get("type") == "way" and e.get("id") == stored_id)
    ]
    save_annotations(ann_path, store)
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
        ``after_node_id``/``lat``/``lon``.

    """
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing required query parameters")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer")

    body = request.get_json(force=True) or {}
    after_node_id = body.get("after_node_id")
    lat = body.get("lat")
    lon = body.get("lon")
    if after_node_id is None or lat is None or lon is None:
        abort(400, "Request body must include after_node_id, lat, lon")

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)

    existing_ids = [a["id"] for a in store.get("added_nodes", []) if a["id"] < 0]
    synth_id = (min(existing_ids) - 1) if existing_ids else -1

    store.setdefault("added_nodes", []).append(
        {
            "id": synth_id,
            "way_id": way_id_int,
            "after_node_id": int(after_node_id),
            "lat": float(lat),
            "lon": float(lon),
        }
    )
    cl = store.setdefault("change_log", [])
    cl.append(
        {
            "type": "add_node",
            "way_id": way_id_int,
            "node_id": synth_id,
            "ts": time.time(),
        }
    )
    save_annotations(ann_path, store)
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
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    node_id_arg = request.args.get("node_id")
    if not filename or way_id is None or node_id_arg is None:
        abort(400, "Missing required query parameters")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
        node_id = int(node_id_arg)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers (way_id can be virtual)")

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    migrate_change_log(store)

    # Synthetic nodes (negative IDs) live in added_nodes, not in the OSM node list
    if node_id < 0:
        store["added_nodes"] = [
            a
            for a in store.get("added_nodes", [])
            if not (a.get("way_id") == way_id_int and a.get("id") == node_id)
        ]
        pos_ov = store.get("node_position_overrides", {}).get(str(way_id_int), {})
        pos_ov.pop(str(node_id), None)
        store["change_log"] = [
            e
            for e in store.get("change_log", [])
            if not (
                e.get("type") == "add_node"
                and e.get("way_id") == way_id_int
                and e.get("node_id") == node_id
            )
        ]
        save_annotations(ann_path, store)
        return Response("", 204)

    dn = store.setdefault("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
        store["deleted_nodes"] = dn

    # Use the full way_id (could be virtual like "123:0") to allow segment-specific deletion
    target_id = way_id if ":" in str(way_id) else way_id_int

    if node_id not in get_deleted_node_ids(store, target_id):
        dn.append({"way_id": target_id, "node_id": node_id})
        cl = store.setdefault("change_log", [])
        if not any(
            e.get("type") == "node"
            and str(e.get("way_id")) == str(target_id)
            and e.get("node_id") == node_id
            for e in cl
        ):
            cl.append(
                {
                    "type": "node",
                    "way_id": target_id,
                    "node_id": node_id,
                    "ts": time.time(),
                },
            )
    save_annotations(ann_path, store)
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
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    node_id_arg = request.args.get("node_id")
    if not filename or way_id is None or node_id_arg is None:
        abort(400, "Missing required query parameters")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
        node_id = int(node_id_arg)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers (way_id can be virtual)")

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)

    # Use the full way_id to match the deletion record
    target_id = way_id if ":" in str(way_id) else way_id_int

    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
    store["deleted_nodes"] = [
        d
        for d in dn
        if not (str(d.get("way_id")) == str(target_id) and d.get("node_id") == node_id)
    ]
    cl = store.get("change_log", [])
    store["change_log"] = [
        e
        for e in cl
        if not (
            e.get("type") == "node"
            and str(e.get("way_id")) == str(target_id)
            and e.get("node_id") == node_id
        )
    ]
    save_annotations(ann_path, store)
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
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing required query parameters")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer")

    body = request.get_json(force=True) or {}
    nodes = body.get("nodes")
    if not isinstance(nodes, list):
        abort(400, "Request body must include 'nodes' list")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    overrides = store.setdefault("node_position_overrides", {})
    way_key = original_way_id_str
    if way_key not in overrides:
        overrides[way_key] = {}
    for n in nodes:
        overrides[way_key][str(n["id"])] = {
            "lat": float(n["lat"]),
            "lon": float(n["lon"]),
        }
    migrate_change_log(store)
    cl = store.setdefault("change_log", [])
    if not any(e.get("type") == "move" and e.get("id") == way_id_int for e in cl):
        cl.append(
            {
                "type": "move",
                "id": way_id_int,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
                "ts": time.time(),
            },
        )
    save_annotations(ann_path, store)
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
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing required query parameters")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer or virtual ID")

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    store.get("node_position_overrides", {}).pop(original_way_id_str, None)
    cl = store.get("change_log", [])
    store["change_log"] = [
        e for e in cl if not (e.get("type") == "move" and e.get("id") == way_id_int)
    ]
    save_annotations(ann_path, store)
    return Response("", 204)


def get_merged_mapdata(filename: str) -> tuple[MapData | None, dict[str, Any] | None]:
    """
    Build a fully-edited, planning/export-ready ``MapData`` for a file.

    Used by :func:`export_mapdata`, :func:`get_cost_grid`, and
    :func:`create_replan`, all of which need a genuinely self-consistent
    ``MapData`` (not just a GeoJSON view of one) to feed to
    :mod:`map_data.utils.serialization` or the path planner. Deep-copies
    the cached ``MapData`` (see the module docstring's "MapData copy
    semantics" section for why deepcopy is used here rather than the
    shallow-copy-plus-manual-Way-copy convention used elsewhere) before:

    1. Applying way/node deletions, splits, and node moves via
       :func:`_apply_way_edits`.
    2. Merging tag overrides directly into each way's ``tags`` (mutating
       the way in place -- safe only because of the deepcopy), then
       re-sorting every way between ``roads_list``/``footways_list`` in
       case an overridden ``highway`` tag changed its category, and
       recomputing ``crossroads_list``.
    3. Synthesizing a new ``Way`` for each freehand annotation
       (``store["annotations"]``, from :func:`add_annotation`): a
       ``"path"`` annotation becomes a road/footway (buffered by its
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
    zn, zl = md.zone_number, md.zone_letter

    _apply_way_edits(md, store)

    tag_overrides = store.get("tag_overrides", {})
    if tag_overrides:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            for w in getattr(md, lst_name):
                original_id = str(w.id).split(":")[0]
                ov = tag_overrides.get(original_id)
                if ov:
                    w.tags = {**(w.tags or {}), **ov}
        new_roads, new_footways = [], []
        for w in md.roads_list:
            (new_footways if w.is_footway() else new_roads).append(w)
        for w in md.footways_list:
            (new_roads if w.is_road() else new_footways).append(w)
        md.roads_list, md.footways_list = new_roads, new_footways
        md.crossroads_list = md.parse_intersections({str(w.id): w for w in md.footways_list})

    ann_id = -1
    node_id = -1
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
                width_m = float(props.get("width", 1.5))
                for e_coord, n_coord in geom.coords:
                    lat, lon = utm.to_latlon(e_coord, n_coord, zn, zl)
                    md.nodes_cache[node_id] = {"lat": lat, "lon": lon, "tags": {}}
                    w.nodes.append(node_id)
                    node_id -= 1
                w.line = geom.buffer(width_m / 2)
                w.is_area = True
            (md.roads_list if w.is_road() else md.footways_list).append(w)
        else:
            w.tags = {"barrier": props.get("barrier", "wall")}
            for k, v in props.items():
                if k != "barrier":
                    w.tags[k] = str(v)
            md.barriers_list.append(w)

    return md, store


@bp.route("/api/export")
def export_mapdata() -> Response:
    """
    Download a mapdata file, with all recorded edits baked in, as JSON.

    Side Effects
    ------------
    None (does not write to the data directory; the merged data is
    streamed directly to the client).

    Returns
    -------
    Response
        The serialized (via
        :func:`~map_data.utils.serialization.map_data_to_dict`) mapdata
        as a downloadable attachment named ``<stem>.exported.mapdata``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing. 404 if the file doesn't exist.

    """
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File not found: {filename}")

    buf = io.BytesIO()
    buf.write(json.dumps(map_data_to_dict(md), indent=2).encode("utf-8"))
    buf.seek(0)
    base = Path(filename).stem
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{base}.exported.mapdata",
        mimetype="application/json",
    )


@bp.route("/api/export/geojson")
def export_geojson() -> Response:
    """
    Download a mapdata file, with all recorded edits baked in, as GeoJSON.

    Unlike :func:`export_mapdata` (which dumps the native ``.mapdata``
    schema), this converts the fully-merged ``MapData`` to a GeoJSON
    ``FeatureCollection`` via
    :func:`~map_data.viewer.helpers.mapdata_to_geojson`, for use in GIS
    tools such as QGIS or geojson.io.

    Side Effects
    ------------
    None (does not write to the data directory; the merged data is
    streamed directly to the client).

    Returns
    -------
    Response
        The GeoJSON ``FeatureCollection`` (see
        :func:`~map_data.viewer.helpers.mapdata_to_geojson`) as a
        downloadable attachment named ``<stem>.geojson``, with
        ``Content-Type: application/geo+json``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing. 404 if the file doesn't exist.

    """
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File not found: {filename}")

    buf = io.BytesIO()
    buf.write(json.dumps(mapdata_to_geojson(md)).encode("utf-8"))
    buf.seek(0)
    base = Path(filename).stem
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{base}.geojson",
        mimetype="application/geo+json",
    )


@bp.route("/api/cost_grid")
def get_cost_grid() -> Response:
    """
    Compute a coarse traversal-cost grid for a bounding box, for visualization.

    Builds a fully-edited ``MapData`` via :func:`get_merged_mapdata`,
    fills a 1m-resolution :class:`~map_data.pathsolver.replan.ReplanPath`
    grid over the requested lat/lon box (obstacles are *not* filtered out
    of the result, unlike planning, so the client can render them), and
    returns every cell as a flat ``[lat, lon, cost]`` point.

    Parameters (query string)
    -----------------------------
    file : str
        Mapdata filename.
    min_lat, min_lon, max_lat, max_lon : float
        Bounding box (WGS84 degrees; order-independent -- min/max are
        computed from the two corners regardless of which is given as
        "min").
    highway_costs, surface_costs : str, optional
        JSON-encoded cost-override dicts; a malformed value is logged and
        ignored (falls back to defaults) rather than erroring the request.

    Returns
    -------
    Response
        JSON list of ``[lat, lon, cost]`` triples, one per grid cell.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if any required parameter is missing. 404 if the file
        doesn't exist.

    """
    filename = request.args.get("file")
    min_lat = request.args.get("min_lat", type=float)
    min_lon = request.args.get("min_lon", type=float)
    max_lat = request.args.get("max_lat", type=float)
    max_lon = request.args.get("max_lon", type=float)

    if filename is None or any(v is None for v in (min_lat, min_lon, max_lat, max_lon)):
        abort(400, "Missing required parameters")
    if filename is None:
        abort(400, "Filename cannot be None")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File {filename} not found")

    zn, zl = md.zone_number, md.zone_letter
    p1 = utm.from_latlon(min_lat, min_lon, zn, zl)
    p2 = utm.from_latlon(max_lat, max_lon, zn, zl)

    low = (min(p1[0], p2[0]), min(p1[1], p2[1]))
    high = (max(p1[0], p2[0]), max(p1[1], p2[1]))

    args = parse_args([])
    args.low = low
    args.high = high
    args.cell_size = 1.0  # Use a coarser grid for visualization performance
    args.inflate_obstacles = 0.0

    # Get custom highway costs from request if provided
    highway_costs_dict = None
    highway_costs = request.args.get("highway_costs")
    if highway_costs:
        try:
            highway_costs_dict = json.loads(highway_costs)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse custom highway costs: %s", e)

    surface_costs_dict = None
    surface_costs = request.args.get("surface_costs")
    if surface_costs:
        try:
            surface_costs_dict = json.loads(surface_costs)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse custom surface costs: %s", e)

    obstacles = ways_to_shapely(md.barriers_list)
    replanner = ReplanPath(
        args, obstacles, highway_costs=highway_costs_dict, surface_costs=surface_costs_dict
    )

    replanner.fill_grid(md, highway_types=["footway", "road"])

    grid = replanner.grid  # [N, 4] -> [x, y, 0, cost]
    # Do not filter out obstacles (cost >= 1.0) so they can be visualized
    visible_grid = grid

    points = []
    for row in visible_grid:
        lat, lon = utm.to_latlon(row[0], row[1], zn, zl)
        points.append([lat, lon, float(row[3])])

    return jsonify(points)


@bp.route("/api/cancel_replan", methods=["POST"])
def cancel_replan_route() -> Response:
    """
    Cancel an in-progress replan started by :func:`create_replan`.

    Parameters (JSON body)
    -----------------------
    transfer_id : str
        ID of the replan to cancel, as passed to
        :func:`~map_data.pathsolver.replan.ReplanPath`.

    Returns
    -------
    Response
        JSON ``{"success": true}`` (always -- an unknown or already
        finished ``transfer_id`` is not reported as an error by
        :func:`~map_data.pathsolver.replan.cancel_replan_backend`).

    """
    transfer_id = request.json.get("transfer_id")
    cancel_replan_backend(transfer_id)
    return jsonify({"success": True})


class WormholeManager:
    """
    Manage ``magic-wormhole`` subprocesses that send a GPX file to a companion app.

    Each transfer runs an actual ``wormhole send`` subprocess against a
    temp file; a background thread scrapes its stdout for the generated
    wormhole code, and cleans up the process and temp directory once the
    transfer finishes, fails, or is cancelled. State for all in-flight
    transfers lives in ``active_transfers``, keyed by a generated
    ``transfer_id``.
    """

    def __init__(self) -> None:
        """Initialize with no active transfers, warning once if the ``wormhole`` CLI is missing."""
        self.active_transfers: dict[str, dict[str, Any]] = {}
        if shutil.which("wormhole") is None:
            # We don't want to crash the whole app if wormhole is missing,
            # just log it and the endpoints will fail gracefully.
            logger.warning("'wormhole' command not found. magic-wormhole is required for sharing.")

    def create_transfer(self, gpx_data: str) -> str:
        """
        Start a ``wormhole send`` subprocess for *gpx_data* and return its transfer ID.

        Writes *gpx_data* to a fresh temp directory, spawns ``wormhole
        send`` on it, and starts :meth:`_capture_wormhole_code_thread` in
        the background to scrape the resulting wormhole code from the
        process's output. The temp directory is *not* cleaned up here on
        success -- that happens in :meth:`_cleanup_transfer` once the
        background thread observes the process finish; it *is* cleaned
        up immediately if the subprocess fails to even start.

        Parameters
        ----------
        gpx_data : str
            Raw GPX file contents to send.

        Returns
        -------
        str
            Newly generated transfer ID, registered in ``active_transfers``.

        Raises
        ------
        RuntimeError
            If the ``wormhole`` subprocess fails to start.

        """
        transfer_id = str(uuid.uuid4())
        temp_dir = Path(tempfile.mkdtemp())
        file_path = temp_dir / "path.gpx"

        with file_path.open("w") as f:
            f.write(gpx_data)

        cmd = ["wormhole", "send", str(file_path)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as e:
            shutil.rmtree(temp_dir)
            msg = f"Failed to start wormhole process: {e}"
            raise RuntimeError(msg) from e

        logger.info("Starting wormhole transfer %s: %s", transfer_id, " ".join(cmd))
        logger.info("Process ID for transfer %s: %s", transfer_id, process.pid)

        self.active_transfers[transfer_id] = {
            "process": process,
            "temp_dir": temp_dir,
            "start_time": time.time(),
            "status": "running",
            "code": None,
        }

        threading.Thread(
            target=self._capture_wormhole_code_thread,
            args=(transfer_id,),
            daemon=True,
        ).start()
        return transfer_id

    def _capture_wormhole_code_thread(self, transfer_id: str) -> None:
        """
        Background-thread target: scrape the wormhole code and await process exit.

        Polls the subprocess's stdout/stderr for the ``"Wormhole code
        is: ..."`` line, records it on ``active_transfers[transfer_id]["code"]``
        as soon as found, then waits (up to 60s) for the process to exit
        and records a final ``"completed"``/``"failed"`` status. Always
        calls :meth:`_cleanup_transfer` on exit (success, failure, or
        exception), which removes the transfer's temp directory and its
        ``active_transfers`` entry.

        Parameters
        ----------
        transfer_id : str
            Transfer to monitor; a no-op if it's no longer in
            ``active_transfers``.

        """
        transfer_info = self.active_transfers.get(transfer_id)
        if not transfer_info:
            return

        process = transfer_info["process"]
        wormhole_code = None
        try:
            while True:
                readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                for stream in readable:
                    line = stream.readline().strip()
                    if line:
                        match = re.search(r"Wormhole code is: (\S+-\S+-\S+)", line)
                        if match:
                            wormhole_code = match.group(1)
                            logger.info(
                                "Wormhole code for transfer %s: %s",
                                transfer_id,
                                wormhole_code,
                            )
                        elif stream == process.stderr:
                            logger.warning("Wormhole stderr (%s): %s", transfer_id, line)

                if wormhole_code:
                    transfer_info["code"] = wormhole_code
                    break

                if process.poll() is not None:
                    break

            process.wait(timeout=60)
            transfer_info["status"] = "completed" if process.returncode == 0 else "failed"
        except Exception:
            logger.exception("Error in wormhole thread for %s", transfer_id)
            transfer_info["status"] = "failed"
            if process.poll() is None:
                process.kill()
        finally:
            self._cleanup_transfer(transfer_id)

    def get_transfer_code(self, transfer_id: str, timeout: float = 10) -> str | None:
        """
        Block (polling every 0.1s) until a transfer's wormhole code is available.

        Parameters
        ----------
        transfer_id : str
            Transfer to wait on.
        timeout : float, default 10
            Maximum seconds to wait.

        Returns
        -------
        str or None
            The wormhole code, or ``None`` if it wasn't captured within
            *timeout* (including if *transfer_id* is unknown throughout).

        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if transfer_id in self.active_transfers and self.active_transfers[transfer_id].get(
                "code",
            ):
                return str(self.active_transfers[transfer_id]["code"])
            time.sleep(0.1)
        return None

    def cancel_transfer(self, transfer_id: str) -> tuple[bool, str]:
        """
        Kill an active transfer's ``wormhole`` subprocess.

        Marks the transfer ``"cancelled"`` but does not remove it from
        ``active_transfers`` or clean up its temp directory here -- that
        still happens via :meth:`_capture_wormhole_code_thread` observing
        the process exit and calling :meth:`_cleanup_transfer`.

        Parameters
        ----------
        transfer_id : str
            Transfer to cancel.

        Returns
        -------
        tuple of (bool, str)
            ``(True, "Transfer cancelled")`` on success, or ``(False,
            "Invalid or unknown transfer ID")`` if *transfer_id* isn't active.

        """
        if transfer_id not in self.active_transfers:
            return False, "Invalid or unknown transfer ID"

        logger.info("Cancelling wormhole transfer %s", transfer_id)
        process = self.active_transfers[transfer_id]["process"]
        if process.poll() is None:
            process.kill()

        self.active_transfers[transfer_id]["status"] = "cancelled"
        return True, "Transfer cancelled"

    def _cleanup_transfer(self, transfer_id: str) -> None:
        """
        Remove a finished transfer's bookkeeping entry and delete its temp directory.

        Parameters
        ----------
        transfer_id : str
            Transfer to remove; a no-op if already removed.

        Side Effects
        ------------
        Deletes the transfer's temp directory tree on disk (errors are
        logged, not raised).

        """
        transfer = self.active_transfers.pop(transfer_id, None)
        if transfer and transfer.get("temp_dir"):
            try:
                shutil.rmtree(transfer["temp_dir"])
            except Exception:
                logger.exception("Error cleaning temp dir for %s", transfer_id)


wormhole_manager = WormholeManager()


@bp.route("/api/create_wormhole", methods=["POST"])
def create_wormhole() -> Response:
    """
    Start sending a GPX path to a companion app via ``magic-wormhole``.

    Blocks up to 15s waiting for the wormhole code to be captured (see
    :meth:`WormholeManager.get_transfer_code`); if it isn't ready in
    time, the transfer is cancelled and an error is returned rather than
    leaving an orphaned transfer running.

    Parameters (JSON body)
    -----------------------
    gpx : str
        Raw GPX file contents to send.

    Returns
    -------
    Response
        JSON ``{"success": true, "code": ..., "transfer_id": ...}`` on
        success (status 200); on failure, ``{"success": false,
        "message": ...}`` with status 400 (missing ``gpx``) or 500
        (wormhole code not captured in time, or an internal error).

    """
    gpx_data = request.json.get("gpx")
    if not gpx_data:
        return jsonify({"success": False, "message": "No GPX data provided"}), 400

    try:
        transfer_id = wormhole_manager.create_transfer(gpx_data)
        code = wormhole_manager.get_transfer_code(transfer_id, timeout=15)
        if code:
            return jsonify({"success": True, "code": code, "transfer_id": transfer_id})
        wormhole_manager.cancel_transfer(transfer_id)
        return jsonify(
            {"success": False, "message": "Failed to capture wormhole code in time"},
        ), 500
    except Exception:
        logger.exception("Error creating wormhole")
        return jsonify({"success": False, "message": "Internal server error"}), 500


@bp.route("/api/cancel_wormhole", methods=["POST"])
def cancel_wormhole() -> Response:
    """
    Cancel an in-progress ``magic-wormhole`` GPX transfer.

    Returns
    -------
    Response
        JSON ``{"success": ..., "message": ...}`` per
        :meth:`WormholeManager.cancel_transfer`.

    """
    transfer_id = request.json.get("transfer_id")
    success, message = wormhole_manager.cancel_transfer(transfer_id)
    return jsonify({"success": success, "message": message})


@bp.route("/api/create_replan", methods=["POST"])
def create_replan() -> Response:
    """
    Replan a path against the (edited) map data, using a grid or graph planner.

    Builds a fully-edited ``MapData`` via :func:`get_merged_mapdata`,
    projects the input path to UTM, computes a planning bounding box
    (a 50m margin around the path, clipped to the map's extent), and
    dispatches to either :class:`~map_data.pathsolver.graph_planner.GraphPlanner`
    (``algorithm="graph"``) or a grid-based
    :class:`~map_data.pathsolver.replan.ReplanPath` (any other
    *algorithm* value, itself further parameterized by *sub_algorithm*,
    e.g. ``"astar"``). For the grid path, barriers are pre-filtered to
    the bounding box for speed.

    Parameters (JSON body)
    -----------------------
    points : list of [lat, lon]
        Path to replan. Required.
    file : str
        Mapdata filename. Required.
    allowed_ways : list of str, default ``["footway"]``
        Highway types the planner may route over.
    transfer_id : str, optional
        ID used to make this replan cancellable via
        :func:`cancel_replan_route` (grid planner only).
    algorithm : str, default ``"rrt"``
        ``"graph"`` for :class:`~map_data.pathsolver.graph_planner.GraphPlanner`;
        anything else selects the grid-based planner.
    sub_algorithm : str, default ``"astar"``
        Search algorithm passed to
        :meth:`~map_data.pathsolver.replan.ReplanPath.replan` (grid
        planner only).
    highway_costs, surface_costs : dict, optional
        Custom per-tag traversal cost overrides (grid planner only).
    cell_size : float, default 0.25
        Grid cell size in meters (grid planner only).
    inflate_obstacles : float, default 0.25
        Obstacle inflation radius in meters (grid planner only).
    simplify_path, smooth_path : bool
        Post-processing toggles (grid planner only).
    grid_cost_weight : float, optional
        Weight of grid traversal cost vs. path length (grid planner only).

    Returns
    -------
    Response
        JSON ``{"retrieveNum": ..., "newPath": [[lat, lon], ...] | None,
        "status": ...}``. ``retrieveNum`` is ``1`` (with ``newPath: None``
        and a ``"status"`` of ``"cancelled"`` or ``"failed"``) if planning
        produced no result, ``0`` if the result differs significantly
        from the input path (by point count or by
        :data:`SIGNIFICANT_CHANGE_TOLERANCE` meters at any matching
        index), or ``-1`` if it's effectively unchanged.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``points`` or ``file`` is missing. 404 if the file doesn't
        exist.

    """
    body = request.get_json(force=True) or {}
    path_data = body.get("points")  # [[lat, lon], ...]
    filename = body.get("file")
    highway_types = body.get("allowed_ways", ["footway"])
    transfer_id = body.get("transfer_id")
    algorithm = body.get("algorithm", "rrt")
    sub_algorithm = body.get("sub_algorithm", "astar")
    highway_costs = body.get("highway_costs")
    surface_costs = body.get("surface_costs")

    cell_size = body.get("cell_size", 0.25)
    inflate_obstacles = body.get("inflate_obstacles", 0.25)
    simplify_path = body.get("simplify_path", True)
    smooth_path = body.get("smooth_path", False)
    grid_cost_weight = body.get("grid_cost_weight")

    if not path_data or not filename:
        abort(400, "Missing points or file parameter")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File {filename} not found")

    zn, zl = md.zone_number, md.zone_letter
    utm_path = []
    for p in path_data:
        e, n, _, _ = utm.from_latlon(float(p[0]), float(p[1]), zn, zl)
        utm_path.append([e, n])
    utm_path = np.array(utm_path, dtype=np.float64)

    # Calculate planning bounding box to limit grid size
    margin = 50.0  # meters
    p_min_x = np.min(utm_path[:, 0]) - margin
    p_max_x = np.max(utm_path[:, 0]) + margin
    p_min_y = np.min(utm_path[:, 1]) - margin
    p_max_y = np.max(utm_path[:, 1]) + margin

    # Clip to map boundaries
    p_low = (max(md.min_x, p_min_x), max(md.min_y, p_min_y))
    p_high = (min(md.max_x, p_max_x), min(md.max_y, p_max_y))

    if algorithm == "graph":
        planner = GraphPlanner(md, highway_types=highway_types)
        res = planner.plan(utm_path)
    else:
        args = parse_args([])
        args.simplify_path = simplify_path
        args.smooth_path = smooth_path
        args.cell_size = cell_size
        args.inflate_obstacles = inflate_obstacles
        args.visualize = False
        args.low = p_low
        args.high = p_high

        # Filter barriers to bounding box for faster processing
        bbox = geometry.box(p_low[0], p_low[1], p_high[0], p_high[1])
        filtered_barriers = [w for w in md.barriers_list if w.line and w.line.intersects(bbox)]

        obstacles = ways_to_shapely(filtered_barriers)
        replanner = ReplanPath(
            args,
            obstacles,
            transfer_id=transfer_id,
            grid_cost_weight=grid_cost_weight,
            highway_costs=highway_costs,
            surface_costs=surface_costs,
        )
        replanner.fill_grid(md, highway_types=highway_types)
        res = replanner.replan(utm_path, algorithm=sub_algorithm)

    if res is None:
        status = "cancelled" if algorithm != "graph" else "failed"
        return jsonify({"retrieveNum": 1, "newPath": None, "status": status})

    new_path = []
    changed = False

    # RRT* result might have more/different points
    if len(res) != len(utm_path):
        changed = True

    for i in range(len(res)):
        lat, lon = utm.to_latlon(res[i][0], res[i][1], zn, zl)
        new_path.append([lat, lon])
        # Simple heuristic to check if it actually changed significantly
        if (
            not changed
            and i < len(utm_path)
            and np.linalg.norm(res[i] - utm_path[i]) > SIGNIFICANT_CHANGE_TOLERANCE
        ):
            changed = True

    if changed:
        return jsonify({"retrieveNum": 0, "newPath": new_path})
    return jsonify({"retrieveNum": -1, "newPath": new_path})
