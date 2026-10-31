"""File routes: viewer page, file listing, mapdata GeoJSON, OSM fetch, uploads, export."""

import copy
import io
import json
import logging
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import utm
from flask import (
    Response,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)
from flask.typing import ResponseReturnValue
from werkzeug.exceptions import HTTPException

from map_data.annotations import apply_tag_overrides, apply_way_edits
from map_data.edits import load_annotations
from map_data.map_data import MapData
from map_data.utils.serialization import map_data_to_dict

from ..cache import load_mapdata_cached, mapdata_geojson_cached
from ..helpers import mapdata_to_geojson
from .common import (
    MAX_FETCH_AREA_KM2,
    _annotation_path,
    _bbox_area_km2,
    _get_data_dir,
    _json_object,
    _mapdata_path,
    _require_args,
    _validated_bbox,
    bp,
    get_merged_mapdata,
)

logger = logging.getLogger(__name__)


@bp.route("/")
def index() -> str:
    """Render the viewer's single-page HTML shell."""
    return render_template("index.html")


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
    result: dict[str, list[str]] = {"mapdata": [], "gpx": []}
    try:
        for p in sorted(_get_data_dir().iterdir()):
            if p.suffix == ".mapdata":
                result["mapdata"].append(p.name)
            elif p.suffix == ".gpx":
                result["gpx"].append(p.name)
    except FileNotFoundError:
        pass
    return jsonify(result)


def _merged_mapdata_geojson(path: Path, store: dict[str, Any]) -> dict[str, Any]:
    """
    Build the ``/api/mapdata`` FeatureCollection for a mapdata *path* edited by *store*.

    Way edits and tag overrides only; freehand annotations are drawn by the
    client. A shallow copy of the cached map suffices: both passes rebind
    the map's attributes and never mutate a shared ``Way``.
    """
    map_data = copy.copy(load_mapdata_cached(str(path)))

    apply_way_edits(map_data, store)
    apply_tag_overrides(map_data, store)
    return mapdata_to_geojson(map_data)


@bp.route("/api/mapdata")
def get_mapdata() -> ResponseReturnValue:
    """
    Return a mapdata file, with its way edits applied, as a GeoJSON FeatureCollection.

    Query: ``file``. The serialized document is cached by
    :func:`~map_data.viewer.cache.mapdata_geojson_cached` under the map's
    signature and the store's digest, which is also the ``ETag``: a matching
    ``If-None-Match`` gets a 304. 400 if ``file`` is missing, 404 if the file
    doesn't exist.
    """
    filename = _require_args("file")
    path = _mapdata_path(filename)
    store = load_annotations(str(_annotation_path(filename)))
    body, etag = mapdata_geojson_cached(
        str(path),
        store,
        lambda: current_app.json.dumps(_merged_mapdata_geojson(path, store)),
    )
    resp = current_app.response_class(body, mimetype="application/json")
    resp.set_etag(etag)
    # An edit changes the body under the same URL, so the browser must ask
    # every time; the ETag is what keeps that ask cheap.
    resp.headers["Cache-Control"] = "no-cache"
    return resp.make_conditional(request)


# Paths picked by a running fetch or GPX upload that has not written its map yet.
_reserved_paths: set[Path] = set()
_reserved_paths_lock = threading.Lock()


def _free_mapdata_path(data_dir: Path, stem: str, *, reserve: bool = False) -> Path:
    """
    Return ``<stem>.mapdata`` in *data_dir*, or ``<stem>_1.mapdata``, ``_2``, ... if taken.

    A name is taken by an existing map, a leftover annotation store (which
    would otherwise be applied to the new map) or a path in
    :data:`_reserved_paths`. With *reserve*, the returned path is added there;
    the caller must discard it once the map is written or abandoned.
    """
    dest, counter = data_dir / f"{stem}.mapdata", 1
    with _reserved_paths_lock:
        while (
            dest.exists()
            or (data_dir / f"{dest.stem}.annotations.json").exists()
            or dest in _reserved_paths
        ):
            dest = data_dir / f"{stem}_{counter}.mapdata"
            counter += 1
        if reserve:
            _reserved_paths.add(dest)
    return dest


def _release_mapdata_path(path: Path) -> None:
    """Drop *path* from :data:`_reserved_paths` (no-op if absent)."""
    with _reserved_paths_lock:
        _reserved_paths.discard(path)


_fetch_tasks: dict[str, dict[str, Any]] = {}

#: How long a terminal ("done"/"failed") fetch task is retained, in seconds,
#: before it is evicted -- either by a poll in :func:`fetch_area_status` or by
#: the sweep in :func:`fetch_area`.
FETCH_TASK_RETENTION_S = 60.0


def _stamp_or_evict_if_stale(task_id: str, task: dict[str, Any], now: float) -> None:
    """
    Stamp a terminal task's ``completed_at``, or evict it once it is older than the retention.

    Non-terminal tasks are left alone. Eviction only pops it from
    ``_fetch_tasks``, so a caller holding *task* can still return it once.
    """
    if task.get("status") not in ("done", "failed"):
        return
    completed_at = task.setdefault("completed_at", now)
    if now - completed_at > FETCH_TASK_RETENTION_S:
        _fetch_tasks.pop(task_id, None)


class _FetchFailed(Exception):
    """Raised by :func:`_query_parse_save` when the Overpass query or parse step fails."""

    def __init__(self, stage: str, error: str) -> None:
        """*stage* is ``"query"`` or ``"parse"``, letting callers pick an appropriate response."""
        super().__init__(error)
        self.stage = stage


def _query_parse_save(
    md: MapData,
    out_path: Path,
    progress_cb: Any = None,
    on_parse_start: Any = None,
) -> dict[str, Any]:
    """
    Query Overpass, parse the result into *md*, and save it to *out_path*.

    *progress_cb* goes to ``md.run_queries``; *on_parse_start* is called
    once querying succeeds.

    Returns
    -------
    dict
        ``{"filename", "roads", "footways", "barriers", "crossroads"}``.

    Raises
    ------
    _FetchFailed
        If the Overpass query comes back empty, or parsing fails.

    """
    md.run_queries(progress_cb=progress_cb)
    if md.osm_data is None:
        raise _FetchFailed("query", "Overpass API unavailable — try again later")
    if on_parse_start:
        on_parse_start()
    if md.run_parse() != 0:
        raise _FetchFailed("parse", "Parsing failed")
    md.save(str(out_path))
    return {
        "filename": out_path.name,
        "roads": len(md.roads_list),
        "footways": len(md.footways_list),
        "barriers": len(md.barriers_list),
        "crossroads": len(md.crossroads_list),
    }


def _run_fetch_task(task_id: str, md: MapData, out_path: Path) -> None:
    """
    Fetch, parse, and save OSM data for a bounding box; the target of :func:`fetch_area`'s thread.

    *md* is the unqueried map of the box. Progress goes to
    ``_fetch_tasks[task_id]``: ``"querying"`` (with a ``"detail"``),
    ``"parsing"``, then ``"done"`` (with ``"result"``) or ``"failed"`` (with
    ``"error"``). Exceptions are recorded as failures, never raised: there is
    no request to report them to.
    """
    try:

        def _report(detail: str) -> None:
            _fetch_tasks[task_id] = {"status": "querying", "detail": detail}

        def _parsing_started() -> None:
            _fetch_tasks[task_id] = {"status": "parsing", "detail": "Parsing OSM data…"}

        _report("Querying Overpass…")
        result = _query_parse_save(
            md, out_path, progress_cb=_report, on_parse_start=_parsing_started
        )
        _fetch_tasks[task_id] = {"status": "done", "result": result}
    except _FetchFailed as e:
        logger.warning("fetch task %s failed at %s: %s", task_id, e.stage, e)
        _fetch_tasks[task_id] = {"status": "failed", "error": str(e)}
    except Exception:
        logger.exception("fetch task %s failed", task_id)
        _fetch_tasks[task_id] = {"status": "failed", "error": "Internal server error"}
    finally:
        _release_mapdata_path(out_path)


@bp.route("/api/fetch_area", methods=["POST"])
def fetch_area() -> Response:
    """
    Start a background OSM fetch for a lat/lon bounding box; poll :func:`fetch_area_status`.

    Parameters (JSON body)
    -----------------------
    min_lat, min_lon, max_lat, max_lon : float
        WGS84 box, min < max on each axis, at most
        :data:`~map_data.viewer.routes.common.MAX_FETCH_AREA_KM2`.
    name : str
        Output stem, sanitized to ``[A-Za-z0-9_-]``; saved as
        ``<name>.mapdata`` or the next free name (:func:`_free_mapdata_path`).
    grid_margin, obstacle_radius, buffer_widths : optional
        Passed through to :class:`~map_data.map_data.MapData`.

    Returns
    -------
    Response
        JSON ``{"task_id": ...}``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if a field is missing or invalid, the box is too large, or
        ``name`` sanitizes to empty.

    """
    body = _json_object()
    for field in ("min_lat", "min_lon", "max_lat", "max_lon", "name"):
        if field not in body:
            abort(400, f"Missing field: {field}")

    min_lat, min_lon, max_lat, max_lon = _validated_bbox(
        body["min_lat"], body["min_lon"], body["max_lat"], body["max_lon"]
    )

    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(body["name"]).strip())
    if not name:
        abort(400, "name is empty after sanitizing")

    grid_margin = body.get("grid_margin")
    obstacle_radius = body.get("obstacle_radius")
    buffer_widths = body.get("buffer_widths")

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    corners = np.array(
        [
            [min_lat, min_lon],
            [min_lat, max_lon],
            [max_lat, min_lon],
            [max_lat, max_lon],
        ],
    )
    easting, northing, zone_number, zone_letter = utm.from_latlon(corners[:, 0], corners[:, 1])
    try:
        md = MapData(
            (np.column_stack([easting, northing]), zone_number, zone_letter),
            coords_type="array",
            grid_margin=grid_margin,
            obstacle_radius=obstacle_radius,
            buffer_widths=buffer_widths,
        )
    except (TypeError, ValueError) as e:
        abort(400, f"Invalid parse options: {e}")
    # The download is the box plus grid_margin, so the limit applies to that.
    area_km2 = _bbox_area_km2(md.min_lat, md.min_long, md.max_lat, md.max_long)
    if area_km2 > MAX_FETCH_AREA_KM2:
        abort(
            400,
            f"Requested area with its margin is {area_km2:.1f} km², which exceeds the "
            f"{MAX_FETCH_AREA_KM2:.0f} km² limit for a single fetch. Draw a smaller area.",
        )

    # Also evict here, or tasks whose client stopped polling would pile up.
    now = time.time()
    for stale_id, task in list(_fetch_tasks.items()):
        _stamp_or_evict_if_stale(stale_id, task, now)
    task_id = str(uuid.uuid4())
    _fetch_tasks[task_id] = {"status": "pending"}
    # Reserved, as the map is only written when the background task finishes.
    out_path = _free_mapdata_path(data_dir, name, reserve=True)
    threading.Thread(
        target=_run_fetch_task,
        args=(task_id, md, out_path),
        daemon=True,
    ).start()
    return jsonify({"task_id": task_id})


@bp.route("/api/fetch_area/<task_id>", methods=["GET"])
def fetch_area_status(task_id: str) -> Response:
    """
    Return the task record of a :func:`fetch_area` job (see :func:`_run_fetch_task`).

    A finished task is kept for :data:`FETCH_TASK_RETENTION_S` after it is
    first seen finished; 404 for an unknown or evicted *task_id*.
    """
    task = _fetch_tasks.get(task_id)
    if task is None:
        abort(404, "Unknown task ID")
    _stamp_or_evict_if_stale(task_id, task, time.time())
    return jsonify(task)


@bp.route("/api/upload_gpx", methods=["POST"])
def upload_gpx() -> Response:
    """
    Build and save a map around an uploaded GPX track, synchronously.

    The GPX itself is only kept in a temp directory. The map is saved as
    ``<name>.mapdata`` or the next free name (:func:`_free_mapdata_path`).

    Parameters (multipart form)
    -----------------------------
    file : file
        The GPX track.
    name : str, optional
        Output stem, sanitized to ``[A-Za-z0-9_-]``; default the upload's stem.
    options : str, optional
        JSON object with optional ``grid_margin``, ``obstacle_radius`` and
        ``buffer_widths``, passed through to :class:`~map_data.map_data.MapData`.

    Returns
    -------
    Response
        JSON ``{"filename", "roads", "footways", "barriers", "crossroads"}``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if the file, ``name`` or ``options`` is invalid or the area is
        too large. 503 if Overpass is unreachable. 500 if parsing fails or
        on an unexpected error.

    """
    if "file" not in request.files:
        abort(400, "No file part")
    file = request.files["file"]
    if not file.filename:
        abort(400, "No selected file")

    name = request.form.get("name")
    if not name:
        name = Path(file.filename).stem

    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
    if not name:
        abort(400, "name is empty after sanitizing")

    try:
        parse_opts = json.loads(request.form.get("options", "{}"))
    except ValueError:
        abort(400, "options must be JSON")
    if not isinstance(parse_opts, dict):
        abort(400, "options must be a JSON object")
    grid_margin = parse_opts.get("grid_margin")
    obstacle_radius = parse_opts.get("obstacle_radius")
    buffer_widths = parse_opts.get("buffer_widths")

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        gpx_tmp_path = Path(tmp_dir) / "upload.gpx"
        file.save(gpx_tmp_path)

        try:
            md = MapData(
                str(gpx_tmp_path),
                coords_type="file",
                grid_margin=grid_margin,
                obstacle_radius=obstacle_radius,
                buffer_widths=buffer_widths,
            )
            area_km2 = _bbox_area_km2(md.min_lat, md.min_long, md.max_lat, md.max_long)
            if area_km2 > MAX_FETCH_AREA_KM2:
                abort(
                    400,
                    f"Track's surrounding area is {area_km2:.1f} km², which exceeds the "
                    f"{MAX_FETCH_AREA_KM2:.0f} km² limit for a single fetch.",
                )

            out_path = _free_mapdata_path(data_dir, name, reserve=True)
            # MapData caches the OSM response next to coords_file: keep it beside the map
            md.coords_file = str(out_path.with_suffix(".gpx"))
            try:
                result = _query_parse_save(md, out_path)
            except _FetchFailed as e:
                logger.warning("GPX upload fetch failed at %s: %s", e.stage, e)
                abort(503 if e.stage == "query" else 500, str(e))
            finally:
                _release_mapdata_path(out_path)

            return jsonify(result)
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.exception("Error processing GPX upload")
            abort(500, "Internal server error")


@bp.route("/api/upload_mapdata", methods=["POST"])
def upload_mapdata() -> Response:
    """
    Save an uploaded ``.mapdata`` file to the data directory.

    Saved as ``<stem>.mapdata`` (lower-case, as the file list matches it
    case-sensitively) or the next free name (:func:`_free_mapdata_path`);
    a file that fails to load is deleted again. Returns ``{"filename": ...}``,
    the name saved under; 400 for a missing, misnamed or unloadable file.
    """
    if "file" not in request.files:
        abort(400, "No file part")
    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".mapdata"):
        abort(400, "File must have a .mapdata extension")

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    dest = _free_mapdata_path(data_dir, Path(file.filename).stem)
    file.save(dest)

    try:
        MapData.load(str(dest))
    except Exception as e:
        logger.warning("Rejected uploaded .mapdata %s: %r", dest.name, e)
        dest.unlink(missing_ok=True)
        abort(400, "Invalid .mapdata file")

    return jsonify({"filename": dest.name})


def _download(
    payload: dict[str, Any],
    download_name: str,
    mimetype: str,
    indent: int | None = None,
) -> Response:
    """JSON-serialize *payload* and send it as a downloadable attachment."""
    buf = io.BytesIO(json.dumps(payload, indent=indent).encode("utf-8"))
    return send_file(buf, as_attachment=True, download_name=download_name, mimetype=mimetype)


@bp.route("/api/export")
def export_mapdata() -> Response:
    """
    Download a mapdata file with all its edits applied as ``<stem>.exported.mapdata``.

    Query: ``file``. Nothing is written to the data directory. 400 if
    ``file`` is missing, 404 if the file doesn't exist.
    """
    filename = _require_args("file")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File not found: {filename}")

    base = Path(filename).stem
    return _download(map_data_to_dict(md), f"{base}.exported.mapdata", "application/json", indent=2)


@bp.route("/api/export/geojson")
def export_geojson() -> Response:
    """
    Download a mapdata file with all its edits applied as ``<stem>.geojson``, for GIS tools.

    Query: ``file``. 400 if ``file`` is missing, 404 if the file doesn't exist.
    """
    filename = _require_args("file")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File not found: {filename}")

    base = Path(filename).stem
    return _download(mapdata_to_geojson(md), f"{base}.geojson", "application/geo+json")
