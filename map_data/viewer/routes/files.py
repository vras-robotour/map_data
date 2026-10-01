"""File routes: viewer page, file listing, mapdata GeoJSON, OSM fetch, uploads, export."""

import copy
import io
import json
import logging
import os
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

from map_data.annotations import apply_way_edits
from map_data.map_data import MapData
from map_data.utils.serialization import map_data_to_dict

from ..cache import load_mapdata_cached, mapdata_geojson_cached
from ..helpers import (
    load_annotations,
    mapdata_to_geojson,
)
from .common import (
    MAX_FETCH_AREA_KM2,
    _annotation_path,
    _apply_tag_override,
    _bbox_area_km2,
    _get_data_dir,
    _mapdata_path,
    _require_args,
    _validated_bbox,
    bp,
    get_merged_mapdata,
)

logger = logging.getLogger(__name__)


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

    Loads the cached ``MapData``, takes a shallow ``copy.copy`` (see the
    :mod:`~map_data.viewer.routes.common` docstring's "MapData copy semantics" section -- this is
    safe because :func:`apply_way_edits` never mutates a ``Way`` object shared
    with the cache), applies deletions/splits/moves via
    :func:`apply_way_edits`, converts to GeoJSON, and finally merges in
    any tag overrides (re-deriving each affected feature's ``road``/
    ``footway`` category from the merged ``highway`` tag).

    Split out of :func:`get_mapdata` so that all of it -- the copy, the
    re-applied edits and the rebuild -- sits behind the one call to
    :func:`~map_data.viewer.cache.mapdata_geojson_cached`.
    """
    map_data = copy.copy(load_mapdata_cached(str(path)))

    apply_way_edits(map_data, store)

    geojson = mapdata_to_geojson(map_data)
    tag_overrides = store.get("tag_overrides", {})
    if tag_overrides:
        for f in geojson["features"]:
            ov = tag_overrides.get(str(f["properties"].get("id", "")))
            if ov:
                f["properties"]["tags"], f["properties"]["category"] = _apply_tag_override(
                    f["properties"].get("tags") or {},
                    ov,
                    f["properties"].get("category"),
                )
    return geojson


@bp.route("/api/mapdata")
def get_mapdata() -> ResponseReturnValue:
    """
    Return a mapdata file, with the user's recorded edits applied, as GeoJSON.

    The document is built by :func:`_merged_mapdata_geojson` and cached,
    already serialized, by :func:`~map_data.viewer.cache.mapdata_geojson_cached`
    under the map file's signature and the annotation store's digest -- so
    panning around an unchanged map costs a stat and a store read, not
    another ``parse_intersections`` plus FeatureCollection rebuild. The
    same key doubles as the ``ETag``: a client that sends it back in
    ``If-None-Match`` gets a 304 and no body at all.

    Returns
    -------
    Response
        A GeoJSON ``FeatureCollection`` (see
        :func:`~map_data.viewer.helpers.mapdata_to_geojson`), or 304 if the
        client's copy is still current.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``file`` is missing. 404 if the file doesn't exist.

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


_fetch_tasks: dict[str, dict[str, Any]] = {}

#: How long a terminal ("done"/"failed") fetch task is retained, in seconds,
#: before it is evicted -- either by a poll in :func:`fetch_area_status` or by
#: the sweep in :func:`fetch_area`.
FETCH_TASK_RETENTION_S = 60.0


def _stamp_or_evict_if_stale(task_id: str, task: dict[str, Any], now: float) -> None:
    """
    Stamp a terminal task's retention clock, or evict it once expired.

    A non-terminal *task* is never touched. A terminal one without a
    ``completed_at`` yet gets stamped with *now*, starting its retention
    clock; once more than :data:`FETCH_TASK_RETENTION_S` has passed since
    then, it's popped from ``_fetch_tasks`` -- though *task* itself (the
    caller's reference) is left intact, so a caller that already holds it
    can still return its final state this one last time.
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

    Shared by :func:`_run_fetch_task` (async, progress-reporting) and
    :func:`upload_gpx` (synchronous, no progress reporting).

    Parameters
    ----------
    progress_cb : callable, optional
        Passed through to ``md.run_queries``.
    on_parse_start : callable, optional
        Called (with no arguments) once querying succeeds, before parsing
        starts.

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
    Writes *out_path* on success. Updates ``_fetch_tasks[task_id]`` as work
    progresses: ``"querying"`` (plus a ``"detail"`` string naming the
    Overpass mirror/attempt in flight) while the Overpass request is out,
    then ``"parsing"``, then a terminal ``"failed"`` (plus an ``"error"``
    message) or ``"done"`` (plus a ``"result"`` summary).

    """
    try:
        md = MapData(
            (waypoints, zone_number, zone_letter),
            coords_type="array",
            grid_margin=grid_margin,
            obstacle_radius=obstacle_radius,
            buffer_widths=buffer_widths,
        )

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
        400 if a required field is missing, a corner isn't a finite in-range number,
        the box is degenerate/inverted, or ``name`` sanitizes to empty.

    """
    body = request.get_json(force=True) or {}
    for field in ("min_lat", "min_lon", "max_lat", "max_lon", "name"):
        if field not in body:
            abort(400, f"Missing field: {field}")

    min_lat, min_lon, max_lat, max_lon = _validated_bbox(
        body["min_lat"], body["min_lon"], body["max_lat"], body["max_lon"]
    )

    area_km2 = _bbox_area_km2(min_lat, min_lon, max_lat, max_lon)
    if area_km2 > MAX_FETCH_AREA_KM2:
        abort(
            400,
            f"Requested area is {area_km2:.1f} km², which exceeds the "
            f"{MAX_FETCH_AREA_KM2:.0f} km² limit for a single fetch. Draw a smaller area.",
        )

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
            [min_lat, min_lon],
            [min_lat, max_lon],
            [max_lat, min_lon],
            [max_lat, max_lon],
        ],
    )
    easting, northing, zone_number, zone_letter = utm.from_latlon(corners[:, 0], corners[:, 1])
    waypoints = np.column_stack([easting, northing])

    # Evict stale terminal tasks here too, so ones whose client never polled them
    # to completion (what normally evicts them, in fetch_area_status) can't pile up.
    now = time.time()
    for stale_id, task in list(_fetch_tasks.items()):
        _stamp_or_evict_if_stale(stale_id, task, now)
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
    kept around for :data:`FETCH_TASK_RETENTION_S` seconds after first
    being observed as terminal (to give the client a chance to see the
    final status), then evicted from ``_fetch_tasks`` on the next poll
    (or by the sweep :func:`fetch_area` runs on each new fetch).

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
    _stamp_or_evict_if_stale(task_id, task, time.time())
    return jsonify(task)


@bp.route("/api/upload_gpx", methods=["POST"])
def upload_gpx() -> Response:
    """
    Upload a GPX track, fetch its surrounding OSM data, parse, and save it.

    Synchronous (unlike :func:`fetch_area`'s background-thread version):
    the GPX is saved to a temp directory (never persisted in the data
    directory), a :class:`~map_data.map_data.MapData` is built from it,
    Overpass is queried and parsed, and the result is saved as
    ``<name>.mapdata``. The temp directory is always removed afterward.

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
    if not file.filename:
        abort(400, "No selected file")

    name = request.form.get("name")
    if not name:
        name = Path(file.filename).stem

    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
    if not name:
        abort(400, "name is empty after sanitizing")

    parse_opts = json.loads(request.form.get("options", "{}"))
    grid_margin = parse_opts.get("grid_margin")
    obstacle_radius = parse_opts.get("obstacle_radius")
    buffer_widths = parse_opts.get("buffer_widths")

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary directory to avoid saving the GPX to the data directory
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
            # Restore the original filename for metadata purposes
            md.coords_file = file.filename

            area_km2 = _bbox_area_km2(md.min_lat, md.min_long, md.max_lat, md.max_long)
            if area_km2 > MAX_FETCH_AREA_KM2:
                abort(
                    400,
                    f"Track's surrounding area is {area_km2:.1f} km², which exceeds the "
                    f"{MAX_FETCH_AREA_KM2:.0f} km² limit for a single fetch.",
                )

            out_path = data_dir / f"{name}.mapdata"
            try:
                result = _query_parse_save(md, out_path)
            except _FetchFailed as e:
                logger.warning("GPX upload fetch failed at %s: %s", e.stage, e)
                abort(503 if e.stage == "query" else 500, str(e))

            return jsonify(result)
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            logger.exception("Error processing GPX upload")
            abort(500, "Internal server error")


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
    filename = _require_args("file")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File not found: {filename}")

    base = Path(filename).stem
    return _download(map_data_to_dict(md), f"{base}.exported.mapdata", "application/json", indent=2)


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
    filename = _require_args("file")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File not found: {filename}")

    base = Path(filename).stem
    return _download(mapdata_to_geojson(md), f"{base}.geojson", "application/geo+json")
