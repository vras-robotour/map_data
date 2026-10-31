"""Path-planning routes: planner defaults, traversability rules, cost grid, (re)planning."""

import base64
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import utm
import yaml
from flask import (
    Response,
    abort,
    current_app,
    jsonify,
    request,
)
from flask.typing import ResponseReturnValue

from map_data.map_data import MapData
from map_data.pathsolver.graph_planner import GraphPlanner
from map_data.pathsolver.replan import ReplanPath, cancel_replan_backend, grid_defaults
from map_data.pathsolver.route import (
    GRAPH_ALGORITHM,
    RoutePlanningError,
    plan_route,
)
from map_data.traversability import (
    DEFAULT_CONFIG_NAME,
    TraversabilityError,
    TraversabilityRules,
    load_traversability,
)
from map_data.utils.config import config_path, load_config
from map_data.utils.parsing import ways_to_shapely
from map_data.utils.serialization import atomic_write_text
from map_data.utils.way import NON_ROUTABLE_HIGHWAY_VALUES

from .common import (
    _MAX_LAT,
    _MAX_LON,
    _MIN_LAT,
    _MIN_LON,
    MAX_CELL_SIZE_M,
    MAX_GRID_CELLS,
    MAX_INFLATE_OBSTACLES_M,
    MIN_CELL_SIZE_M,
    _bbox_area_km2,
    _json_object,
    _validated_bbox,
    _validated_cost_dict,
    _validated_number,
    bp,
    get_merged_mapdata,
)

logger = logging.getLogger(__name__)

#: route_planner parameters (its PLANNER_PARAMETERS) that seed the planner toggles.
NODE_PLANNER_PARAMETERS = ("cell_size", "inflate_obstacles", "simplify_path", "smooth_path")


@lru_cache(maxsize=4)
def _graph_planner(
    md: MapData,
    highway_types: tuple[str, ...],
    traversability: TraversabilityRules,
    highway_costs: tuple[tuple[str, float], ...] | None,
    surface_costs: tuple[tuple[str, float], ...] | None,
) -> GraphPlanner:
    """
    The :class:`GraphPlanner` ``plan_route`` would build for these arguments, built once.

    Keyed on *md*'s identity, which is safe because *md* is the shared,
    read-only map from :func:`get_merged_mapdata` (a new one after every edit)
    and the cached planner keeps it alive. Cost tables come as sorted item
    tuples so they can be part of the key.
    """
    return GraphPlanner(
        md,
        highway_types=list(highway_types),
        traversability=traversability,
        highway_costs=None if highway_costs is None else dict(highway_costs),
        surface_costs=None if surface_costs is None else dict(surface_costs),
    )


@bp.route("/api/planner_defaults")
def get_planner_defaults() -> Response:
    """
    Return the default path-planning parameters as JSON.

    When the viewer runs under ROS and the tracker's ``planner_node`` answers, its
    :data:`NODE_PLANNER_PARAMETERS` replace the file's values. The viewer only reads
    them: the toggles a plan request sends are what it plans with.

    Returns
    -------
    Response
        JSON dict of planner defaults from ``config/planner_defaults.yaml``, plus
        ``source`` (``"route_planner"`` or ``"yaml"``) and, for ``"route_planner"``,
        ``node`` (the node's name).

    """
    from ..tracker_routes import TRACKER_EXTENSION  # tracker_routes imports this package

    cfg = load_config("planner_defaults.yaml")
    node = current_app.extensions.get(TRACKER_EXTENSION)
    if node is not None and (params := node.planner_parameters(NODE_PLANNER_PARAMETERS)):
        return jsonify({**cfg, **params, "source": "route_planner", "node": node.planner_node})
    return jsonify({**cfg, "source": "yaml"})


def _traversability_path() -> Path:
    """The rule file the viewer's planner reads when a request brings no rules of its own."""
    return config_path(DEFAULT_CONFIG_NAME)


def _request_rules(body: dict[str, Any]) -> TraversabilityRules:
    """
    Rules from the body's ``traversability`` YAML text, or from the rule file when absent.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if the text (or the file) is not a valid rule set.

    """
    text = body.get("traversability")
    try:
        if text is None:
            return load_traversability(_traversability_path())
        if not isinstance(text, str):
            abort(400, "traversability must be YAML text")
        return TraversabilityRules.from_dict(yaml.safe_load(text), source="viewer")
    except (yaml.YAMLError, TraversabilityError) as e:
        abort(400, str(e))


@bp.route("/api/traversability")
def get_traversability() -> Response:
    """Return the rule file as ``{"yaml": text, "path": str}`` (empty text if it is missing)."""
    path = _traversability_path()
    return jsonify({"yaml": path.read_text() if path.is_file() else "", "path": str(path)})


@bp.route("/api/traversability", methods=["PUT"])
def save_traversability() -> Response:
    """
    Validate the body's ``traversability`` YAML text and write it to the rule file.

    The text is written verbatim, so comments survive. route_planner picks it
    up on its next goal; osm_cloud reads it only at startup and needs a restart.
    """
    body = _json_object()
    if body.get("traversability") is None:
        abort(400, "Missing traversability")
    _request_rules(body)
    path = _traversability_path()
    atomic_write_text(path, body["traversability"])
    return jsonify({"path": str(path)})


@bp.route("/api/traversability/blocked", methods=["POST"])
def get_traversability_blocked() -> Response:
    """
    List the ways of a file the *Paths only* planner will not route over.

    JSON body: ``file``, optional ``traversability`` YAML text (default: the
    rule file) and ``allowed_ways`` (default ``["footway"]``). A way is
    blocked when its category is not allowed or the rules (plus the always
    excluded stairs) refuse it. Returns ``{"blocked": [[way_id, reason], ...]}``.
    """
    body = _json_object()
    filename = body.get("file")
    if not isinstance(filename, str) or not filename:
        abort(400, "Missing file")
    allowed = body.get("allowed_ways", ["footway"])
    if not isinstance(allowed, list):
        abort(400, "allowed_ways must be a list of strings")
    rules = _request_rules(body).extend(NON_ROUTABLE_HIGHWAY_VALUES)
    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File {filename} not found")

    blocked = []
    for cat, ways in (("footway", md.footways_list), ("road", md.roads_list)):
        for way in ways:
            if cat not in allowed:
                blocked.append([way.id, f"{cat}s not enabled"])
                continue
            verdict = rules.evaluate(way.tags)
            if not verdict.traversable:
                blocked.append([way.id, verdict.reason])
    return jsonify({"blocked": blocked})


def _cost_grid_image(
    costs: np.ndarray,
    low: tuple[float, float],
    cell_size: float,
    off_path_cost: float,
    path_cost_cap: float,
    zone_number: int,
    zone_letter: str,
) -> tuple[np.ndarray, list[list[float]]]:
    """
    Resample a UTM cost grid onto a lat/lon-aligned RGBA raster for ``L.imageOverlay``.

    The grid is axis-aligned in UTM, which is rotated against lat/lon by the
    meridian convergence, so each pixel centre is projected to UTM and takes
    the colour of the grid cell it falls in; pixels outside the grid stay
    transparent. Black for obstacles (cost >= 1), translucent grey for
    off-path cells (cost >= ``off_path_cost``), green-to-red for paths
    (``cost / path_cost_cap``).

    Parameters
    ----------
    costs : np.ndarray
        ``(ny, nx)`` cell costs, row 0 at the grid's low edge, ``inf`` for obstacles
        (``PathGrid.grid_2d_cache``).
    low : tuple of float
        UTM ``(x, y)`` corner the grid was built from.
    off_path_cost, path_cost_cap : float
        The planner's ``default_off_path_cost`` and ``path_cost_cap``.

    Returns
    -------
    rgba : np.ndarray
        ``(H, W, 4)`` uint8 image, row 0 at the north edge.
    bounds : list of list of float
        ``[[south, west], [north, east]]`` of the image.

    """
    ny, nx = costs.shape
    off_path_cost = float(np.float32(off_path_cost))  # the grid holds float32 costs
    xs = low[0] + np.array([0, nx, 0, nx]) * cell_size
    ys = low[1] + np.array([0, 0, ny, ny]) * cell_size
    lats, lons = utm.to_latlon(xs, ys, zone_number, zone_letter)
    south, north, west, east = lats.min(), lats.max(), lons.min(), lons.max()

    # ~cell_size pixels. shortcut: rows are linear in latitude but Leaflet stretches
    # linearly in Web Mercator (< 0.2 m off at MAX_GRID_CELLS), reproject if the cap grows.
    m_per_deg = 111_320.0
    height = max(1, int(np.ceil((north - south) * m_per_deg / cell_size)))
    width = max(1, int(np.ceil((east - west) * m_per_deg * np.cos(np.radians(north)) / cell_size)))
    lat_c = north - (np.arange(height) + 0.5) * (north - south) / height
    lon_c = west + (np.arange(width) + 0.5) * (east - west) / width
    lon2d, lat2d = np.meshgrid(lon_c, lat_c)
    px, py, *_ = utm.from_latlon(lat2d.ravel(), lon2d.ravel(), zone_number, zone_letter)
    pix = np.floor((px - low[0]) / cell_size).astype(int)
    piy = np.floor((py - low[1]) / cell_size).astype(int)
    inside = (pix >= 0) & (pix < nx) & (piy >= 0) & (piy < ny)
    cost = np.full(pix.shape, np.nan)
    cost[inside] = costs[piy[inside], pix[inside]]

    rgba = np.zeros((cost.size, 4), dtype=np.uint8)
    with np.errstate(invalid="ignore"):  # NaN (outside / no cell) stays transparent
        obstacle = cost >= 1.0
        off_path = (cost >= off_path_cost) & ~obstacle
        on_path = (cost < off_path_cost) & ~obstacle
    rgba[obstacle] = (0, 0, 0, 255)
    rgba[off_path] = (0x44, 0x44, 0x44, 102)  # fillOpacity 0.4
    norm = cost[on_path] / path_cost_cap
    rgba[on_path, 0] = np.floor(255 * np.minimum(1, norm))
    rgba[on_path, 1] = np.floor(255 * np.maximum(0, 1 - norm))
    rgba[on_path, 3] = 153  # fillOpacity 0.6
    return rgba.reshape(height, width, 4), [[south, west], [north, east]]


@bp.route("/api/cost_grid")
def get_cost_grid() -> Response:
    """
    Render the edited map's 1 m traversal-cost grid over a bounding box as one PNG.

    Obstacles stay in the grid (as ``inf``) so they are drawn too (see
    :func:`_cost_grid_image`).

    Parameters (query string)
    -----------------------------
    file : str
        Mapdata filename.
    min_lat, min_lon, max_lat, max_lon : float
        WGS84 box; min < max on each axis, within
        :data:`~map_data.viewer.routes.common.MAX_GRID_CELLS` 1 m cells.
    highway_costs, surface_costs : str, optional
        JSON-encoded cost-override dicts (str -> finite non-negative number).

    Returns
    -------
    Response
        JSON ``{"image": "data:image/png;base64,...", "bounds": [[s, w], [n, e]]}``
        for ``L.imageOverlay`` (both ``null`` for an empty grid).

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if a parameter is missing or invalid or the box is too large.
        404 if the file doesn't exist.

    """
    filename = request.args.get("file")
    min_lat_arg = request.args.get("min_lat", type=float)
    min_lon_arg = request.args.get("min_lon", type=float)
    max_lat_arg = request.args.get("max_lat", type=float)
    max_lon_arg = request.args.get("max_lon", type=float)

    if filename is None or any(
        v is None for v in (min_lat_arg, min_lon_arg, max_lat_arg, max_lon_arg)
    ):
        abort(400, "Missing required parameters")

    min_lat, min_lon, max_lat, max_lon = _validated_bbox(
        min_lat_arg,
        min_lon_arg,
        max_lat_arg,
        max_lon_arg,
    )

    cell_size = 1.0  # coarser than planning, for rendering speed
    cells = _bbox_area_km2(min_lat, min_lon, max_lat, max_lon) * 1e6 / (cell_size * cell_size)
    if cells > MAX_GRID_CELLS:
        abort(
            400,
            f"Requested area needs ~{cells / 1e6:.1f} million grid cells at "
            f"cell_size={cell_size} m, exceeding the {MAX_GRID_CELLS / 1e6:.0f} million "
            "cell limit. Request a smaller area or a larger cell size.",
        )

    cost_dicts: dict[str, dict[str, float] | None] = {}
    for name in ("highway_costs", "surface_costs"):
        raw = request.args.get(name)
        cost_dicts[name] = None
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                abort(400, f"{name} must be valid JSON")
            cost_dicts[name] = _validated_cost_dict(parsed, name)
    highway_costs_dict = cost_dicts["highway_costs"]
    surface_costs_dict = cost_dicts["surface_costs"]

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File {filename} not found")

    zn, zl = md.zone_number, md.zone_letter
    p1 = utm.from_latlon(min_lat, min_lon, zn, zl)
    p2 = utm.from_latlon(max_lat, max_lon, zn, zl)

    low = (min(p1[0], p2[0]), min(p1[1], p2[1]))
    high = (max(p1[0], p2[0]), max(p1[1], p2[1]))

    obstacles = ways_to_shapely(md.barriers_list)
    replanner = ReplanPath(
        obstacles,
        highway_costs=highway_costs_dict,
        surface_costs=surface_costs_dict,
        low=low,
        high=high,
        cell_size=cell_size,
        inflate_obstacles=0.0,
    )

    replanner.fill_grid(md, highway_types=["footway", "road"])

    # Obstacle cells are inf, not filtered out as in planning, so they can be visualized too.
    path_grid = replanner.path_grid
    assert path_grid.grid_2d_cache is not None  # fill_grid() ran above
    costs = np.asarray(path_grid.grid_2d_cache, dtype=float)
    if costs.size == 0:
        return jsonify({"image": None, "bounds": None})
    rgba, bounds = _cost_grid_image(
        costs,
        low,
        cell_size,
        path_grid.default_off_path_cost,
        path_grid.path_cost_cap,
        zn,
        zl,
    )
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        abort(500, "Failed to encode the cost grid image")
    image = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")
    return jsonify({"image": image, "bounds": bounds})


@bp.route("/api/cancel_replan", methods=["POST"])
def cancel_replan_route() -> ResponseReturnValue:
    """
    Cancel the grid replan started by :func:`create_replan` with the body's ``transfer_id``.

    Returns ``{"success": true}`` even for an unknown or finished ID; 400 if
    the body is not JSON or lacks ``transfer_id``.
    """
    transfer_id = _json_object().get("transfer_id")
    if not transfer_id:
        abort(400, "No transfer_id provided")
    cancel_replan_backend(transfer_id)
    return jsonify({"success": True})


@bp.route("/api/create_replan", methods=["POST"])
def create_replan() -> Response:
    """
    Replan a path on the edited map with :func:`~map_data.pathsolver.route.plan_route`.

    The graph planner is cached per map and parameters (:func:`_graph_planner`).

    Parameters (JSON body)
    -----------------------
    points : list of [lat, lon]
        Path to replan. Required.
    file : str
        Mapdata filename. Required.
    allowed_ways : list of str, default ``["footway"]``
        Highway types the planner may route over.
    transfer_id : str, optional
        Makes a grid replan cancellable via :func:`cancel_replan_route`.
    algorithm : str, default ``"rrt"``
        ``"graph"`` routes along the ways; ``"astar"``/``"rrt"`` run the grid planner.
    sub_algorithm : str, default ``"astar"``
        Grid search algorithm.
    highway_costs, surface_costs : dict, optional
        Per-tag cost overrides, for both planners.
    cell_size, inflate_obstacles, simplify_path, smooth_path : optional
        Grid planner settings, defaulting to
        :func:`~map_data.pathsolver.replan.grid_defaults`; the two sizes are
        bounded by the ``common`` ``*_M`` limits.
    grid_cost_weight : float, optional
        Weight of grid traversal cost vs. path length (grid planner only).
    traversability : str, optional
        Rule YAML text (default: the rule file).

    Returns
    -------
    Response
        JSON ``{"retrieveNum": 0 | -1, "newPath": [[lat, lon], ...]}``: ``0``
        if the path changed (:attr:`~map_data.pathsolver.route.RouteResult.changed`),
        ``-1`` if not. On a planning failure ``{"retrieveNum": 1, "newPath": null,
        "status": "cancelled" | "failed", "reason": ...}``.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``points`` or ``file`` is missing, a parameter is invalid, the
        planning area exceeds :data:`~map_data.viewer.routes.common.MAX_GRID_CELLS`,
        or there are too few points. 404 if the file doesn't exist.

    """
    body = _json_object()
    path_data = body.get("points")  # [[lat, lon], ...]
    filename = body.get("file")
    highway_types = body.get("allowed_ways", ["footway"])
    transfer_id = body.get("transfer_id")
    algorithm = body.get("algorithm", "rrt")
    sub_algorithm = body.get("sub_algorithm", "astar")

    if not path_data or not filename:
        abort(400, "Missing points or file parameter")
    if not isinstance(filename, str):
        abort(400, "file must be a string")
    if not isinstance(path_data, list):
        abort(400, "points must be a list of [lat, lon] pairs")
    if not isinstance(highway_types, list) or not all(isinstance(h, str) for h in highway_types):
        abort(400, "allowed_ways must be a list of strings")
    if transfer_id is not None and not isinstance(transfer_id, str):
        abort(400, "transfer_id must be a string")
    if not isinstance(algorithm, str):
        abort(400, "algorithm must be a string")
    if not isinstance(sub_algorithm, str):
        abort(400, "sub_algorithm must be a string")

    highway_costs = _validated_cost_dict(body.get("highway_costs"), "highway_costs")
    surface_costs = _validated_cost_dict(body.get("surface_costs"), "surface_costs")

    defaults = grid_defaults()
    cell_size = _validated_number(
        body.get("cell_size", defaults["cell_size"]),
        "cell_size",
        MIN_CELL_SIZE_M,
        MAX_CELL_SIZE_M,
    )
    inflate_obstacles = _validated_number(
        body.get("inflate_obstacles", defaults["inflate_obstacles"]),
        "inflate_obstacles",
        0.0,
        MAX_INFLATE_OBSTACLES_M,
    )
    simplify_path = body.get("simplify_path", defaults["simplify_path"])
    smooth_path = body.get("smooth_path", defaults["smooth_path"])
    if not isinstance(simplify_path, bool) or not isinstance(smooth_path, bool):
        abort(400, "simplify_path and smooth_path must be booleans")
    grid_cost_weight = body.get("grid_cost_weight")
    if grid_cost_weight is not None:
        grid_cost_weight = _validated_number(grid_cost_weight, "grid_cost_weight", 0.0, 1000.0)
    traversability = _request_rules(body)

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File {filename} not found")

    points: list[tuple[float, float]] = []
    for i, p in enumerate(path_data):
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            abort(400, "points must be a list of [lat, lon] pairs")
        lat = _validated_number(p[0], f"points[{i}] latitude", _MIN_LAT, _MAX_LAT)
        lon = _validated_number(p[1], f"points[{i}] longitude", _MIN_LON, _MAX_LON)
        points.append((lat, lon))

    planner = None
    if algorithm == GRAPH_ALGORITHM:
        planner = _graph_planner(
            md,
            tuple(highway_types or ["footway"]),
            traversability,
            None if highway_costs is None else tuple(sorted(highway_costs.items())),
            None if surface_costs is None else tuple(sorted(surface_costs.items())),
        )

    try:
        result = plan_route(
            md,
            points,
            algorithm=algorithm,
            sub_algorithm=sub_algorithm,
            highway_types=highway_types,
            cell_size=cell_size,
            inflate_obstacles=inflate_obstacles,
            simplify_path=simplify_path,
            smooth_path=smooth_path,
            grid_cost_weight=grid_cost_weight,
            highway_costs=highway_costs,
            surface_costs=surface_costs,
            traversability=traversability,
            transfer_id=transfer_id,
            max_grid_cells=MAX_GRID_CELLS,
            planner=planner,
        )
    except RoutePlanningError as e:
        if e.reason in ("grid_too_large", "too_few_points", "bad_parameter"):
            abort(400, e.message)
        status = "cancelled" if e.reason == "cancelled" else "failed"
        logger.warning("Replan failed (%s): %s", e.reason, e.message)
        return jsonify({"retrieveNum": 1, "newPath": None, "status": status, "reason": e.reason})

    new_path = [[lat, lon] for lat, lon in result.latlon]
    return jsonify({"retrieveNum": 0 if result.changed else -1, "newPath": new_path})
