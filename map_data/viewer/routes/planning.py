"""Path-planning routes: planner defaults, traversability rules, cost grid, (re)planning."""

import copy
import json
import logging
from pathlib import Path
from typing import Any

import utm
import yaml
from flask import (
    Response,
    abort,
    jsonify,
    request,
)
from flask.typing import ResponseReturnValue

from map_data.pathsolver.replan import DEFAULT_ARGS, ReplanPath, cancel_replan_backend
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
    _check_grid_cells,
    _validated_bbox,
    _validated_cost_dict,
    _validated_number,
    bp,
    get_merged_mapdata,
)

logger = logging.getLogger(__name__)


@bp.route("/api/planner_defaults")
def get_planner_defaults() -> Response:
    """
    Return the default path-planning parameters as JSON.

    Returns
    -------
    Response
        JSON dict of planner defaults from ``config/planner_defaults.yaml``.

    """
    return jsonify(load_config("planner_defaults.yaml"))


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
    body = request.get_json(force=True) or {}
    if body.get("traversability") is None:
        abort(400, "Missing traversability")
    _request_rules(body)
    path = _traversability_path()
    path.write_text(body["traversability"])
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
    body = request.get_json(force=True) or {}
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


@bp.route("/api/cost_grid")
def get_cost_grid() -> Response:
    """
    Compute a coarse traversal-cost grid for a bounding box, for visualization.

    Builds a fully-edited ``MapData`` via :func:`~map_data.viewer.routes.common.get_merged_mapdata`,
    fills a 1m-resolution :class:`~map_data.pathsolver.replan.ReplanPath`
    grid over the requested lat/lon box (obstacles are *not* filtered out
    of the result, unlike planning, so the client can render them), and
    returns every cell as a flat ``[lat, lon, cost]`` point.

    Parameters (query string)
    -----------------------------
    file : str
        Mapdata filename.
    min_lat, min_lon, max_lat, max_lon : float
        Bounding box (WGS84 degrees); min must be strictly less than max
        on each axis, and the box must fit within the
        :data:`~map_data.viewer.routes.common.MAX_GRID_CELLS` cell budget at the fixed 1 m
        resolution.
    highway_costs, surface_costs : str, optional
        JSON-encoded cost-override dicts (str -> finite non-negative
        number); a malformed value aborts the request with 400.

    Returns
    -------
    Response
        JSON list of ``[lat, lon, cost]`` triples, one per grid cell.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if any required parameter is missing, the bounding box is
        malformed (non-finite, out of lat/lon range, or min >= max),
        the box exceeds the grid-cell budget, or a cost-override dict is
        malformed. 404 if the file doesn't exist.

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

    cell_size = 1.0  # Use a coarser grid for visualization performance
    _check_grid_cells(_bbox_area_km2(min_lat, min_lon, max_lat, max_lon) * 1e6, cell_size)

    # Get custom highway/surface costs from the request, if provided
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

    args = copy.copy(DEFAULT_ARGS)
    args.low = low
    args.high = high
    args.cell_size = cell_size
    args.inflate_obstacles = 0.0

    obstacles = ways_to_shapely(md.barriers_list)
    replanner = ReplanPath(
        args, obstacles, highway_costs=highway_costs_dict, surface_costs=surface_costs_dict
    )

    replanner.fill_grid(md, highway_types=["footway", "road"])

    # grid is [N, 4] -> [x, y, 0, cost]; obstacles (cost >= 1.0) are not filtered
    # out, unlike planning, so they can be visualized too.
    points = []
    for row in replanner.grid:
        lat, lon = utm.to_latlon(row[0], row[1], zn, zl)
        points.append([lat, lon, float(row[3])])

    return jsonify(points)


@bp.route("/api/cancel_replan", methods=["POST"])
def cancel_replan_route() -> ResponseReturnValue:
    """
    Cancel an in-progress replan started by :func:`create_replan`.

    Parameters (JSON body)
    -----------------------
    transfer_id : str
        ID of the replan to cancel, as passed to
        :func:`~map_data.pathsolver.replan.ReplanPath`.

    Returns
    -------
    ResponseReturnValue
        JSON ``{"success": true}`` for any provided ``transfer_id`` (an
        unknown or already finished one is not reported as an error by
        :func:`~map_data.pathsolver.replan.cancel_replan_backend`);
        ``{"success": false, "message": ...}`` with status 400 if the
        request body is not JSON or lacks ``transfer_id``.

    """
    transfer_id = (request.get_json(silent=True) or {}).get("transfer_id")
    if not transfer_id:
        return jsonify({"success": False, "message": "No transfer_id provided"}), 400
    cancel_replan_backend(transfer_id)
    return jsonify({"success": True})


@bp.route("/api/create_replan", methods=["POST"])
def create_replan() -> Response:
    """
    Replan a path against the (edited) map data, using a grid or graph planner.

    Builds a fully-edited ``MapData`` via :func:`~map_data.viewer.routes.common.get_merged_mapdata`,
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
        Grid cell size in meters (grid planner only); must be within
        [:data:`~map_data.viewer.routes.common.MIN_CELL_SIZE_M`,
        :data:`~map_data.viewer.routes.common.MAX_CELL_SIZE_M`].
    inflate_obstacles : float, default 0.25
        Obstacle inflation radius in meters (grid planner only); must be
        within [0, :data:`~map_data.viewer.routes.common.MAX_INFLATE_OBSTACLES_M`].
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
        from the input path (by point count or by more than a small
        per-point tolerance, per :attr:`~map_data.pathsolver.route.RouteResult.changed`),
        or ``-1`` if it's effectively unchanged.

    Raises
    ------
    werkzeug.exceptions.HTTPException
        400 if ``points`` or ``file`` is missing, or any parameter fails
        validation: non-numeric/out-of-range points, ``cell_size``
        outside [:data:`~map_data.viewer.routes.common.MIN_CELL_SIZE_M`,
        :data:`~map_data.viewer.routes.common.MAX_CELL_SIZE_M`], ``inflate_obstacles`` outside [0,
        :data:`~map_data.viewer.routes.common.MAX_INFLATE_OBSTACLES_M`], malformed cost dicts,
        wrongly-typed flags/strings, or a planning area exceeding the
        :data:`~map_data.viewer.routes.common.MAX_GRID_CELLS` cell budget. 404 if the file doesn't
        exist.

    """
    body = request.get_json(force=True) or {}
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

    cell_size = _validated_number(
        body.get("cell_size", 0.25),
        "cell_size",
        MIN_CELL_SIZE_M,
        MAX_CELL_SIZE_M,
    )
    inflate_obstacles = _validated_number(
        body.get("inflate_obstacles", 0.25),
        "inflate_obstacles",
        0.0,
        MAX_INFLATE_OBSTACLES_M,
    )
    simplify_path = body.get("simplify_path", True)
    smooth_path = body.get("smooth_path", False)
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
        )
    except RoutePlanningError as e:
        if e.reason == "grid_too_large":
            abort(400, e.message)
        if e.reason == "too_few_points":
            abort(400, e.message)
        # The grid planner cannot tell "no path" from "cancelled"; keep the
        # historical status strings the frontend expects.
        status = "cancelled" if algorithm != GRAPH_ALGORITHM else "failed"
        logger.warning("Replan failed (%s): %s", e.reason, e.message)
        return jsonify({"retrieveNum": 1, "newPath": None, "status": status, "reason": e.reason})

    new_path = [[lat, lon] for lat, lon in result.latlon]
    changed = result.changed

    if changed:
        return jsonify({"retrieveNum": 0, "newPath": new_path})
    return jsonify({"retrieveNum": -1, "newPath": new_path})
