"""
Central data class for OSM-based map data.

This module provides the MapData class which orchestrates downloading,
caching, and parsing OpenStreetMap data for use in path planning.
"""

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import overpy
import utm
from gpxpy import parse as gpxparse
from shapely import geometry

from map_data.traversability import TraversabilityRules
from map_data.utils.config import load_config
from map_data.utils.overpass import REQUEST_TIMEOUT, OverpassClient
from map_data.utils.parsing import (
    parse_osm_nodes,
    parse_osm_rels,
    parse_osm_ways,
    separate_ways,
)
from map_data.utils.serialization import load_mapdata, save_mapdata
from map_data.utils.way import Way

logger = logging.getLogger(__name__)


_DEFAULTS = load_config("planner_defaults.yaml")
GRID_MARGIN: float = _DEFAULTS.get("grid_margin", 150)
BBOX_LEN = 4


class CoordsData:
    """Bounding box and coordinate metadata for map data."""

    def __init__(self, min_long: float, max_long: float, min_lat: float, max_lat: float) -> None:
        self.min_long = min_long
        self.max_long = max_long
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.x_margin, self.y_margin = self._compute_margin()

    def _compute_margin(self) -> tuple[float, float]:
        margin = max(
            (self.max_lat - self.min_lat) * 0.1,
            (self.max_long - self.min_long) * 0.1,
        )
        return margin, margin


class MapData:
    """
    Central data class for OSM-based map data.

    Parses GPS waypoints from a GPX file (or a raw coordinate array),
    downloads the corresponding OSM features via the Overpass API, and
    exposes them as three categorised lists of :class:`~map_data.utils.way.Way`
    objects: ``roads_list``, ``footways_list``, and ``barriers_list``.

    The class can be used entirely without ROS2. A running ROS2 context is
    only required for the ``create_mapdata`` CLI node.

    Attributes
    ----------
    waypoints : np.ndarray
        ``(N, 2)`` array of UTM easting/northing coordinates parsed from the
        input file.
    zone_number : int
        UTM zone number inferred from the waypoints.
    zone_letter : str
        UTM zone letter inferred from the waypoints.
    roads_list : list of Way
        Parsed road ways (vehicle-intended highways).
    footways_list : list of Way
        Parsed footway ways (pedestrian paths).
    barriers_list : list of Way
        Parsed barrier features (walls, buildings, fences, water, …).
    crossroads_list : list of Way
        Footway intersection points detected during parsing.

    """

    def __init__(
        self,
        coords: str | tuple[np.ndarray, int, str],
        coords_type: str = "file",
        current_robot_position: np.ndarray | None = None,
        *,
        flip: bool = False,
        grid_margin: float | None = None,
        obstacle_radius: float | None = None,
        buffer_widths: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize MapData from a GPX file or coordinate array.

        Parameters
        ----------
        coords : str or array-like
            If *coords_type* is ``"file"``, the path to a ``.gpx`` file.
            If *coords_type* is ``"array"``, a tuple
            ``(waypoints, zone_number, zone_letter)`` where *waypoints* is
            an ``(N, 2)`` array of UTM easting/northing coordinates.
        coords_type : str
            ``"file"`` (default) to parse a GPX file, or ``"array"`` to
            supply pre-converted UTM coordinates directly.
        current_robot_position : np.ndarray, optional
            If provided, prepended to the waypoint array so the robot's
            current position is included in the bounding box calculation.
            Accepts a single ``(2,)``/``(3,)`` position or an ``(M, 2)``/
            ``(M, 3)`` array. The column count is reconciled with the
            waypoints: a missing elevation column is padded with ``0``
            (matching the GPX/YAML parsers' default), a superfluous one is
            dropped; any other shape raises :class:`ValueError`.
        flip : bool
            If ``True``, reverse the order of the parsed waypoints.

        """
        if coords_type == "file":
            # coords_type == "file" guarantees coords is a str here, but coords_type
            # isn't a Literal type so mypy can't correlate the two params.
            with Path(coords).open() as f:  # type: ignore[arg-type]
                gpx_object = gpxparse(f)
            self.coords_file: str | None = coords  # type: ignore[assignment] # see arg-type note above

            points = []
            if gpx_object.waypoints:
                points = [[p.latitude, p.longitude] for p in gpx_object.waypoints]
            elif gpx_object.tracks:
                for track in gpx_object.tracks:
                    for segment in track.segments:
                        points.extend([[p.latitude, p.longitude] for p in segment.points])
            elif gpx_object.routes:
                for route in gpx_object.routes:
                    points.extend([[p.latitude, p.longitude] for p in route.points])

            if not points:
                msg = f"No points (waypoints, tracks or routes) found in {coords}"
                raise ValueError(msg)

            latlon = np.array(points)
            self.waypoints, self.zone_number, self.zone_letter = self._latlon_to_utm(latlon)
        elif coords_type == "array":
            # coords_type == "array" guarantees coords is the (ndarray, int, str)
            # tuple here, but mypy can't correlate the two params (see above).
            self.waypoints = np.array(coords[0])
            self.zone_number = coords[1]  # type: ignore[assignment]
            self.zone_letter = coords[2]
            self.coords_file = None
        else:
            msg = f"Unknown coords_type: {coords_type!r}"
            raise ValueError(msg)

        if flip:
            self.waypoints = np.flip(self.waypoints, 0)

        if current_robot_position is not None:
            position = np.atleast_2d(np.asarray(current_robot_position, dtype=float))
            n_cols = self.waypoints.shape[1]
            if position.shape[1] == n_cols:
                pass
            elif position.shape[1] == 2 and n_cols == 3:
                # Waypoints carry an elevation column (GPX/YAML paths default
                # missing elevations to 0) — pad the position the same way.
                position = np.column_stack([position, np.zeros(len(position))])
            elif position.shape[1] == 3 and n_cols == 2:
                position = position[:, :2]
            else:
                msg = (
                    f"current_robot_position has {position.shape[1]} column(s); "
                    f"expected 2 (easting, northing) or 3 (easting, northing, "
                    f"elevation) to match waypoints with {n_cols} column(s)"
                )
                raise ValueError(msg)
            self.waypoints = np.concatenate([position, self.waypoints])

        _margin = grid_margin if grid_margin is not None else GRID_MARGIN
        self.max_x = float(np.max(self.waypoints[:, 0]) + _margin)
        self.min_x = float(np.min(self.waypoints[:, 0]) - _margin)
        self.max_y = float(np.max(self.waypoints[:, 1]) + _margin)
        self.min_y = float(np.min(self.waypoints[:, 1]) - _margin)
        self._obstacle_radius = obstacle_radius
        self._buffer_widths = buffer_widths

        self.max_lat, self.max_long = utm.to_latlon(
            self.max_x,
            self.max_y,
            self.zone_number,
            self.zone_letter,
        )
        self.min_lat, self.min_long = utm.to_latlon(
            self.min_x,
            self.min_y,
            self.zone_number,
            self.zone_letter,
        )

        self.coords_data = CoordsData(self.min_long, self.max_long, self.min_lat, self.max_lat)
        self._check_utm_zone_boundary()
        self.points = [
            geometry.Point(x, y)
            for x, y in zip(self.waypoints[:, 0], self.waypoints[:, 1], strict=True)
        ]

        self.nodes_cache: dict[int, dict[str, Any]] = {}
        self.roads_list: list[Way] = []
        self.footways_list: list[Way] = []
        self.barriers_list: list[Way] = []
        self.crossroads_list: list[Way] = []
        #: ``{reason: count}`` of the last :meth:`apply_traversability` call.
        self.traversability_removed: dict[str, int] = {}

        # Raw data stored temporarily during parsing
        self.osm_ways_data: overpy.Result | None = None
        self.osm_rels_data: overpy.Result | None = None
        self.osm_nodes_data: overpy.Result | None = None

        self._load_tag_configs()

    def _check_utm_zone_boundary(self) -> None:
        corners = [
            (self.min_lat, self.min_long),
            (self.min_lat, self.max_long),
            (self.max_lat, self.min_long),
            (self.max_lat, self.max_long),
        ]
        zones = set()
        for lat, lon in corners:
            _, _, zn, zl = utm.from_latlon(lat, lon)
            zones.add((zn, zl))
        if len(zones) > 1:
            zone_strs = ", ".join(f"{zn}{zl}" for zn, zl in sorted(zones))
            logger.warning(
                "Waypoints span multiple UTM zones (%s). Geometry near zone boundaries "
                "may be distorted. Consider splitting the area into smaller regions.",
                zone_strs,
            )

    def _load_tag_configs(self) -> None:
        try:
            from ament_index_python.resources import get_resource

            _, package_path = get_resource("packages", "map_data")
            params_path = Path(package_path) / "share" / "map_data" / "parameters"
        except (ImportError, LookupError):
            params_path = (Path(__file__).parent / ".." / "parameters").resolve()

        self.BARRIER_TAGS: dict[str, list[str]] = self._csv_to_dict(
            params_path / "barrier_tags.csv",
        )
        self.NOT_BARRIER_TAGS: dict[str, list[str]] = self._csv_to_dict(
            params_path / "not_barrier_tags.csv",
        )
        self.ANTI_BARRIER_TAGS: dict[str, list[str]] = self._csv_to_dict(
            params_path / "anti_barrier_tags.csv",
        )
        self.OBSTACLE_TAGS: dict[str, list[str]] = self._csv_to_dict(
            params_path / "obstacle_tags.csv",
        )
        self.NOT_OBSTACLE_TAGS: dict[str, list[str]] = self._csv_to_dict(
            params_path / "not_obstacle_tags.csv",
        )

    @staticmethod
    def _csv_to_dict(path: str | Path) -> dict[str, list[str]]:
        # atleast_2d: for a single-row CSV genfromtxt returns a 1-D array,
        # whose "rows" would iterate as individual strings (characters).
        arr = np.atleast_2d(np.genfromtxt(path, dtype=str, delimiter=","))
        result: dict[str, list[str]] = {}
        for row in arr:
            result.setdefault(row[0], []).append(row[1])
        return result

    @staticmethod
    def _latlon_to_utm(latlon: np.ndarray) -> tuple[np.ndarray, int, str]:
        easting, northing, zone_number, zone_letter = utm.from_latlon(latlon[:, 0], latlon[:, 1])
        return np.column_stack([easting, northing]), zone_number, zone_letter

    def _get_osm_cache_path(self) -> Path | None:
        if not self.coords_file:
            return None
        return Path(self.coords_file).with_suffix(".osm_cache.json")

    def _save_osm_cache(self, raw: str) -> None:
        path = self._get_osm_cache_path()
        if not path:
            return
        cache_data = {
            "bbox": [self.min_lat, self.min_long, self.max_lat, self.max_long],
            "raw": raw,
        }
        try:
            # Atomic write: dump to a temp file next to the target, then
            # replace it, so a crash or full disk mid-dump cannot truncate a
            # previously good cache file.
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f)
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            logger.info("Saved OSM response cache to %s", path)
        except (OSError, TypeError) as e:
            logger.warning("Could not save OSM cache: %s", e)

    def _load_osm_cache(self) -> str | None:
        path = self._get_osm_cache_path()
        if not path or not path.exists():
            return None

        try:
            with path.open(encoding="utf-8") as f:
                cache_data = json.load(f)

            # Validate bbox
            stored_bbox = cache_data.get("bbox")
            current_bbox = [self.min_lat, self.min_long, self.max_lat, self.max_long]
            if not stored_bbox or len(stored_bbox) != BBOX_LEN:
                return None

            # 1e-6 degree tolerance (~11cm at equator)
            if not np.allclose(stored_bbox, current_bbox, atol=1e-6):
                logger.info("OSM cache found but bounding box has changed. Re-querying.")
                return None

            logger.info("Using cached OSM responses from %s", path)
            return cache_data["raw"]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.debug("Could not load OSM cache: %s", e)
            return None

    def run_queries(
        self,
        *,
        use_cache: bool = True,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        """
        Download OSM ways, relations, and nodes from the Overpass API.

        Fires a single Overpass query covering the bounding box of the
        loaded waypoints — the union of ways, their nodes, the relations
        referencing them, and all standalone nodes in the box — and stores
        the parsed result internally. Call :meth:`run_parse` afterwards to
        convert it into :class:`~map_data.utils.way.Way` objects.

        Parameters
        ----------
        use_cache : bool
            If ``True`` (default), attempt to load the response from a
            local ``.osm_cache.json`` file before querying the API.
        progress_cb : callable, optional
            Called with a short human-readable status string before each
            Overpass request attempt (e.g. which mirror/attempt is in
            flight), so a caller can surface fetch progress to a user.

        """
        client = OverpassClient()

        if use_cache:
            cached_raw = self._load_osm_cache()
            if cached_raw is not None:
                try:
                    result = client.api.parse_json(cached_raw)
                except (overpy.exception.OverPyException, json.JSONDecodeError):
                    logger.warning("Cached OSM response is invalid. Re-querying.")
                else:
                    self.osm_ways_data = result
                    self.osm_rels_data = result
                    self.osm_nodes_data = result
                    return

        bbox = f"{self.min_lat},{self.min_long},{self.max_lat},{self.max_long}"
        # Server-side timeout stays comfortably below OverpassClient's HTTP
        # timeout so a slow query is reported by Overpass (as a proper
        # error) rather than by the client giving up mid-response.
        ql_timeout = REQUEST_TIMEOUT - 10
        query = (
            f"[out:json][timeout:{ql_timeout}];way({bbox})->.w;(.w; .w >; .w <; node({bbox}););out;"
        )

        def _on_attempt(endpoint: str, attempt: int, retries: int) -> None:
            if progress_cb is not None:
                progress_cb(f"Querying {endpoint} (attempt {attempt}/{retries})")

        raw = client.query_raw(query, on_attempt=_on_attempt)
        if raw is None:
            logger.error("Overpass query failed.")
            return

        # query_raw validates 200 bodies, but stay defensive: a malformed
        # response that slips through must not kill the run with an
        # uncaught overpy/json traceback.
        try:
            result = client.api.parse_json(raw)
        except (overpy.exception.OverPyException, json.JSONDecodeError):
            logger.exception("Overpass returned an unparseable response.")
            return

        self.osm_ways_data = result
        self.osm_rels_data = result
        self.osm_nodes_data = result

        logger.info("OSM query finished.")
        self._save_osm_cache(raw)

    def run_parse(self) -> int:
        """
        Parse the downloaded OSM data into categorised Way lists.

        Populates :attr:`roads_list`, :attr:`footways_list`,
        :attr:`barriers_list`, and :attr:`crossroads_list`. The raw OSM
        response buffers are cleared after parsing to free memory.

        Returns
        -------
        int
            ``0`` on success, ``1`` if OSM data has not been downloaded yet
            (call :meth:`run_queries` first).

        """
        if any(d is None for d in (self.osm_ways_data, self.osm_rels_data, self.osm_nodes_data)):
            logger.error("Missing OSM data. Run run_queries() first.")
            return 1

        logger.info("Parsing OSM data.")
        ways_dict = parse_osm_ways(
            self.osm_ways_data, self.nodes_cache, self.zone_number, self.zone_letter
        )
        parse_osm_rels(self.osm_rels_data, ways_dict)

        way_node_ids = {nid for w in ways_dict.values() for nid in w.nodes}
        node_barriers = parse_osm_nodes(
            self.osm_nodes_data,
            self.nodes_cache,
            way_node_ids,
            self.OBSTACLE_TAGS,
            self.NOT_OBSTACLE_TAGS,
            obstacle_radius=self._obstacle_radius,
            force_zone_number=self.zone_number,
            force_zone_letter=self.zone_letter,
        )

        self.roads_list, self.footways_list, parsed_barriers = separate_ways(
            ways_dict,
            self.BARRIER_TAGS,
            self.NOT_BARRIER_TAGS,
            self.ANTI_BARRIER_TAGS,
            buffer_widths=self._buffer_widths,
        )
        self.barriers_list = parsed_barriers + node_barriers
        self.crossroads_list = self.parse_intersections(ways_dict)

        self.osm_ways_data = None
        self.osm_rels_data = None
        self.osm_nodes_data = None

        logger.info("Parsing finished.")
        return 0

    def exclude_ways(self, highway_values: Iterable[str]) -> int:
        """
        Drop every road/footway whose ``highway`` tag is in *highway_values*.

        A shortcut for :meth:`apply_traversability` with rules that only deny
        those ``highway`` values (see
        :meth:`~map_data.traversability.TraversabilityRules.extend`), kept
        because the ``exclude_highway`` parameter of the nodes, the loader and
        the graph planner is written in those terms.

        Parameters
        ----------
        highway_values : iterable of str
            ``highway`` tag values to remove. An empty iterable is a no-op.

        Returns
        -------
        int
            Number of ways removed.

        """
        return self.apply_traversability(TraversabilityRules().extend(highway_values))

    def apply_traversability(self, rules: "TraversabilityRules") -> int:
        """
        Remove every road/footway *rules* declares non-traversable.

        The classification is stored in the ``.mapdata`` file as OSM tagged it,
        so ways a robot must not drive on (stairs, muddy shortcuts, bridges on
        some maps) have to be removed at use time rather than at parse time —
        the viewer must still show them. The crossroads are recomputed from the
        remaining footways, so junctions that only existed because of a removed
        way disappear with it.

        The per-reason counts are logged and also kept in
        :attr:`traversability_removed` for callers that report them (the ROS
        nodes log them through their own logger).

        Parameters
        ----------
        rules : TraversabilityRules
            The rule set; one with no rules is a no-op.

        Returns
        -------
        int
            Number of ways removed.

        """
        if not rules.rules and rules.default.traversable:
            self.traversability_removed = {}
            return 0
        counts: dict[str, int] = {}
        removed = 0
        for lst_name in ("footways_list", "roads_list"):
            ways = getattr(self, lst_name)
            kept = []
            for way in ways:
                verdict = rules.evaluate(way.tags)
                if verdict.traversable:
                    kept.append(way)
                else:
                    counts[verdict.reason] = counts.get(verdict.reason, 0) + 1
                    removed += 1
            setattr(self, lst_name, kept)
        self.traversability_removed = counts
        if removed:
            self.crossroads_list = self.parse_intersections(
                {w.id: w for w in self.footways_list + self.roads_list},
            )
            for reason, count in counts.items():
                logger.info("Removed %d way(s): %s", count, reason)
            logger.info(
                "Traversability rules removed %d way(s); %d crossroads remain",
                removed,
                len(self.crossroads_list),
            )
        return removed

    def centre_line(self, way: Way) -> geometry.LineString | None:
        """
        Rebuild a way's unbuffered centre line in UTM from its node ids.

        ``footways_list``/``roads_list`` hold the *buffered* geometry (a 3 m
        wide footway, a 7 m wide road), so the stored ``line`` cannot be used
        to find where two ways cross: a line intersected with a corridor gives
        the run inside it, not a crossing point. The node ids survive in
        ``nodes_cache``, so the true centre line can be rebuilt from them.

        A way that still carries an unbuffered ``LineString`` (a raw parse, an
        annotation's own centre line) is already what is wanted and is returned
        as-is. Returns ``None`` when the way has fewer than two known nodes, in
        which case the caller has to fall back to the stored geometry.
        """
        if way.line is not None and way.line.geom_type == "LineString":
            return way.line
        coords: list[tuple[float, float]] = []
        for node_id in way.nodes:
            node_data = self.nodes_cache.get(node_id)
            if node_data is None:
                return None
            e, n, _, _ = utm.from_latlon(
                node_data["lat"],
                node_data["lon"],
                force_zone_number=self.zone_number,
                force_zone_letter=self.zone_letter,
            )
            if not coords or (e, n) != coords[-1]:
                coords.append((e, n))
        return geometry.LineString(coords) if len(coords) >= 2 else None

    def parse_intersections(self, ways_dict: dict[Any, Way]) -> list[Way]:
        """
        Identify the routable nodes where ways actually branch, as crossroad Ways.

        Every footway *and* road is considered: the robot may be routed over
        roads (``highway_types`` includes ``road``), and a footway meeting a
        service road is as much a junction as two footways meeting. Ways that
        are neither (barriers, untagged areas) are ignored, so the whole
        ``ways_dict`` of a fresh parse can be passed in.

        A node is a crossroad when more than two *distinct* neighbouring nodes
        leave it. Counting the ways that use the node instead — the obvious
        reading of "shared by several ways" — reports a junction wherever the
        same corridor is mapped twice, which is common: a cycleway or an
        annotated path drawn along an existing footway reuses its node ids and
        would otherwise turn every node of the shared run into a crossroad. Two
        ways that run through a node between the same neighbours are the same
        path, not a fork; three directions out of a node are.
        """
        neighbours: dict[int, set[int]] = {}
        way_count: dict[int, int] = {}

        ways = [w for w in ways_dict.values() if w.is_footway() or w.is_road()]

        for way in ways:
            node_ids = way.nodes
            for i, node_id in enumerate(node_ids):
                way_count[node_id] = way_count.get(node_id, 0) + 1
                seen = neighbours.setdefault(node_id, set())
                if i > 0:
                    seen.add(node_ids[i - 1])
                if i < len(node_ids) - 1:
                    seen.add(node_ids[i + 1])

        crossroads = []
        for node_id, seen in neighbours.items():
            seen.discard(node_id)  # a way listing the same node twice in a row
            if len(seen) > 2:
                count = way_count[node_id]
                node_data = self.nodes_cache.get(node_id)
                if node_data is None:
                    continue
                e, n, _, _ = utm.from_latlon(
                    node_data["lat"],
                    node_data["lon"],
                    force_zone_number=self.zone_number,
                    force_zone_letter=self.zone_letter,
                )
                crossroads.append(
                    Way(
                        id=node_id,
                        is_area=True,
                        tags={"type": "footway_intersection", "count": str(count)},
                        line=geometry.Point(e, n).buffer(1.5),
                    ),
                )
        return crossroads

    @staticmethod
    def geometric_intersections(
        lines: list[tuple[Way, geometry.LineString]],
        others: list[tuple[Way, geometry.base.BaseGeometry]],
        touch_tolerance: float = 1.0,
        radius: float = 1.5,
    ) -> list[Way]:
        """
        Detect crossroads geometrically for ways that share no OSM node ids.

        Used for manually annotated paths: a crossroad is created where an
        annotated centre line crosses another way's centre line, where the two
        stop running together, or where one of its endpoints lies within
        ``touch_tolerance`` metres of another way (a T-junction).

        Both sides must be **centre lines**. The geometry stored on a parsed
        ``Way`` is the way buffered to its width, and intersecting a line with
        a 3 m wide corridor yields the stretch of line inside it rather than a
        crossing: a path merely running alongside a footway then reports a
        junction it never reaches, and a real crossing is reported half a
        corridor away from where it happens. :meth:`centre_line` rebuilds the
        unbuffered line from a way's node ids for exactly this.

        Parameters
        ----------
        lines : list of (Way, LineString)
            Annotated ways with their (unbuffered) centre lines, in UTM.
        others : list of (Way, BaseGeometry)
            Ways to test against, each with its centre line (the annotated way
            itself is skipped). A way whose centre line could not be rebuilt
            may be passed with its stored geometry, at the cost above.
        touch_tolerance : float
            Endpoint-to-way distance (m) that still counts as touching.
        radius : float
            Buffer radius (m) of the resulting crossroad polygons.

        Returns
        -------
        list of Way
            Crossroad ways (``type: annotation_intersection``) with negative ids.
        """
        crossroads: list[Way] = []
        seen: list[geometry.Point] = []
        next_id = -1_000_000

        def add(pt: geometry.Point, count: int) -> None:
            nonlocal next_id
            if any(pt.distance(q) < radius for q in seen):
                return
            seen.append(pt)
            crossroads.append(
                Way(
                    id=next_id,
                    is_area=True,
                    tags={"type": "annotation_intersection", "count": str(count)},
                    line=pt.buffer(radius),
                ),
            )
            next_id -= 1

        def junction_points(geom: geometry.base.BaseGeometry) -> list[geometry.Point]:
            """The junctions an intersection geometry stands for."""
            parts = getattr(geom, "geoms", [geom])
            points: list[geometry.Point] = []
            for part in parts:
                if part.is_empty:
                    continue
                if part.geom_type == "Point":
                    points.append(part)
                elif part.geom_type in ("LineString", "LinearRing"):
                    # The two ways run together along this stretch: they meet
                    # where it starts and separate where it ends.
                    points.append(geometry.Point(part.coords[0]))
                    points.append(geometry.Point(part.coords[-1]))
                else:
                    points.append(part.representative_point())
            return points

        for way, line in lines:
            if line is None or line.is_empty or line.geom_type != "LineString":
                continue
            ends = [geometry.Point(line.coords[0]), geometry.Point(line.coords[-1])]
            for other, other_line in others:
                if other is way or other_line is None or other_line.is_empty:
                    continue
                inter = line.intersection(other_line)
                if not inter.is_empty:
                    for pt in junction_points(inter):
                        add(pt, 2)
                    continue
                for end in ends:
                    if end.distance(other_line) <= touch_tolerance:
                        add(end, 2)
        return crossroads

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def run_all(self, *, save: bool = True) -> None:
        """
        Download OSM data, parse it, and optionally save the result.

        Convenience wrapper that calls :meth:`run_queries`, :meth:`run_parse`,
        and :meth:`save` in sequence.

        Parameters
        ----------
        save : bool
            If ``True`` (default), write a ``.mapdata`` file after successful
            parsing.

        """
        self.run_queries()
        if self.run_parse() == 0 and save:
            self.save()

    def save(self, path: str | None = None) -> None:
        """
        Serialize this object to a ``.mapdata`` file.

        Parameters
        ----------
        path : str, optional
            Output file path. Defaults to the source GPX filename with its
            extension replaced by ``.mapdata``. Logs an error and returns
            without writing if no path can be determined.

        """
        if path is None:
            if self.coords_file:
                path = str(Path(self.coords_file).with_suffix(".mapdata"))
            else:
                logger.error("No save path provided and no source file available.")
                return
        save_mapdata(self, path)
        logger.info("Map data saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MapData":
        """
        Load a previously saved ``.mapdata`` file.

        The node-based crossroads are recomputed rather than trusted: they are
        derived from the ways, and a file written by an older version carries
        the junctions its detector found (footways only, one per node of a
        corridor mapped twice). Crossroads that cannot be recomputed from node
        ids, the ``annotation_intersection`` ones a drawn path contributes, are
        kept as they were saved.

        Parameters
        ----------
        path : str
            Path to the ``.mapdata`` file to load.

        Returns
        -------
        MapData
            Restored instance with all way lists populated.

        """
        md = load_mapdata(cls, path)
        geometric = [
            c for c in md.crossroads_list if c.tags.get("type") == "annotation_intersection"
        ]
        md.crossroads_list = (
            md.parse_intersections({str(w.id): w for w in md.footways_list + md.roads_list})
            + geometric
        )
        return md

    def __str__(self) -> str:
        source = f"File: {self.coords_file}" if self.coords_file else "Array"
        return (
            f"MapData Object\n"
            f"  Source: {source}\n"
            f"  Waypoints: {len(self.waypoints)}\n"
            f"  UTM Zone: {self.zone_number}{self.zone_letter}\n"
            f"  Bounds: X[{self.min_x:.1f}, {self.max_x:.1f}], "
            f"Y[{self.min_y:.1f}, {self.max_y:.1f}]\n"
            f"  Features: {len(self.roads_list)} roads, "
            f"{len(self.footways_list)} footways, {len(self.barriers_list)} barriers"
        )

    def get_points(self, z: float = 0.0) -> dict[int, np.ndarray]:
        """
        Return all cached OSM nodes as a dictionary of UTM coordinates.

        Parameters
        ----------
        z : float
            Z-coordinate to assign to every node (default ``0.0``).

        Returns
        -------
        dict of {int: np.ndarray}
            Mapping of OSM node ID → column vector of shape ``(3, 1)``
            containing ``[easting, northing, z]``.

        """
        if not self.nodes_cache:
            return {}
        node_ids = list(self.nodes_cache.keys())
        lats = np.array([self.nodes_cache[nid]["lat"] for nid in node_ids])
        lons = np.array([self.nodes_cache[nid]["lon"] for nid in node_ids])
        eastings, northings, _, _ = utm.from_latlon(
            lats, lons, force_zone_number=self.zone_number, force_zone_letter=self.zone_letter
        )
        return {
            nid: np.array([e, n, z]).reshape(3, 1)
            for nid, e, n in zip(node_ids, eastings, northings, strict=True)
        }

    def get_ways(self) -> dict[str, list[Way]]:
        """
        Return all parsed way lists grouped by category.

        Returns
        -------
        dict of {str: list of Way}
            Keys are ``"roads"``, ``"footways"``, ``"barriers"``, and
            ``"crossroads"``.

        """
        return {
            "roads": self.roads_list,
            "footways": self.footways_list,
            "barriers": self.barriers_list,
            "crossroads": self.crossroads_list,
        }
