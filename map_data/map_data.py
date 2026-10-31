"""MapData: download, cache and parse the OpenStreetMap data of an area."""

import json
import logging
import time
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import overpy
import utm
from gpxpy import parse as gpxparse
from shapely import STRtree, geometry

from map_data.traversability import TraversabilityRules
from map_data.utils.config import load_config, package_share
from map_data.utils.overpass import REQUEST_TIMEOUT, OverpassClient
from map_data.utils.parsing import (
    parse_osm_nodes,
    parse_osm_rels,
    parse_osm_ways,
    separate_ways,
)
from map_data.utils.serialization import atomic_write_json, load_mapdata, map_data_to_dict
from map_data.utils.way import Way

logger = logging.getLogger(__name__)


_DEFAULTS = load_config("planner_defaults.yaml")
GRID_MARGIN: float = _DEFAULTS.get("grid_margin", 150)
BBOX_LEN = 4
# geometric_intersections: endpoint-to-way distance (m) that still counts as
# touching, and buffer radius (m) of the resulting crossroad polygons.
TOUCH_TOLERANCE = 1.0
CROSSROAD_RADIUS = 1.5
OSM_CACHE_MAX_AGE = 30 * 24 * 60 * 60


class MapData:
    """
    OSM features of the area around a set of GPS waypoints.

    Parses the waypoints from a GPX file (or a UTM array), downloads the OSM
    features of their bounding box via Overpass and sorts them into
    ``roads_list``, ``footways_list`` and ``barriers_list`` of
    :class:`~map_data.utils.way.Way`. Needs no ROS.

    Attributes
    ----------
    waypoints : np.ndarray
        ``(N, 2)`` UTM easting/northing of the input points.
    crossroads_list : list of Way
        Junction discs where footways/roads branch, plus the
        ``annotation_intersection`` ones of drawn paths.

    """

    zone_number: int
    zone_letter: str
    # Tag tables loaded from parameters/<name>.csv by _load_tag_configs.
    BARRIER_TAGS: dict[str, list[str]]
    NOT_BARRIER_TAGS: dict[str, list[str]]
    ANTI_BARRIER_TAGS: dict[str, list[str]]
    OBSTACLE_TAGS: dict[str, list[str]]
    NOT_OBSTACLE_TAGS: dict[str, list[str]]
    ANTI_OBSTACLE_TAGS: dict[str, list[str]]

    def __init__(
        self,
        coords: str | tuple[np.ndarray, int, str],
        coords_type: str = "file",
        *,
        grid_margin: float | None = None,
        obstacle_radius: float | None = None,
        buffer_widths: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize MapData from a GPX file or coordinate array.

        Parameters
        ----------
        coords : str or tuple
            A ``.gpx`` path (``coords_type="file"``) or
            ``(waypoints, zone_number, zone_letter)`` with ``(N, 2)`` UTM
            waypoints (``coords_type="array"``).

        """
        if coords_type == "file":
            # coords is a str here; mypy can't correlate it with coords_type.
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
            easting, northing, self.zone_number, self.zone_letter = utm.from_latlon(
                latlon[:, 0], latlon[:, 1]
            )
            self.waypoints = np.column_stack([easting, northing])
        elif coords_type == "array":
            # coords is the (ndarray, int, str) tuple here (see above).
            self.waypoints = np.array(coords[0])
            self.zone_number = coords[1]  # type: ignore[assignment]
            self.zone_letter = coords[2]
            self.coords_file = None
        else:
            msg = f"Unknown coords_type: {coords_type!r}"
            raise ValueError(msg)

        _margin = grid_margin if grid_margin is not None else GRID_MARGIN
        self.max_x = float(np.max(self.waypoints[:, 0]) + _margin)
        self.min_x = float(np.min(self.waypoints[:, 0]) - _margin)
        self.max_y = float(np.max(self.waypoints[:, 1]) + _margin)
        self.min_y = float(np.min(self.waypoints[:, 1]) - _margin)
        self._obstacle_radius = obstacle_radius
        self._buffer_widths = buffer_widths

        corners = utm.to_latlon(
            np.array([self.min_x, self.max_x, self.min_x, self.max_x]),
            np.array([self.min_y, self.min_y, self.max_y, self.max_y]),
            self.zone_number,
            self.zone_letter,
        )
        self.min_lat, self.max_lat = float(np.min(corners[0])), float(np.max(corners[0]))
        self.min_long, self.max_long = float(np.min(corners[1])), float(np.max(corners[1]))

        self._check_utm_zone_boundary()

        self.nodes_cache: dict[int, dict[str, Any]] = {}
        self.roads_list: list[Way] = []
        self.footways_list: list[Way] = []
        self.barriers_list: list[Way] = []
        self.crossroads_list: list[Way] = []
        #: ``{reason: count}`` of the last :meth:`apply_traversability` call.
        self.traversability_removed: dict[str, int] = {}

        # Overpass result, held from run_queries until run_parse.
        self.osm_data: overpy.Result | None = None

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
            zones.add((zn, zl[-1]))
        if len(zones) > 1:
            zone_strs = ", ".join(f"{zn}{zl}" for zn, zl in sorted(zones))
            logger.warning(
                "Waypoints span multiple UTM zones (%s). Geometry near zone boundaries "
                "may be distorted. Consider splitting the area into smaller regions.",
                zone_strs,
            )

    def _load_tag_configs(self) -> None:
        params_path = package_share("parameters")
        for attr in (
            "BARRIER_TAGS",
            "NOT_BARRIER_TAGS",
            "ANTI_BARRIER_TAGS",
            "OBSTACLE_TAGS",
            "NOT_OBSTACLE_TAGS",
            "ANTI_OBSTACLE_TAGS",
        ):
            setattr(self, attr, self._csv_to_dict(params_path / f"{attr.lower()}.csv"))

    @staticmethod
    def _csv_to_dict(path: str | Path) -> dict[str, list[str]]:
        # atleast_2d: for a single-row CSV genfromtxt returns a 1-D array,
        # whose "rows" would iterate as individual strings (characters).
        arr = np.atleast_2d(np.genfromtxt(path, dtype=str, delimiter=","))
        result: dict[str, list[str]] = {}
        for row in arr:
            result.setdefault(row[0], []).append(row[1])
        return result

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
            "created": time.time(),
            "query_hash": self._osm_query_hash(),
            "raw": raw,
        }
        try:
            atomic_write_json(path, cache_data)
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

            stored_bbox = cache_data.get("bbox")
            current_bbox = [self.min_lat, self.min_long, self.max_lat, self.max_long]
            if not stored_bbox or len(stored_bbox) != BBOX_LEN:
                return None

            # 1e-6 degree tolerance (~11cm at equator)
            if not np.allclose(stored_bbox, current_bbox, atol=1e-6):
                logger.info("OSM cache found but bounding box has changed. Re-querying.")
                return None
            if cache_data.get("query_hash") != self._osm_query_hash():
                logger.info("OSM cache found but the query has changed. Re-querying.")
                return None
            if time.time() - cache_data.get("created", 0) > OSM_CACHE_MAX_AGE:
                logger.info("OSM cache is stale. Re-querying.")
                return None

            logger.info("Using cached OSM responses from %s", path)
            return cache_data["raw"]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.debug("Could not load OSM cache: %s", e)
            return None

    def _osm_query_hash(self) -> str:
        return sha256(self._osm_query().encode()).hexdigest()

    def _osm_query(self) -> str:
        bbox = f"{self.min_lat},{self.min_long},{self.max_lat},{self.max_long}"
        ql_timeout = REQUEST_TIMEOUT - 10
        return (
            f"[out:json][timeout:{ql_timeout}];way({bbox})->.w;(.w; .w >; .w <; node({bbox}););out;"
        )

    def run_queries(
        self,
        *,
        use_cache: bool = True,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        """
        Download the bounding box's ways, their nodes and relations, and all nodes.

        One Overpass query; the result is kept for :meth:`run_parse`.

        Parameters
        ----------
        use_cache : bool
            Try the ``.osm_cache.json`` next to the GPX file first.
        progress_cb : callable, optional
            Called with a status string before each Overpass request attempt.

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
                    self.osm_data = result
                    return

        bbox = f"{self.min_lat},{self.min_long},{self.max_lat},{self.max_long}"
        # Server-side timeout below the HTTP one, so Overpass reports a slow query
        # as an error instead of the client giving up mid-response.
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

        # A malformed body that passes query_raw's check must not crash the run.
        try:
            result = client.api.parse_json(raw)
        except (overpy.exception.OverPyException, json.JSONDecodeError):
            logger.exception("Overpass returned an unparseable response.")
            return

        self.osm_data = result

        logger.info("OSM query finished.")
        self._save_osm_cache(raw)

    def run_parse(self) -> int:
        """
        Parse the downloaded OSM data into categorised Way lists.

        Fills the way and crossroad lists and drops the raw Overpass result.

        Returns
        -------
        int
            ``0`` on success, ``1`` if :meth:`run_queries` has not succeeded.

        """
        if self.osm_data is None:
            logger.error("Missing OSM data. Run run_queries() first.")
            return 1

        logger.info("Parsing OSM data.")
        ways_dict = parse_osm_ways(
            self.osm_data, self.nodes_cache, self.zone_number, self.zone_letter
        )
        parse_osm_rels(self.osm_data, ways_dict)

        way_node_ids = {nid for w in ways_dict.values() for nid in w.nodes}
        node_barriers = parse_osm_nodes(
            self.osm_data,
            self.nodes_cache,
            way_node_ids,
            self.OBSTACLE_TAGS,
            self.NOT_OBSTACLE_TAGS,
            self.ANTI_OBSTACLE_TAGS,
            self._obstacle_radius,
            self.zone_number,
            self.zone_letter,
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

        self.osm_data = None

        logger.info("Parsing finished.")
        return 0

    def apply_traversability(self, rules: "TraversabilityRules") -> int:
        """
        Remove every road/footway *rules* declares non-traversable.

        Done at use time, not parse time, because the ``.mapdata`` keeps OSM's
        classification for the viewer. Crossroads are recomputed from the
        remaining ways; per-reason counts are logged and kept in
        :attr:`traversability_removed`.

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
            self.recompute_crossroads()
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

        Stored road/footway geometry is buffered, so it cannot locate where two
        ways cross. A way that still holds a ``LineString`` is returned as-is.
        Returns ``None`` when a node is missing from ``nodes_cache`` or fewer
        than two distinct points remain.
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
        Return a crossroad Way at every footway/road node where the ways branch.

        A node is a crossroad when more than two distinct neighbouring nodes
        leave it, not when several ways share it: a corridor mapped twice (a
        cycleway drawn along a footway) shares every node without forking.
        Ways that are neither footway nor road are ignored.
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
    ) -> list[Way]:
        """
        Detect crossroads geometrically for ways that share no OSM node ids.

        A crossroad is placed where an annotated centre line crosses another
        way's, at both ends of a stretch where they run together, and at an
        endpoint within ``TOUCH_TOLERANCE`` m of another way. Both sides should
        be centre lines (:meth:`centre_line`): against buffered geometry a path
        running alongside a way reports a junction it never reaches.

        Parameters
        ----------
        lines : list of (Way, LineString)
            Annotated ways with their centre lines, in UTM.
        others : list of (Way, BaseGeometry)
            Ways to test against with their centre lines; a way in *lines* is
            not tested against itself.

        Returns
        -------
        list of Way
            ``annotation_intersection`` crossroads with negative ids.
        """
        crossroads: list[Way] = []
        seen: list[geometry.Point] = []
        next_id = -1_000_000

        def add(pt: geometry.Point, count: int) -> None:
            nonlocal next_id
            if any(pt.distance(q) < CROSSROAD_RADIUS for q in seen):
                return
            seen.append(pt)
            crossroads.append(
                Way(
                    id=next_id,
                    is_area=True,
                    tags={"type": "annotation_intersection", "count": str(count)},
                    line=pt.buffer(CROSSROAD_RADIUS),
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
                    if end.distance(other_line) <= TOUCH_TOLERANCE:
                        add(end, 2)
        return crossroads

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def run_all(self, *, save: bool = True) -> None:
        """Download (bypassing the cache), parse and, if *save*, save the map."""
        self.run_queries(use_cache=False)
        if self.run_parse() == 0 and save:
            self.save()

    def save(self, path: str | None = None) -> None:
        """
        Serialize this object to a ``.mapdata`` file.

        Parameters
        ----------
        path : str, optional
            Defaults to the source GPX path with a ``.mapdata`` suffix; without
            either, logs an error and writes nothing.

        """
        if path is None:
            if self.coords_file:
                path = str(Path(self.coords_file).with_suffix(".mapdata"))
            else:
                logger.error("No save path provided and no source file available.")
                return
        atomic_write_json(path, map_data_to_dict(self), indent=2)
        logger.info("Map data saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MapData":
        """
        Load a ``.mapdata`` file.

        Node-based crossroads are recomputed from the ways rather than read from
        the file; saved ``annotation_intersection`` crossroads are kept.
        """
        md = load_mapdata(cls, path)
        md.recompute_crossroads()
        return md

    def recompute_crossroads(self) -> None:
        """
        Recompute the node-based crossroads and keep the drawn-path ones.

        The ``annotation_intersection`` crossroads cannot be recomputed from node
        ids, so they are kept as they are. A drawn path joined to the network by a
        real node (:func:`map_data.annotations.join_ways`) forks at that node too;
        that recomputed crossroad is dropped, the junction is already listed.
        """
        node_based = self.parse_intersections(
            {str(w.id): w for w in self.footways_list + self.roads_list}
        )
        geometric = [
            c for c in self.crossroads_list if c.tags.get("type") == "annotation_intersection"
        ]
        if node_based and geometric:
            tree = STRtree([c.line for c in geometric])
            node_based = [c for c in node_based if not len(tree.query(c.line, "intersects"))]
        self.crossroads_list = node_based + geometric

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
        """Map every cached node id to a ``(3, 1)`` ``[easting, northing, z]`` column."""
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
        """Way lists keyed ``"roads"``, ``"footways"``, ``"barriers"``, ``"crossroads"``."""
        return {
            "roads": self.roads_list,
            "footways": self.footways_list,
            "barriers": self.barriers_list,
            "crossroads": self.crossroads_list,
        }
