import concurrent.futures
import json
import logging
import os
from typing import Any

import numpy as np
import shapely.geometry as geometry
import utm
from gpxpy import parse as gpxparse

from map_data.utils.config import load_config
from map_data.utils.overpass import OverpassClient
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
OSM_MARGIN: float = _DEFAULTS.get("osm_margin", 100)
RESERVE: float = _DEFAULTS.get("reserve_margin", 50)


class CoordsData:
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
        coords: Any,
        coords_type: str = "file",
        current_robot_position: np.ndarray | None = None,
        flip: bool = False,
    ) -> None:
        """
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
        flip : bool
            If ``True``, reverse the order of the parsed waypoints.
        """
        if coords_type == "file":
            with open(coords) as f:
                gpx_object = gpxparse(f)
            self.coords_file: str | None = coords

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
                raise ValueError(f"No points (waypoints, tracks or routes) found in {coords}")

            latlon = np.array(points)
            self.waypoints, self.zone_number, self.zone_letter = self._latlon_to_utm(latlon)
        elif coords_type == "array":
            self.waypoints = np.array(coords[0])
            self.zone_number = coords[1]
            self.zone_letter = coords[2]
            self.coords_file = None
        else:
            raise ValueError(f"Unknown coords_type: {coords_type!r}")

        if flip:
            self.waypoints = np.flip(self.waypoints, 0)

        if current_robot_position is not None:
            self.waypoints = np.concatenate([current_robot_position, self.waypoints])

        self.max_x = float(np.max(self.waypoints[:, 0]) + RESERVE + OSM_MARGIN)
        self.min_x = float(np.min(self.waypoints[:, 0]) - (RESERVE + OSM_MARGIN))
        self.max_y = float(np.max(self.waypoints[:, 1]) + RESERVE + OSM_MARGIN)
        self.min_y = float(np.min(self.waypoints[:, 1]) - (RESERVE + OSM_MARGIN))

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
            geometry.Point(x, y) for x, y in zip(self.waypoints[:, 0], self.waypoints[:, 1])
        ]

        self.nodes_cache: dict[int, Any] = {}
        self.roads_list: list[Way] = []
        self.footways_list: list[Way] = []
        self.barriers_list: list[Way] = []
        self.crossroads_list: list[Way] = []

        # Raw data stored temporarily during parsing
        self.osm_ways_data: Any | None = None
        self.osm_rels_data: Any | None = None
        self.osm_nodes_data: Any | None = None

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
            params_path = os.path.join(package_path, "share", "map_data", "parameters")
        except Exception:
            params_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "parameters")
            )

        self.BARRIER_TAGS: dict[str, list[str]] = self._csv_to_dict(
            os.path.join(params_path, "barrier_tags.csv")
        )
        self.NOT_BARRIER_TAGS: dict[str, list[str]] = self._csv_to_dict(
            os.path.join(params_path, "not_barrier_tags.csv")
        )
        self.ANTI_BARRIER_TAGS: dict[str, list[str]] = self._csv_to_dict(
            os.path.join(params_path, "anti_barrier_tags.csv")
        )
        self.OBSTACLE_TAGS: dict[str, list[str]] = self._csv_to_dict(
            os.path.join(params_path, "obstacle_tags.csv")
        )
        self.NOT_OBSTACLE_TAGS: dict[str, list[str]] = self._csv_to_dict(
            os.path.join(params_path, "not_obstacle_tags.csv")
        )

    @staticmethod
    def _csv_to_dict(path: str) -> dict[str, list[str]]:
        arr = np.genfromtxt(path, dtype=str, delimiter=",")
        result: dict[str, list[str]] = {}
        for row in arr:
            result.setdefault(row[0], []).append(row[1])
        return result

    @staticmethod
    def _latlon_to_utm(latlon: np.ndarray) -> tuple[np.ndarray, int, str]:
        easting, northing, zone_number, zone_letter = utm.from_latlon(latlon[:, 0], latlon[:, 1])
        return np.column_stack([easting, northing]), zone_number, zone_letter

    def _get_osm_cache_path(self) -> str | None:
        if not self.coords_file:
            return None
        return self.coords_file.rsplit(".", 1)[0] + ".osm_cache.json"

    def _save_osm_cache(self, ways_raw: str, rels_raw: str, nodes_raw: str) -> None:
        path = self._get_osm_cache_path()
        if not path:
            return
        cache_data = {
            "bbox": [self.min_lat, self.min_long, self.max_lat, self.max_long],
            "ways": ways_raw,
            "rels": rels_raw,
            "nodes": nodes_raw,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
            logger.info(f"Saved OSM response cache to {path}")
        except Exception as e:
            logger.warning(f"Could not save OSM cache: {e}")

    def _load_osm_cache(self) -> dict[str, str] | None:
        path = self._get_osm_cache_path()
        if not path or not os.path.exists(path):
            return None

        try:
            with open(path, encoding="utf-8") as f:
                cache_data = json.load(f)

            # Validate bbox
            stored_bbox = cache_data.get("bbox")
            current_bbox = [self.min_lat, self.min_long, self.max_lat, self.max_long]
            if not stored_bbox or len(stored_bbox) != 4:
                return None

            # 1e-6 degree tolerance (~11cm at equator)
            if not np.allclose(stored_bbox, current_bbox, atol=1e-6):
                logger.info("OSM cache found but bounding box has changed. Re-querying.")
                return None

            logger.info(f"Using cached OSM responses from {path}")
            return {
                "ways": cache_data["ways"],
                "rels": cache_data["rels"],
                "nodes": cache_data["nodes"],
            }
        except Exception as e:
            logger.debug(f"Could not load OSM cache: {e}")
            return None

    def run_queries(self, use_cache: bool = True) -> None:
        """
        Download OSM ways, relations, and nodes from the Overpass API.

        Fires three concurrent Overpass queries covering the bounding box of
        the loaded waypoints and stores the raw JSON responses internally.
        Call :meth:`run_parse` afterwards to convert the responses into
        :class:`~map_data.utils.way.Way` objects.

        Parameters
        ----------
        use_cache : bool
            If ``True`` (default), attempt to load responses from a local
            ``.osm_cache.json`` file before querying the API.
        """
        if use_cache:
            cache = self._load_osm_cache()
            if cache:
                client = OverpassClient()
                self.osm_ways_data = client.api.parse_json(cache["ways"])
                self.osm_rels_data = client.api.parse_json(cache["rels"])
                self.osm_nodes_data = client.api.parse_json(cache["nodes"])
                return

        bbox = f"{self.min_lat},{self.min_long},{self.max_lat},{self.max_long}"
        queries = {
            "ways": f"[out:json]; (way({bbox}); >; ); out;",
            "rels": f"[out:json]; (way({bbox}); <; ); out;",
            "nodes": f"[out:json]; (node({bbox}); ); out;",
        }

        client = OverpassClient()

        def _fetch(q: str) -> str | None:
            return client.query_raw(q)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futs = {name: ex.submit(_fetch, q) for name, q in queries.items()}
            ways_raw = futs["ways"].result()
            rels_raw = futs["rels"].result()
            nodes_raw = futs["nodes"].result()

        if any(r is None for r in (ways_raw, rels_raw, nodes_raw)):
            logger.error("One or more Overpass queries failed.")
            return

        self.osm_ways_data = client.api.parse_json(ways_raw)
        self.osm_rels_data = client.api.parse_json(rels_raw)
        self.osm_nodes_data = client.api.parse_json(nodes_raw)

        logger.info("All OSM queries finished.")
        self._save_osm_cache(ways_raw, rels_raw, nodes_raw)

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
        ways_dict = parse_osm_ways(self.osm_ways_data, self.nodes_cache)
        parse_osm_rels(self.osm_rels_data, ways_dict)

        way_node_ids = {nid for w in ways_dict.values() for nid in w.nodes}
        node_barriers = parse_osm_nodes(
            self.osm_nodes_data,
            self.nodes_cache,
            way_node_ids,
            self.OBSTACLE_TAGS,
            self.NOT_OBSTACLE_TAGS,
        )

        self.roads_list, self.footways_list, parsed_barriers = separate_ways(
            ways_dict, self.BARRIER_TAGS, self.NOT_BARRIER_TAGS, self.ANTI_BARRIER_TAGS
        )
        self.barriers_list = parsed_barriers + node_barriers
        self.crossroads_list = self.parse_intersections(ways_dict)

        self.osm_ways_data = None
        self.osm_rels_data = None
        self.osm_nodes_data = None

        logger.info("Parsing finished.")
        return 0

    def parse_intersections(self, ways_dict: dict[int, Way]) -> list[Way]:
        """
        Identify nodes shared by multiple footways and return them as crossroad Ways.
        """
        node_usage: dict[int, list[bool]] = {}

        footways = [w for w in ways_dict.values() if w.is_footway()]

        for way in footways:
            for i, node_id in enumerate(way.nodes):
                is_endpoint = i == 0 or i == len(way.nodes) - 1
                if node_id not in node_usage:
                    node_usage[node_id] = []
                node_usage[node_id].append(is_endpoint)

        crossroads = []
        for node_id, usages in node_usage.items():
            count = len(usages)
            is_crossroad = count > 2 or (count == 2 and not (usages[0] and usages[1]))
            if is_crossroad:
                node_data = self.nodes_cache.get(node_id)
                if node_data is None:
                    continue
                e, n, _, _ = utm.from_latlon(node_data["lat"], node_data["lon"])
                crossroads.append(
                    Way(
                        id=node_id,
                        is_area=True,
                        tags={"type": "footway_intersection", "count": str(count)},
                        line=geometry.Point(e, n).buffer(1.5),
                    )
                )
        return crossroads

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def run_all(self, save: bool = True) -> None:
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
                path = self.coords_file.rsplit(".", 1)[0] + ".mapdata"
            else:
                logger.error("No save path provided and no source file available.")
                return
        save_mapdata(self, path)
        logger.info(f"Map data saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MapData":
        """
        Load a previously saved ``.mapdata`` file.

        Parameters
        ----------
        path : str
            Path to the ``.mapdata`` file to load.

        Returns
        -------
        MapData
            Restored instance with all way lists populated.
        """
        return load_mapdata(cls, path)

    def __str__(self) -> str:
        source = f"File: {self.coords_file}" if self.coords_file else "Array"
        return (
            f"MapData Object\n"
            f"  Source: {source}\n"
            f"  Waypoints: {len(self.waypoints)}\n"
            f"  UTM Zone: {self.zone_number}{self.zone_letter}\n"
            f"  Bounds: X[{self.min_x:.1f}, {self.max_x:.1f}], Y[{self.min_y:.1f}, {self.max_y:.1f}]\n"
            f"  Features: {len(self.roads_list)} roads, {len(self.footways_list)} footways, {len(self.barriers_list)} barriers"
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
        points = {}
        for node_id, data in self.nodes_cache.items():
            e, n, _, _ = utm.from_latlon(data["lat"], data["lon"])
            points[node_id] = np.array([e, n, z]).reshape(3, 1)
        return points

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
