import os
import logging
import utm
import numpy as np
import shapely.geometry as geometry
from gpxpy import parse as gpxparse
from typing import Optional, Dict, List, Any, Tuple

from map_data.way import Way
from map_data.overpass import OverpassClient
from map_data.parsing import (
    parse_osm_ways,
    parse_osm_rels,
    parse_osm_nodes,
    separate_ways,
)
from map_data.serialization import save_mapdata, load_mapdata

logger = logging.getLogger(__name__)

OSM_RECTANGLE_MARGIN = 100  # meters, expansion margin for OSM bounding box query
RESERVE = 50  # meters, safety margin added to waypoint bounds


class CoordsData:
    def __init__(
        self, min_long: float, max_long: float, min_lat: float, max_lat: float
    ):
        self.min_long = min_long
        self.max_long = max_long
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.x_margin, self.y_margin = self._compute_margin()

    def _compute_margin(self) -> Tuple[float, float]:
        margin = max(
            (self.max_lat - self.min_lat) * 0.1,
            (self.max_long - self.min_long) * 0.1,
        )
        return margin, margin


class MapData:
    def __init__(
        self,
        coords: Any,
        coords_type: str = "file",
        current_robot_position: Optional[np.ndarray] = None,
        flip: bool = False,
    ):
        self.coords_type = coords_type
        if coords_type == "file":
            with open(coords, "r") as f:
                gpx_object = gpxparse(f)
            self.coords_file = coords
            latlon = np.array([[p.latitude, p.longitude] for p in gpx_object.waypoints])
            self.waypoints, self.zone_number, self.zone_letter = self._latlon_to_utm(
                latlon
            )
        elif coords_type == "array":
            self.waypoints = np.array(coords[0])
            self.zone_number = coords[1]
            self.zone_letter = coords[2]
            self.coords_file = None
        else:
            raise ValueError(f"Unknown coords_type: {coords_type!r}")

        self.flip = flip
        if flip:
            self.waypoints = np.flip(self.waypoints, 0)

        if current_robot_position is not None:
            self.robot_position_first_point = True
            self.waypoints = np.concatenate([current_robot_position, self.waypoints])
        else:
            self.robot_position_first_point = False

        self.max_x = float(np.max(self.waypoints[:, 0]) + RESERVE)
        self.min_x = float(np.min(self.waypoints[:, 0]) - RESERVE)
        self.max_y = float(np.max(self.waypoints[:, 1]) + RESERVE)
        self.min_y = float(np.min(self.waypoints[:, 1]) - RESERVE)

        self.max_lat, self.max_long = utm.to_latlon(
            self.max_x + OSM_RECTANGLE_MARGIN,
            self.max_y + OSM_RECTANGLE_MARGIN,
            self.zone_number,
            self.zone_letter,
        )
        self.min_lat, self.min_long = utm.to_latlon(
            self.min_x - OSM_RECTANGLE_MARGIN,
            self.min_y - OSM_RECTANGLE_MARGIN,
            self.zone_number,
            self.zone_letter,
        )

        self.coords_data = CoordsData(
            self.min_long, self.max_long, self.min_lat, self.max_lat
        )
        self.points = [
            geometry.Point(x, y)
            for x, y in zip(self.waypoints[:, 0], self.waypoints[:, 1])
        ]

        self.nodes_cache: Dict[int, Any] = {}
        self.roads_list: List[Way] = []
        self.footways_list: List[Way] = []
        self.barriers_list: List[Way] = []
        self.crossroads_list: List[Way] = []

        # Raw data stored temporarily during parsing
        self.osm_ways_data: Optional[Any] = None
        self.osm_rels_data: Optional[Any] = None
        self.osm_nodes_data: Optional[Any] = None

        self._load_tag_configs()
        self.overpass = OverpassClient()

    def _load_tag_configs(self):
        try:
            from ament_index_python.resources import get_resource

            _, package_path = get_resource("packages", "map_data")
            params_path = os.path.join(package_path, "share", "map_data", "parameters")
        except Exception:
            params_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "parameters")
            )

        self.BARRIER_TAGS = self._csv_to_dict(
            os.path.join(params_path, "barrier_tags.csv")
        )
        self.NOT_BARRIER_TAGS = self._csv_to_dict(
            os.path.join(params_path, "not_barrier_tags.csv")
        )
        self.ANTI_BARRIER_TAGS = self._csv_to_dict(
            os.path.join(params_path, "anti_barrier_tags.csv")
        )
        self.OBSTACLE_TAGS = self._csv_to_dict(
            os.path.join(params_path, "obstacle_tags.csv")
        )
        self.NOT_OBSTACLE_TAGS = self._csv_to_dict(
            os.path.join(params_path, "not_obstacle_tags.csv")
        )

    @staticmethod
    def _csv_to_dict(path: str) -> Dict[str, List[str]]:
        arr = np.genfromtxt(path, dtype=str, delimiter=",")
        result = {}
        for row in arr:
            result.setdefault(row[0], []).append(row[1])
        return result

    @staticmethod
    def _latlon_to_utm(latlon: np.ndarray) -> Tuple[np.ndarray, int, str]:
        easting, northing, zone_number, zone_letter = utm.from_latlon(
            latlon[:, 0], latlon[:, 1]
        )
        return np.column_stack([easting, northing]), zone_number, zone_letter

    def run_queries(self):
        bbox = f"{self.min_lat},{self.min_long},{self.max_lat},{self.max_long}"
        queries = {
            "ways": f"[out:json]; (way({bbox}); >; ); out;",
            "rels": f"[out:json]; (way({bbox}); <; ); out;",
            "nodes": f"[out:json]; (node({bbox}); ); out;",
        }
        self.osm_ways_data = self.overpass.query(queries["ways"])
        self.osm_rels_data = self.overpass.query(queries["rels"])
        self.osm_nodes_data = self.overpass.query(queries["nodes"])
        logger.info("All OSM queries finished.")

    def run_parse(self) -> int:
        if any(
            d is None
            for d in (self.osm_ways_data, self.osm_rels_data, self.osm_nodes_data)
        ):
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
        logger.info("Parsing finished.")
        return 0

    def parse_intersections(self, ways_dict: Dict[int, Way]) -> List[Way]:
        """Identify nodes shared by multiple footways and return them as crossroad Ways."""
        node_usage: Dict[int, List[bool]] = {}

        footways = [w for w in ways_dict.values() if w.is_footway()]

        for way in footways:
            for i, node_id in enumerate(way.nodes):
                is_endpoint = (i == 0 or i == len(way.nodes) - 1)
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

    def run_all(self, save: bool = True):
        self.run_queries()
        if self.run_parse() == 0 and save:
            self.save()

    def save(self, path: Optional[str] = None):
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

    def get_points(self, z: float = 0.0) -> Dict[int, np.ndarray]:
        points = {}
        for node_id, data in self.nodes_cache.items():
            e, n, _, _ = utm.from_latlon(data["lat"], data["lon"])
            points[node_id] = np.array([e, n, z]).reshape(3, 1)
        return points

    def get_ways(self) -> Dict[str, List[Way]]:
        return {
            "roads": self.roads_list,
            "footways": self.footways_list,
            "barriers": self.barriers_list,
            "crossroads": self.crossroads_list,
        }
