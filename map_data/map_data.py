import os
import re
import json
import logging
import time
import pickle
import urllib.error
import urllib.parse
import urllib.request

import overpy
import utm
import numpy as np
from tqdm import tqdm
import shapely.geometry as geometry
from shapely.ops import linemerge
from gpxpy import parse as gpxparse

from map_data.way import Way


logger = logging.getLogger(__name__)

OBSTACLE_RADIUS = 2        # meters, radius of the circle around isolated obstacle nodes
OSM_RECTANGLE_MARGIN = 100 # meters, expansion margin for OSM bounding box query
RESERVE = 50               # meters, safety margin added to waypoint bounds

# Tried in order on failure; overpass-api.de is checked via /api/status before
# each attempt so we sleep until a free slot is available.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def _fetch_overpass(url: str, query: str, timeout: int = 180) -> dict:
    """POST a query to the Overpass API and return the parsed JSON dict.

    Sends the query as a form-encoded 'data' parameter (correct Overpass format).
    The result is passed to overpy for parsing into native result objects.
    """
    body = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "map_data/1.0 (https://github.com/map_data)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ------------------------------------------------------------------

class CoordsData:
    def __init__(self, min_long, max_long, min_lat, max_lat):
        self.min_long = min_long
        self.max_long = max_long
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.x_margin, self.y_margin = self._compute_margin()

    def _compute_margin(self):
        margin = max(
            (self.max_lat - self.min_lat) * 0.1,
            (self.max_long - self.min_long) * 0.1,
        )
        return margin, margin


class MapData:
    def __init__(self, coords, coords_type="file", current_robot_position=None, flip=False):
        self._endpoint_index = 0

        self.coords_type = coords_type
        if coords_type == "file":
            with open(coords, "r") as f:
                gpx_object = gpxparse(f)
            self.coords_file = coords
            latlon = np.array([[p.latitude, p.longitude] for p in gpx_object.waypoints])
            self.waypoints, self.zone_number, self.zone_letter = self._latlon_to_utm(latlon)
        elif coords_type == "array":
            self.waypoints = np.array(coords[0])
            self.zone_number = coords[1]
            self.zone_letter = coords[2]
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

        self.max_x = np.max(self.waypoints[:, 0]) + RESERVE
        self.min_x = np.min(self.waypoints[:, 0]) - RESERVE
        self.max_y = np.max(self.waypoints[:, 1]) + RESERVE
        self.min_y = np.min(self.waypoints[:, 1]) - RESERVE

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

        self.coords_data = CoordsData(self.min_long, self.max_long, self.min_lat, self.max_lat)
        self.points = [
            geometry.Point(x, y)
            for x, y in zip(self.waypoints[:, 0], self.waypoints[:, 1])
        ]

        self.way_node_ids = set()
        self.osm_ways_data = None
        self.osm_rels_data = None
        self.osm_nodes_data = None

        self.roads = set()
        self.footways = set()
        self.barriers = set()
        self.roads_list = []
        self.footways_list = []
        self.barriers_list = []
        self.ways = {}

        self._load_tag_configs()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_tag_configs(self):
        try:
            from ament_index_python.resources import get_resource
            _, package_path = get_resource("packages", "map_data")
            params_path = os.path.join(package_path, "share", "map_data", "parameters")
        except Exception:
            params_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "parameters")
            )

        self.BARRIER_TAGS = self._csv_to_dict(os.path.join(params_path, "barrier_tags.csv"))
        self.NOT_BARRIER_TAGS = self._csv_to_dict(os.path.join(params_path, "not_barrier_tags.csv"))
        self.ANTI_BARRIER_TAGS = self._csv_to_dict(os.path.join(params_path, "anti_barrier_tags.csv"))
        self.OBSTACLE_TAGS = self._csv_to_dict(os.path.join(params_path, "obstacle_tags.csv"))
        self.NOT_OBSTACLE_TAGS = self._csv_to_dict(os.path.join(params_path, "not_obstacle_tags.csv"))

    @staticmethod
    def _csv_to_dict(path):
        arr = np.genfromtxt(path, dtype=str, delimiter=",")
        result = {}
        for row in arr:
            result.setdefault(row[0], []).append(row[1])
        return result

    @staticmethod
    def _latlon_to_utm(latlon):
        easting, northing, zone_number, zone_letter = utm.from_latlon(latlon[:, 0], latlon[:, 1])
        return np.column_stack([easting, northing]), zone_number, zone_letter

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    def get_way_query(self):
        return "[out:json]; (way({}, {}, {}, {}); >; ); out;".format(
            self.min_lat, self.min_long, self.max_lat, self.max_long
        )

    def get_rel_query(self):
        return "[out:json]; (way({}, {}, {}, {}); <; ); out;".format(
            self.min_lat, self.min_long, self.max_lat, self.max_long
        )

    def get_node_query(self):
        return "[out:json]; (node({}, {}, {}, {}); ); out;".format(
            self.min_lat, self.min_long, self.max_lat, self.max_long
        )

    # ------------------------------------------------------------------
    # OSM querying
    # ------------------------------------------------------------------

    def run_queries(self):
        """Download raw OSM data for ways, relations, and nodes."""
        self.osm_ways_data = self._run_query("ways", self.get_way_query())
        self.osm_rels_data = self._run_query("relations", self.get_rel_query())
        self.osm_nodes_data = self._run_query("nodes", self.get_node_query())
        logger.info("All OSM queries finished.")

    def _run_query(self, name, query_str, retries=3, wait=5):
        _api = overpy.Overpass()
        for attempt in range(1, retries + 1):
            endpoint = OVERPASS_ENDPOINTS[self._endpoint_index % len(OVERPASS_ENDPOINTS)]
            self._wait_for_slot(endpoint)
            logger.info(f"OSM query '{name}' via {endpoint} (attempt {attempt}/{retries})")
            try:
                data = _fetch_overpass(endpoint, query_str)
                return _api.parse_json(json.dumps(data))
            except urllib.error.HTTPError as e:
                if e.code in (406, 429):
                    logger.warning(
                        f"Overpass rate limited (HTTP {e.code}) on {endpoint}, "
                        f"waiting 60s then trying next endpoint..."
                    )
                    time.sleep(60)
                else:
                    logger.warning(f"Query '{name}' HTTP {e.code} on {endpoint}: {e}")
                self._endpoint_index += 1
            except Exception as e:
                logger.warning(f"Query '{name}' failed on {endpoint}: {e}")
                self._endpoint_index += 1
                if attempt < retries:
                    time.sleep(wait)
        logger.error(f"OSM query '{name}' failed after {retries} attempts.")
        return None

    @staticmethod
    def _wait_for_slot(endpoint, max_wait=300):
        """For overpass-api.de: parse /api/status and sleep until a slot is free."""
        if "overpass-api.de" not in endpoint:
            return
        status_url = endpoint.replace("/api/interpreter", "/api/status")
        try:
            with urllib.request.urlopen(status_url, timeout=10) as resp:
                text = resp.read().decode()
            m_slots = re.search(r"(\d+) slots available now", text)
            if m_slots and int(m_slots.group(1)) > 0:
                return
            m_wait = re.search(r"in (\d+) seconds", text)
            wait_secs = min(int(m_wait.group(1)) + 2 if m_wait else 60, max_wait)
            logger.info(f"Overpass rate limit active — waiting {wait_secs}s for a free slot...")
            time.sleep(wait_secs)
        except Exception as e:
            logger.debug(f"Could not read Overpass status: {e}")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_ways(self):
        """Populate self.ways from the raw OSM way data."""
        for way in tqdm(self.osm_ways_data.ways, desc="Parse ways"):
            lats = np.array([float(n.lat) for n in way.nodes])
            lons = np.array([float(n.lon) for n in way.nodes])
            easting, northing, _, _ = utm.from_latlon(lats, lons)
            coords = list(zip(easting, northing))

            self.way_node_ids.update(n.id for n in way.nodes)

            is_area = coords[0] == coords[-1]
            self.ways[way.id] = Way(
                id=way.id,
                is_area=is_area,
                nodes=way.nodes,
                tags=way.tags or {},
                line=geometry.Polygon(coords) if is_area else geometry.LineString(coords),
            )

    def parse_rels(self):
        """Apply relations: combine connected ways and propagate relation tags."""
        for rel in tqdm(self.osm_rels_data.relations, desc="Parse rels"):
            keys = self.ways.keys()
            outer_ids, inner_ids = [], []

            for member in rel.members:
                if member._type_value == "way" and int(member.ref) in keys:
                    (outer_ids if member.role == "outer" else inner_ids).append(int(member.ref))

            outer_ids = self.combine_ways(outer_ids)
            rel_tags = rel.tags or {}

            for id in outer_ids:
                self.ways[id].in_out = "outer"
                self.ways[id].tags.update(rel_tags)

            for id in inner_ids:
                self.ways[id].in_out = "inner"

    def parse_nodes(self):
        """Convert standalone obstacle nodes to small circular barrier polygons."""
        for node in tqdm(self.osm_nodes_data.nodes, desc="Parse nodes"):
            if node.id in self.way_node_ids:
                continue
            if any(
                key in self.OBSTACLE_TAGS
                and (
                    node.tags[key] in self.OBSTACLE_TAGS[key]
                    or (
                        "*" in self.OBSTACLE_TAGS[key]
                        and node.tags[key] not in self.NOT_OBSTACLE_TAGS.get(key, [])
                    )
                )
                for key in node.tags
            ):
                easting, northing, _, _ = utm.from_latlon(float(node.lat), float(node.lon))
                self.barriers.add(Way(
                    id=node.id,
                    is_area=True,
                    tags=node.tags,
                    line=geometry.Point(easting, northing).buffer(OBSTACLE_RADIUS),
                ))

    def combine_ways(self, ids):
        """Merge ways that share endpoint nodes into longer continuous ways."""
        ways = [self.ways[id] for id in ids]
        i = 0
        while i < len(ways):
            j = 0
            while j < len(ways):
                if i != j and not ways[i].is_area and not ways[j].is_area:
                    if ways[i].nodes[0].id == ways[j].nodes[0].id:
                        ways[i].nodes.reverse()
                    elif ways[i].nodes[-1].id == ways[j].nodes[-1].id:
                        ways[j].nodes.reverse()

                    if ways[i].nodes[-1].id == ways[j].nodes[0].id:
                        new_nodes = ways[i].nodes + ways[j].nodes[1:]
                        new_tags = {**(ways[i].tags or {}), **(ways[j].tags or {})}
                        new_line = linemerge([ways[i].line, ways[j].line])
                        is_area = new_nodes[0].id == new_nodes[-1].id

                        new_id = int(-(10**15) * np.random.random())
                        while new_id in self.ways:
                            new_id = int(-(10**15) * np.random.random())

                        new_way = Way(
                            id=new_id,
                            is_area=is_area,
                            nodes=new_nodes,
                            tags=new_tags,
                            line=geometry.Polygon(new_line.coords) if is_area else new_line,
                        )
                        self.ways[new_id] = new_way
                        ways[j] = new_way
                        ids[j] = new_id
                        ids.pop(i)
                        ways.pop(i)
                        i -= 1
                        j -= 1
                        break
                j += 1
            i += 1
        return ids

    def separate_ways(self):
        """Classify ways into roads, footways, and barriers."""
        for way in tqdm(self.ways.values(), desc="Separate ways"):
            if way.is_road():
                self.roads.add(self._buffer_line(way, width=7))
            elif way.is_footway():
                self.footways.add(self._buffer_line(way, width=3))
            elif way.is_barrier(self.BARRIER_TAGS, self.NOT_BARRIER_TAGS, self.ANTI_BARRIER_TAGS):
                if not way.is_area:
                    way = self._buffer_line(way, width=2)
                self.barriers.add(way)

    @staticmethod
    def _buffer_line(way, width):
        way.line = way.line.buffer(width / 2)
        way.is_area = True
        return way

    def sets_to_lists(self):
        self.roads_list = list(self.roads)
        self.footways_list = list(self.footways)
        self.barriers_list = list(self.barriers)

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def run_parse(self):
        """Parse downloaded OSM data into roads/footways/barriers. Returns 0 on success, 1 on failure."""
        missing = [
            name for name, data in [
                ("ways", self.osm_ways_data),
                ("relations", self.osm_rels_data),
                ("nodes", self.osm_nodes_data),
            ]
            if data is None
        ]
        if missing:
            logger.error(f"Missing OSM data for: {missing}. Run run_queries() first.")
            return 1

        logger.info("Parsing OSM data.")
        self.parse_ways()
        self.parse_rels()
        self.parse_nodes()
        self.separate_ways()
        self.sets_to_lists()
        logger.info("Parsing finished.")
        return 0

    def run_all(self, save=True):
        """Download OSM data, parse it, and optionally save to disk."""
        self.run_queries()
        if self.run_parse():
            return
        if save:
            self.save_to_pickle()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Raw overpy Result objects are not reliably picklable; clear them.
        # All useful data has already been extracted into roads/footways/barriers.
        state["osm_ways_data"] = None
        state["osm_rels_data"] = None
        state["osm_nodes_data"] = None
        return state

    def save_to_pickle(self, filename=None):
        fn = getattr(self, "coords_file", None) if self.coords_type == "file" else filename
        if fn is None:
            logger.error("No filename given.")
            return
        out_path = fn[:-4] + ".mapdata"
        try:
            with open(out_path, "wb") as fh:
                pickle.dump(self, fh, protocol=2)
            logger.info(f"Map data saved to {out_path}")
        except Exception as e:
            logger.error(f"Error saving map data: {e}")

    # ------------------------------------------------------------------
    # Data accessors (used by osm_cloud)
    # ------------------------------------------------------------------

    def get_points(self, z=0):
        """Return all OSM nodes as {id: utm_column_vector} dict."""
        points = {}
        for node in self.osm_nodes_data.nodes:
            e, n, _, _ = utm.from_latlon(float(node.lat), float(node.lon))
            points[node.id] = np.array([e, n, z]).reshape(3, 1)
        for way in self.osm_ways_data.ways:
            for node in way.nodes:
                e, n, _, _ = utm.from_latlon(float(node.lat), float(node.lon))
                points[node.id] = np.array([e, n, z]).reshape(3, 1)
        return points

    def get_ways(self):
        return {
            "roads": self.roads_list,
            "footways": self.footways_list,
            "barriers": self.barriers_list,
        }

    # ------------------------------------------------------------------
    # Legacy public aliases for backward compatibility
    # ------------------------------------------------------------------

    def waypoints_to_utm(self, waypoints):
        return self._latlon_to_utm(waypoints)

    def csv_to_dict(self, f):
        return self._csv_to_dict(f)

    def point_to_polygon(self, point, r=1):
        return point.buffer(r)

    def line_to_polygon(self, way, width=4):
        return self._buffer_line(way, width)
