import logging
import numpy as np
import utm
from tqdm import tqdm
import shapely.geometry as geometry
from shapely.ops import linemerge
from typing import Dict, List, Set, Any
from map_data.way import Way

logger = logging.getLogger(__name__)

OBSTACLE_RADIUS = 2


def parse_osm_ways(osm_ways_data: Any, nodes_cache: Dict[int, Any]) -> Dict[int, Way]:
    ways = {}
    for way in tqdm(osm_ways_data.ways, desc="Parse ways"):
        lats = np.array([float(n.lat) for n in way.nodes])
        lons = np.array([float(n.lon) for n in way.nodes])
        easting, northing, _, _ = utm.from_latlon(lats, lons)
        coords = list(zip(easting, northing))

        for n in way.nodes:
            if n.id not in nodes_cache:
                nodes_cache[n.id] = {
                    "lat": float(n.lat),
                    "lon": float(n.lon),
                    "tags": dict(n.tags) if n.tags else {},
                }

        is_area = coords[0] == coords[-1]
        ways[way.id] = Way(
            id=way.id,
            is_area=is_area,
            nodes=[n.id for n in way.nodes],
            tags=dict(way.tags) if way.tags else {},
            line=geometry.Polygon(coords) if is_area else geometry.LineString(coords),
        )
    return ways


def parse_osm_rels(osm_rels_data: Any, ways: Dict[int, Way]):
    for rel in tqdm(osm_rels_data.relations, desc="Parse rels"):
        outer_ids, inner_ids = [], []

        for member in rel.members:
            if member._type_value == "way" and int(member.ref) in ways:
                (outer_ids if member.role == "outer" else inner_ids).append(
                    int(member.ref)
                )

        outer_ids = combine_ways(outer_ids, ways)
        rel_tags = dict(rel.tags) if rel.tags else {}

        for wid in outer_ids:
            ways[wid].in_out = "outer"
            ways[wid].tags.update(rel_tags)

        for wid in inner_ids:
            ways[wid].in_out = "inner"


def parse_osm_nodes(
    osm_nodes_data: Any,
    nodes_cache: Dict[int, Any],
    way_node_ids: Set[int],
    obstacle_tags: Dict[str, List[str]],
    not_obstacle_tags: Dict[str, List[str]],
) -> List[Way]:
    barriers = []
    for node in tqdm(osm_nodes_data.nodes, desc="Parse nodes"):
        if node.id not in nodes_cache:
            nodes_cache[node.id] = {
                "lat": float(node.lat),
                "lon": float(node.lon),
                "tags": dict(node.tags) if node.tags else {},
            }

        if node.id in way_node_ids:
            continue

        is_obstacle = any(
            key in obstacle_tags
            and (
                node.tags[key] in obstacle_tags[key]
                or (
                    "*" in obstacle_tags[key]
                    and node.tags[key] not in not_obstacle_tags.get(key, [])
                )
            )
            for key in node.tags
        )

        if is_obstacle:
            easting, northing, _, _ = utm.from_latlon(float(node.lat), float(node.lon))
            barriers.append(
                Way(
                    id=node.id,
                    is_area=True,
                    tags=dict(node.tags) if node.tags else {},
                    line=geometry.Point(easting, northing).buffer(OBSTACLE_RADIUS),
                )
            )
    return barriers


def combine_ways(ids: List[int], ways: Dict[int, Way]) -> List[int]:
    if not ids:
        return ids

    ways_to_merge = [ways[wid] for wid in ids if wid in ways and not ways[wid].is_area]
    area_ids = [wid for wid in ids if wid in ways and ways[wid].is_area]

    if not ways_to_merge:
        return ids

    endpoint_map = {}
    for way in ways_to_merge:
        # way.nodes is a list of IDs now
        endpoint_map.setdefault(way.nodes[0], []).append(way)
        endpoint_map.setdefault(way.nodes[-1], []).append(way)

    merged_ids = []
    used_ways = set()

    for start_way in ways_to_merge:
        if start_way.id in used_ways:
            continue

        current_nodes = list(start_way.nodes)
        current_tags = dict(start_way.tags)
        current_lines = [start_way.line]
        used_ways.add(start_way.id)

        while True:
            last_node_id = current_nodes[-1]
            possible_next = [
                w for w in endpoint_map.get(last_node_id, []) if w.id not in used_ways
            ]
            if not possible_next:
                break
            next_way = possible_next[0]
            used_ways.add(next_way.id)

            if next_way.nodes[0] == last_node_id:
                current_nodes.extend(next_way.nodes[1:])
            else:
                current_nodes.extend(reversed(next_way.nodes[:-1]))

            current_tags.update(next_way.tags)
            current_lines.append(next_way.line)

            if current_nodes[0] == current_nodes[-1]:
                break

        if current_nodes[0] != current_nodes[-1]:
            while True:
                first_node_id = current_nodes[0]
                possible_prev = [
                    w
                    for w in endpoint_map.get(first_node_id, [])
                    if w.id not in used_ways
                ]
                if not possible_prev:
                    break
                prev_way = possible_prev[0]
                used_ways.add(prev_way.id)

                if prev_way.nodes[-1] == first_node_id:
                    current_nodes = list(prev_way.nodes[:-1]) + current_nodes
                else:
                    current_nodes = list(reversed(prev_way.nodes[1:])) + current_nodes

                current_tags.update(prev_way.tags)
                current_lines.insert(0, prev_way.line)

        if len(current_lines) > 1:
            is_area = current_nodes[0] == current_nodes[-1]
            merged_line = linemerge(current_lines)

            new_id = int(-(10**15) * np.random.random())
            while new_id in ways:
                new_id = int(-(10**15) * np.random.random())

            new_way = Way(
                id=new_id,
                is_area=is_area,
                nodes=current_nodes,
                tags=current_tags,
                line=geometry.Polygon(merged_line.coords) if is_area else merged_line,
            )
            ways[new_id] = new_way
            merged_ids.append(new_id)
        else:
            merged_ids.append(start_way.id)

    return area_ids + merged_ids


def separate_ways(
    ways: Dict[int, Way],
    barrier_tags: Dict[str, List[str]],
    not_barrier_tags: Dict[str, List[str]],
    anti_barrier_tags: Dict[str, List[str]],
):
    roads, footways, barriers = [], [], []
    for way in tqdm(ways.values(), desc="Separate ways"):
        if way.is_road():
            roads.append(buffer_line(way, width=7))
        elif way.is_footway():
            footways.append(buffer_line(way, width=3))
        elif way.is_barrier(barrier_tags, not_barrier_tags, anti_barrier_tags):
            if not way.is_area:
                way = buffer_line(way, width=2)
            barriers.append(way)
    return roads, footways, barriers


def buffer_line(way: Way, width: float) -> Way:
    way.line = way.line.buffer(width / 2)
    way.is_area = True
    return way
