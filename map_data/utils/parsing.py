import logging
from typing import Any

import numpy as np
import overpy
import utm
from shapely import geometry
from shapely.errors import GEOSException
from shapely.ops import linemerge, unary_union
from shapely.validation import make_valid

from map_data.utils.config import load_config
from map_data.utils.way import Way

logger = logging.getLogger(__name__)

_DEFAULTS = load_config("planner_defaults.yaml")
OBSTACLE_RADIUS = _DEFAULTS.get("obstacle_radius", 2.0)
BUFFER_WIDTHS = _DEFAULTS.get("buffer_widths", {"road": 7, "footway": 3, "barrier": 2})
CLASSIFYING_TAGS = frozenset({"highway", "barrier", "building", "natural", "landuse", "waterway"})
LINEAR_BARRIER_TAGS = frozenset({"barrier", "natural"})
LINEAR_BARRIER_VALUES = frozenset(
    {
        "fence",
        "wall",
        "hedge",
        "cliff",
        "city_wall",
        "retaining_wall",
        "ditch",
        "guard_rail",
        "kerb",
        "handrail",
        "cable_barrier",
        "jersey_barrier",
        "chain",
        "rope",
    }
)


def parse_osm_ways(
    osm_ways_data: overpy.Result,
    nodes_cache: dict[int, dict[str, Any]],
    force_zone_number: int | None = None,
    force_zone_letter: str | None = None,
) -> dict[int, Way]:
    ways = {}
    lats: list[float] = []
    lons: list[float] = []
    for way in osm_ways_data.ways:
        lats.extend(float(n.lat) for n in way.nodes)
        lons.extend(float(n.lon) for n in way.nodes)
    eastings, northings, _, _ = utm.from_latlon(
        np.array(lats),
        np.array(lons),
        force_zone_number=force_zone_number,
        force_zone_letter=force_zone_letter,
    )
    cursor = 0
    for way in osm_ways_data.ways:
        end = cursor + len(way.nodes)
        coords = list(zip(eastings[cursor:end], northings[cursor:end], strict=True))
        cursor = end

        for n in way.nodes:
            if n.id not in nodes_cache:
                nodes_cache[n.id] = {
                    "lat": float(n.lat),
                    "lon": float(n.lon),
                    "tags": dict(n.tags) if n.tags else {},
                }

        # A closed linear barrier is a ring, not the area it encloses.
        is_ring = coords[0] == coords[-1]
        is_area = is_ring and (way.tags.get("area") == "yes" or not _is_linear_barrier(way.tags))
        # Skip, not raise: shapely needs 2 coordinates for a LineString and 4
        # (3 distinct + the closing one) for a Polygon.
        min_coords = 4 if is_area else 2
        if len(coords) < min_coords:
            logger.warning(
                "Skipping degenerate way %s: %d coordinate(s) is too few for a %s",
                way.id,
                len(coords),
                "Polygon" if is_area else "LineString",
            )
            continue
        ways[way.id] = Way(
            id=way.id,
            is_area=is_area,
            nodes=[n.id for n in way.nodes],
            tags=dict(way.tags) if way.tags else {},
            line=geometry.Polygon(coords) if is_area else geometry.LineString(coords),
        )
    return ways


#: Relation types whose outer/inner member ways form area rings.
AREA_RELATION_TYPES = frozenset({"multipolygon", "boundary"})


def _is_linear_barrier(tags: dict[str, str]) -> bool:
    return any(tags.get(key) in LINEAR_BARRIER_VALUES for key in LINEAR_BARRIER_TAGS)


def _has_classifying_tags(tags: dict[str, str]) -> bool:
    return any(key in CLASSIFYING_TAGS for key in tags)


def parse_osm_rels(osm_rels_data: overpy.Result, ways: dict[int, Way]) -> None:
    consumed_ids: set[int] = set()
    kept_ids: set[int] = set()

    for rel in osm_rels_data.relations:
        # Only area relations have rings to merge; chaining a route relation would
        # make one way carrying every member's tags (one bridge blocks the route).
        rel_type = (rel.tags or {}).get("type")
        if rel_type not in AREA_RELATION_TYPES:
            continue
        outer_ids: list[int] = []
        inner_ids: list[int] = []

        for member in rel.members:
            if member._type_value == "way" and int(member.ref) in ways:
                # An empty multipolygon member role is treated as "outer".
                if member.role == "outer" or (not member.role and rel_type == "multipolygon"):
                    outer_ids.append(int(member.ref))
                elif member.role == "inner":
                    inner_ids.append(int(member.ref))

        outer_ids = combine_ways(outer_ids, ways, consumed=consumed_ids)
        inner_ids = combine_ways(inner_ids, ways, consumed=consumed_ids)
        kept_ids.update(outer_ids)
        kept_ids.update(inner_ids)
        rel_tags = dict(rel.tags) if rel.tags else {}

        # Inner rings are holes (e.g. a courtyard): subtract them from the outers.
        inner_rings = [
            ways[wid].line
            for wid in inner_ids
            if ways[wid].is_area and isinstance(ways[wid].line, geometry.Polygon)
        ]

        for wid in outer_ids:
            way = ways[wid]
            way.in_out = "outer"
            if not _has_classifying_tags(way.tags):
                way.tags.update(rel_tags)
            if inner_rings and way.is_area and isinstance(way.line, geometry.Polygon):
                holes = [ring for ring in inner_rings if way.line.intersects(ring)]
                if holes:
                    _subtract_holes(way, holes)

        for wid in inner_ids:
            ways[wid].in_out = "inner"

    # Drop merged members so separate_ways does not classify them twice; only
    # after all relations, as another relation may have kept a way unmerged.
    for wid in consumed_ids - kept_ids:
        ways.pop(wid, None)


def parse_osm_nodes(
    osm_nodes_data: overpy.Result,
    nodes_cache: dict[int, dict[str, Any]],
    way_node_ids: set[int],
    obstacle_tags: dict[str, list[str]],
    not_obstacle_tags: dict[str, list[str]],
    anti_obstacle_tags: dict[str, list[str]],
    obstacle_radius: float | None = None,
    force_zone_number: int | None = None,
    force_zone_letter: str | None = None,
) -> list[Way]:
    barriers = []
    for node in osm_nodes_data.nodes:
        if node.id not in nodes_cache:
            nodes_cache[node.id] = {
                "lat": float(node.lat),
                "lon": float(node.lon),
                "tags": dict(node.tags) if node.tags else {},
            }

        if node.id in way_node_ids:
            continue

        if Way(tags=node.tags).is_barrier(obstacle_tags, not_obstacle_tags, anti_obstacle_tags):
            easting, northing, _, _ = utm.from_latlon(
                float(node.lat),
                float(node.lon),
                force_zone_number=force_zone_number,
                force_zone_letter=force_zone_letter,
            )
            radius = obstacle_radius if obstacle_radius is not None else OBSTACLE_RADIUS
            barriers.append(
                Way(
                    id=node.id,
                    is_area=True,
                    tags=dict(node.tags) if node.tags else {},
                    line=geometry.Point(easting, northing).buffer(radius),
                ),
            )
    return barriers


def combine_ways(
    ids: list[int],
    ways: dict[int, Way],
    consumed: set[int] | None = None,
) -> list[int]:
    """
    Merge chains of member ways into new ways added to *ways* under negative ids.

    Returns the ids standing for *ids* afterwards. *consumed* collects the ids of
    member ways a merged way replaces.
    """
    if not ids:
        return ids

    ways_to_merge = []
    area_ids = []
    for wid in ids:
        if wid not in ways:
            continue
        if ways[wid].is_area or _has_classifying_tags(ways[wid].tags):
            area_ids.append(wid)
        else:
            ways_to_merge.append(ways[wid])

    if not ways_to_merge:
        return ids

    endpoint_map: dict[int, list[Way]] = {}
    for way in ways_to_merge:
        endpoint_map.setdefault(way.nodes[0], []).append(way)
        endpoint_map.setdefault(way.nodes[-1], []).append(way)

    merged_ids = []
    used_ways = set()

    def extend(at_end: bool) -> bool:
        """Attach the next unused way at the chain's end (or start); False if none."""
        joint = current_nodes[-1] if at_end else current_nodes[0]
        candidates = [w for w in endpoint_map.get(joint, []) if w.id not in used_ways]
        if not candidates:
            return False
        way = candidates[0]
        used_ways.add(way.id)
        used_ways_this_loop.append(way.id)
        if at_end:
            nodes = way.nodes if way.nodes[0] == joint else way.nodes[::-1]
            current_nodes.extend(nodes[1:])
            current_lines.append(way.line)
        else:
            nodes = way.nodes if way.nodes[-1] == joint else way.nodes[::-1]
            current_nodes[:0] = nodes[:-1]
            current_lines.insert(0, way.line)
        current_tags.update(way.tags)
        return True

    for start_way in ways_to_merge:
        if start_way.id in used_ways:
            continue

        current_nodes = list(start_way.nodes)
        current_tags = dict(start_way.tags)
        current_lines = [start_way.line]
        used_ways.add(start_way.id)
        used_ways_this_loop = [start_way.id]

        while extend(at_end=True):
            if current_nodes[0] == current_nodes[-1]:
                break  # closed ring

        if current_nodes[0] != current_nodes[-1]:
            while extend(at_end=False):
                pass

        if len(current_lines) > 1:
            is_area = current_nodes[0] == current_nodes[-1]
            merged_line = linemerge(current_lines)

            if merged_line.geom_type == "MultiLineString":
                logger.warning("linemerge yielded MultiLineString, keeping member ways unmerged")
                merged_ids.extend(used_ways_this_loop)
                continue

            new_id = -1
            while new_id in ways:
                new_id -= 1

            new_way = Way(
                id=new_id,
                is_area=is_area,
                nodes=current_nodes,
                tags=current_tags,
                line=geometry.Polygon(merged_line.coords) if is_area else merged_line,
            )
            ways[new_id] = new_way
            merged_ids.append(new_id)
            if consumed is not None:
                consumed.update(used_ways_this_loop)
        else:
            merged_ids.append(start_way.id)

    return sorted(area_ids) + merged_ids


def _subtract_holes(way: Way, holes: list[geometry.base.BaseGeometry]) -> None:
    """Subtract *holes* from *way*, accepting a self-intersecting outer ring."""
    if way.line is None:
        return
    try:
        outer = way.line if way.line.is_valid else make_valid(way.line)
        way.line = outer.difference(unary_union(holes))
    except GEOSException as e:
        logger.warning("Could not subtract holes from way %s: %s", way.id, e)


def separate_ways(
    ways: dict[int, Way],
    barrier_tags: dict[str, list[str]],
    not_barrier_tags: dict[str, list[str]],
    anti_barrier_tags: dict[str, list[str]],
    buffer_widths: dict[str, float] | None = None,
) -> tuple[list[Way], list[Way], list[Way]]:
    bw = buffer_widths if buffer_widths is not None else BUFFER_WIDTHS
    roads: list[Way] = []
    footways: list[Way] = []
    barriers: list[Way] = []
    for way in ways.values():
        if way.is_road():
            roads.append(buffer_line(way, width=bw.get("road", 7)))
        elif way.is_footway():
            footways.append(buffer_line(way, width=bw.get("footway", 3)))
        elif way.is_barrier(barrier_tags, not_barrier_tags, anti_barrier_tags):
            if not way.is_area:
                buffer_line(way, width=bw.get("barrier", 2))
            barriers.append(way)
    return roads, footways, barriers


def buffer_line(way: Way, width: float) -> Way:
    line = way.line
    if line is None:
        raise ValueError(f"buffer_line: way {way.id} has no geometry to buffer")
    # A closed road/footway (roundabout) is parsed as a Polygon; buffer its ring,
    # not its interior.
    if isinstance(line, geometry.Polygon) and way.tags.get("area") != "yes":
        line = geometry.LineString(line.exterior.coords)
    way.line = line.buffer(width / 2)
    way.is_area = True
    return way


def ways_to_shapely(ways: list[Way]) -> list[geometry.base.BaseGeometry]:
    """Geometries of the *ways* that have one."""
    return [w.line for w in ways if w.line]
