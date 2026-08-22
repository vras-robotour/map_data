import argparse
import logging
import math
from pathlib import Path

import numpy as np
import utm

from map_data.map_data import MapData
from map_data.utils.config import setup_logging
from map_data.utils.way import Way

logger = logging.getLogger(__name__)

_METADATA_FIELDS = ("zone_number", "zone_letter", "min_x", "max_x", "min_y", "max_y")


def _way_node_ids(way: Way) -> list:
    # way.nodes holds plain ids after JSON round-trip, overpy objects otherwise
    return [getattr(n, "id", n) for n in way.nodes or []]


def _footway_centerline_length(way: Way, md: MapData) -> float:
    """
    Length of a footway's centerline in metres.

    By save time footways have been buffered into Polygons, so ``line.length``
    is the buffer's *perimeter* (~2x the walked distance plus the end caps),
    not the walked distance. Prefer reconstructing the centerline from the
    way's node coordinates via ``nodes_cache``; fall back to inverting the
    round-capped buffer formulas (perimeter ``P = 2L + pi*w``, area
    ``A = L*w + pi*w^2/4`` give ``L = sqrt(P^2 - 4*pi*A) / 2``).
    """
    line = way.line
    if line is None:
        return 0.0
    if line.geom_type == "LineString":
        return float(line.length)

    node_ids = _way_node_ids(way)
    nodes_cache = getattr(md, "nodes_cache", None) or {}
    zone_number = getattr(md, "zone_number", None)
    zone_letter = getattr(md, "zone_letter", None)
    min_nodes = 2
    if len(node_ids) >= min_nodes and zone_number and all(n in nodes_cache for n in node_ids):
        lats = np.array([nodes_cache[n]["lat"] for n in node_ids])
        lons = np.array([nodes_cache[n]["lon"] for n in node_ids])
        easting, northing, _, _ = utm.from_latlon(
            lats, lons, force_zone_number=zone_number, force_zone_letter=zone_letter
        )
        return float(np.sum(np.hypot(np.diff(easting), np.diff(northing))))

    discriminant = line.length**2 - 4.0 * math.pi * line.area
    if discriminant <= 0:
        return 0.0  # ~circular blob, no meaningful centerline
    return math.sqrt(discriminant) / 2.0


def _footway_components(footways: list[Way]) -> int:
    """
    Count connected components of the footway network (ways joined by shared nodes).
    """
    node_ids = [set(_way_node_ids(w)) for w in footways]
    parent = list(range(len(footways)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    node_to_way: dict = {}
    for i, ids in enumerate(node_ids):
        for nid in ids:
            if nid in node_to_way:
                parent[find(i)] = find(node_to_way[nid])
            else:
                node_to_way[nid] = i
    return len({find(i) for i in range(len(footways))})


def validate_mapdata(md: MapData) -> list[str]:
    """
    Check a loaded MapData object for common structural issues.

    Returns a list of human-readable issue descriptions (empty = no issues).
    """
    issues = []

    for field in _METADATA_FIELDS:
        if getattr(md, field, None) is None:
            issues.append(f"metadata: missing '{field}'")

    categories = {
        "road": md.roads_list,
        "footway": md.footways_list,
        "barrier": md.barriers_list,
    }
    if not any(categories.values()):
        issues.append("content: no roads, footways, or barriers")

    seen_ids: dict = {}
    nodes_cache = getattr(md, "nodes_cache", None) or {}
    for cat, ways in categories.items():
        for w in ways:
            if w.line is None:
                issues.append(f"{cat} {w.id}: missing geometry")
            if w.id in seen_ids and seen_ids[w.id] != cat:
                issues.append(f"{cat} {w.id}: duplicate id (also a {seen_ids[w.id]})")
            seen_ids.setdefault(w.id, cat)
            if nodes_cache:
                missing = [n for n in _way_node_ids(w) if n not in nodes_cache]
                if missing:
                    issues.append(f"{cat} {w.id}: {len(missing)} node(s) missing from nodes_cache")

    connected_footways = [w for w in md.footways_list if w.nodes]
    if len(connected_footways) > 1:
        components = _footway_components(connected_footways)
        if components > 1:
            issues.append(f"connectivity: footway network has {components} disconnected components")

    return issues


def validate(path: str) -> int:
    p = Path(path)
    if not p.is_file():
        logger.error("File not found: %s", path)
        return 1

    try:
        md = MapData.load(path)
    except Exception:
        logger.exception("Failed to load map data")
        return 1

    issues = validate_mapdata(md)
    print(f"Validating {p.name}: ", end="")
    if not issues:
        print("no issues found")
        return 0
    print(f"{len(issues)} issue(s)")
    for issue in issues:
        print(f"  - {issue}")
    return 1


def get_stats(path: str) -> None:
    p = Path(path)
    if not p.is_file():
        logger.error("File not found: %s", path)
        return

    try:
        md = MapData.load(path)
    except Exception:
        logger.exception("Failed to load map data")
        return

    print("=" * 40)
    print(f"MAP DATA STATISTICS: {p.name}")
    print("=" * 40)

    source = f"File: {md.coords_file}" if md.coords_file else "Array"
    print(f"Source:      {source}")
    print(f"UTM Zone:    {md.zone_number}{md.zone_letter}")

    area = (md.max_x - md.min_x) * (md.max_y - md.min_y)
    print(f"Bounds X:    [{md.min_x:.1f}, {md.max_x:.1f}]")
    print(f"Bounds Y:    [{md.min_y:.1f}, {md.max_y:.1f}]")
    print(f"Total Area:  {area:,.0f} m²")

    print("-" * 40)
    print(f"Roads:       {len(md.roads_list)}")
    print(f"Footways:    {len(md.footways_list)}")
    print(f"Barriers:    {len(md.barriers_list)}")

    total_footway_len = sum(_footway_centerline_length(w, md) for w in md.footways_list)
    print(f"Total Footway Distance: {total_footway_len:.1f} m")

    # Check for annotations sidecar
    ann_path = p.with_suffix(".annotations.json")
    if ann_path.is_file():
        import json

        with ann_path.open() as f:
            ann_data = json.load(f)
            anns = ann_data.get("annotations", [])
            print(f"Annotations: {len(anns)} (manual edits)")

    print("=" * 40)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Display statistics for a .mapdata file")
    parser.add_argument("file", help="Path to the .mapdata file")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check the file for structural issues (missing geometry/metadata, "
        "disconnected footways); exits non-zero if any are found",
    )
    args = parser.parse_args()

    if args.validate:
        raise SystemExit(validate(args.file))
    get_stats(args.file)


if __name__ == "__main__":
    main()
