import json
import logging
from typing import Any

import numpy as np
from shapely import wkt

from map_data.utils.way import Way

logger = logging.getLogger(__name__)


def way_to_dict(way: Way) -> dict[str, Any]:
    return {
        "id": way.id,
        "is_area": way.is_area,
        "nodes": way.nodes,
        "tags": way.tags,
        "line": wkt.dumps(way.line) if way.line else None,
        "in_out": way.in_out,
    }


def way_from_dict(data: dict[str, Any]) -> Way:
    line = wkt.loads(data["line"]) if data.get("line") else None
    return Way(
        id=data["id"],
        is_area=data["is_area"],
        nodes=data["nodes"],
        tags=data["tags"],
        line=line,
        in_out=data["in_out"],
    )


def map_data_to_dict(md: Any) -> dict[str, Any]:
    return {
        "metadata": {
            "zone_number": md.zone_number,
            "zone_letter": md.zone_letter,
            "min_x": md.min_x,
            "max_x": md.max_x,
            "min_y": md.min_y,
            "max_y": md.max_y,
            "min_lat": md.min_lat,
            "max_lat": md.max_lat,
            "min_long": md.min_long,
            "max_long": md.max_long,
            "coords_file": getattr(md, "coords_file", None),
        },
        "waypoints": md.waypoints.tolist(),
        "roads": [way_to_dict(w) for w in md.roads_list],
        "footways": [way_to_dict(w) for w in md.footways_list],
        "barriers": [way_to_dict(w) for w in md.barriers_list],
        "crossroads": [way_to_dict(w) for w in md.crossroads_list],
        "nodes_cache": md.nodes_cache,
    }


def save_mapdata(md: Any, path: str) -> None:
    data = map_data_to_dict(md)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_mapdata(md_class: Any, path: str) -> Any:
    # Check if it's a legacy pickle file (starts with 0x80)
    with open(path, "rb") as f:
        header = f.read(1)

    if header == b"\x80":
        logger.error(
            "Detected legacy pickle format for %s. "
            "Pickle support has been removed for security reasons. "
            "Please re-parse the data from the original GPX/YAML file.",
            path,
        )
        raise ValueError(f"Legacy pickle format no longer supported: {path}")

    # Try JSON
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    md = md_class.__new__(md_class)

    md.zone_number = meta["zone_number"]
    md.zone_letter = meta["zone_letter"]
    md.min_x = meta["min_x"]
    md.max_x = meta["max_x"]
    md.min_y = meta["min_y"]
    md.max_y = meta["max_y"]
    md.min_lat = meta["min_lat"]
    md.max_lat = meta["max_lat"]
    md.min_long = meta["min_long"]
    md.max_long = meta["max_long"]
    md.coords_file = meta["coords_file"]

    from map_data.map_data import CoordsData  # local import to avoid circular dep

    md.coords_data = CoordsData(md.min_long, md.max_long, md.min_lat, md.max_lat)

    md.osm_ways_data = None
    md.osm_rels_data = None
    md.osm_nodes_data = None

    md.waypoints = np.array(data["waypoints"])
    md.nodes_cache = {int(k): v for k, v in data["nodes_cache"].items()}

    md.roads_list = [way_from_dict(w) for w in data["roads"]]
    md.footways_list = [way_from_dict(w) for w in data["footways"]]
    md.barriers_list = [way_from_dict(w) for w in data["barriers"]]
    md.crossroads_list = [way_from_dict(w) for w in data.get("crossroads", [])]

    return md
