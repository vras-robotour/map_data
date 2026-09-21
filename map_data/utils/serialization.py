import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely import geometry, wkt

from map_data.utils.way import Way

if TYPE_CHECKING:
    from map_data.map_data import MapData

#: Version of the ``.mapdata`` JSON schema written by :func:`map_data_to_dict`.
#: Bump it whenever the schema changes; a reader that only knows an older
#: version then says so instead of failing on a key it does not understand.
FORMAT_VERSION = 1

#: Version assumed for a file written before the marker existed. Such files are
#: read as they are: every key added since is optional on load.
LEGACY_FORMAT_VERSION = 0


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
    # Every key is optional: a file written by an older version lacks the ones
    # added since, and the Way defaults are what that version meant by them.
    line = wkt.loads(data["line"]) if data.get("line") else None
    return Way(
        id=data.get("id", -1),
        is_area=data.get("is_area", False),
        nodes=data.get("nodes") or [],
        tags=data.get("tags") or {},
        line=line,
        in_out=data.get("in_out", ""),
    )


def map_data_to_dict(md: "MapData") -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
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


def atomic_write_json(path: str | Path, data: Any, **dump_kwargs: Any) -> None:
    """Dump JSON to a temp file next to ``path``, then replace it, so a crash or
    full disk mid-dump cannot truncate a previously good file. A symlinked ``path``
    is written through to its target (colcon symlink-install data files)."""
    p = Path(path).resolve()
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **dump_kwargs)
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_mapdata(md_class: type["MapData"], path: str | Path) -> "MapData":
    p = Path(path)
    # A legacy pickle file (starts with 0x80) is invalid UTF-8, so json.load
    # already raises a clear (ValueError-subclass) error on it; pickle
    # support has been removed for security reasons regardless.
    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    # No marker means a file written before the schema was versioned; anything
    # newer than this build knows may hold keys whose meaning it would guess wrong.
    version = data.get("format_version", LEGACY_FORMAT_VERSION)
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise ValueError(
            f"{p}: .mapdata format version {version!r} is not supported by this build, which "
            f"reads up to version {FORMAT_VERSION}. Update map_data, or re-save the file with "
            "the version that wrote it.",
        )

    md = md_class.__new__(md_class)
    md.__dict__.update(data["metadata"])  # zone_number/letter, min_/max_ x/y/lat/long, coords_file

    md.osm_ways_data = None
    md.osm_rels_data = None
    md.osm_nodes_data = None

    md.waypoints = np.array(data["waypoints"])
    # Constructed instances expose the waypoints as shapely Points too;
    # restore the attribute for parity (MapData.__init__ builds the same list).
    md.points = (
        [geometry.Point(x, y) for x, y in zip(md.waypoints[:, 0], md.waypoints[:, 1], strict=True)]
        if md.waypoints.ndim == 2
        else []
    )
    md.nodes_cache = {int(k): v for k, v in data["nodes_cache"].items()}

    md.roads_list = [way_from_dict(w) for w in data["roads"]]
    md.footways_list = [way_from_dict(w) for w in data["footways"]]
    md.barriers_list = [way_from_dict(w) for w in data["barriers"]]
    md.crossroads_list = [way_from_dict(w) for w in data.get("crossroads", [])]

    # Restore tag configs and private fields needed for run_queries()/run_parse()
    md._obstacle_radius = None
    md._buffer_widths = None
    md._load_tag_configs()

    return md
