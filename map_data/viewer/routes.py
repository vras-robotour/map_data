import copy
import io
import json
import logging
import os
import re
import select
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import utm
from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)
from shapely import geometry

from map_data.map_data import MapData
from map_data.pathsolver.graph_planner import GraphPlanner
from map_data.pathsolver.replan import (
    ReplanPath,
    cancel_replan_backend,
    load_planner_defaults,
    parse_args,
)
from map_data.utils.parsing import ways_to_shapely
from map_data.utils.serialization import map_data_to_dict
from map_data.utils.way import FOOTWAY_VALUES, Way

from .cache import load_mapdata_cached
from .helpers import (
    apply_node_position_overrides,
    geojson_geom_to_utm,
    geom_to_geojson,
    get_deleted_node_ids,
    get_deleted_way_ids,
    get_node_position_overrides,
    get_split_node_ids,
    load_annotations,
    mapdata_to_geojson,
    migrate_change_log,
    rebuild_way_without_nodes,
    save_annotations,
    split_way,
)

SIGNIFICANT_CHANGE_TOLERANCE = 0.1

logger = logging.getLogger(__name__)
bp = Blueprint("viewer", __name__)


@bp.route("/api/planner_defaults")
def get_planner_defaults() -> Response:
    return jsonify(load_planner_defaults())


_CAT_FOR_LIST = {
    "roads_list": "road",
    "footways_list": "footway",
    "barriers_list": "barrier",
}


def _apply_way_edits(md: MapData, store: dict[str, Any]) -> None:
    """
    Apply deletions, node deletions, splits, and position overrides to md in-place.
    """
    zn, zl = md.zone_number, md.zone_letter
    nodes_cache = getattr(md, "nodes_cache", {})

    deleted_way_ids = get_deleted_way_ids(store)
    has_node_dels = bool(store.get("deleted_nodes"))
    has_splits = bool(store.get("split_ways"))

    if deleted_way_ids or has_node_dels or has_splits:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            cat = _CAT_FOR_LIST[lst_name]
            new_lst = []
            for w in getattr(md, lst_name):
                if w.id in deleted_way_ids:
                    continue
                del_nids = get_deleted_node_ids(store, w.id)
                if del_nids:
                    w = rebuild_way_without_nodes(w, del_nids, zn, zl, nodes_cache, category=cat)  # noqa: PLW2901
                    if w is None:
                        continue
                split_nids = get_split_node_ids(store, w.id)
                if split_nids:
                    segments = split_way(w, split_nids, zn, zl, nodes_cache)
                    for i, seg in enumerate(segments):
                        virtual_id = f"{w.id}:{i}"
                        seg.id = virtual_id
                        if virtual_id in deleted_way_ids:
                            continue
                        seg_del_nids = get_deleted_node_ids(store, virtual_id)
                        if seg_del_nids:
                            seg = rebuild_way_without_nodes(  # noqa: PLW2901
                                seg, seg_del_nids, zn, zl, nodes_cache, category=cat,
                            )
                            if seg is None:
                                continue
                        new_lst.append(seg)
                else:
                    new_lst.append(w)
            setattr(md, lst_name, new_lst)
        md.crossroads_list = md.parse_intersections({str(w.id): w for w in md.footways_list})

    node_pos_store = store.get("node_position_overrides", {})
    if node_pos_store:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            new_lst = []
            for w in getattr(md, lst_name):
                ov = get_node_position_overrides(store, w.id)
                if ov:
                    result = apply_node_position_overrides(
                        w,
                        ov,
                        zn,
                        zl,
                        nodes_cache,
                        category=_CAT_FOR_LIST[lst_name],
                    )
                    new_lst.append(result or w)
                else:
                    new_lst.append(w)
            setattr(md, lst_name, new_lst)


def _get_data_dir() -> Path:
    if current_app.config.get("DATA_DIR"):
        return Path(current_app.config["DATA_DIR"])
    try:
        from ament_index_python.resources import get_resource

        _, pkg = get_resource("packages", "map_data")
        return Path(pkg) / "share" / "map_data" / "data"
    except (ImportError, LookupError):
        # Fallback to the local data directory relative to this file
        return (Path(__file__).parent / ".." / ".." / "data").resolve()


def _safe_data_path(filename: str) -> Path:
    """
    Resolve a user-supplied filename within the data directory.

    Aborts with 400 if the resolved path would escape the data directory
    (e.g. via '../' traversal sequences).
    """
    data_dir = _get_data_dir().resolve()
    resolved = (data_dir / filename).resolve()
    if not (resolved == data_dir or data_dir in resolved.parents):
        abort(400, "Invalid file path")
    return resolved


def _annotation_path(filename: str) -> Path:
    base = Path(filename).stem
    return _get_data_dir().resolve() / f"{base}.annotations.json"


@bp.route("/")
def index() -> str:
    api_key_thunderforest = os.getenv("THUNDERFOREST_API_KEY")
    api_key_seznam = os.getenv("SEZNAM_API_KEY")
    return render_template(
        "index.html",
        apikey_thunderforest=api_key_thunderforest,
        apikey_seznam=api_key_seznam,
    )


@bp.route("/api/files")
def list_files() -> Response:
    result = {"mapdata": [], "gpx": []}
    try:
        for p in sorted(_get_data_dir().iterdir()):
            if p.suffix == ".mapdata":
                result["mapdata"].append(p.name)
            elif p.suffix == ".gpx":
                result["gpx"].append(p.name)
    except FileNotFoundError:
        pass
    return jsonify(result)


@bp.route("/api/mapdata")
def get_mapdata() -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = _safe_data_path(filename)
    if not path.is_file():
        abort(404, f"File not found: {filename}")
    map_data = copy.copy(load_mapdata_cached(str(path)))
    store = load_annotations(str(_annotation_path(filename)))

    _apply_way_edits(map_data, store)

    geojson = mapdata_to_geojson(map_data)
    tag_overrides = store.get("tag_overrides", {})
    if tag_overrides:
        for f in geojson["features"]:
            ov = tag_overrides.get(str(f["properties"].get("id", "")))
            if ov:
                merged = {**(f["properties"].get("tags") or {}), **ov}
                f["properties"]["tags"] = merged
                cat = f["properties"].get("category")
                if cat in ("road", "footway"):
                    hw = merged.get("highway", "")
                    f["properties"]["category"] = "footway" if hw in FOOTWAY_VALUES else "road"
    return jsonify(geojson)


@bp.route("/api/annotations")
def get_annotations() -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    return jsonify(load_annotations(_annotation_path(filename)))


@bp.route("/api/annotations", methods=["POST"])
def add_annotation() -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    ann = {
        "id": str(uuid.uuid4()),
        "type": body.get("type", "obstacle"),
        "geometry": body["geometry"],
        "properties": body.get("properties", {}),
    }
    store["annotations"].append(ann)
    save_annotations(ann_path, store)
    return jsonify(ann), 201


@bp.route("/api/annotations/<ann_id>", methods=["PUT"])
def update_annotation(ann_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    ann = next((a for a in store["annotations"] if a["id"] == ann_id), None)
    if ann is None:
        abort(404, "Annotation not found")
    ann["geometry"] = body["geometry"]
    if "type" in body:
        ann["type"] = body["type"]
    if "properties" in body:
        ann["properties"] = body["properties"]
    save_annotations(ann_path, store)
    return jsonify(ann)


@bp.route("/api/annotations/<ann_id>", methods=["DELETE"])
def delete_annotation(ann_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    before = len(store["annotations"])
    store["annotations"] = [a for a in store["annotations"] if a["id"] != ann_id]
    if len(store["annotations"]) == before:
        abort(404, "Annotation not found")
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/fetch_area", methods=["POST"])
def fetch_area() -> Response:
    body = request.get_json(force=True) or {}
    for field in ("min_lat", "min_lon", "max_lat", "max_lon", "name"):
        if field not in body:
            abort(400, f"Missing field: {field}")

    if body["min_lat"] >= body["max_lat"] or body["min_lon"] >= body["max_lon"]:
        abort(400, "min_lat/min_lon must be strictly less than max_lat/max_lon")

    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(body["name"]).strip())
    if not name:
        abort(400, "name is empty after sanitizing")

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{name}.mapdata"

    corners = np.array(
        [
            [body["min_lat"], body["min_lon"]],
            [body["min_lat"], body["max_lon"]],
            [body["max_lat"], body["min_lon"]],
            [body["max_lat"], body["max_lon"]],
        ],
    )
    easting, northing, zone_number, zone_letter = utm.from_latlon(corners[:, 0], corners[:, 1])
    waypoints = np.column_stack([easting, northing])

    md = MapData([waypoints, zone_number, zone_letter], coords_type="array")
    md.run_queries()
    if any(d is None for d in (md.osm_ways_data, md.osm_rels_data, md.osm_nodes_data)):
        abort(503, "Overpass API unavailable — try again later")

    if md.run_parse() != 0:
        abort(500, "Parsing failed")

    md.save(out_path)

    return jsonify(
        {
            "filename": f"{name}.mapdata",
            "roads": len(md.roads_list),
            "footways": len(md.footways_list),
            "barriers": len(md.barriers_list),
            "crossroads": len(md.crossroads_list),
        },
    )


@bp.route("/api/upload_gpx", methods=["POST"])
def upload_gpx() -> Response:
    if "file" not in request.files:
        abort(400, "No file part")
    file = request.files["file"]
    if file.filename == "":
        abort(400, "No selected file")

    name = request.form.get("name")
    if not name:
        name = Path(file.filename).stem

    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
    if not name:
        abort(400, "name is empty after sanitizing")

    data_dir = _get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary file to avoid saving the GPX to the data directory
    with tempfile.NamedTemporaryFile(suffix=".gpx", delete=False) as tmp:
        file.save(tmp.name)
        gpx_tmp_path = Path(tmp.name)

    try:
        md = MapData(str(gpx_tmp_path), coords_type="file")
        # Restore the original filename for metadata purposes
        md.coords_file = file.filename

        md.run_queries()
        if any(d is None for d in (md.osm_ways_data, md.osm_rels_data, md.osm_nodes_data)):
            abort(503, "Overpass API unavailable — try again later")

        if md.run_parse() != 0:
            abort(500, "Parsing failed")

        out_path = data_dir / f"{name}.mapdata"
        md.save(str(out_path))

        return jsonify(
            {
                "filename": f"{name}.mapdata",
                "roads": len(md.roads_list),
                "footways": len(md.footways_list),
                "barriers": len(md.barriers_list),
                "crossroads": len(md.crossroads_list),
            },
        )
    except Exception as e:
        logger.exception("Error processing GPX upload")
        abort(500, str(e))
    finally:
        if gpx_tmp_path.exists():
            gpx_tmp_path.unlink()


@bp.route("/api/way_nodes")
def get_way_nodes() -> Response:
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing 'file' or 'way_id' query parameter")
    path = _safe_data_path(filename)
    if not path.is_file():
        abort(404, f"File not found: {filename}")

    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))

    original_way_id_str = str(way_id).split(":")[0]
    try:
        search_id = int(original_way_id_str)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer or virtual ID")

    way = None
    category = None
    for lst_name, cat in (
        ("roads_list", "road"),
        ("footways_list", "footway"),
        ("barriers_list", "barrier"),
    ):
        for w in getattr(md, lst_name):
            if w.id == search_id:
                way = copy.copy(w)
                category = cat
                break
        if way:
            break

    if way is None:
        abort(404, f"Way {search_id} not found")

    zn, zl = md.zone_number, md.zone_letter
    nodes_cache = getattr(md, "nodes_cache", {})

    # Apply deletions and overrides (consistent with get_way)
    del_nids = get_deleted_node_ids(store, search_id)
    if del_nids:
        way = rebuild_way_without_nodes(way, del_nids, zn, zl, nodes_cache, category=category)
        if way is None:
            return jsonify({"way_id": way_id, "nodes": []})

    pos_overrides = get_node_position_overrides(store, search_id)
    if pos_overrides:
        way = (
            apply_node_position_overrides(
                way, pos_overrides, zn, zl, nodes_cache, category=category,
            )
            or way
        )

    # Handle split segments if it's a virtual ID
    if ":" in str(way_id):
        try:
            segment_idx = int(str(way_id).split(":")[1])
        except ValueError:
            abort(400, "Invalid virtual ID")

        split_nids = get_split_node_ids(store, search_id)
        if split_nids:
            segments = split_way(way, split_nids, zn, zl, nodes_cache)
            if segment_idx < len(segments):
                way = segments[segment_idx]
                # Apply segment-specific deletions
                seg_del_nids = get_deleted_node_ids(store, way_id)
                if seg_del_nids:
                    way = rebuild_way_without_nodes(
                        way, seg_del_nids, zn, zl, nodes_cache, category=category,
                    )
                    if way is None:
                        return jsonify({"way_id": way_id, "nodes": []})
            else:
                abort(404, "Segment not found")

    nodes = []
    geom_latlon = None
    for i, nid_obj in enumerate(way.nodes):
        nid = getattr(nid_obj, "id", nid_obj)
        if nid in nodes_cache:
            nd = nodes_cache[nid]
            nodes.append({"id": nid, "lat": nd["lat"], "lon": nd["lon"], "tags": nd["tags"]})
        else:
            # Fallback to geometry
            if geom_latlon is None:
                geom = way.line
                raw = list(geom.exterior.coords if hasattr(geom, "exterior") else geom.coords)
                geom_latlon = [utm.to_latlon(e, n, zn, zl) for e, n in raw]
            if i < len(geom_latlon):
                lat, lon = geom_latlon[i]
                nodes.append({"id": nid, "lat": lat, "lon": lon, "tags": {}})

    # centroid fallback
    if not nodes and way.line:
        centroid = way.line.centroid
        lat, lon = utm.to_latlon(centroid.x, centroid.y, zn, zl)
        nodes = [{"id": search_id, "lat": lat, "lon": lon, "tags": {}}]

    # Ensure position overrides are applied to the resulting nodes list
    if pos_overrides:
        for n in nodes:
            if n["id"] in pos_overrides:
                n["lat"] = pos_overrides[n["id"]]["lat"]
                n["lon"] = pos_overrides[n["id"]]["lon"]

    return jsonify({"way_id": way_id, "nodes": nodes})


@bp.route("/api/ways/<way_id>")
def get_way(way_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = _safe_data_path(filename)
    if not path.is_file():
        abort(404, f"File not found: {filename}")

    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))
    nodes_cache = getattr(md, "nodes_cache", {})

    way = None
    category = None

    # Virtual ID handling: split by colon
    original_way_id_str = str(way_id).split(":")[0]
    try:
        search_id = int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    for lst_name, cat in (
        ("roads_list", "road"),
        ("footways_list", "footway"),
        ("barriers_list", "barrier"),
    ):
        for w in getattr(md, lst_name):
            if w.id == search_id:
                way = copy.copy(w)
                category = cat
                break
        if way:
            break

    if way is None:
        abort(404, f"Way {way_id} not found")

    zn, zl = md.zone_number, md.zone_letter
    del_nids = get_deleted_node_ids(store, search_id)
    if del_nids:
        way = rebuild_way_without_nodes(
            way, del_nids, zn, zl, getattr(md, "nodes_cache", {}), category=category,
        )
        if way is None:
            abort(404, f"Way {way_id} reduced to nothing by node deletions")

    pos_overrides = get_node_position_overrides(store, search_id)
    if pos_overrides:
        way = (
            apply_node_position_overrides(
                way,
                pos_overrides,
                zn,
                zl,
                getattr(md, "nodes_cache", {}),
                category=category,
            )
            or way
        )

    # Handle split segments if it's a virtual ID
    if ":" in str(way_id):
        try:
            segment_idx = int(str(way_id).split(":")[1])
        except ValueError:
            abort(400, "Invalid virtual ID")

        split_nids = get_split_node_ids(store, search_id)
        if split_nids:
            segments = split_way(way, split_nids, zn, zl, nodes_cache)
            if segment_idx < len(segments):
                way = segments[segment_idx]
                # Apply segment-specific deletions
                seg_del_nids = get_deleted_node_ids(store, way_id)
                if seg_del_nids:
                    way = rebuild_way_without_nodes(
                        way, seg_del_nids, zn, zl, nodes_cache, category=category,
                    )
                    if way is None:
                        abort(
                            404,
                            "Segment reduced to nothing by segment-specific deletions",
                        )
            else:
                abort(404, "Segment not found")

    geom = geom_to_geojson(way.line, zn, zl)
    if geom is None:
        abort(500, "Could not convert geometry")

    feature = {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "id": way_id,
            "category": category,
            "is_node": category == "barrier" and not bool(way.nodes),
            "tags": way.tags or {},
            "in_out": way.in_out,
        },
    }

    ov = store.get("tag_overrides", {}).get(original_way_id_str)
    if ov:
        merged = {**(feature["properties"]["tags"]), **ov}
        feature["properties"]["tags"] = merged
        if category in ("road", "footway"):
            hw = merged.get("highway", "")
            feature["properties"]["category"] = "footway" if hw in FOOTWAY_VALUES else "road"

    return jsonify(feature)


@bp.route("/api/ways/<way_id>", methods=["DELETE"])
def delete_way(way_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    way_id_str = str(way_id)
    original_way_id_str = way_id_str.split(":")[0]
    try:
        int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    # Segments (virtual IDs like "123:0") are stored as strings so that only the
    # specific segment is suppressed on reload, not the whole original way.
    stored_id: int | str = way_id_str if ":" in way_id_str else int(original_way_id_str)

    body = request.get_json(force=True) or {}
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    migrate_change_log(store)
    if stored_id not in get_deleted_way_ids(store):
        store.setdefault("deleted_ways", []).append(
            {
                "id": stored_id,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
            },
        )
        cl = store.setdefault("change_log", [])
        if not any(e.get("type") == "way" and e.get("id") == stored_id for e in cl):
            cl.append({"type": "way", "id": stored_id, "ts": time.time()})
    save_annotations(ann_path, store)
    return Response("", 204)


@bp.route("/api/ways/<way_id>/tags", methods=["PUT"])
def update_way_tags(way_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    original_way_id_str = str(way_id).split(":")[0]

    body = request.get_json(force=True) or {}
    tags = body.get("tags")
    if not isinstance(tags, dict):
        abort(400, "Request body must include 'tags' dict")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    migrate_change_log(store)
    store.setdefault("tag_overrides", {})[original_way_id_str] = tags
    store.setdefault("tag_override_meta", {})[original_way_id_str] = {
        "category": body.get("category", "unknown"),
        "label": body.get("label", ""),
    }
    cl = store.setdefault("change_log", [])
    if not any(e.get("type") == "tag" and e.get("id") == original_way_id_str for e in cl):
        cl.append({"type": "tag", "id": original_way_id_str, "ts": time.time()})
    save_annotations(ann_path, store)
    return Response("", 204)


@bp.route("/api/ways/<way_id>/tags", methods=["DELETE"])
def delete_way_tags(way_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    store.get("tag_overrides", {}).pop(str(way_id), None)
    store.get("tag_override_meta", {}).pop(str(way_id), None)
    cl = store.get("change_log", [])
    store["change_log"] = [e for e in cl if not (e.get("type") == "tag" and e.get("id") == way_id)]
    save_annotations(ann_path, store)
    return Response("", 204)


@bp.route("/api/ways/<way_id>/segments")
def get_way_segments(way_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    original_way_id = str(way_id).split(":")[0]
    segments = _get_way_segments_geojson(filename, original_way_id)
    return jsonify({"segments": segments})


def _get_way_segments_geojson(filename: str, original_way_id: str) -> list[dict[str, Any]]:
    path = _safe_data_path(filename)
    md = load_mapdata_cached(str(path))
    store = load_annotations(str(_annotation_path(filename)))

    original_way = None
    category = None
    search_id = int(original_way_id)

    for lst_name, cat in (
        ("roads_list", "road"),
        ("footways_list", "footway"),
        ("barriers_list", "barrier"),
    ):
        for w in getattr(md, lst_name):
            if w.id == search_id:
                original_way = copy.copy(w)
                category = cat
                break
        if original_way:
            break

    if not original_way:
        return []

    # Apply node deletions first (as in get_mapdata)
    zn, zl = md.zone_number, md.zone_letter
    del_nids = get_deleted_node_ids(store, search_id)
    if del_nids:
        original_way = rebuild_way_without_nodes(
            original_way,
            del_nids,
            zn,
            zl,
            getattr(md, "nodes_cache", {}),
            category=category,
        )
        if original_way is None:
            return []

    # Apply position overrides
    pos_overrides = get_node_position_overrides(store, search_id)
    if pos_overrides:
        original_way = (
            apply_node_position_overrides(
                original_way,
                pos_overrides,
                zn,
                zl,
                getattr(md, "nodes_cache", {}),
                category=category,
            )
            or original_way
        )

    split_nids = get_split_node_ids(store, search_id)
    segments = split_way(original_way, split_nids, zn, zl, md.nodes_cache)

    features = []
    tag_overrides = store.get("tag_overrides", {})
    ov = tag_overrides.get(str(original_way_id))

    for i, seg in enumerate(segments):
        virtual_id = f"{original_way_id}:{i}"

        # Apply segment-specific deletions to segment geometry
        seg_del_nids = get_deleted_node_ids(store, virtual_id)
        if seg_del_nids:
            seg = rebuild_way_without_nodes(  # noqa: PLW2901
                seg, seg_del_nids, zn, zl, md.nodes_cache, category=category,
            )
            if seg is None:
                continue

        feat_cat = category
        tags = seg.tags or {}
        if ov:
            tags = {**tags, **ov}
            if feat_cat in ("road", "footway"):
                hw = tags.get("highway", "")
                feat_cat = "footway" if hw in FOOTWAY_VALUES else "road"

        features.append(
            {
                "type": "Feature",
                "geometry": geom_to_geojson(seg.line, zn, zl),
                "properties": {
                    "id": virtual_id,
                    "category": feat_cat,
                    "is_node": feat_cat == "barrier" and not bool(seg.nodes),
                    "tags": tags,
                    "in_out": seg.in_out,
                },
            },
        )
    return features


@bp.route("/api/ways/split", methods=["POST"])
def split_way_endpoint() -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True) or {}
    way_id = body.get("way_id")
    node_id = body.get("node_id")

    try:
        way_id_val = str(way_id)
        node_id_int = int(node_id)
    except (ValueError, TypeError):
        abort(400, "Invalid way_id or node_id")

    # If way_id is virtual (e.g. 123:0), get original ID
    original_way_id = way_id_val.split(":", maxsplit=1)[0]

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    migrate_change_log(store)
    splits = store.setdefault("split_ways", {})
    way_splits = splits.setdefault(original_way_id, [])

    if node_id_int not in way_splits:
        way_splits.append(node_id_int)
        cl = store.setdefault("change_log", [])
        if not any(
            e.get("type") == "split"
            and e.get("way_id") == int(original_way_id)
            and e.get("node_id") == node_id_int
            for e in cl
        ):
            cl.append(
                {
                    "type": "split",
                    "way_id": int(original_way_id),
                    "node_id": node_id_int,
                    "ts": time.time(),
                },
            )

    save_annotations(ann_path, store)

    segments = _get_way_segments_geojson(filename, original_way_id)
    return jsonify({"success": True, "segments": segments})


@bp.route("/api/ways/split", methods=["DELETE"])
def undo_way_split() -> Response:
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    node_id = request.args.get("node_id")
    if not filename or way_id is None or node_id is None:
        abort(400, "Missing required query parameters")
    try:
        way_id_int = int(way_id)
        node_id_int = int(node_id)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    splits = store.get("split_ways", {})
    if str(way_id_int) in splits:
        splits[str(way_id_int)] = [nid for nid in splits[str(way_id_int)] if nid != node_id_int]
        if not splits[str(way_id_int)]:
            del splits[str(way_id_int)]

    cl = store.get("change_log", [])
    store["change_log"] = [
        e
        for e in cl
        if not (
            e.get("type") == "split"
            and e.get("way_id") == way_id_int
            and e.get("node_id") == node_id_int
        )
    ]
    save_annotations(ann_path, store)

    segments = _get_way_segments_geojson(filename, str(way_id_int))
    return jsonify({"segments": segments})


@bp.route("/api/ways/<way_id>/hide", methods=["PUT"])
def hide_way(way_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    body = request.get_json(force=True) or {}
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    hw = store.setdefault("hidden_ways", [])
    existing_ids = {(d["id"] if isinstance(d, dict) else d) for d in hw}
    if way_id_int not in existing_ids:
        hw.append(
            {
                "id": way_id_int,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
            },
        )
    save_annotations(ann_path, store)
    return Response("", 204)


@bp.route("/api/ways/<way_id>/show", methods=["PUT"])
def show_way(way_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    hw = store.get("hidden_ways", [])
    store["hidden_ways"] = [d for d in hw if (d["id"] if isinstance(d, dict) else d) != way_id_int]
    save_annotations(ann_path, store)
    return Response("", 204)


@bp.route("/api/ways/<way_id>/restore", methods=["PUT"])
def restore_way(way_id: str) -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    way_id_str = str(way_id)
    original_way_id_str = way_id_str.split(":")[0]
    try:
        int(original_way_id_str)
    except ValueError:
        abort(400, "Invalid way ID")

    stored_id: int | str = way_id_str if ":" in way_id_str else int(original_way_id_str)

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    dw = store.get("deleted_ways", [])
    store["deleted_ways"] = [d for d in dw if (d["id"] if isinstance(d, dict) else d) != stored_id]
    cl = store.get("change_log", [])
    store["change_log"] = [
        e for e in cl if not (e.get("type") == "way" and e.get("id") == stored_id)
    ]
    save_annotations(ann_path, store)
    return Response("", 204)


@bp.route("/api/way_node", methods=["DELETE"])
def delete_way_node() -> Response:
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    node_id_arg = request.args.get("node_id")
    if not filename or way_id is None or node_id_arg is None:
        abort(400, "Missing required query parameters")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
        node_id = int(node_id_arg)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers (way_id can be virtual)")

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    migrate_change_log(store)

    dn = store.setdefault("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
        store["deleted_nodes"] = dn

    # Use the full way_id (could be virtual like "123:0") to allow segment-specific deletion
    target_id = way_id if ":" in str(way_id) else way_id_int

    if node_id not in get_deleted_node_ids(store, target_id):
        dn.append({"way_id": target_id, "node_id": node_id})
        cl = store.setdefault("change_log", [])
        if not any(
            e.get("type") == "node"
            and str(e.get("way_id")) == str(target_id)
            and e.get("node_id") == node_id
            for e in cl
        ):
            cl.append(
                {
                    "type": "node",
                    "way_id": target_id,
                    "node_id": node_id,
                    "ts": time.time(),
                },
            )
    save_annotations(ann_path, store)
    return Response("", 204)


@bp.route("/api/way_node/restore", methods=["PUT"])
def restore_way_node() -> Response:
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    node_id_arg = request.args.get("node_id")
    if not filename or way_id is None or node_id_arg is None:
        abort(400, "Missing required query parameters")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
        node_id = int(node_id_arg)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers (way_id can be virtual)")

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)

    # Use the full way_id to match the deletion record
    target_id = way_id if ":" in str(way_id) else way_id_int

    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
    store["deleted_nodes"] = [
        d
        for d in dn
        if not (str(d.get("way_id")) == str(target_id) and d.get("node_id") == node_id)
    ]
    cl = store.get("change_log", [])
    store["change_log"] = [
        e
        for e in cl
        if not (
            e.get("type") == "node"
            and str(e.get("way_id")) == str(target_id)
            and e.get("node_id") == node_id
        )
    ]
    save_annotations(ann_path, store)
    return Response("", 204)


@bp.route("/api/way_nodes/move", methods=["PUT"])
def move_way_nodes() -> Response:
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing required query parameters")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer")

    body = request.get_json(force=True) or {}
    nodes = body.get("nodes")
    if not isinstance(nodes, list):
        abort(400, "Request body must include 'nodes' list")
    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    overrides = store.setdefault("node_position_overrides", {})
    way_key = original_way_id_str
    if way_key not in overrides:
        overrides[way_key] = {}
    for n in nodes:
        overrides[way_key][str(n["id"])] = {
            "lat": float(n["lat"]),
            "lon": float(n["lon"]),
        }
    migrate_change_log(store)
    cl = store.setdefault("change_log", [])
    if not any(e.get("type") == "move" and e.get("id") == way_id_int for e in cl):
        cl.append(
            {
                "type": "move",
                "id": way_id_int,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
                "ts": time.time(),
            },
        )
    save_annotations(ann_path, store)
    return Response("", 204)


@bp.route("/api/way_nodes/move", methods=["DELETE"])
def undo_move_way_nodes() -> Response:
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing required query parameters")

    original_way_id_str = str(way_id).split(":")[0]
    try:
        way_id_int = int(original_way_id_str)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer or virtual ID")

    ann_path = str(_annotation_path(filename))
    store = load_annotations(ann_path)
    store.get("node_position_overrides", {}).pop(original_way_id_str, None)
    cl = store.get("change_log", [])
    store["change_log"] = [
        e for e in cl if not (e.get("type") == "move" and e.get("id") == way_id_int)
    ]
    save_annotations(ann_path, store)
    return Response("", 204)


def get_merged_mapdata(filename: str) -> tuple[MapData | None, dict[str, Any] | None]:
    path = _safe_data_path(filename)
    if not path.is_file():
        return None, None

    store = load_annotations(str(_annotation_path(filename)))
    md = copy.deepcopy(load_mapdata_cached(str(path)))
    zn, zl = md.zone_number, md.zone_letter

    _apply_way_edits(md, store)

    tag_overrides = store.get("tag_overrides", {})
    if tag_overrides:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            for w in getattr(md, lst_name):
                original_id = str(w.id).split(":")[0]
                ov = tag_overrides.get(original_id)
                if ov:
                    w.tags = {**(w.tags or {}), **ov}
        new_roads, new_footways = [], []
        for w in md.roads_list:
            (new_footways if w.is_footway() else new_roads).append(w)
        for w in md.footways_list:
            (new_roads if w.is_road() else new_footways).append(w)
        md.roads_list, md.footways_list = new_roads, new_footways
        md.crossroads_list = md.parse_intersections({str(w.id): w for w in md.footways_list})

    ann_id = -1
    node_id = -1
    if not hasattr(md, "nodes_cache") or md.nodes_cache is None:
        md.nodes_cache = {}
    for ann in store.get("annotations", []):
        geom = geojson_geom_to_utm(ann["geometry"], zn, zl)
        if geom is None:
            continue
        props = ann.get("properties", {})
        ann_type = ann.get("type", "obstacle")

        w = Way()
        w.id = ann_id
        ann_id -= 1
        w.line = geom
        w.nodes = []
        w.in_out = ""

        if ann_type == "path":
            hw = props.get("highway", "path")
            w.tags = {"highway": hw}
            if "width" in props:
                w.tags["width"] = str(props["width"])
            for k, v in props.items():
                if k not in ("highway", "width"):
                    w.tags[k] = str(v)
            if geom.geom_type == "LineString":
                width_m = float(props.get("width", 1.5))
                for e_coord, n_coord in geom.coords:
                    lat, lon = utm.to_latlon(e_coord, n_coord, zn, zl)
                    md.nodes_cache[node_id] = {"lat": lat, "lon": lon, "tags": {}}
                    w.nodes.append(node_id)
                    node_id -= 1
                w.line = geom.buffer(width_m / 2)
                w.is_area = True
            (md.roads_list if w.is_road() else md.footways_list).append(w)
        else:
            w.tags = {"barrier": props.get("barrier", "wall")}
            for k, v in props.items():
                if k != "barrier":
                    w.tags[k] = str(v)
            md.barriers_list.append(w)

    return md, store


@bp.route("/api/export")
def export_mapdata() -> Response:
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File not found: {filename}")

    buf = io.BytesIO()
    buf.write(json.dumps(map_data_to_dict(md), indent=2).encode("utf-8"))
    buf.seek(0)
    base = Path(filename).stem
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{base}.exported.mapdata",
        mimetype="application/json",
    )


@bp.route("/api/cost_grid")
def get_cost_grid() -> Response:
    filename = request.args.get("file")
    min_lat = request.args.get("min_lat", type=float)
    min_lon = request.args.get("min_lon", type=float)
    max_lat = request.args.get("max_lat", type=float)
    max_lon = request.args.get("max_lon", type=float)

    if not all([filename, min_lat, min_lon, max_lat, max_lon]):
        abort(400, "Missing required parameters")
    if filename is None:
        abort(400, "Filename cannot be None")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File {filename} not found")

    zn, zl = md.zone_number, md.zone_letter
    p1 = utm.from_latlon(min_lat, min_lon, zn, zl)
    p2 = utm.from_latlon(max_lat, max_lon, zn, zl)

    low = (min(p1[0], p2[0]), min(p1[1], p2[1]))
    high = (max(p1[0], p2[0]), max(p1[1], p2[1]))

    args = parse_args([])
    args.low = low
    args.high = high
    args.cell_size = 1.0  # Use a coarser grid for visualization performance
    args.inflate_obstacles = 0.0

    obstacles = ways_to_shapely(md.barriers_list)
    replanner = ReplanPath(args, obstacles)

    # Get custom highway costs from request if provided
    highway_costs = request.args.get("highway_costs")
    if highway_costs:
        try:
            custom_costs = json.loads(highway_costs)
            replanner.HIGHWAY_COSTS = custom_costs
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse custom highway costs: %s", e)

    surface_costs = request.args.get("surface_costs")
    if surface_costs:
        try:
            custom_surf_costs = json.loads(surface_costs)
            replanner.SURFACE_COSTS = custom_surf_costs
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse custom surface costs: %s", e)

    replanner.fill_grid(md, highway_types=["footway", "road"])

    grid = replanner.grid  # [N, 4] -> [x, y, 0, cost]
    # Do not filter out obstacles (cost >= 1.0) so they can be visualized
    visible_grid = grid

    features = []
    for row in visible_grid:
        lat, lon = utm.to_latlon(row[0], row[1], zn, zl)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"cost": float(row[3])},
            },
        )

    return jsonify({"type": "FeatureCollection", "features": features})


@bp.route("/api/cancel_replan", methods=["POST"])
def cancel_replan_route() -> Response:
    transfer_id = request.json.get("transfer_id")
    cancel_replan_backend(transfer_id)
    return jsonify({"success": True})


class WormholeManager:
    def __init__(self) -> None:
        self.active_transfers: dict[str, dict[str, Any]] = {}
        if shutil.which("wormhole") is None:
            # We don't want to crash the whole app if wormhole is missing,
            # just log it and the endpoints will fail gracefully.
            logger.warning("'wormhole' command not found. magic-wormhole is required for sharing.")

    def create_transfer(self, gpx_data: str) -> str:
        transfer_id = str(uuid.uuid4())
        temp_dir = Path(tempfile.mkdtemp())
        file_path = temp_dir / "path.gpx"

        with file_path.open("w") as f:
            f.write(gpx_data)

        cmd = ["wormhole", "send", str(file_path)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as e:
            shutil.rmtree(temp_dir)
            msg = f"Failed to start wormhole process: {e}"
            raise RuntimeError(msg) from e

        logger.info("Starting wormhole transfer %s: %s", transfer_id, " ".join(cmd))
        logger.info("Process ID for transfer %s: %s", transfer_id, process.pid)

        self.active_transfers[transfer_id] = {
            "process": process,
            "temp_dir": temp_dir,
            "start_time": time.time(),
            "status": "running",
            "code": None,
        }

        threading.Thread(
            target=self._capture_wormhole_code_thread, args=(transfer_id,), daemon=True,
        ).start()
        return transfer_id

    def _capture_wormhole_code_thread(self, transfer_id: str) -> None:
        transfer_info = self.active_transfers.get(transfer_id)
        if not transfer_info:
            return

        process = transfer_info["process"]
        wormhole_code = None
        try:
            while True:
                readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                for stream in readable:
                    line = stream.readline().strip()
                    if line:
                        match = re.search(r"Wormhole code is: (\S+-\S+-\S+)", line)
                        if match:
                            wormhole_code = match.group(1)
                            logger.info(
                                "Wormhole code for transfer %s: %s", transfer_id, wormhole_code,
                            )
                        elif stream == process.stderr:
                            logger.warning("Wormhole stderr (%s): %s", transfer_id, line)

                if wormhole_code:
                    transfer_info["code"] = wormhole_code
                    break

                if process.poll() is not None:
                    break

            process.wait(timeout=60)
            transfer_info["status"] = "completed" if process.returncode == 0 else "failed"
        except Exception:
            logger.exception("Error in wormhole thread for %s", transfer_id)
            transfer_info["status"] = "failed"
            if process.poll() is None:
                process.kill()
        finally:
            self._cleanup_transfer(transfer_id)

    def get_transfer_code(self, transfer_id: str, timeout: float = 10) -> str | None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if transfer_id in self.active_transfers and self.active_transfers[transfer_id].get(
                "code",
            ):
                return str(self.active_transfers[transfer_id]["code"])
            time.sleep(0.1)
        return None

    def cancel_transfer(self, transfer_id: str) -> tuple[bool, str]:
        if transfer_id not in self.active_transfers:
            return False, "Invalid or unknown transfer ID"

        logger.info("Cancelling wormhole transfer %s", transfer_id)
        process = self.active_transfers[transfer_id]["process"]
        if process.poll() is None:
            process.kill()

        self.active_transfers[transfer_id]["status"] = "cancelled"
        return True, "Transfer cancelled"

    def _cleanup_transfer(self, transfer_id: str) -> None:
        transfer = self.active_transfers.pop(transfer_id, None)
        if transfer and transfer.get("temp_dir"):
            try:
                shutil.rmtree(transfer["temp_dir"])
            except Exception:
                logger.exception("Error cleaning temp dir for %s", transfer_id)


wormhole_manager = WormholeManager()


@bp.route("/api/create_wormhole", methods=["POST"])
def create_wormhole() -> Response:
    gpx_data = request.json.get("gpx")
    if not gpx_data:
        return jsonify({"success": False, "message": "No GPX data provided"}), 400

    try:
        transfer_id = wormhole_manager.create_transfer(gpx_data)
        code = wormhole_manager.get_transfer_code(transfer_id, timeout=15)
        if code:
            return jsonify({"success": True, "code": code, "transfer_id": transfer_id})
        wormhole_manager.cancel_transfer(transfer_id)
        return jsonify(
            {"success": False, "message": "Failed to capture wormhole code in time"},
        ), 500
    except Exception as e:
        logger.exception("Error creating wormhole")
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route("/api/cancel_wormhole", methods=["POST"])
def cancel_wormhole() -> Response:
    transfer_id = request.json.get("transfer_id")
    success, message = wormhole_manager.cancel_transfer(transfer_id)
    return jsonify({"success": success, "message": message})


@bp.route("/api/create_replan", methods=["POST"])
def create_replan() -> Response:
    body = request.get_json(force=True) or {}
    path_data = body.get("points")  # [[lat, lon], ...]
    filename = body.get("file")
    highway_types = body.get("allowed_ways", ["footway"])
    transfer_id = body.get("transfer_id")
    algorithm = body.get("algorithm", "rrt")
    sub_algorithm = body.get("sub_algorithm", "astar")
    highway_costs = body.get("highway_costs")
    surface_costs = body.get("surface_costs")

    cell_size = body.get("cell_size", 0.25)
    inflate_obstacles = body.get("inflate_obstacles", 0.25)
    simplify_path = body.get("simplify_path", True)
    smooth_path = body.get("smooth_path", False)

    if not path_data or not filename:
        abort(400, "Missing points or file parameter")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File {filename} not found")

    zn, zl = md.zone_number, md.zone_letter
    utm_path = []
    for p in path_data:
        e, n, _, _ = utm.from_latlon(float(p[0]), float(p[1]), zn, zl)
        utm_path.append([e, n])
    utm_path = np.array(utm_path, dtype=np.float64)

    # Calculate planning bounding box to limit grid size
    margin = 50.0  # meters
    p_min_x = np.min(utm_path[:, 0]) - margin
    p_max_x = np.max(utm_path[:, 0]) + margin
    p_min_y = np.min(utm_path[:, 1]) - margin
    p_max_y = np.max(utm_path[:, 1]) + margin

    # Clip to map boundaries
    p_low = (max(md.min_x, p_min_x), max(md.min_y, p_min_y))
    p_high = (min(md.max_x, p_max_x), min(md.max_y, p_max_y))

    if algorithm == "graph":
        planner = GraphPlanner(md, highway_types=highway_types)
        res = planner.plan(utm_path)
    else:
        args = parse_args([])
        args.simplify_path = simplify_path
        args.smooth_path = smooth_path
        args.cell_size = cell_size
        args.inflate_obstacles = inflate_obstacles
        args.visualize = False
        args.low = p_low
        args.high = p_high

        # Filter barriers to bounding box for faster processing
        bbox = geometry.box(p_low[0], p_low[1], p_high[0], p_high[1])
        filtered_barriers = [w for w in md.barriers_list if w.line and w.line.intersects(bbox)]

        obstacles = ways_to_shapely(filtered_barriers)
        replanner = ReplanPath(args, obstacles, transfer_id=transfer_id)
        if highway_costs:
            try:
                replanner.HIGHWAY_COSTS = highway_costs
            except (ValueError, TypeError) as e:
                logger.warning("Failed to set custom highway costs: %s", e)
        if surface_costs:
            try:
                replanner.SURFACE_COSTS = surface_costs
            except (ValueError, TypeError) as e:
                logger.warning("Failed to set custom surface costs: %s", e)
        replanner.fill_grid(md, highway_types=highway_types)
        res = replanner.replan(utm_path, algorithm=sub_algorithm)

    if res is None:
        status = "cancelled" if algorithm != "graph" else "failed"
        return jsonify({"retrieveNum": 1, "newPath": None, "status": status})

    new_path = []
    changed = False

    # RRT* result might have more/different points
    if len(res) != len(utm_path):
        changed = True

    for i in range(len(res)):
        lat, lon = utm.to_latlon(res[i][0], res[i][1], zn, zl)
        new_path.append([lat, lon])
        # Simple heuristic to check if it actually changed significantly
        if (
            not changed
            and i < len(utm_path)
            and np.linalg.norm(res[i] - utm_path[i]) > SIGNIFICANT_CHANGE_TOLERANCE
        ):
            changed = True

    if changed:
        return jsonify({"retrieveNum": 0, "newPath": new_path})
    return jsonify({"retrieveNum": -1, "newPath": new_path})
