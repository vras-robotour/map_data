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

import numpy as np
import utm
from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)

from map_data.map_data import MapData
from map_data.pathsolver.replan import ReplanPath, cancel_replan_backend, parse_args
from map_data.pathsolver.graph_planner import GraphPlanner
from map_data.utils.parsing import ways_to_shapely
from map_data.utils.serialization import map_data_to_dict
from map_data.utils.way import FOOTWAY_VALUES, Way

from .cache import load_mapdata_cached
from .helpers import (
    apply_node_position_overrides,
    geom_to_geojson,
    get_deleted_node_ids,
    get_deleted_way_ids,
    get_node_position_overrides,
    geojson_geom_to_utm,
    load_annotations,
    mapdata_to_geojson,
    migrate_change_log,
    rebuild_way_without_nodes,
    save_annotations,
)

bp = Blueprint("viewer", __name__)
logger = logging.getLogger(__name__)


def _get_data_dir():
    if current_app.config.get("DATA_DIR"):
        return current_app.config["DATA_DIR"]
    try:
        from ament_index_python.resources import get_resource

        _, pkg = get_resource("packages", "map_data")
        return os.path.join(pkg, "share", "map_data", "data")
    except Exception:
        # Fallback to the local data directory relative to this file
        return os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data")
        )


def _annotation_path(filename):
    base = filename.rsplit(".", 1)[0]
    return os.path.join(_get_data_dir(), f"{base}.annotations.json")


@bp.route("/")
def index():
    api_key_thunderforest = os.getenv("THUNDERFOREST_API_KEY")
    api_key_seznam = os.getenv("SEZNAM_API_KEY")
    return render_template(
        "index.html",
        apikey_thunderforest=api_key_thunderforest,
        apikey_seznam=api_key_seznam,
    )


@bp.route("/api/files")
def list_files():
    result = {"mapdata": [], "gpx": []}
    try:
        for name in sorted(os.listdir(_get_data_dir())):
            if name.endswith(".mapdata"):
                result["mapdata"].append(name)
            elif name.endswith(".gpx"):
                result["gpx"].append(name)
    except FileNotFoundError:
        pass
    return jsonify(result)


@bp.route("/api/mapdata")
def get_mapdata():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = os.path.join(_get_data_dir(), filename)
    if not os.path.isfile(path):
        abort(404, f"File not found: {filename}")
    map_data = copy.copy(load_mapdata_cached(path))
    store = load_annotations(_annotation_path(filename))
    deleted_way_ids = get_deleted_way_ids(store)
    has_node_dels = bool(store.get("deleted_nodes"))
    if deleted_way_ids or has_node_dels:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            new_lst = []
            for w in getattr(map_data, lst_name):
                if w.id in deleted_way_ids:
                    continue
                del_nids = get_deleted_node_ids(store, w.id)
                if del_nids:
                    w = rebuild_way_without_nodes(
                        w,
                        del_nids,
                        map_data.zone_number,
                        map_data.zone_letter,
                        getattr(map_data, "nodes_cache", {}),
                    )
                    if w is None:
                        continue
                new_lst.append(w)
            setattr(map_data, lst_name, new_lst)
        map_data.crossroads_list = map_data.parse_intersections(
            {w.id: w for w in map_data.footways_list}
        )
    node_pos_store = store.get("node_position_overrides", {})
    if node_pos_store:
        zn, zl = map_data.zone_number, map_data.zone_letter
        _cat_for_list = {
            "roads_list": "road",
            "footways_list": "footway",
            "barriers_list": "barrier",
        }
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            new_lst = []
            for w in getattr(map_data, lst_name):
                ov = get_node_position_overrides(store, w.id)
                if ov:
                    w = (
                        apply_node_position_overrides(
                            w,
                            ov,
                            zn,
                            zl,
                            getattr(map_data, "nodes_cache", {}),
                            category=_cat_for_list[lst_name],
                        )
                        or w
                    )
                new_lst.append(w)
            setattr(map_data, lst_name, new_lst)

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
                    f["properties"]["category"] = (
                        "footway" if hw in FOOTWAY_VALUES else "road"
                    )
    return jsonify(geojson)


@bp.route("/api/annotations")
def get_annotations():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    return jsonify(load_annotations(_annotation_path(filename)))


@bp.route("/api/annotations", methods=["POST"])
def add_annotation():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = _annotation_path(filename)
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
def update_annotation(ann_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = _annotation_path(filename)
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
def delete_annotation(ann_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    before = len(store["annotations"])
    store["annotations"] = [a for a in store["annotations"] if a["id"] != ann_id]
    if len(store["annotations"]) == before:
        abort(404, "Annotation not found")
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/fetch_area", methods=["POST"])
def fetch_area():
    body = request.get_json(force=True) or {}
    for field in ("min_lat", "min_lon", "max_lat", "max_lon", "name"):
        if field not in body:
            abort(400, f"Missing field: {field}")

    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(body["name"]).strip())
    if not name:
        abort(400, "name is empty after sanitizing")

    data_dir = _get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{name}.mapdata")

    corners = np.array(
        [
            [body["min_lat"], body["min_lon"]],
            [body["min_lat"], body["max_lon"]],
            [body["max_lat"], body["min_lon"]],
            [body["max_lat"], body["max_lon"]],
        ]
    )
    easting, northing, zone_number, zone_letter = utm.from_latlon(
        corners[:, 0], corners[:, 1]
    )
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
        }
    )


@bp.route("/api/way_nodes")
def get_way_nodes():
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing 'file' or 'way_id' query parameter")
    path = os.path.join(_get_data_dir(), filename)
    if not os.path.isfile(path):
        abort(404, f"File not found: {filename}")

    md = load_mapdata_cached(path)

    try:
        wid = int(way_id)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer")

    way = next(
        (
            w
            for lst in (md.roads_list, md.footways_list, md.barriers_list)
            for w in lst
            if w.id == wid
        ),
        None,
    )
    if way is None:
        abort(404, f"Way {wid} not found")

    nodes_cache = getattr(md, "nodes_cache", {})
    zn, zl = md.zone_number, md.zone_letter

    geom_latlon = None
    nodes = []
    for i, nid in enumerate(way.nodes):
        if nid in nodes_cache:
            nd = nodes_cache[nid]
            nodes.append(
                {"id": nid, "lat": nd["lat"], "lon": nd["lon"], "tags": nd["tags"]}
            )
        else:
            if geom_latlon is None:
                geom = way.line
                raw = list(
                    geom.exterior.coords if hasattr(geom, "exterior") else geom.coords
                )
                geom_latlon = [utm.to_latlon(e, n, zn, zl) for e, n in raw]
            if i < len(geom_latlon):
                lat, lon = geom_latlon[i]
                nodes.append({"id": nid, "lat": lat, "lon": lon, "tags": {}})

    # For ways with no nodes (e.g. individual barrier nodes), synthesize a centroid node
    # so the object can be selected and dragged in edit mode.
    if not nodes:
        centroid = way.line.centroid
        lat, lon = utm.to_latlon(centroid.x, centroid.y, zn, zl)
        nodes = [{"id": wid, "lat": lat, "lon": lon, "tags": {}}]

    store = load_annotations(_annotation_path(filename))
    deleted_node_ids = get_deleted_node_ids(store, wid)
    if deleted_node_ids:
        nodes = [n for n in nodes if n["id"] not in deleted_node_ids]

    pos_overrides = get_node_position_overrides(store, wid)
    if pos_overrides:
        for n in nodes:
            if n["id"] in pos_overrides:
                n["lat"] = pos_overrides[n["id"]]["lat"]
                n["lon"] = pos_overrides[n["id"]]["lon"]

    return jsonify({"way_id": wid, "nodes": nodes})


@bp.route("/api/ways/<signed_int:way_id>")
def get_way(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = os.path.join(_get_data_dir(), filename)
    if not os.path.isfile(path):
        abort(404, f"File not found: {filename}")

    md = load_mapdata_cached(path)
    store = load_annotations(_annotation_path(filename))

    way = None
    category = None
    for lst_name, cat in (
        ("roads_list", "road"),
        ("footways_list", "footway"),
        ("barriers_list", "barrier"),
    ):
        for w in getattr(md, lst_name):
            if w.id == way_id:
                way = copy.copy(w)
                category = cat
                break
        if way:
            break

    if way is None:
        abort(404, f"Way {way_id} not found")

    zn, zl = md.zone_number, md.zone_letter
    del_nids = get_deleted_node_ids(store, way_id)
    if del_nids:
        way = rebuild_way_without_nodes(
            way, del_nids, zn, zl, getattr(md, "nodes_cache", {}), category=category
        )
        if way is None:
            abort(404, f"Way {way_id} reduced to nothing by node deletions")

    pos_overrides = get_node_position_overrides(store, way_id)
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

    geom = geom_to_geojson(way.line, zn, zl)
    if geom is None:
        abort(500, "Could not convert geometry")

    feature = {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "id": way.id,
            "category": category,
            "is_node": category == "barrier" and not bool(way.nodes),
            "tags": way.tags or {},
            "in_out": way.in_out,
        },
    }

    ov = store.get("tag_overrides", {}).get(str(way_id))
    if ov:
        merged = {**(feature["properties"]["tags"]), **ov}
        feature["properties"]["tags"] = merged
        if category in ("road", "footway"):
            hw = merged.get("highway", "")
            feature["properties"]["category"] = (
                "footway" if hw in FOOTWAY_VALUES else "road"
            )

    return jsonify(feature)


@bp.route("/api/ways/<signed_int:way_id>", methods=["DELETE"])
def delete_way(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True) or {}
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    migrate_change_log(store)
    if way_id not in get_deleted_way_ids(store):
        store.setdefault("deleted_ways", []).append(
            {
                "id": way_id,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
            }
        )
        cl = store.setdefault("change_log", [])
        if not any(e.get("type") == "way" and e.get("id") == way_id for e in cl):
            cl.append({"type": "way", "id": way_id, "ts": time.time()})
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/ways/<signed_int:way_id>/tags", methods=["PUT"])
def update_way_tags(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True) or {}
    tags = body.get("tags")
    if not isinstance(tags, dict):
        abort(400, "Request body must include 'tags' dict")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    migrate_change_log(store)
    store.setdefault("tag_overrides", {})[str(way_id)] = tags
    store.setdefault("tag_override_meta", {})[str(way_id)] = {
        "category": body.get("category", "unknown"),
        "label": body.get("label", ""),
    }
    cl = store.setdefault("change_log", [])
    if not any(e.get("type") == "tag" and e.get("id") == way_id for e in cl):
        cl.append({"type": "tag", "id": way_id, "ts": time.time()})
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/ways/<signed_int:way_id>/tags", methods=["DELETE"])
def delete_way_tags(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    store.get("tag_overrides", {}).pop(str(way_id), None)
    store.get("tag_override_meta", {}).pop(str(way_id), None)
    cl = store.get("change_log", [])
    store["change_log"] = [
        e for e in cl if not (e.get("type") == "tag" and e.get("id") == way_id)
    ]
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/ways/<signed_int:way_id>/hide", methods=["PUT"])
def hide_way(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True) or {}
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    hw = store.setdefault("hidden_ways", [])
    existing_ids = {(d["id"] if isinstance(d, dict) else d) for d in hw}
    if way_id not in existing_ids:
        hw.append(
            {
                "id": way_id,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
            }
        )
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/ways/<signed_int:way_id>/show", methods=["PUT"])
def show_way(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    hw = store.get("hidden_ways", [])
    store["hidden_ways"] = [
        d for d in hw if (d["id"] if isinstance(d, dict) else d) != way_id
    ]
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/ways/<signed_int:way_id>/restore", methods=["PUT"])
def restore_way(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    dw = store.get("deleted_ways", [])
    store["deleted_ways"] = [
        d for d in dw if (d["id"] if isinstance(d, dict) else d) != way_id
    ]
    cl = store.get("change_log", [])
    store["change_log"] = [
        e for e in cl if not (e.get("type") == "way" and e.get("id") == way_id)
    ]
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/way_node", methods=["DELETE"])
def delete_way_node():
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    node_id = request.args.get("node_id")
    if not filename or way_id is None or node_id is None:
        abort(400, "Missing required query parameters")
    try:
        way_id = int(way_id)
        node_id = int(node_id)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    migrate_change_log(store)
    dn = store.setdefault("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
        store["deleted_nodes"] = dn
    if node_id not in get_deleted_node_ids(store, way_id):
        dn.append({"way_id": way_id, "node_id": node_id})
        cl = store.setdefault("change_log", [])
        if not any(
            e.get("type") == "node"
            and e.get("way_id") == way_id
            and e.get("node_id") == node_id
            for e in cl
        ):
            cl.append(
                {
                    "type": "node",
                    "way_id": way_id,
                    "node_id": node_id,
                    "ts": time.time(),
                }
            )
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/way_node/restore", methods=["PUT"])
def restore_way_node():
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    node_id = request.args.get("node_id")
    if not filename or way_id is None or node_id is None:
        abort(400, "Missing required query parameters")
    try:
        way_id = int(way_id)
        node_id = int(node_id)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
    store["deleted_nodes"] = [
        d for d in dn if not (d["way_id"] == way_id and d["node_id"] == node_id)
    ]
    cl = store.get("change_log", [])
    store["change_log"] = [
        e
        for e in cl
        if not (
            e.get("type") == "node"
            and e.get("way_id") == way_id
            and e.get("node_id") == node_id
        )
    ]
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/way_nodes/move", methods=["PUT"])
def move_way_nodes():
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing required query parameters")
    try:
        way_id = int(way_id)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer")
    body = request.get_json(force=True) or {}
    nodes = body.get("nodes")
    if not isinstance(nodes, list):
        abort(400, "Request body must include 'nodes' list")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    overrides = store.setdefault("node_position_overrides", {})
    way_key = str(way_id)
    if way_key not in overrides:
        overrides[way_key] = {}
    for n in nodes:
        overrides[way_key][str(n["id"])] = {
            "lat": float(n["lat"]),
            "lon": float(n["lon"]),
        }
    migrate_change_log(store)
    cl = store.setdefault("change_log", [])
    if not any(e.get("type") == "move" and e.get("id") == way_id for e in cl):
        cl.append(
            {
                "type": "move",
                "id": way_id,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
                "ts": time.time(),
            }
        )
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/way_nodes/move", methods=["DELETE"])
def undo_move_way_nodes():
    filename = request.args.get("file")
    way_id = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing required query parameters")
    try:
        way_id = int(way_id)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    store.get("node_position_overrides", {}).pop(str(way_id), None)
    cl = store.get("change_log", [])
    store["change_log"] = [
        e for e in cl if not (e.get("type") == "move" and e.get("id") == way_id)
    ]
    save_annotations(ann_path, store)
    return "", 204


def get_merged_mapdata(filename):
    path = os.path.join(_get_data_dir(), filename)
    if not os.path.isfile(path):
        return None, None

    store = load_annotations(_annotation_path(filename))
    md = copy.deepcopy(load_mapdata_cached(path))
    zn, zl = md.zone_number, md.zone_letter

    deleted_way_ids = get_deleted_way_ids(store)
    has_node_dels = bool(store.get("deleted_nodes"))
    if deleted_way_ids or has_node_dels:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            new_lst = []
            for w in getattr(md, lst_name):
                if w.id in deleted_way_ids:
                    continue
                del_nids = get_deleted_node_ids(store, w.id)
                if del_nids:
                    w = rebuild_way_without_nodes(
                        w, del_nids, zn, zl, getattr(md, "nodes_cache", {})
                    )
                    if w is None:
                        continue
                new_lst.append(w)
            setattr(md, lst_name, new_lst)

    node_pos_store = store.get("node_position_overrides", {})
    if node_pos_store:
        _cat_for_list = {
            "roads_list": "road",
            "footways_list": "footway",
            "barriers_list": "barrier",
        }
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            new_lst = []
            for w in getattr(md, lst_name):
                ov = get_node_position_overrides(store, w.id)
                if ov:
                    w = (
                        apply_node_position_overrides(
                            w,
                            ov,
                            zn,
                            zl,
                            getattr(md, "nodes_cache", {}),
                            category=_cat_for_list[lst_name],
                        )
                        or w
                    )
                new_lst.append(w)
            setattr(md, lst_name, new_lst)

    tag_overrides = store.get("tag_overrides", {})
    if tag_overrides:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            for w in getattr(md, lst_name):
                ov = tag_overrides.get(str(w.id))
                if ov:
                    w.tags = {**(w.tags or {}), **ov}
        new_roads, new_footways = [], []
        for w in md.roads_list:
            (new_footways if w.is_footway() else new_roads).append(w)
        for w in md.footways_list:
            (new_roads if w.is_road() else new_footways).append(w)
        md.roads_list, md.footways_list = new_roads, new_footways

    if deleted_way_ids or has_node_dels or tag_overrides:
        md.crossroads_list = md.parse_intersections({w.id: w for w in md.footways_list})

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
def export_mapdata():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")

    md, _ = get_merged_mapdata(filename)
    if md is None:
        abort(404, f"File not found: {filename}")

    buf = io.BytesIO()
    buf.write(json.dumps(map_data_to_dict(md), indent=2).encode("utf-8"))
    buf.seek(0)
    base = filename.rsplit(".", 1)[0]
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{base}.exported.mapdata",
        mimetype="application/json",
    )


@bp.route("/api/cancel_replan", methods=["POST"])
def cancel_replan_route():
    transfer_id = request.json.get("transfer_id")
    cancel_replan_backend(transfer_id)
    return jsonify({"success": True})


class WormholeManager:
    def __init__(self):
        self.active_transfers = {}
        if shutil.which("wormhole") is None:
            # We don't want to crash the whole app if wormhole is missing,
            # just log it and the endpoints will fail gracefully.
            logger.warning(
                "'wormhole' command not found. magic-wormhole is required for sharing."
            )

    def create_transfer(self, gpx_data):
        transfer_id = str(uuid.uuid4())
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "path.gpx")

        with open(file_path, "w") as f:
            f.write(gpx_data)

        cmd = ["wormhole", "send", file_path]
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
            raise RuntimeError(f"Failed to start wormhole process: {e}")

        logger.info(f"Starting wormhole transfer {transfer_id}: {' '.join(cmd)}")
        logger.info(f"Process ID for transfer {transfer_id}: {process.pid}")

        self.active_transfers[transfer_id] = {
            "process": process,
            "temp_dir": temp_dir,
            "start_time": time.time(),
            "status": "running",
            "code": None,
        }

        threading.Thread(
            target=self._capture_wormhole_code_thread, args=(transfer_id,), daemon=True
        ).start()
        return transfer_id

    def _capture_wormhole_code_thread(self, transfer_id):
        transfer_info = self.active_transfers.get(transfer_id)
        if not transfer_info:
            return

        process = transfer_info["process"]
        wormhole_code = None
        try:
            while True:
                readable, _, _ = select.select(
                    [process.stdout, process.stderr], [], [], 0.1
                )
                for stream in readable:
                    line = stream.readline().strip()
                    if line:
                        match = re.search(r"Wormhole code is: (\S+-\S+-\S+)", line)
                        if match:
                            wormhole_code = match.group(1)
                            logger.info(
                                f"Wormhole code captured for transfer {transfer_id}: {wormhole_code}"
                            )
                        elif stream == process.stderr:
                            logger.warning(f"Wormhole stderr ({transfer_id}): {line}")

                if wormhole_code:
                    transfer_info["code"] = wormhole_code
                    break

                if process.poll() is not None:
                    break

            process.wait(timeout=60)
            transfer_info["status"] = (
                "completed" if process.returncode == 0 else "failed"
            )
        except Exception as e:
            logger.error(f"Error in wormhole thread for {transfer_id}: {e}")
            transfer_info["status"] = "failed"
            if process.poll() is None:
                process.kill()
        finally:
            self._cleanup_transfer(transfer_id)

    def get_transfer_code(self, transfer_id, timeout=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if transfer_id in self.active_transfers and self.active_transfers[
                transfer_id
            ].get("code"):
                return self.active_transfers[transfer_id]["code"]
            time.sleep(0.1)
        return None

    def cancel_transfer(self, transfer_id):
        if transfer_id not in self.active_transfers:
            return False, "Invalid or unknown transfer ID"

        logger.info(f"Cancelling wormhole transfer {transfer_id}")
        process = self.active_transfers[transfer_id]["process"]
        if process.poll() is None:
            process.kill()

        self.active_transfers[transfer_id]["status"] = "cancelled"
        return True, "Transfer cancelled"

    def _cleanup_transfer(self, transfer_id):
        transfer = self.active_transfers.pop(transfer_id, None)
        if transfer and transfer.get("temp_dir"):
            try:
                shutil.rmtree(transfer["temp_dir"])
            except Exception as e:
                logger.error(f"Error cleaning temp dir for {transfer_id}: {e}")


wormhole_manager = WormholeManager()


@bp.route("/api/create_wormhole", methods=["POST"])
def create_wormhole():
    gpx_data = request.json.get("gpx")
    if not gpx_data:
        return jsonify({"success": False, "message": "No GPX data provided"}), 400

    try:
        transfer_id = wormhole_manager.create_transfer(gpx_data)
        code = wormhole_manager.get_transfer_code(transfer_id, timeout=15)
        if code:
            return jsonify({"success": True, "code": code, "transfer_id": transfer_id})
        else:
            wormhole_manager.cancel_transfer(transfer_id)
            return jsonify(
                {"success": False, "message": "Failed to capture wormhole code in time"}
            ), 500
    except Exception as e:
        logger.error(f"Error creating wormhole: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route("/api/cancel_wormhole", methods=["POST"])
def cancel_wormhole():
    transfer_id = request.json.get("transfer_id")
    success, message = wormhole_manager.cancel_transfer(transfer_id)
    return jsonify({"success": success, "message": message})


@bp.route("/api/create_replan", methods=["POST"])
def create_replan():
    body = request.get_json(force=True) or {}
    path_data = body.get("points")  # [[lat, lon], ...]
    filename = body.get("file")
    highway_types = body.get("allowed_ways", ["footway"])
    transfer_id = body.get("transfer_id")
    algorithm = body.get("algorithm", "rrt")

    cell_size = body.get("cell_size", 0.25)
    inflate_obstacles = body.get("inflate_obstacles", 0.25)
    simplify_path = body.get("simplify_path", True)

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

    if algorithm == "graph":
        planner = GraphPlanner(md, highway_types=highway_types)
        res = planner.plan(utm_path)
    else:
        args = parse_args([])
        args.simplify_path = simplify_path
        args.cell_size = cell_size
        args.inflate_obstacles = inflate_obstacles
        args.visualize = False
        args.low = (md.min_x, md.min_y)
        args.high = (md.max_x, md.max_y)

        obstacles = ways_to_shapely(md.barriers_list)
        replanner = ReplanPath(args, obstacles, transfer_id=transfer_id)
        replanner.fill_grid(md, highway_types=highway_types)
        res = replanner.replan_rrt(utm_path)

    if res is None:
        return jsonify({"retrieveNum": 1, "newPath": None, "status": "cancelled"})

    new_path = []
    changed = False

    # RRT* result might have more/different points
    if len(res) != len(utm_path):
        changed = True

    for i in range(len(res)):
        lat, lon = utm.to_latlon(res[i][0], res[i][1], zn, zl)
        new_path.append([lat, lon])
        # Simple heuristic to check if it actually changed significantly
        if not changed and i < len(utm_path):
            if np.linalg.norm(res[i] - utm_path[i]) > 0.1:
                changed = True

    if changed:
        return jsonify({"retrieveNum": 0, "newPath": new_path})
    else:
        return jsonify({"retrieveNum": -1, "newPath": new_path})
