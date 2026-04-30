import os
import re
import io
import copy
import uuid
import pickle
import logging
import utm
import numpy as np
from flask import (
    Blueprint,
    jsonify,
    request,
    abort,
    send_file,
    render_template,
    current_app,
)
from map_data.map_data import MapData
from map_data.way import Way, FOOTWAY_VALUES
from .cache import load_mapdata_cached
from .helpers import (
    mapdata_to_geojson,
    load_annotations,
    save_annotations,
    get_deleted_way_ids,
    get_deleted_node_ids,
    rebuild_way_without_nodes,
    geom_to_geojson,
    geojson_geom_to_utm,
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
    return render_template("index.html")


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

    with open(out_path, "wb") as fh:
        pickle.dump(md, fh, protocol=2)

    return jsonify(
        {
            "filename": f"{name}.mapdata",
            "roads": len(md.roads_list),
            "footways": len(md.footways_list),
            "barriers": len(md.barriers_list),
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
    for i, node_or_id in enumerate(way.nodes):
        nid = getattr(node_or_id, "id", node_or_id)
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

    store = load_annotations(_annotation_path(filename))
    deleted_node_ids = get_deleted_node_ids(store, wid)
    if deleted_node_ids:
        nodes = [n for n in nodes if n["id"] not in deleted_node_ids]

    return jsonify({"way_id": wid, "nodes": nodes})


@bp.route("/api/ways/<int:way_id>")
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
            way, del_nids, zn, zl, getattr(md, "nodes_cache", {})
        )
        if way is None:
            abort(404, f"Way {way_id} reduced to nothing by node deletions")

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


@bp.route("/api/ways/<int:way_id>", methods=["DELETE"])
def delete_way(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True) or {}
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    if way_id not in get_deleted_way_ids(store):
        store.setdefault("deleted_ways", []).append(
            {
                "id": way_id,
                "category": body.get("category", "unknown"),
                "label": body.get("label", ""),
            }
        )
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/ways/<int:way_id>/tags", methods=["PUT"])
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
    store.setdefault("tag_overrides", {})[str(way_id)] = tags
    store.setdefault("tag_override_meta", {})[str(way_id)] = {
        "category": body.get("category", "unknown"),
        "label": body.get("label", ""),
    }
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/ways/<int:way_id>/tags", methods=["DELETE"])
def delete_way_tags(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = _annotation_path(filename)
    store = load_annotations(ann_path)
    store.get("tag_overrides", {}).pop(str(way_id), None)
    store.get("tag_override_meta", {}).pop(str(way_id), None)
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/ways/<int:way_id>/hide", methods=["PUT"])
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


@bp.route("/api/ways/<int:way_id>/show", methods=["PUT"])
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


@bp.route("/api/ways/<int:way_id>/restore", methods=["PUT"])
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
    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
        store["deleted_nodes"] = dn
    if node_id not in get_deleted_node_ids(store, way_id):
        dn.append({"way_id": way_id, "node_id": node_id})
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
    save_annotations(ann_path, store)
    return "", 204


@bp.route("/api/export")
def export_mapdata():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = os.path.join(_get_data_dir(), filename)
    if not os.path.isfile(path):
        abort(404, f"File not found: {filename}")

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

    ann_id = -1
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
            (md.roads_list if w.is_road() else md.footways_list).append(w)
        else:
            w.tags = {"barrier": props.get("barrier", "wall")}
            for k, v in props.items():
                if k != "barrier":
                    w.tags[k] = str(v)
            md.barriers_list.append(w)

    buf = io.BytesIO()
    pickle.dump(md, buf, protocol=2)
    buf.seek(0)
    base = filename.rsplit(".", 1)[0]
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{base}.exported.mapdata",
        mimetype="application/octet-stream",
    )
