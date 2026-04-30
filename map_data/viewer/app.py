import copy
import io
import os
import re
import json
import pickle
import logging
import uuid

import utm
import numpy as np
from flask import Flask, jsonify, request, render_template, abort, send_file
from shapely.geometry import LineString as _SLS, Polygon as _SPoly, MultiPolygon as _SMPoly

from map_data.map_data import MapData  # noqa: F401 – ensures MapData class is resolvable on pickle load
from map_data.way import Way, FOOTWAY_VALUES  # noqa: F401 – Way needed for export reconstruction


app = Flask(__name__)
logger = logging.getLogger(__name__)


def _get_data_dir():
    if app.config.get("DATA_DIR"):
        return app.config["DATA_DIR"]
    try:
        from ament_index_python.resources import get_resource

        _, pkg = get_resource("packages", "map_data")
        return os.path.join(pkg, "share", "map_data", "data")
    except Exception:
        return os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data")
        )


# ------------------------------------------------------------------
# GeoJSON conversion helpers
# ------------------------------------------------------------------


def _ring_to_latlon(coords, zone_number, zone_letter):
    result = []
    for x, y in coords:
        lat, lon = utm.to_latlon(x, y, zone_number, zone_letter)
        result.append([lon, lat])
    return result


def _geom_to_geojson(geom, zone_number, zone_letter):
    gtype = geom.geom_type
    if gtype == "Polygon":
        exterior = _ring_to_latlon(geom.exterior.coords, zone_number, zone_letter)
        interiors = [
            _ring_to_latlon(r.coords, zone_number, zone_letter) for r in geom.interiors
        ]
        return {"type": "Polygon", "coordinates": [exterior] + interiors}
    if gtype == "MultiPolygon":
        polygons = []
        for poly in geom.geoms:
            exterior = _ring_to_latlon(poly.exterior.coords, zone_number, zone_letter)
            interiors = [
                _ring_to_latlon(r.coords, zone_number, zone_letter)
                for r in poly.interiors
            ]
            polygons.append([exterior] + interiors)
        return {"type": "MultiPolygon", "coordinates": polygons}
    if gtype == "LineString":
        return {
            "type": "LineString",
            "coordinates": _ring_to_latlon(geom.coords, zone_number, zone_letter),
        }
    return None


def _mapdata_to_geojson(map_data):
    features = []
    zn, zl = map_data.zone_number, map_data.zone_letter

    def add_ways(ways, category):
        for way in ways:
            try:
                geom = _geom_to_geojson(way.line, zn, zl)
            except Exception:
                continue
            if geom is None:
                continue
            features.append(
                {
                    "type": "Feature",
                    "id": str(way.id),
                    "geometry": geom,
                    "properties": {
                        "id": way.id,
                        "category": category,
                        "tags": way.tags or {},
                        "in_out": way.in_out,
                    },
                }
            )

    add_ways(map_data.roads_list, "road")
    add_ways(map_data.footways_list, "footway")
    add_ways(map_data.barriers_list, "barrier")

    for i, (x, y) in enumerate(map_data.waypoints):
        lat, lon = utm.to_latlon(x, y, zn, zl)
        features.append(
            {
                "type": "Feature",
                "id": f"wp_{i}",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"category": "waypoint", "index": i},
            }
        )

    return {"type": "FeatureCollection", "features": features}


# ------------------------------------------------------------------
# Annotation helpers
# ------------------------------------------------------------------


def _annotation_path(filename):
    base = filename.rsplit(".", 1)[0]
    return os.path.join(_get_data_dir(), f"{base}.annotations.json")


def _load_annotations(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"version": 1, "annotations": []}


def _save_annotations(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ------------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------------


def _geojson_geom_to_utm(geometry, zone_number, zone_letter):
    """GeoJSON geometry (lon/lat) → Shapely geometry (UTM, same zone as mapdata)."""
    def pt(c):
        e, n, _, _ = utm.from_latlon(
            c[1], c[0],
            force_zone_number=zone_number,
            force_zone_letter=zone_letter,
        )
        return (e, n)

    gtype = geometry.get("type")
    if gtype == "LineString":
        return _SLS([pt(c) for c in geometry["coordinates"]])
    if gtype == "Polygon":
        rings = [[pt(c) for c in ring] for ring in geometry["coordinates"]]
        return _SPoly(rings[0], rings[1:])
    if gtype == "MultiPolygon":
        polys = [
            _SPoly([pt(c) for c in pc[0]], [[pt(c) for c in r] for r in pc[1:]])
            for pc in geometry["coordinates"]
        ]
        return _SMPoly(polys)
    return None


def _rebuild_way_without_nodes(way, del_nids, zone_number=None, zone_letter=None, nodes_cache=None):
    """Return a shallow copy of way with del_nids removed, or None if geometry becomes invalid.

    For buffered Polygon ways (all roads/footways/barriers after separate_ways), the polygon
    exterior coords have far more vertices than way.nodes, so index-based removal is wrong.
    Instead, we rebuild the centerline from node lat/lon positions and re-buffer with the
    same radius (estimated from the original polygon geometry).
    """
    node_ids = [getattr(n, "id", n) for n in way.nodes]
    keep = [i for i, nid in enumerate(node_ids) if nid not in del_nids]
    if len(keep) < 2:
        return None
    w = copy.copy(way)
    w.nodes = [way.nodes[i] for i in keep]
    geom = way.line

    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        new_coords = [coords[i] for i in keep if i < len(coords)]
        if len(new_coords) < 2:
            return None
        w.line = _SLS(new_coords)

    elif geom.geom_type == "Polygon":
        if zone_number is not None:
            # Rebuild centerline from node positions, then re-buffer with the same radius.
            nc = nodes_cache or {}
            utm_coords = []
            for n in w.nodes:
                lat = getattr(n, "lat", None)
                lon = getattr(n, "lon", None)
                if lat is None:  # node stored as plain ID — look up in cache
                    nd = nc.get(getattr(n, "id", n))
                    if nd:
                        lat, lon = nd["lat"], nd["lon"]
                if lat is not None and lon is not None:
                    e, nn, _, _ = utm.from_latlon(
                        float(lat), float(lon),
                        force_zone_number=zone_number,
                        force_zone_letter=zone_letter,
                    )
                    utm_coords.append((e, nn))
            if len(utm_coords) < 2:
                return None
            ls = _SLS(utm_coords)
            # Estimate original buffer radius from polygon area and perimeter:
            # area = 2*r*L + π*r²  and  perimeter = 2*L + 2*π*r
            # → π*r² - perimeter*r + area = 0
            p = geom.length
            a = geom.area
            disc = p * p - 4 * np.pi * a
            r = (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi) if disc >= 0 else a / p
            try:
                w.line = ls.buffer(r)
            except Exception:
                w.line = ls  # fallback: leave as LineString
        else:
            # No zone info — fall back to simple index-based removal (only correct for
            # unmodified closed areas where coords align with nodes).
            coords = list(geom.exterior.coords)
            new_coords = [coords[i] for i in keep if i < len(coords)]
            if len(new_coords) < 3:
                return None
            if new_coords[0] != new_coords[-1]:
                new_coords.append(new_coords[0])
            w.line = _SPoly(new_coords)

    else:
        return None  # MultiPolygon: not supported for partial node deletion

    return w


def _get_deleted_way_ids(store):
    """Return set of deleted way IDs, handling both old (int list) and new (dict list) formats."""
    dw = store.get("deleted_ways", [])
    return {(d["id"] if isinstance(d, dict) else d) for d in dw}


def _get_deleted_node_ids(store, way_id):
    """Return set of deleted node IDs for a given way_id, handling both storage formats."""
    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        return set(dn.get(str(way_id), []))
    return {d["node_id"] for d in dn if d["way_id"] == way_id}


# ------------------------------------------------------------------
# Mapdata cache  (keyed by (abs_path, mtime); holds at most 3 files)
# ------------------------------------------------------------------

_mapdata_cache: dict = {}


def _load_mapdata_cached(path):
    mtime = os.path.getmtime(path)
    key = (path, mtime)
    if key not in _mapdata_cache:
        # Evict stale entries for the same path
        for k in [k for k in _mapdata_cache if k[0] == path]:
            del _mapdata_cache[k]
        # Evict oldest entry if over capacity
        while len(_mapdata_cache) >= 3:
            del _mapdata_cache[next(iter(_mapdata_cache))]
        with open(path, "rb") as f:
            _mapdata_cache[key] = pickle.load(f)
    return _mapdata_cache[key]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/files")
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


@app.route("/api/mapdata")
def get_mapdata():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = os.path.join(_get_data_dir(), filename)
    if not os.path.isfile(path):
        abort(404, f"File not found: {filename}")
    map_data = copy.copy(_load_mapdata_cached(path))  # shallow copy — setattr won't mutate cache
    store = _load_annotations(_annotation_path(filename))
    deleted_way_ids = _get_deleted_way_ids(store)
    has_node_dels = bool(store.get("deleted_nodes"))
    if deleted_way_ids or has_node_dels:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            new_lst = []
            for w in getattr(map_data, lst_name):
                if w.id in deleted_way_ids:
                    continue
                del_nids = _get_deleted_node_ids(store, w.id)
                if del_nids:
                    w = _rebuild_way_without_nodes(w, del_nids)
                    if w is None:
                        continue
                new_lst.append(w)
            setattr(map_data, lst_name, new_lst)
    geojson = _mapdata_to_geojson(map_data)
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


@app.route("/api/annotations")
def get_annotations():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    return jsonify(_load_annotations(_annotation_path(filename)))


@app.route("/api/annotations", methods=["POST"])
def add_annotation():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = _annotation_path(filename)
    store = _load_annotations(ann_path)
    ann = {
        "id": str(uuid.uuid4()),
        "type": body.get("type", "obstacle"),
        "geometry": body["geometry"],
        "properties": body.get("properties", {}),
    }
    store["annotations"].append(ann)
    _save_annotations(ann_path, store)
    return jsonify(ann), 201


@app.route("/api/annotations/<ann_id>", methods=["PUT"])
def update_annotation(ann_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True)
    if not body or "geometry" not in body:
        abort(400, "Request body must include 'geometry'")
    ann_path = _annotation_path(filename)
    store = _load_annotations(ann_path)
    ann = next((a for a in store["annotations"] if a["id"] == ann_id), None)
    if ann is None:
        abort(404, "Annotation not found")
    ann["geometry"] = body["geometry"]
    if "type" in body:
        ann["type"] = body["type"]
    if "properties" in body:
        ann["properties"] = body["properties"]
    _save_annotations(ann_path, store)
    return jsonify(ann)


@app.route("/api/annotations/<ann_id>", methods=["DELETE"])
def delete_annotation(ann_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = _annotation_path(filename)
    store = _load_annotations(ann_path)
    before = len(store["annotations"])
    store["annotations"] = [a for a in store["annotations"] if a["id"] != ann_id]
    if len(store["annotations"]) == before:
        abort(404, "Annotation not found")
    _save_annotations(ann_path, store)
    return "", 204


@app.route("/api/fetch_area", methods=["POST"])
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


@app.route("/api/way_nodes")
def get_way_nodes():
    filename = request.args.get("file")
    way_id   = request.args.get("way_id")
    if not filename or way_id is None:
        abort(400, "Missing 'file' or 'way_id' query parameter")
    path = os.path.join(_get_data_dir(), filename)
    if not os.path.isfile(path):
        abort(404, f"File not found: {filename}")

    md = _load_mapdata_cached(path)

    try:
        wid = int(way_id)
    except (ValueError, TypeError):
        abort(400, "way_id must be an integer")

    way = next(
        (w for lst in (md.roads_list, md.footways_list, md.barriers_list)
         for w in lst if w.id == wid),
        None,
    )
    if way is None:
        abort(404, f"Way {wid} not found")

    nodes_cache = getattr(md, "nodes_cache", {})
    zn, zl = md.zone_number, md.zone_letter

    # Lazy fallback: derive lat/lon from geometry coords when cache misses
    geom_latlon = None
    nodes = []
    for i, node_or_id in enumerate(way.nodes):
        nid = getattr(node_or_id, "id", node_or_id)
        if nid in nodes_cache:
            nd = nodes_cache[nid]
            nodes.append({"id": nid, "lat": nd["lat"], "lon": nd["lon"], "tags": nd["tags"]})
        else:
            if geom_latlon is None:
                geom = way.line
                raw = list(geom.exterior.coords if hasattr(geom, "exterior") else geom.coords)
                geom_latlon = [utm.to_latlon(e, n, zn, zl) for e, n in raw]
            if i < len(geom_latlon):
                lat, lon = geom_latlon[i]
                nodes.append({"id": nid, "lat": lat, "lon": lon, "tags": {}})

    store = _load_annotations(_annotation_path(filename))
    deleted_node_ids = _get_deleted_node_ids(store, wid)
    if deleted_node_ids:
        nodes = [n for n in nodes if n["id"] not in deleted_node_ids]

    return jsonify({"way_id": wid, "nodes": nodes})


@app.route("/api/ways/<int:way_id>")
def get_way(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = os.path.join(_get_data_dir(), filename)
    if not os.path.isfile(path):
        abort(404, f"File not found: {filename}")

    md = _load_mapdata_cached(path)
    store = _load_annotations(_annotation_path(filename))

    way = None
    category = None
    for lst_name, cat in (("roads_list", "road"), ("footways_list", "footway"), ("barriers_list", "barrier")):
        for w in getattr(md, lst_name):
            if w.id == way_id:
                way = copy.copy(w)
                category = cat
                break
        if way:
            break

    if way is None:
        abort(404, f"Way {way_id} not found")

    del_nids = _get_deleted_node_ids(store, way_id)
    if del_nids:
        way = _rebuild_way_without_nodes(way, del_nids)
        if way is None:
            abort(404, f"Way {way_id} reduced to nothing by node deletions")

    zn, zl = md.zone_number, md.zone_letter
    geom = _geom_to_geojson(way.line, zn, zl)
    if geom is None:
        abort(500, "Could not convert geometry")

    feature = {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "id": way.id,
            "category": category,
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
            feature["properties"]["category"] = "footway" if hw in FOOTWAY_VALUES else "road"

    return jsonify(feature)


@app.route("/api/ways/<int:way_id>", methods=["DELETE"])
def delete_way(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True) or {}
    ann_path = _annotation_path(filename)
    store = _load_annotations(ann_path)
    if way_id not in _get_deleted_way_ids(store):
        store.setdefault("deleted_ways", []).append({
            "id": way_id,
            "category": body.get("category", "unknown"),
            "label": body.get("label", ""),
        })
    _save_annotations(ann_path, store)
    return "", 204


@app.route("/api/ways/<int:way_id>/tags", methods=["PUT"])
def update_way_tags(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    body = request.get_json(force=True) or {}
    tags = body.get("tags")
    if not isinstance(tags, dict):
        abort(400, "Request body must include 'tags' dict")
    ann_path = _annotation_path(filename)
    store = _load_annotations(ann_path)
    store.setdefault("tag_overrides", {})[str(way_id)] = tags
    store.setdefault("tag_override_meta", {})[str(way_id)] = {
        "category": body.get("category", "unknown"),
        "label": body.get("label", ""),
    }
    _save_annotations(ann_path, store)
    return "", 204


@app.route("/api/ways/<int:way_id>/tags", methods=["DELETE"])
def delete_way_tags(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = _annotation_path(filename)
    store = _load_annotations(ann_path)
    store.get("tag_overrides", {}).pop(str(way_id), None)
    store.get("tag_override_meta", {}).pop(str(way_id), None)
    _save_annotations(ann_path, store)
    return "", 204


@app.route("/api/ways/<int:way_id>/restore", methods=["PUT"])
def restore_way(way_id):
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    ann_path = _annotation_path(filename)
    store = _load_annotations(ann_path)
    dw = store.get("deleted_ways", [])
    store["deleted_ways"] = [
        d for d in dw if (d["id"] if isinstance(d, dict) else d) != way_id
    ]
    _save_annotations(ann_path, store)
    return "", 204


@app.route("/api/way_node", methods=["DELETE"])
def delete_way_node():
    filename = request.args.get("file")
    way_id   = request.args.get("way_id")
    node_id  = request.args.get("node_id")
    if not filename or way_id is None or node_id is None:
        abort(400, "Missing required query parameters")
    try:
        way_id  = int(way_id)
        node_id = int(node_id)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers")
    ann_path = _annotation_path(filename)
    store = _load_annotations(ann_path)
    # Migrate old dict format to list format if needed
    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
        store["deleted_nodes"] = dn
    if node_id not in _get_deleted_node_ids(store, way_id):
        dn.append({"way_id": way_id, "node_id": node_id})
    _save_annotations(ann_path, store)
    return "", 204


@app.route("/api/way_node/restore", methods=["PUT"])
def restore_way_node():
    filename = request.args.get("file")
    way_id   = request.args.get("way_id")
    node_id  = request.args.get("node_id")
    if not filename or way_id is None or node_id is None:
        abort(400, "Missing required query parameters")
    try:
        way_id  = int(way_id)
        node_id = int(node_id)
    except (ValueError, TypeError):
        abort(400, "way_id and node_id must be integers")
    ann_path = _annotation_path(filename)
    store = _load_annotations(ann_path)
    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
    store["deleted_nodes"] = [
        d for d in dn if not (d["way_id"] == way_id and d["node_id"] == node_id)
    ]
    _save_annotations(ann_path, store)
    return "", 204


@app.route("/api/export")
def export_mapdata():
    filename = request.args.get("file")
    if not filename:
        abort(400, "Missing 'file' query parameter")
    path = os.path.join(_get_data_dir(), filename)
    if not os.path.isfile(path):
        abort(404, f"File not found: {filename}")

    store = _load_annotations(_annotation_path(filename))
    md = copy.deepcopy(_load_mapdata_cached(path))
    zn, zl = md.zone_number, md.zone_letter

    # Apply OSM deletions
    deleted_way_ids = _get_deleted_way_ids(store)
    has_node_dels   = bool(store.get("deleted_nodes"))
    if deleted_way_ids or has_node_dels:
        for lst_name in ("roads_list", "footways_list", "barriers_list"):
            new_lst = []
            for w in getattr(md, lst_name):
                if w.id in deleted_way_ids:
                    continue
                del_nids = _get_deleted_node_ids(store, w.id)
                if del_nids:
                    w = _rebuild_way_without_nodes(w, del_nids)
                    if w is None:
                        continue
                new_lst.append(w)
            setattr(md, lst_name, new_lst)

    # Apply tag overrides
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
        geom = _geojson_geom_to_utm(ann["geometry"], zn, zl)
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


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive map data viewer")
    parser.add_argument("--data-dir", help="Directory containing .mapdata files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.data_dir:
        app.config["DATA_DIR"] = os.path.realpath(args.data_dir)

    logging.basicConfig(level=logging.INFO)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
