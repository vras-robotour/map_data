import os
import re
import json
import pickle
import logging
import uuid

import utm
import numpy as np
from flask import Flask, jsonify, request, render_template, abort

from map_data.map_data import MapData  # noqa: F401 – ensures MapData class is resolvable on pickle load


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
    with open(path, "rb") as f:
        map_data = pickle.load(f)
    return jsonify(_mapdata_to_geojson(map_data))


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
