import os
import copy
import json
import utm
import numpy as np
from shapely.geometry import (
    LineString as _SLS,
    Polygon as _SPoly,
    MultiPolygon as _SMPoly,
)
from map_data.way import FOOTWAY_VALUES

# ------------------------------------------------------------------
# GeoJSON conversion helpers
# ------------------------------------------------------------------

def ring_to_latlon(coords, zone_number, zone_letter):
    result = []
    for x, y in coords:
        lat, lon = utm.to_latlon(x, y, zone_number, zone_letter)
        result.append([lon, lat])
    return result


def geom_to_geojson(geom, zone_number, zone_letter):
    gtype = geom.geom_type
    if gtype == "Polygon":
        exterior = ring_to_latlon(geom.exterior.coords, zone_number, zone_letter)
        interiors = [
            ring_to_latlon(r.coords, zone_number, zone_letter) for r in geom.interiors
        ]
        return {"type": "Polygon", "coordinates": [exterior] + interiors}
    if gtype == "MultiPolygon":
        polygons = []
        for poly in geom.geoms:
            exterior = ring_to_latlon(poly.exterior.coords, zone_number, zone_letter)
            interiors = [
                ring_to_latlon(r.coords, zone_number, zone_letter)
                for r in poly.interiors
            ]
            polygons.append([exterior] + interiors)
        return {"type": "MultiPolygon", "coordinates": polygons}
    if gtype == "LineString":
        return {
            "type": "LineString",
            "coordinates": ring_to_latlon(geom.coords, zone_number, zone_letter),
        }
    return None


def mapdata_to_geojson(map_data):
    features = []
    zn, zl = map_data.zone_number, map_data.zone_letter

    def add_ways(ways, category):
        for way in ways:
            try:
                geom = geom_to_geojson(way.line, zn, zl)
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
                        "is_node": category == "barrier" and not bool(way.nodes),
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

def load_annotations(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"version": 1, "annotations": []}


def save_annotations(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_deleted_way_ids(store):
    """Return set of deleted way IDs, handling both old (int list) and new (dict list) formats."""
    dw = store.get("deleted_ways", [])
    return {(d["id"] if isinstance(d, dict) else d) for d in dw}


def get_deleted_node_ids(store, way_id):
    """Return set of deleted node IDs for a given way_id, handling both storage formats."""
    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        return set(dn.get(str(way_id), []))
    return {d["node_id"] for d in dn if d["way_id"] == way_id}


# ------------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------------

def geojson_geom_to_utm(geometry, zone_number, zone_letter):
    """GeoJSON geometry (lon/lat) → Shapely geometry (UTM, same zone as mapdata)."""

    def pt(c):
        e, n, _, _ = utm.from_latlon(
            c[1],
            c[0],
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


def rebuild_way_without_nodes(
    way, del_nids, zone_number=None, zone_letter=None, nodes_cache=None
):
    """Return a shallow copy of way with del_nids removed, or None if geometry becomes invalid."""
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
            nc = nodes_cache or {}
            utm_coords = []
            for n in w.nodes:
                lat = getattr(n, "lat", None)
                lon = getattr(n, "lon", None)
                if lat is None:
                    nd = nc.get(getattr(n, "id", n))
                    if nd:
                        lat, lon = nd["lat"], nd["lon"]
                if lat is not None and lon is not None:
                    e, nn, _, _ = utm.from_latlon(
                        float(lat),
                        float(lon),
                        force_zone_number=zone_number,
                        force_zone_letter=zone_letter,
                    )
                    utm_coords.append((e, nn))
            if len(utm_coords) < 2:
                return None
            ls = _SLS(utm_coords)
            p = geom.length
            a = geom.area
            disc = p * p - 4 * np.pi * a
            r = (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi) if disc >= 0 else a / p
            try:
                w.line = ls.buffer(r)
            except Exception:
                w.line = ls
        else:
            coords = list(geom.exterior.coords)
            new_coords = [coords[i] for i in keep if i < len(coords)]
            if len(new_coords) < 3:
                return None
            if new_coords[0] != new_coords[-1]:
                new_coords.append(new_coords[0])
            w.line = _SPoly(new_coords)

    else:
        return None

    return w
