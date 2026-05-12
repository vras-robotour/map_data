import copy
import json
import os

import numpy as np
import utm
from shapely.affinity import translate as _affine_translate
from shapely.geometry import (
    LineString as _SLS,
    MultiPolygon as _SMPoly,
    Polygon as _SPoly,
)

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
    if hasattr(map_data, "crossroads_list"):
        add_ways(map_data.crossroads_list, "crossroad")

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


def get_split_node_ids(store, way_id):
    """Return list of node IDs where the given way should be split."""
    splits = store.get("split_ways") or {}
    way_splits = splits.get(str(way_id), [])
    # Ensure they are integers for comparison with OSM node IDs
    return [int(nid) for nid in way_splits]


def split_way(way, split_nids, zone_number=None, zone_letter=None, nodes_cache=None):
    """Split a way at specified node IDs into a list of new Way objects."""
    if not split_nids:
        return [way]

    geom = way.line
    if geom is None:
        return [way]

    node_ids = [getattr(n, "id", n) for n in way.nodes]
    if not node_ids:
        return [way]

    # Check if closed
    if node_ids[0] == node_ids[-1]:
        return [way]

    # If it's a Polygon, we assume it's a buffered path since it's not closed at node level
    is_buffered = geom.geom_type == "Polygon"

    # Calculate buffer radius if buffered
    radius = 0
    if is_buffered:
        p = geom.length
        a = geom.area
        disc = p * p - 4 * np.pi * a
        radius = (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi) if disc >= 0 else a / p

    # Reconstruct centerline coordinates
    raw_coords = []
    if not is_buffered and geom.geom_type == "LineString":
        raw_coords = list(geom.coords)
    elif zone_number is not None and zone_letter is not None:
        # Reconstruct centerline from node positions
        nc = nodes_cache or {}
        for nid in node_ids:
            lat, lon = None, None
            # Check if node object has lat/lon
            for n_obj in way.nodes:
                if getattr(n_obj, "id", None) == nid:
                    lat, lon = getattr(n_obj, "lat", None), getattr(n_obj, "lon", None)
                    break
            # Fallback to cache
            if lat is None and nid in nc:
                lat, lon = nc[nid]["lat"], nc[nid]["lon"]

            if lat is not None and lon is not None:
                e, nn, _, _ = utm.from_latlon(
                    float(lat),
                    float(lon),
                    force_zone_number=zone_number,
                    force_zone_letter=zone_letter,
                )
                raw_coords.append((e, nn))

    if not raw_coords:
        return [way]

    segments = []
    current_nodes = []
    current_coords = []

    split_set = set(int(nid) for nid in split_nids)
    for i, nid in enumerate(node_ids):
        current_nodes.append(way.nodes[i])
        if i < len(raw_coords):
            current_coords.append(raw_coords[i])

        if nid in split_set and i > 0 and i < len(node_ids) - 1:
            # Split point reached
            if len(current_nodes) >= 2:
                w = copy.copy(way)
                w.nodes = current_nodes
                ls = _SLS(current_coords)
                w.line = ls.buffer(radius) if is_buffered else ls
                segments.append(w)

            # Start new segment with the split node
            current_nodes = [way.nodes[i]]
            current_coords = [raw_coords[i]]

    # Last segment
    if len(current_nodes) >= 2:
        w = copy.copy(way)
        w.nodes = current_nodes
        ls = _SLS(current_coords)
        w.line = ls.buffer(radius) if is_buffered else ls
        segments.append(w)

    if not segments or len(segments) <= 1:
        return [way]

    # Update IDs to virtual IDs original_id:index
    for i, seg in enumerate(segments):
        seg.id = f"{way.id}:{i}"

    return segments


_MIGRATION_VERSION = "v2"


def migrate_change_log(store):
    """Ensure change_log covers all existing changes with proportional way/node interleaving.

    On first call (or when migration is outdated): entries without a "ts" key are
    considered legacy and are replaced by a fresh proportionally-interleaved block.
    Entries that carry a "ts" (recorded by user actions) are preserved as-is.
    Idempotent once migration_version matches.
    """
    if store.get("change_log_migration") != _MIGRATION_VERSION:
        # Drop legacy entries (no "ts") added by a previous grouped-order migration;
        # keep user-action entries that already have a timestamp.
        cl_kept = [e for e in store.get("change_log", []) if "ts" in e]
        store["change_log"] = cl_kept

    cl = store.setdefault("change_log", [])
    tracked_ways = {e["id"] for e in cl if e.get("type") == "way"}
    tracked_tags = {e["id"] for e in cl if e.get("type") == "tag"}
    tracked_nodes = {(e["way_id"], e["node_id"]) for e in cl if e.get("type") == "node"}

    untracked_ways = []
    for d in store.get("deleted_ways", []):
        wid = d["id"] if isinstance(d, dict) else d
        if wid not in tracked_ways:
            untracked_ways.append({"type": "way", "id": wid})

    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        dn = [{"way_id": int(k), "node_id": v} for k, vs in dn.items() for v in vs]
    untracked_nodes = []
    for d in dn:
        if isinstance(d, dict):
            key = (d["way_id"], d["node_id"])
            if key not in tracked_nodes:
                untracked_nodes.append(
                    {"type": "node", "way_id": d["way_id"], "node_id": d["node_id"]}
                )

    untracked_tags = []
    for wid_str in store.get("tag_overrides", {}):
        wid = int(wid_str)
        if wid not in tracked_tags:
            untracked_tags.append({"type": "tag", "id": wid})

    tracked_moves = {e["id"] for e in cl if e.get("type") == "move"}
    untracked_moves = []
    for wid_str in store.get("node_position_overrides", {}):
        wid = int(wid_str)
        if wid not in tracked_moves:
            untracked_moves.append(
                {"type": "move", "id": wid, "category": "unknown", "label": ""}
            )

    tracked_splits = {
        (e.get("way_id"), e.get("node_id")) for e in cl if e.get("type") == "split"
    }
    untracked_splits = []
    for wid_str, nids in store.get("split_ways", {}).items():
        wid = int(wid_str)
        for nid in nids:
            if (wid, nid) not in tracked_splits:
                untracked_splits.append(
                    {"type": "split", "way_id": wid, "node_id": nid}
                )

    if (
        untracked_ways
        or untracked_nodes
        or untracked_tags
        or untracked_moves
        or untracked_splits
    ):
        nw, nn = len(untracked_ways), len(untracked_nodes)
        if nw == 0 or nn == 0:
            interleaved = untracked_ways + untracked_nodes
        else:
            # Spread ways and nodes uniformly using normalised midpoint positions.
            items = [((i + 0.5) / nw, e) for i, e in enumerate(untracked_ways)] + [
                ((j + 0.5) / nn, e) for j, e in enumerate(untracked_nodes)
            ]
            items.sort(key=lambda x: x[0])
            interleaved = [e for _, e in items]
        store["change_log"] = (
            interleaved + untracked_tags + untracked_moves + untracked_splits + cl
        )

    store["change_log_migration"] = _MIGRATION_VERSION


def get_node_position_overrides(store, way_id):
    """Return {node_id (int): {lat, lon}} for position overrides on a given way."""
    return {
        int(k): v
        for k, v in store.get("node_position_overrides", {})
        .get(str(way_id), {})
        .items()
    }


def apply_node_position_overrides(
    way, overrides, zone_number, zone_letter, nodes_cache=None, category=None
):
    """Return a copy of way with geometry updated from node position overrides.

    overrides: {node_id (int): {"lat": float, "lon": float}}
    For non-overridden nodes, nodes_cache is consulted before falling back to
    geometry coordinates (geometry coords are unreliable for buffered polygons).
    """
    if not overrides:
        return way
    geom = way.line
    node_ids = [getattr(n, "id", n) for n in way.nodes]
    if not node_ids:
        # No OSM nodes (e.g. individual barrier node): translate geometry by centroid shift.
        if overrides:
            first_ov = next(iter(overrides.values()))
            e_new, n_new, _, _ = utm.from_latlon(
                float(first_ov["lat"]),
                float(first_ov["lon"]),
                force_zone_number=zone_number,
                force_zone_letter=zone_letter,
            )
            centroid = geom.centroid
            w = copy.copy(way)
            w.line = _affine_translate(
                geom, xoff=e_new - centroid.x, yoff=n_new - centroid.y
            )
            return w
        return way

    nc = nodes_cache or {}
    # Geometry coords as fallback only — may be unreliable for buffered polygons
    raw = list(geom.exterior.coords if hasattr(geom, "exterior") else geom.coords)
    geom_latlon = [utm.to_latlon(e, n, zone_number, zone_letter) for e, n in raw]

    utm_coords = []
    for i, nid in enumerate(node_ids):
        if nid in overrides:
            lat = float(overrides[nid]["lat"])
            lon = float(overrides[nid]["lon"])
        elif nid in nc:
            lat = float(nc[nid]["lat"])
            lon = float(nc[nid]["lon"])
        elif i < len(geom_latlon):
            lat, lon = geom_latlon[i]
        else:
            continue
        e, n, _, _ = utm.from_latlon(
            lat,
            lon,
            force_zone_number=zone_number,
            force_zone_letter=zone_letter,
        )
        utm_coords.append((e, n))

    if len(utm_coords) < 2:
        return way

    # Preserve closure. Primary signal: closed OSM ways have the same node ID at first
    # and last position (node_ids[0] == node_ids[-1]).  Fallback: geom.is_closed (only
    # valid for LineString).  Apply before the geom-type branch so both LineString ways
    # and buffered Polygon barriers get a closed centerline — otherwise ls.buffer(r)
    # produces a capsule instead of a ring for closed barriers.
    _is_closed = (len(node_ids) >= 2 and node_ids[0] == node_ids[-1]) or (
        geom.geom_type == "LineString" and geom.is_closed
    )
    if _is_closed and utm_coords[0] != utm_coords[-1]:
        utm_coords.append(utm_coords[0])

    w = copy.copy(way)
    if geom.geom_type == "LineString":
        w.line = _SLS(utm_coords)
    elif geom.geom_type == "Polygon":
        if len(utm_coords) < 2:
            return way
        if _is_closed:
            if category == "barrier":
                # Closed barrier area: reconstruct as flat Polygon from the node ring
                ring = (
                    utm_coords
                    if utm_coords[0] == utm_coords[-1]
                    else utm_coords + [utm_coords[0]]
                )
                if len(ring) < 4:
                    return way
                try:
                    w.line = _SPoly(ring)
                except Exception:
                    return way
            else:
                # Closed road/footway: flat Polygon if area=yes, else re-buffer the loop
                is_area_way = (getattr(way, "tags", None) or {}).get("area") == "yes"
                if utm_coords[0] != utm_coords[-1]:
                    utm_coords.append(utm_coords[0])
                if is_area_way:
                    if len(utm_coords) < 4:
                        return way
                    try:
                        w.line = _SPoly(utm_coords)
                    except Exception:
                        return way
                else:
                    ls = _SLS(utm_coords)
                    p = geom.length
                    a = geom.area
                    disc = p * p - 4 * np.pi * a
                    r = (
                        (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi)
                        if disc >= 0
                        else (a / p if p else 0)
                    )
                    try:
                        w.line = ls.buffer(max(r, 0.01))
                    except Exception:
                        if len(utm_coords) >= 4:
                            try:
                                w.line = _SPoly(utm_coords)
                            except Exception:
                                return way
                        else:
                            return way
        else:
            # Open way stored as buffered polygon: re-buffer the centerline
            ls = _SLS(utm_coords)
            p = geom.length
            a = geom.area
            disc = p * p - 4 * np.pi * a
            r = (
                (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi)
                if disc >= 0
                else (a / p if p else 0)
            )
            try:
                w.line = ls.buffer(max(r, 0.01))
            except Exception:
                if len(utm_coords) >= 3:
                    closed = utm_coords + [utm_coords[0]]
                    try:
                        w.line = _SPoly(closed)
                    except Exception:
                        return way
                else:
                    return way
    else:
        return way
    return w


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
    way, del_nids, zone_number=None, zone_letter=None, nodes_cache=None, category=None
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
        # Preserve closure: primary signal is node_ids[0] == node_ids[-1] (closed OSM
        # way); geom.is_closed is the fallback for geometries that stored the repeat.
        _is_closed = (
            len(node_ids) >= 2 and node_ids[0] == node_ids[-1]
        ) or geom.is_closed
        if _is_closed and new_coords[0] != new_coords[-1]:
            new_coords.append(new_coords[0])
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
            _is_closed_orig = len(node_ids) >= 2 and node_ids[0] == node_ids[-1]
            if _is_closed_orig:
                if category == "barrier":
                    # Closed barrier area: reconstruct as flat Polygon from the node ring
                    ring = (
                        utm_coords
                        if utm_coords[0] == utm_coords[-1]
                        else utm_coords + [utm_coords[0]]
                    )
                    if len(ring) < 4:
                        return None
                    try:
                        w.line = _SPoly(ring)
                    except Exception:
                        return None
                else:
                    # Closed road/footway: flat Polygon if area=yes, else re-buffer the loop
                    is_area_way = (getattr(way, "tags", None) or {}).get(
                        "area"
                    ) == "yes"
                    if utm_coords[0] != utm_coords[-1]:
                        utm_coords.append(utm_coords[0])
                    if is_area_way:
                        if len(utm_coords) < 4:
                            return None
                        try:
                            w.line = _SPoly(utm_coords)
                        except Exception:
                            return None
                    else:
                        ls = _SLS(utm_coords)
                        p = geom.length
                        a = geom.area
                        disc = p * p - 4 * np.pi * a
                        r = (
                            (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi)
                            if disc >= 0
                            else a / p
                        )
                        try:
                            w.line = ls.buffer(r)
                        except Exception:
                            w.line = ls
            else:
                # Open way stored as buffered polygon: re-buffer the centerline
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
