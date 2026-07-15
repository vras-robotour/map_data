"""
Support functions for the interactive way-editing viewer.

This module has two halves. The first converts between :class:`MapData`
UTM geometry and lon/lat GeoJSON for the web viewer, and persists user
annotations (deletions, splits, tag overrides, node moves/additions) to a
JSON store. The second half is the "way-edit pipeline": functions that
take a stored edit (a set of deleted/moved/added/split node IDs) and
rebuild a :class:`~map_data.utils.way.Way`'s ``nodes`` list and Shapely
``line`` geometry to reflect it, handling both plain ``LineString`` ways
and ways stored as a buffered ``Polygon`` (paths/roads drawn with width).
"""

import contextlib
import copy
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import utm
from shapely.affinity import translate as _affine_translate
from shapely.geometry import (
    LineString as _LineString,
)
from shapely.geometry import (
    MultiPolygon as _SMPoly,
)
from shapely.geometry import (
    Polygon as _SPoly,
)

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from map_data.map_data import MapData
    from map_data.utils.way import Way

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# GeoJSON conversion helpers
# ------------------------------------------------------------------


def ring_to_latlon(
    coords: list[tuple[float, float]],
    zone_number: int,
    zone_letter: str,
) -> list[list[float]]:
    """
    Convert a ring/line of UTM coordinates to GeoJSON-ordered lon/lat pairs.

    Parameters
    ----------
    coords : list of (float, float)
        Sequence of ``(easting, northing)`` UTM coordinates.
    zone_number : int
        UTM zone number shared by all coordinates.
    zone_letter : str
        UTM zone letter shared by all coordinates.

    Returns
    -------
    list of [float, float]
        One ``[lon, lat]`` pair per input coordinate, in GeoJSON axis order
        (longitude first).

    """
    result = []
    for x, y in coords:
        lat, lon = utm.to_latlon(x, y, zone_number, zone_letter)
        result.append([lon, lat])
    return result


def geom_to_geojson(
    geom: "_LineString | _SPoly | _SMPoly",
    zone_number: int,
    zone_letter: str,
) -> dict[str, Any] | None:
    """
    Convert a single Shapely UTM geometry to a GeoJSON geometry dict.

    Handles ``Polygon`` (including interior holes), ``MultiPolygon``, and
    ``LineString`` geometries; all other geometry types return ``None``.
    ``None`` is also returned if *geom*'s declared ``geom_type`` does not
    match its actual Python class (e.g. a proxy/mixed object), since the
    coordinate extraction below is type-specific.

    Parameters
    ----------
    geom : LineString, Polygon, or MultiPolygon
        Shapely geometry in UTM coordinates.
    zone_number : int
        UTM zone number of *geom*.
    zone_letter : str
        UTM zone letter of *geom*.

    Returns
    -------
    dict or None
        GeoJSON geometry object (``{"type": ..., "coordinates": ...}``)
        with coordinates converted to lon/lat via :func:`ring_to_latlon`,
        or ``None`` if the geometry type is unsupported.

    """
    gtype = geom.geom_type
    if gtype == "Polygon":
        if not isinstance(geom, _SPoly):
            return None
        exterior = ring_to_latlon(geom.exterior.coords, zone_number, zone_letter)
        interiors = [ring_to_latlon(r.coords, zone_number, zone_letter) for r in geom.interiors]
        return {"type": "Polygon", "coordinates": [exterior, *interiors]}
    if gtype == "MultiPolygon":
        if not isinstance(geom, _SMPoly):
            return None
        polygons = []
        for poly in geom.geoms:
            exterior = ring_to_latlon(poly.exterior.coords, zone_number, zone_letter)
            interiors = [ring_to_latlon(r.coords, zone_number, zone_letter) for r in poly.interiors]
            polygons.append([exterior, *interiors])
        return {"type": "MultiPolygon", "coordinates": polygons}
    if gtype == "LineString":
        if not isinstance(geom, _LineString):
            return None
        return {
            "type": "LineString",
            "coordinates": ring_to_latlon(geom.coords, zone_number, zone_letter),
        }
    return None


def mapdata_to_geojson(map_data: "MapData") -> dict[str, Any]:
    """
    Convert a :class:`MapData` instance into a single GeoJSON FeatureCollection.

    Emits one Feature per way in ``roads_list``, ``footways_list``,
    ``barriers_list``, and (if present) ``crossroads_list``, tagged with a
    ``"category"`` property (``"road"``, ``"footway"``, ``"barrier"``, or
    ``"crossroad"``), plus one Point Feature per waypoint (category
    ``"waypoint"``). Ways whose geometry fails to convert (e.g. an
    unsupported type, per :func:`geom_to_geojson`) are logged and skipped
    rather than raising.

    Each way Feature's ``properties`` include:

    - ``is_node`` : ``True`` only for barrier features that have no
      constituent OSM nodes (i.e. features synthesized from a single
      obstacle node rather than parsed from a way).
    - ``in_out`` : the way's :attr:`~map_data.utils.way.Way.in_out`
      direction hint, as set during parsing.

    Parameters
    ----------
    map_data : MapData
        Source of ways, waypoints, and the UTM zone used to project
        everything back to lon/lat.

    Returns
    -------
    dict
        A GeoJSON ``{"type": "FeatureCollection", "features": [...]}`` dict.

    """
    features = []
    zn, zl = map_data.zone_number, map_data.zone_letter

    def add_ways(ways: list["Way"], category: str) -> None:
        for way in ways:
            try:
                geom = geom_to_geojson(way.line, zn, zl) if way.line else None  # type: ignore[arg-type]
            except (ValueError, TypeError) as e:
                logger.warning("Failed to convert geometry for way %s: %s", way.id, e)
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
                },
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
            },
        )

    return {"type": "FeatureCollection", "features": features}


# Per-file lock for thread-safe annotation writes
_annotation_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_annotation_lock(path: str) -> threading.Lock:
    """
    Return the process-wide lock guarding writes to a given annotation file.

    Locks are created lazily and keyed by *path* (as given, not resolved),
    so callers should pass a consistent path string for the same file to
    actually serialize on the same lock. Locks are never removed, so the
    ``_annotation_locks`` dict grows by one entry per distinct path for the
    life of the process.

    Parameters
    ----------
    path : str
        Path to the annotation JSON file, used as the lock dict key.

    Returns
    -------
    threading.Lock
        The lock instance associated with *path*.

    """
    with _locks_lock:
        if path not in _annotation_locks:
            _annotation_locks[path] = threading.Lock()
        return _annotation_locks[path]


def load_annotations(path: str) -> dict[str, Any]:
    """
    Load the annotation store from *path*, or return a fresh empty store.

    An empty/default store (``{"version": 1, "annotations": []}``) is
    returned both when *path* does not exist and when it exists but
    contains invalid JSON (in the latter case a warning is logged; the
    corrupt file itself is left untouched on disk).

    Parameters
    ----------
    path : str
        Path to the annotation JSON file.

    Returns
    -------
    dict
        The parsed annotation store, or a fresh default store.

    """
    p = Path(path)
    if p.is_file():
        try:
            with p.open() as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Corrupt annotation file %s, returning empty store", path)
    return {"version": 1, "annotations": []}


def save_annotations(path: str, data: dict[str, Any]) -> None:
    """
    Atomically write the annotation store to *path*.

    Serializes *data* as indented JSON to a temp file in the same
    directory, then ``os.replace``s it over *path* — so readers never see a
    partially written file. Writes are serialized per-path via
    :func:`_get_annotation_lock` to guard against concurrent writers in the
    same process; the temp file is cleaned up if writing fails.

    Parameters
    ----------
    path : str
        Destination path for the annotation JSON file.
    data : dict
        Annotation store to serialize.

    """
    lock = _get_annotation_lock(path)
    with lock:
        p = Path(path)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, str(p))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


def get_deleted_way_ids(store: dict[str, Any]) -> set[int | str]:
    """
    Return the set of way IDs deleted in *store*.

    Supports both the legacy ``store["deleted_ways"]`` format (a flat list
    of raw IDs) and the current format (a list of dicts with an ``"id"``
    key, which additionally carries metadata like a timestamp). Way IDs may
    be ``int`` (original OSM ways) or ``str`` (virtual IDs of the form
    ``"<original_id>:<segment_index>"`` produced by :func:`split_way`).

    Parameters
    ----------
    store : dict
        Annotation store, as returned by :func:`load_annotations`.

    Returns
    -------
    set of int or str
        IDs of all deleted ways.

    """
    dw = store.get("deleted_ways", [])
    return {(d["id"] if isinstance(d, dict) else d) for d in dw}


def get_deleted_node_ids(store: dict[str, Any], way_id: int | str) -> set[int]:
    """
    Return the set of deleted node IDs belonging to a given way.

    Supports both the legacy ``store["deleted_nodes"]`` format (a dict
    mapping ``str(way_id)`` to a list of node IDs) and the current format
    (a flat list of ``{"way_id": ..., "node_id": ...}`` dicts).

    Parameters
    ----------
    store : dict
        Annotation store, as returned by :func:`load_annotations`.
    way_id : int or str
        ID of the way to look up deleted nodes for.

    Returns
    -------
    set of int
        OSM node IDs deleted from *way_id*.

    """
    dn = store.get("deleted_nodes", [])
    if isinstance(dn, dict):
        return set(dn.get(str(way_id), []))
    return {d["node_id"] for d in dn if d["way_id"] == way_id}


def get_split_node_ids(store: dict[str, Any], way_id: int | str) -> list[int]:
    """
    Return the node IDs at which a given way should be split.

    Looked up in ``store["split_ways"]`` under the string key
    ``str(way_id)``. Intended to be passed straight to :func:`split_way`
    along with the way object.

    Parameters
    ----------
    store : dict
        Annotation store, as returned by :func:`load_annotations`.
    way_id : int or str
        ID of the way to look up split points for.

    Returns
    -------
    list of int
        Node IDs (as ints, coerced for comparison against OSM node IDs) at
        which *way_id* should be split. Empty if no splits are recorded.

    """
    splits = store.get("split_ways") or {}
    way_splits = splits.get(str(way_id), [])
    # Ensure they are integers for comparison with OSM node IDs
    return [int(nid) for nid in way_splits]


def split_way(
    way: "Way",
    split_nids: list[int],
    zone_number: int | None = None,
    zone_letter: str | None = None,
    nodes_cache: dict[int, dict[str, Any]] | None = None,
) -> list["Way"]:
    """
    Split a way into segments at the given interior node IDs.

    Only interior nodes are honoured as split points — a split ID matching
    the first or last node of *way* is ignored, since it would produce a
    zero-length segment. Returns the original *way* unchanged (as a
    single-element list) for a number of cases where splitting is not
    applicable or not possible: no *split_nids* given, no geometry, no
    nodes, a closed way (first and last node IDs equal — splitting a loop
    is not supported), or fewer than two resulting segments.

    For ways whose geometry is a buffered ``Polygon`` (paths/roads with a
    drawn width rather than a bare centerline), the buffer radius is
    reconstructed from the isoperimetric relation between the polygon's
    perimeter and area (``r = (p - sqrt(p^2 - 4*pi*a)) / (2*pi)``, falling
    back to ``a / p`` if the discriminant is negative), and each output
    segment's geometry is re-buffered from its centerline by that same
    radius. For a plain ``LineString`` way, node coordinates are taken
    directly from the geometry; otherwise (buffered case, or missing
    node-level coordinates) the centerline is reconstructed from node
    lat/lon — first from ``way.nodes`` objects' own ``lat``/``lon``
    attributes if present, else from *nodes_cache* — which requires
    *zone_number* and *zone_letter* to reproject back to UTM.

    Parameters
    ----------
    way : Way
        Way to split. Must have a non-``None`` ``line`` and a non-closed
        ``nodes`` list to be split at all.
    split_nids : list of int
        Interior node IDs at which to cut the way.
    zone_number : int, optional
        UTM zone number, required to reconstruct centerline coordinates
        from node lat/lon when the geometry itself doesn't carry per-node
        coordinates (buffered-polygon ways).
    zone_letter : str, optional
        UTM zone letter, paired with *zone_number*.
    nodes_cache : dict, optional
        Fallback ``{node_id: {"lat": ..., "lon": ...}}`` lookup used when a
        node object in ``way.nodes`` doesn't carry its own coordinates.

    Returns
    -------
    list of Way
        Two or more segments, each a shallow copy of *way* with its own
        ``nodes`` slice and recomputed ``line``, and with ``id`` rewritten
        to ``f"{way.id}:{index}"`` (a virtual ID); or ``[way]`` unchanged if
        splitting did not apply or did not produce multiple segments.

    """
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

    split_set = {int(nid) for nid in split_nids}
    for i, nid in enumerate(node_ids):
        current_nodes.append(way.nodes[i])
        if i < len(raw_coords):
            current_coords.append(raw_coords[i])

        if nid in split_set and i > 0 and i < len(node_ids) - 1:
            # Split point reached
            if len(current_nodes) >= 2:
                w = copy.copy(way)
                w.nodes = current_nodes
                ls = _LineString(current_coords)
                w.line = ls.buffer(radius) if is_buffered else ls
                segments.append(w)

            # Start new segment with the split node
            current_nodes = [way.nodes[i]]
            current_coords = [raw_coords[i]]

    # Last segment
    if len(current_nodes) >= 2:
        w = copy.copy(way)
        w.nodes = current_nodes
        ls = _LineString(current_coords)
        w.line = ls.buffer(radius) if is_buffered else ls
        segments.append(w)

    if not segments or len(segments) <= 1:
        return [way]

    # Update IDs to virtual IDs original_id:index
    for i, seg in enumerate(segments):
        seg.id = f"{way.id}:{i}"

    return segments


_MIGRATION_VERSION = "v2"


def migrate_change_log(store: dict[str, Any]) -> None:
    """
    Ensure change_log covers all existing changes with proportional way/node interleaving.

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
                    {"type": "node", "way_id": d["way_id"], "node_id": d["node_id"]},
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
            untracked_moves.append({"type": "move", "id": wid, "category": "unknown", "label": ""})

    tracked_splits = {(e.get("way_id"), e.get("node_id")) for e in cl if e.get("type") == "split"}
    untracked_splits = []
    for wid_str, nids in store.get("split_ways", {}).items():
        wid = int(wid_str)
        untracked_splits.extend(
            {"type": "split", "way_id": wid, "node_id": nid}
            for nid in nids
            if (wid, nid) not in tracked_splits
        )

    if untracked_ways or untracked_nodes or untracked_tags or untracked_moves or untracked_splits:
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
        store["change_log"] = interleaved + untracked_tags + untracked_moves + untracked_splits + cl

    store["change_log_migration"] = _MIGRATION_VERSION


def get_node_position_overrides(
    store: dict[str, Any],
    way_id: int | str,
) -> dict[int, dict[str, float]]:
    """
    Return the node position overrides recorded for a given way.

    *way_id* is normalized to its original OSM way ID by stripping any
    ``":<segment_index>"`` suffix, so overrides recorded before a way was
    split (under the original ID) are still found via any of its split
    segment IDs.

    Parameters
    ----------
    store : dict
        Annotation store, as returned by :func:`load_annotations`.
    way_id : int or str
        ID of the way (original or a ``"<original_id>:<index>"`` split
        segment ID) to look up overrides for.

    Returns
    -------
    dict of {int: dict}
        Maps node ID to ``{"lat": float, "lon": float}``. Empty if no
        overrides are recorded for the way.

    """
    original_way_id_str = str(way_id).split(":")[0]
    return {
        int(k): v
        for k, v in store.get("node_position_overrides", {}).get(original_way_id_str, {}).items()
    }


def apply_node_position_overrides(
    way: "Way",
    overrides: dict[int, dict[str, float]],
    zone_number: int,
    zone_letter: str,
    nodes_cache: dict[int, dict[str, Any]] | None = None,
    category: str | None = None,
) -> "Way":
    """
    Return a copy of *way* with node positions moved per *overrides*.

    For each node, the new lat/lon is resolved in priority order: an entry
    in *overrides*, else *nodes_cache*, else (last resort) the node's
    coordinate as currently stored in ``way.line`` — geometry coordinates
    are the least trustworthy source because for a buffered ``Polygon``
    way the exterior ring is the buffer outline, not the original
    centerline. A way with no OSM nodes at all (e.g. a synthetic barrier
    point) is instead translated as a rigid body: its geometry is shifted
    so its centroid moves to the first override's position.

    Geometry reconstruction depends on ``way.line``'s type:

    - ``LineString`` : rebuilt directly from the new coordinates.
    - ``Polygon`` : interpreted as a buffered centerline (unless *category*
      is ``"barrier"`` and the way is closed, or the way has an
      ``area=yes`` tag, in which case it is a genuine flat area and is
      rebuilt as a ``Polygon`` from the node ring directly). Otherwise the
      buffer radius is recovered from the polygon's perimeter/area via the
      isoperimetric relation (clamped to a minimum of ``0.01``) and the new
      centerline is re-buffered by that radius. If re-buffering fails and
      there are enough points, falls back to a flat ``Polygon``; genuinely
      unreconstructable geometry causes the function to return ``None``
      (breaking the ``-> "Way"`` type hint — callers should guard for it).

    Closure (first node ID equals last node ID, or — for a ``LineString``
    with no node IDs — ``geom.is_closed``) is preserved by appending the
    first coordinate to the end of the new coordinate list before
    dispatching on geometry type; this must happen for closed barriers too,
    since buffering an open polyline instead of a closed ring produces a
    rounded capsule shape rather than a proper ring buffer.

    Parameters
    ----------
    way : Way
        Way whose geometry should be updated. Returned unchanged if
        *overrides* is falsy, if fewer than two usable coordinates can be
        resolved, or if reconstruction is not possible for the geometry
        type.
    overrides : dict of {int: dict}
        ``{node_id: {"lat": float, "lon": float}}`` new positions to apply.
    zone_number : int
        UTM zone number to project overridden/cached lat/lon back into.
    zone_letter : str
        UTM zone letter, paired with *zone_number*.
    nodes_cache : dict, optional
        Fallback ``{node_id: {"lat": ..., "lon": ...}}`` lookup for nodes
        not present in *overrides*.
    category : str, optional
        Way category (e.g. ``"barrier"``); only ``"barrier"`` changes
        behaviour, selecting flat-polygon reconstruction for closed areas
        instead of ring re-buffering.

    Returns
    -------
    Way
        A shallow copy of *way* with ``line`` replaced by the updated
        geometry, or the original *way* if no update was applicable.

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
            w.line = _affine_translate(geom, xoff=e_new - centroid.x, yoff=n_new - centroid.y)
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
        w.line = _LineString(utm_coords)
    elif geom.geom_type == "Polygon":
        if len(utm_coords) < 2:
            return way
        if _is_closed:
            if category == "barrier":
                # Closed barrier area: reconstruct as flat Polygon from the node ring
                ring = (
                    utm_coords if utm_coords[0] == utm_coords[-1] else [*utm_coords, utm_coords[0]]
                )
                if len(ring) < 4:
                    return way
                try:
                    w.line = _SPoly(ring)
                except (ValueError, TypeError):
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
                    except (ValueError, TypeError):
                        return way
                else:
                    ls = _LineString(utm_coords)
                    p = geom.length
                    a = geom.area
                    disc = p * p - 4 * np.pi * a
                    r = (
                        (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi)
                        if disc >= 0
                        else (a / p if p else 0)
                    )
                    r = max(r, 0.01)
                    try:
                        w.line = ls.buffer(r)
                    except (ValueError, TypeError):
                        if len(utm_coords) >= 4:
                            try:
                                w.line = _SPoly(utm_coords)
                            except (ValueError, TypeError):
                                return None
                        else:
                            return None
        else:
            # Open way stored as buffered polygon: re-buffer the centerline
            ls = _LineString(utm_coords)
            p = geom.length
            a = geom.area
            disc = p * p - 4 * np.pi * a
            r = (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi) if disc >= 0 else (a / p if p else 0)
            r = max(r, 0.01)
            try:
                w.line = ls.buffer(r)
            except (ValueError, TypeError):
                if len(utm_coords) >= 3:
                    closed = [*utm_coords, utm_coords[0]]
                    try:
                        w.line = _SPoly(closed)
                    except (ValueError, TypeError):
                        return None
                else:
                    return None

    else:
        return way
    return w


# ------------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------------


def geojson_geom_to_utm(
    geometry: dict[str, Any],
    zone_number: int,
    zone_letter: str,
) -> "BaseGeometry | None":
    """
    Convert a GeoJSON geometry dict (lon/lat) to a Shapely geometry (UTM).

    Inverse of :func:`geom_to_geojson`. Supports ``LineString``,
    ``Polygon`` (exterior ring plus any interior/hole rings), and
    ``MultiPolygon``; any other ``"type"`` returns ``None``. Every
    coordinate is projected independently via `utm.from_latlon` with the
    zone forced to *zone_number*/*zone_letter*, so the result lands in the
    same UTM zone as the rest of the map data regardless of where the
    point would naturally fall.

    Parameters
    ----------
    geometry : dict
        GeoJSON geometry object, e.g. ``{"type": "Polygon", "coordinates": [...]}``
        with coordinates in ``[lon, lat]`` order.
    zone_number : int
        UTM zone number to force all coordinates into.
    zone_letter : str
        UTM zone letter to force all coordinates into.

    Returns
    -------
    BaseGeometry or None
        The corresponding ``LineString``, ``Polygon``, or ``MultiPolygon``
        in UTM coordinates, or ``None`` for an unsupported geometry type.

    """

    def pt(c: list[float] | tuple[float, float]) -> tuple[float, float]:
        e, n, _, _ = utm.from_latlon(
            c[1],
            c[0],
            force_zone_number=zone_number,
            force_zone_letter=zone_letter,
        )
        return (e, n)

    gtype = geometry.get("type")
    if gtype == "LineString":
        return _LineString([pt(c) for c in geometry["coordinates"]])
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


def apply_added_nodes(
    way: "Way",
    store: dict[str, Any],
    zone_number: int,
    zone_letter: str,
) -> "Way":
    """
    Return a copy of *way* with user-added synthetic nodes spliced in.

    Reads ``store["added_nodes"]`` entries matching this way's original ID
    (the ``":<index>"`` split suffix, if any, is stripped for the lookup),
    each of the form ``{"id": synth_id, "after_node_id": ..., "lat": ...,
    "lon": ...}``, and inserts ``synth_id`` into ``way.nodes`` immediately
    after ``after_node_id``. An entry whose ``after_node_id`` is not found
    in the current node list (e.g. because that node was itself deleted)
    is silently skipped. Position for the new node comes from a matching
    ``node_position_overrides`` entry if present, else the ``lat``/``lon``
    recorded on the ``added_nodes`` entry itself.

    Only ``LineString`` and ``Polygon`` geometries are handled; any other
    geometry type is returned unchanged. For a ``Polygon`` way, the ring
    stored is the *buffer outline*, not the centerline, so inserting a
    centerline coordinate there would create a spike — geometry is left
    untouched and only ``way.nodes`` is updated, meaning the visual
    line/buffer will not reflect the new node until the way is otherwise
    rebuilt from its node list. For a closed ``LineString`` (last
    coordinate repeats the first), an insertion at the very end is placed
    just before that repeated closing coordinate rather than after it.

    Parameters
    ----------
    way : Way
        Way to augment. Returned unchanged if there are no matching
        ``added_nodes`` entries, if its geometry is not a ``LineString`` or
        ``Polygon``, or if every entry's ``after_node_id`` failed to
        resolve.
    store : dict
        Annotation store, as returned by :func:`load_annotations`.
    zone_number : int
        UTM zone number to project added-node lat/lon into.
    zone_letter : str
        UTM zone letter, paired with *zone_number*.

    Returns
    -------
    Way
        A shallow copy of *way* with updated ``nodes`` (and, for
        ``LineString`` geometry, an updated ``line``), or the original
        *way* if no node was actually inserted.

    """
    original_id = int(str(way.id).split(":")[0])
    added_for_way = [a for a in store.get("added_nodes", []) if a.get("way_id") == original_id]
    if not added_for_way:
        return way

    pos_ov_raw = store.get("node_position_overrides", {}).get(str(original_id), {})

    w = copy.copy(way)
    w.nodes = [getattr(n, "id", n) for n in way.nodes]
    node_ids: list = w.nodes  # mutable reference

    geom = way.line
    is_linestring = geom.geom_type == "LineString"
    is_polygon = geom.geom_type == "Polygon"
    if not is_linestring and not is_polygon:
        return way

    # For Polygon (buffered) ways the exterior ring is the buffer outline, not the
    # centerline.  Inserting a centerline coord there would create a spike and corrupt
    # the buffer shape, so geometry is left unchanged; only way.nodes is updated.
    coords = list(geom.coords) if is_linestring else None

    offset = 0
    for a in added_for_way:
        synth_id = a["id"]
        after_id = a["after_node_id"]

        try:
            idx = node_ids.index(after_id)
        except ValueError:
            continue  # after_node_id was deleted or not found; skip

        insert_pos = idx + 1 + offset
        node_ids.insert(insert_pos, synth_id)

        if coords is not None:
            ov = pos_ov_raw.get(str(synth_id))
            lat = float(ov["lat"] if ov else a["lat"])
            lon = float(ov["lon"] if ov else a["lon"])
            e, n_utm, _, _ = utm.from_latlon(
                lat,
                lon,
                force_zone_number=zone_number,
                force_zone_letter=zone_letter,
            )
            # For closed LineStrings the last coord repeats the first — insert before it.
            _is_closed = len(node_ids) >= 2 and node_ids[0] == node_ids[-1]
            coord_limit = len(coords) - 1 if _is_closed and len(coords) > 1 else len(coords)
            coord_pos = min(insert_pos, coord_limit)
            coords.insert(coord_pos, (e, n_utm))

        offset += 1

    if offset == 0:
        return way

    w.nodes = node_ids
    if coords is not None:
        try:
            w.line = _LineString(coords)
        except (ValueError, TypeError):
            return way

    return w


def rebuild_way_without_nodes(
    way: "Way",
    del_nids: set[int] | list[int],
    zone_number: int | None = None,
    zone_letter: str | None = None,
    nodes_cache: dict[int, dict[str, Any]] | None = None,
    category: str | None = None,
) -> "Way | None":
    """
    Return a copy of *way* with the given node IDs removed from it.

    Filters ``way.nodes`` to drop *del_nids*, then rebuilds ``line`` to
    match. Deletion never merely drops points from the existing geometry
    without also considering closure/buffering, since either could corrupt
    the shape; this largely mirrors the geometry-type handling in
    :func:`apply_node_position_overrides`, keyed off whether *zone_number*
    is given rather than off explicit overrides:

    - ``LineString`` : coordinates are filtered by node index directly
      (``zone_number`` is not needed). Closure (first node ID equals last,
      or ``geom.is_closed`` as a fallback) is preserved by re-appending the
      first coordinate if dropping nodes broke it.
    - ``Polygon`` *with* *zone_number* given : treated as a buffered
      centerline (or, for a closed ``category="barrier"`` way or an
      ``area=yes`` tagged way, a flat area) and reconstructed the same way
      as in :func:`apply_node_position_overrides` — remaining node
      positions are read from each node object's own ``lat``/``lon`` or
      else *nodes_cache*, the loop is re-closed if needed, and (for the
      re-buffer case) a new buffer radius is derived from the *original*
      geometry's perimeter/area via the isoperimetric relation. If
      re-buffering raises, falls back to the bare (unbuffered) centerline
      rather than failing outright.
    - ``Polygon`` *without* *zone_number* : falls back to filtering the
      polygon's own exterior-ring coordinates by index (no node-level
      lat/lon available), re-closing the ring if needed. Requires at least
      3 remaining coordinates.

    Any other geometry type, or a case where filtering leaves fewer than 2
    (LineString) / 3 (Polygon fallback) / 4 (reconstructed ring) usable
    coordinates, causes ``None`` to be returned instead of a Way — this
    signals to callers that the way became degenerate and should be
    dropped entirely, not just left unmodified.

    Parameters
    ----------
    way : Way
        Way to filter. If fewer than 2 nodes remain after removing
        *del_nids*, ``None`` is returned immediately.
    del_nids : set or list of int
        Node IDs to remove from *way*.
    zone_number : int, optional
        UTM zone number; if given, node positions for ``Polygon``
        reconstruction are resolved from node/``nodes_cache`` lat/lon
        instead of the (buffer-outline) geometry coordinates.
    zone_letter : str, optional
        UTM zone letter, paired with *zone_number*.
    nodes_cache : dict, optional
        Fallback ``{node_id: {"lat": ..., "lon": ...}}`` lookup used when a
        node object doesn't carry its own coordinates.
    category : str, optional
        Way category; only ``"barrier"`` changes behaviour, selecting
        flat-polygon reconstruction for closed areas instead of ring
        re-buffering.

    Returns
    -------
    Way or None
        A shallow copy of *way* with ``nodes`` and ``line`` updated, or
        ``None`` if the deletion left the way with unusable/degenerate
        geometry.

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
        # Preserve closure: primary signal is node_ids[0] == node_ids[-1] (closed OSM
        # way); geom.is_closed is the fallback for geometries that stored the repeat.
        _is_closed = (len(node_ids) >= 2 and node_ids[0] == node_ids[-1]) or geom.is_closed
        if _is_closed and new_coords[0] != new_coords[-1]:
            new_coords.append(new_coords[0])
        w.line = _LineString(new_coords)

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
                        else [*utm_coords, utm_coords[0]]
                    )
                    if len(ring) < 4:
                        return None
                    try:
                        w.line = _SPoly(ring)
                    except (ValueError, TypeError):
                        return None
                else:
                    # Closed road/footway: flat Polygon if area=yes, else re-buffer the loop
                    is_area_way = (getattr(way, "tags", None) or {}).get("area") == "yes"
                    if utm_coords[0] != utm_coords[-1]:
                        utm_coords.append(utm_coords[0])
                    if is_area_way:
                        if len(utm_coords) < 4:
                            return None
                        try:
                            w.line = _SPoly(utm_coords)
                        except (ValueError, TypeError):
                            return None
                    else:
                        ls = _LineString(utm_coords)
                        p = geom.length
                        a = geom.area
                        disc = p * p - 4 * np.pi * a
                        r = (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi) if disc >= 0 else a / p
                        try:
                            w.line = ls.buffer(r)
                        except (ValueError, TypeError):
                            w.line = ls
            else:
                # Open way stored as buffered polygon: re-buffer the centerline
                ls = _LineString(utm_coords)
                p = geom.length
                a = geom.area
                disc = p * p - 4 * np.pi * a
                r = (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi) if disc >= 0 else a / p
                try:
                    w.line = ls.buffer(r)
                except (ValueError, TypeError):
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
