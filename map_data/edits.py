"""
The JSON annotation store and the way-edit pipeline.

The pipeline rebuilds a :class:`~map_data.utils.way.Way`'s ``nodes`` and
``line`` from stored edits (deleted/moved/added/split nodes), for both
``LineString`` ways and buffered-``Polygon`` ways.
"""

import contextlib
import copy
import json
import logging
import os
import threading
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
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

from map_data.utils.serialization import atomic_write_json

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from map_data.utils.way import Way

logger = logging.getLogger(__name__)


def _to_utm(lat: float, lon: float, zone_number: int, zone_letter: str) -> tuple[float, float]:
    """Project lat/lon to ``(easting, northing)``, forced into the given UTM zone."""
    e, n, _, _ = utm.from_latlon(
        lat, lon, force_zone_number=zone_number, force_zone_letter=zone_letter
    )
    return e, n


# Per-file lock for thread-safe annotation access, held by annotation_store()
# across a whole load -> mutate -> save cycle.
_annotation_locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)


class AnnotationStoreError(Exception):
    """The annotation store exists but is not a readable JSON object."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{Path(path).name}: {reason}")
        self.path = path


def load_annotations(path: str) -> dict[str, Any]:
    """
    Load the annotation store from *path*, or ``{"version": 1, "annotations": []}`` if absent.

    Raises
    ------
    AnnotationStoreError
        If *path* exists but cannot be read, is not valid JSON, or is not a
        JSON object. Treating it as empty would let the next edit overwrite
        every edit it holds.

    """
    p = Path(path)
    if not p.is_file():
        return {"version": 1, "annotations": []}
    try:
        with p.open(encoding="utf-8") as f:
            store = json.load(f)
    except (OSError, ValueError) as e:
        raise AnnotationStoreError(path, str(e)) from e
    if not isinstance(store, dict):
        raise AnnotationStoreError(path, "not a JSON object")
    return store


def quarantine_annotation_store(path: str) -> Path | None:
    """
    Move an unreadable store at *path* aside to ``<path>.corrupt-<timestamp>``.

    Runs under the store's lock and re-checks the file first, so a store
    another request already replaced with a good one is left alone.

    Returns
    -------
    Path or None
        The backup path, or ``None`` if the store is readable (or gone).

    """
    with _annotation_locks[path]:
        try:
            load_annotations(path)
            return None
        except AnnotationStoreError as e:
            backup = Path(f"{path}.corrupt-{datetime.now():%Y%m%dT%H%M%S%f}")
            os.replace(path, backup)
            logger.error("Unreadable annotation store (%s), moved it to %s", e, backup)
            return backup


@contextlib.contextmanager
def annotation_store(path: str) -> Iterator[dict[str, Any]]:
    """
    Yield the store at *path* for in-place mutation, under its lock, and save it on exit.

    The per-path lock is held across load -> mutate -> save, so concurrent
    edits cannot overwrite each other. The store is written (atomically)
    only when the body exits normally; if it raises (including ``abort()``)
    nothing is saved. An unreadable store raises :class:`AnnotationStoreError`
    before the body runs, so it is never overwritten.
    """
    with _annotation_locks[path]:
        store = load_annotations(path)
        yield store
        atomic_write_json(path, store, indent=2)


def get_deleted_way_ids(store: dict[str, Any]) -> set[int | str]:
    """
    Return the IDs of deleted ways in *store*.

    An ID is an ``int`` for a whole way or a ``"<original_id>:<segment_index>"``
    string for a split segment.
    """
    return {d["id"] for d in store.get("deleted_ways", [])}


def get_deleted_node_ids(store: dict[str, Any], way_id: int | str) -> set[int]:
    """
    Return the node IDs deleted from *way_id* (matched exactly, segment suffix included).

    ``store["deleted_nodes"]`` is a flat list of ``{"way_id", "node_id"}`` dicts.
    """
    return {d["node_id"] for d in store.get("deleted_nodes", []) if d["way_id"] == way_id}


def get_split_node_ids(store: dict[str, Any], way_id: int | str) -> list[int]:
    """Return the node IDs *way_id* is split at (``store["split_ways"][str(way_id)]``), as ints."""
    splits = store.get("split_ways") or {}
    way_splits = splits.get(str(way_id), [])
    # Ensure they are integers for comparison with OSM node IDs
    return [int(nid) for nid in way_splits]


def get_detached_node_ids(store: dict[str, Any], way_id: int | str) -> dict[int, int]:
    """
    Map each detached split node of a way to the node that starts the next segment.

    A split made in the viewer detaches the segments: the one after the split
    starts at its own synthetic copy of the split node (``store["detached_nodes"]``
    entries ``{"way_id", "node_id", "id"}``), so each end can be moved alone and
    the planner does not join them. A split without an entry keeps sharing
    the node. *way_id* may carry a ``":<segment_index>"`` suffix.
    """
    original_id = int(str(way_id).split(":")[0])
    return {
        d["node_id"]: d["id"] for d in store.get("detached_nodes", []) if d["way_id"] == original_id
    }


def next_synthetic_node_id(store: dict[str, Any]) -> int:
    """Next free negative node ID, shared by added nodes and detached split ends."""
    ids = [a["id"] for a in store.get("added_nodes", [])]
    ids += [d["id"] for d in store.get("detached_nodes", [])]
    return min([0, *ids]) - 1


def edited_nodes_cache(
    store: dict[str, Any],
    nodes_cache: dict[int, dict[str, Any]] | None,
    way_id: int | str | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Return *nodes_cache* plus the store's synthetic nodes, with moved positions applied.

    Adds user-added nodes and detached split ends, then overlays
    ``node_position_overrides`` -- for *way_id* only (segment suffix
    stripped), or for every way when ``None``. *nodes_cache* is not mutated.
    """
    base = nodes_cache or {}
    wanted = None if way_id is None else int(str(way_id).split(":")[0])
    extra: dict[int, dict[str, Any]] = {}

    for a in store.get("added_nodes", []):
        if wanted in (None, a.get("way_id")):
            extra[a["id"]] = {"lat": float(a["lat"]), "lon": float(a["lon"]), "tags": {}}
    for d in store.get("detached_nodes", []):
        src = extra.get(d["node_id"]) or base.get(d["node_id"])
        if wanted in (None, d["way_id"]) and src:
            extra[d["id"]] = {"lat": src["lat"], "lon": src["lon"], "tags": {}}
    # Overrides are per way but the cache is per node: with way_id=None a node moved
    # in several ways keeps the last one here; apply_way_edits gives such a node a
    # copy per way and restores the original position.
    for wid, way_ov in store.get("node_position_overrides", {}).items():
        if wanted not in (None, int(wid)):
            continue
        for nid_str, pos in way_ov.items():
            nid = int(nid_str)
            src = extra.get(nid) or base.get(nid)
            if src:
                extra[nid] = {**src, "lat": float(pos["lat"]), "lon": float(pos["lon"])}

    return {**base, **extra} if extra else base


def _buffer_radius(geom: "BaseGeometry", min_radius: float = 0.0) -> float:
    """
    Recover a buffered way's original buffer radius from its polygon.

    Uses the isoperimetric relation between perimeter and area (``r = (p -
    sqrt(p^2 - 4*pi*a)) / (2*pi)``), falling back to ``a / p`` if the
    discriminant is negative (or ``0`` if *geom* also has zero perimeter).
    """
    p = geom.length
    a = geom.area
    disc = p * p - 4 * np.pi * a
    r = (p - np.sqrt(max(disc, 0.0))) / (2 * np.pi) if disc >= 0 else (a / p if p else 0)
    return max(r, min_radius)


class _PolygonTooSmall(Exception):
    """Too few coordinates to rebuild a closed ring or flat polygon."""


def _rebuild_polygon(
    geom: "BaseGeometry",
    utm_coords: list[tuple[float, float]],
    is_closed: bool,
    category: str | None,
    tags: dict[str, Any] | None,
    min_radius: float = 0.0,
) -> "BaseGeometry":
    """
    Rebuild a ``Polygon`` way's geometry from new ring/centerline coords.

    A closed barrier, or a closed ``area=yes`` way, is a flat area rebuilt
    directly from *utm_coords*; anything else is a centerline re-buffered by
    :func:`_buffer_radius`. For a closed non-barrier way, *utm_coords* is
    closed in place.

    Raises
    ------
    _PolygonTooSmall
        Too few coordinates for the flat polygon. Buffering errors propagate;
        each caller has its own fallback.

    """
    if is_closed:
        if category == "barrier":
            ring = utm_coords if utm_coords[0] == utm_coords[-1] else [*utm_coords, utm_coords[0]]
            if len(ring) < 4:
                raise _PolygonTooSmall
            try:
                return _SPoly(ring)
            except (ValueError, TypeError):
                raise _PolygonTooSmall from None
        is_area_way = (tags or {}).get("area") == "yes"
        if utm_coords[0] != utm_coords[-1]:
            utm_coords.append(utm_coords[0])
        if is_area_way:
            if len(utm_coords) < 4:
                raise _PolygonTooSmall
            try:
                return _SPoly(utm_coords)
            except (ValueError, TypeError):
                raise _PolygonTooSmall from None
    ls = _LineString(utm_coords)
    return ls.buffer(_buffer_radius(geom, min_radius))


def split_way(
    way: "Way",
    split_nids: list[int],
    zone_number: int,
    zone_letter: str,
    nodes_cache: dict[int, dict[str, Any]] | None,
    detached: dict[int, int] | None = None,
) -> list["Way"]:
    """
    Split a way into segments at the given interior node IDs.

    A ``LineString`` way is cut along its own coordinates; a buffered
    ``Polygon`` way's centerline is rebuilt from *nodes_cache* and each
    segment re-buffered by :func:`_buffer_radius`. Returns ``[way]``
    unchanged when there is nothing to split: no *split_nids*, no geometry
    or nodes, a closed way, or fewer than two resulting segments. Split IDs
    at the first or last node are ignored.

    Parameters
    ----------
    nodes_cache : dict or None
        ``{node_id: {"lat": ..., "lon": ...}}``; needed for ``Polygon`` ways
        and detached split ends.
    detached : dict of {int: int}, optional
        Split node ID -> synthetic node ID that starts the following segment
        instead (see :func:`get_detached_node_ids`).

    Returns
    -------
    list of Way
        Shallow copies of *way* with their own ``nodes`` and ``line`` and
        virtual IDs ``f"{way.id}:{index}"``, or ``[way]``.

    """
    if not split_nids:
        return [way]

    geom = way.line
    if geom is None:
        return [way]

    node_ids = way.nodes
    if not node_ids:
        return [way]

    # Splitting a loop is not supported.
    if node_ids[0] == node_ids[-1]:
        return [way]

    # The way is open at node level, so a Polygon is a buffered centerline.
    is_buffered = geom.geom_type == "Polygon"

    radius = _buffer_radius(geom) if is_buffered else 0

    nc = nodes_cache or {}
    raw_coords = []
    if not is_buffered and geom.geom_type == "LineString":
        raw_coords = list(geom.coords)
    else:
        for nid in node_ids:
            nd = nc.get(nid)
            if nd and nd["lat"] is not None and nd["lon"] is not None:
                raw_coords.append(
                    _to_utm(float(nd["lat"]), float(nd["lon"]), zone_number, zone_letter)
                )

    if not raw_coords:
        return [way]

    segments = []
    current_nodes: list[int] = []
    current_coords: list[tuple[float, float]] = []

    def emit() -> None:
        if len(current_nodes) >= 2:
            w = copy.copy(way)
            w.nodes = current_nodes
            ls = _LineString(current_coords)
            w.line = ls.buffer(radius) if is_buffered else ls
            segments.append(w)

    split_set = {int(nid) for nid in split_nids}
    for i, nid in enumerate(node_ids):
        current_nodes.append(way.nodes[i])
        if i < len(raw_coords):
            current_coords.append(raw_coords[i])

        if nid in split_set and i > 0 and i < len(node_ids) - 1:
            emit()

            # The next segment starts at the split node, or at its detached copy.
            current_nodes = [way.nodes[i]]
            current_coords = [raw_coords[i]]
            did = (detached or {}).get(nid)
            if did is not None:
                current_nodes = [did]
                pos = nc.get(did)
                if pos:
                    current_coords = [
                        _to_utm(float(pos["lat"]), float(pos["lon"]), zone_number, zone_letter)
                    ]

    emit()

    if len(segments) <= 1:
        return [way]

    for i, seg in enumerate(segments):
        seg.id = f"{way.id}:{i}"

    return segments


def get_node_position_overrides(
    store: dict[str, Any],
    way_id: int | str,
) -> dict[int, dict[str, float]]:
    """
    Return ``{node_id: {"lat", "lon"}}`` moves recorded for *way_id*'s original way.

    Overrides are keyed by the original way ID, so any ``":<segment_index>"``
    suffix is stripped and every segment shares them.
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
) -> "Way | None":
    """
    Return a copy of *way* with node positions moved per *overrides*.

    Each node's position comes from *overrides*, else *nodes_cache*, else
    ``way.line`` (least reliable: a ``Polygon``'s ring is the buffer
    outline, not the centerline). A ``LineString`` is rebuilt from the new
    coordinates, a ``Polygon`` by :func:`_rebuild_polygon`. A way without
    nodes (a point barrier) is translated so its centroid lands on the
    first override.

    Parameters
    ----------
    overrides : dict of {int: dict}
        ``{node_id: {"lat": float, "lon": float}}``.
    category : str, optional
        ``"barrier"`` makes a closed ``Polygon`` a flat area rather than a
        buffered ring.

    Returns
    -------
    Way or None
        The edited copy; *way* itself when there is nothing to apply or it
        cannot be rebuilt; ``None`` when a ``Polygon`` can be neither
        re-buffered nor rebuilt as a flat ring (callers use ``result or way``).

    """
    if not overrides:
        return way
    # way.line is typed Optional at the Way level, but every Way reaching this
    # function already has a concrete geometry from OSM parsing / prior edits.
    geom: BaseGeometry = way.line
    node_ids = way.nodes
    if not node_ids:
        first_ov = next(iter(overrides.values()))
        e_new, n_new = _to_utm(
            float(first_ov["lat"]), float(first_ov["lon"]), zone_number, zone_letter
        )
        centroid = geom.centroid
        w = copy.copy(way)
        w.line = _affine_translate(geom, xoff=e_new - centroid.x, yoff=n_new - centroid.y)
        return w

    nc = nodes_cache or {}
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
        utm_coords.append(_to_utm(lat, lon, zone_number, zone_letter))

    if len(utm_coords) < 2:
        return way

    # Close the coords before branching on type: buffering an open centerline of a
    # closed way gives a capsule instead of a ring.
    _is_closed = (len(node_ids) >= 2 and node_ids[0] == node_ids[-1]) or (
        geom.geom_type == "LineString" and geom.is_closed
    )
    if _is_closed and utm_coords[0] != utm_coords[-1]:
        utm_coords.append(utm_coords[0])

    w = copy.copy(way)
    if geom.geom_type == "LineString":
        w.line = _LineString(utm_coords)
    elif geom.geom_type == "Polygon":
        try:
            w.line = _rebuild_polygon(
                geom,
                utm_coords,
                _is_closed,
                category,
                getattr(way, "tags", None),
                min_radius=0.01,
            )
        except _PolygonTooSmall:
            return way
        except (ValueError, TypeError) as e:
            # Buffering the centerline failed; fall back to a flat polygon from the ring.
            logger.warning("Way %s: polygon rebuild failed (%r), using a flat ring", way.id, e)
            ring = utm_coords if utm_coords[0] == utm_coords[-1] else [*utm_coords, utm_coords[0]]
            if len(ring) < 4:
                return None
            try:
                w.line = _SPoly(ring)
            except (ValueError, TypeError) as e2:
                logger.warning("Way %s: flat ring is invalid too (%r), dropping it", way.id, e2)
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
    Convert a lon/lat GeoJSON ``LineString``/``Polygon``/``MultiPolygon`` to Shapely UTM.

    Inverse of :func:`geom_to_geojson`; ``None`` for any other type. Every
    point is forced into the given zone, so the result matches the map's
    zone wherever it lies.
    """

    def pt(c: list[float] | tuple[float, float]) -> tuple[float, float]:
        return _to_utm(c[1], c[0], zone_number, zone_letter)

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
    Return a copy of *way* with its user-added synthetic nodes spliced in.

    Each ``store["added_nodes"]`` entry of the original way (``{"id",
    "after_node_id", "lat", "lon"}``) is inserted after ``after_node_id``,
    at its moved position if it has one; an entry whose anchor is not in
    the node list is skipped. A ``LineString`` gets the coordinate too; a
    ``Polygon`` keeps its geometry, since its ring is the buffer outline.
    Returns *way* itself when nothing was inserted, the geometry is another
    type, or the new line is invalid.
    """
    original_id = int(str(way.id).split(":")[0])
    added_for_way = [a for a in store.get("added_nodes", []) if a.get("way_id") == original_id]
    if not added_for_way:
        return way

    pos_ov_raw = store.get("node_position_overrides", {}).get(str(original_id), {})

    w = copy.copy(way)
    w.nodes = list(way.nodes)
    node_ids: list = w.nodes  # mutable reference

    # way.line is typed Optional at the Way level, but every Way reaching this
    # function already has a concrete geometry from OSM parsing / prior edits.
    geom: BaseGeometry = way.line
    is_linestring = geom.geom_type == "LineString"
    is_polygon = geom.geom_type == "Polygon"
    if not is_linestring and not is_polygon:
        return way

    # A Polygon's ring is the buffer outline: a centerline coord inserted there is a spike.
    coords = list(geom.coords) if is_linestring else None

    nodes_added = 0
    for a in added_for_way:
        synth_id = a["id"]
        after_id = a["after_node_id"]

        try:
            idx = node_ids.index(after_id)
        except ValueError:
            continue  # after_node_id was deleted or not found; skip

        insert_pos = idx + 1
        node_ids.insert(insert_pos, synth_id)

        if coords is not None:
            ov = pos_ov_raw.get(str(synth_id))
            lat = float(ov["lat"] if ov else a["lat"])
            lon = float(ov["lon"] if ov else a["lon"])
            e, n_utm = _to_utm(lat, lon, zone_number, zone_letter)
            # For closed LineStrings the last coord repeats the first — insert before it.
            _is_closed = len(node_ids) >= 2 and node_ids[0] == node_ids[-1]
            coord_limit = len(coords) - 1 if _is_closed and len(coords) > 1 else len(coords)
            coord_pos = min(insert_pos, coord_limit)
            coords.insert(coord_pos, (e, n_utm))

        nodes_added += 1

    if nodes_added == 0:
        return way

    w.nodes = node_ids
    if coords is not None:
        try:
            w.line = _LineString(coords)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Way %s: added nodes left an invalid line (%r), ignoring them", way.id, e
            )
            return way

    return w


def rebuild_way_without_nodes(
    way: "Way",
    del_nids: set[int] | list[int],
    zone_number: int,
    zone_letter: str,
    nodes_cache: dict[int, dict[str, Any]] | None,
    category: str | None = None,
) -> "Way | None":
    """
    Return a copy of *way* with the nodes *del_nids* removed and ``line`` rebuilt.

    A ``LineString`` keeps the coordinates of the remaining nodes (re-closed
    if the way was closed). A ``Polygon`` is rebuilt from the remaining
    nodes' *nodes_cache* positions by :func:`_rebuild_polygon`, falling back
    to the bare centerline if re-buffering fails.

    Returns
    -------
    Way or None
        The edited copy, or ``None`` when the way became degenerate (too
        few nodes or coordinates, or an unsupported geometry type) and
        should be dropped.

    """
    node_ids = way.nodes
    keep = [i for i, nid in enumerate(node_ids) if nid not in del_nids]
    if len(keep) < 2:
        return None
    w = copy.copy(way)
    w.nodes = [way.nodes[i] for i in keep]
    # way.line is typed Optional at the Way level, but every Way reaching this
    # function already has a concrete geometry from OSM parsing / prior edits.
    geom: BaseGeometry = way.line

    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        new_coords = [coords[i] for i in keep if i < len(coords)]
        if len(new_coords) < 2:
            return None
        _is_closed = (len(node_ids) >= 2 and node_ids[0] == node_ids[-1]) or geom.is_closed
        if _is_closed and new_coords[0] != new_coords[-1]:
            new_coords.append(new_coords[0])
        w.line = _LineString(new_coords)

    elif geom.geom_type == "Polygon":
        nc = nodes_cache or {}
        utm_coords = []
        for n in w.nodes:
            nd = nc.get(n)
            if nd and nd["lat"] is not None and nd["lon"] is not None:
                utm_coords.append(
                    _to_utm(float(nd["lat"]), float(nd["lon"]), zone_number, zone_letter)
                )
        if len(utm_coords) < 2:
            return None
        _is_closed_orig = len(node_ids) >= 2 and node_ids[0] == node_ids[-1]
        try:
            w.line = _rebuild_polygon(
                geom,
                utm_coords,
                _is_closed_orig,
                category,
                getattr(way, "tags", None),
            )
        except _PolygonTooSmall:
            return None
        except (ValueError, TypeError) as e:
            # Buffering the centerline failed; keep the (unbuffered) centerline itself.
            logger.warning("Way %s: polygon rebuild failed (%r), keeping the centerline", way.id, e)
            w.line = _LineString(utm_coords)

    else:
        return None

    return w


def update_segment_annotations_for_split_change(
    store: dict[str, Any],
    original_way_id: int,
    way: "Way",
    old_split_nids: list[int],
    new_split_nids: list[int],
) -> None:
    """
    Update segment-specific annotations in the store when split points change.

    When a way's split points are modified, the virtual segment IDs (original_way_id:index)
    change meaning. This function maps deleted ways, deleted nodes, and change_log entries
    from the old segments to the new segments.
    """
    node_ids = way.nodes
    if not node_ids or node_ids[0] == node_ids[-1]:
        return

    old_splits = {int(nid) for nid in old_split_nids}
    new_splits = {int(nid) for nid in new_split_nids}

    def get_segs(splits: set[int]) -> list[list[int]]:
        segs: list[list[int]] = []
        curr: list[int] = []
        for i, nid in enumerate(node_ids):
            curr.append(nid)
            is_interior = 0 < i < len(node_ids) - 1
            if is_interior and int(nid) in splits:
                segs.append(curr)
                curr = [nid]
        segs.append(curr)
        return segs

    old_segs = get_segs(old_splits)
    new_segs = get_segs(new_splits)

    def edge_to_seg(segs: list[list[int]]) -> dict[int, int]:
        mapping = {}
        e_idx = 0
        for s_idx, seg in enumerate(segs):
            for _ in range(len(seg) - 1):
                mapping[e_idx] = s_idx
                e_idx += 1
        return mapping

    old_edges = edge_to_seg(old_segs)
    new_edges = edge_to_seg(new_segs)
    num_edges = len(node_ids) - 1

    deleted_ways = store.get("deleted_ways", [])
    old_deleted_idxs = set()
    new_deleted_ways = []

    prefix = f"{original_way_id}:"

    def seg_id(new_idx: int) -> int | str:
        # Back to one segment: edits move to the plain way id, which is all an unsplit way matches.
        return f"{prefix}{new_idx}" if len(new_segs) > 1 else int(original_way_id)

    for d in deleted_ways:
        wid = d["id"]
        if str(wid).startswith(prefix):
            old_deleted_idxs.add(int(str(wid).split(":")[1]))
        else:
            new_deleted_ways.append(d)

    for new_idx in range(len(new_segs)):
        edges_in_new = [e for e, s in new_edges.items() if s == new_idx]
        if edges_in_new and all(old_edges[e] in old_deleted_idxs for e in edges_in_new):
            new_deleted_ways.append({"id": seg_id(new_idx)})

    store["deleted_ways"] = new_deleted_ways

    def map_node(old_idx: int, nid: int) -> set[int]:
        try:
            i = node_ids.index(nid)
        except ValueError:
            return set()

        E = []
        if i > 0:
            E.append(i - 1)
        if i < num_edges:
            E.append(i)

        Ej = {e for e in E if old_edges[e] == old_idx}

        mapped = set()
        for new_idx in {new_edges.get(e, -1) for e in E}:
            if new_idx == -1:
                continue
            Ek = {e for e in E if new_edges[e] == new_idx}
            if Ej & Ek:
                mapped.add(new_idx)
        return mapped

    new_dn = []
    for d in store.get("deleted_nodes", []):
        wid = d["way_id"]
        if str(wid).startswith(prefix):
            old_idx = int(str(wid).split(":")[1])
            nid = d["node_id"]
            mapped_new_idxs = map_node(old_idx, nid)
            for new_idx in mapped_new_idxs:
                new_dn.append({"way_id": seg_id(new_idx), "node_id": nid})
        else:
            new_dn.append(d)

    store["deleted_nodes"] = new_dn

    cl = store.get("change_log", [])
    new_cl = []
    for e in cl:
        if e.get("type") == "way":
            wid = e["id"]
            if str(wid).startswith(prefix):
                old_idx = int(str(wid).split(":")[1])
                mapped_new_idxs = set()
                for new_idx in range(len(new_segs)):
                    edges_in_new = [e for e, s in new_edges.items() if s == new_idx]
                    if edges_in_new and all(old_edges[edge] == old_idx for edge in edges_in_new):
                        mapped_new_idxs.add(new_idx)
                for new_idx in mapped_new_idxs:
                    new_entry = dict(e)
                    new_entry["id"] = seg_id(new_idx)
                    new_cl.append(new_entry)
            else:
                new_cl.append(e)
        elif e.get("type") == "node":
            wid = e.get("way_id")
            if str(wid).startswith(prefix):
                old_idx = int(str(wid).split(":")[1])
                nid = e["node_id"]
                mapped_new_idxs = map_node(old_idx, nid)
                for new_idx in mapped_new_idxs:
                    new_entry = dict(e)
                    new_entry["way_id"] = seg_id(new_idx)
                    new_cl.append(new_entry)
            else:
                new_cl.append(e)
        else:
            new_cl.append(e)

    store["change_log"] = new_cl
