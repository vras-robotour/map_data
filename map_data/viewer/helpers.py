"""UTM -> lon/lat GeoJSON conversion for the viewer."""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import utm
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
    from map_data.map_data import MapData
    from map_data.utils.way import Way

logger = logging.getLogger(__name__)

#: Geometries kept by :func:`geom_to_geojson` (a ~4 MB map holds ~3000).
_GEOM_CACHE_SIZE = 16384


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
    xy = np.asarray(coords, dtype=float)
    if len(xy) == 0:
        return []
    # One vectorised call: utm.to_latlon is ~50x slower per point than per array.
    lat, lon = utm.to_latlon(xy[:, 0], xy[:, 1], zone_number, zone_letter)
    return np.column_stack([lon, lat]).tolist()


def way_feature(
    way: "Way",
    way_id: int | str,
    category: str | None,
    tags: dict[str, Any],
    geom: dict[str, Any] | None,
    feature_id: str | None = None,
) -> dict[str, Any]:
    """
    Build a way's GeoJSON Feature; a top-level ``"id"`` only when *feature_id* is given.

    ``is_node`` marks a barrier with no OSM nodes (synthesized from a single obstacle node).
    """
    feature: dict[str, Any] = {"type": "Feature"}
    if feature_id is not None:
        feature["id"] = feature_id
    feature["geometry"] = geom
    feature["properties"] = {
        "id": way_id,
        "category": category,
        "is_node": category == "barrier" and not bool(way.nodes),
        "tags": tags,
        "in_out": way.in_out,
    }
    return feature


@lru_cache(maxsize=_GEOM_CACHE_SIZE)
def geom_to_geojson(
    geom: "_LineString | _SPoly | _SMPoly",
    zone_number: int,
    zone_letter: str,
) -> dict[str, Any] | None:
    """
    Convert a Shapely UTM ``LineString``/``Polygon``/``MultiPolygon`` to a lon/lat GeoJSON dict.

    Returns ``None`` for any other type, or when ``geom_type`` does not
    match the object's class (the coordinate extraction is type-specific).

    Cached on the geometry's *value* (shapely hashes and compares by coordinates) and
    the zone, so there is nothing to invalidate: an edited way is a different key and
    an unchanged one, or a recomputed crossroad equal to the old one, hits. The dict is
    shared between callers: never mutate it.
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
    Convert a :class:`MapData` into one GeoJSON FeatureCollection.

    One Feature per road/footway/barrier/crossroad way (see
    :func:`way_feature`) plus one Point per waypoint, each with a
    ``"category"`` property. Ways whose geometry cannot be converted are
    skipped.
    """
    features = []
    zn, zl = map_data.zone_number, map_data.zone_letter

    def add_ways(ways: list["Way"], category: str) -> None:
        for way in ways:
            try:
                geom = geom_to_geojson(way.line, zn, zl) if way.line else None
            except (ValueError, TypeError) as e:
                logger.warning("Failed to convert geometry for way %s: %s", way.id, e)
                continue
            if geom is None:
                continue
            features.append(way_feature(way, way.id, category, way.tags or {}, geom, str(way.id)))

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
