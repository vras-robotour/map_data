"""
Flask routes for the interactive OSM map-data viewer/editor.

Exposes a JSON/GeoJSON API (registered on ``bp``, mounted by the Flask app
factory) that the browser-side viewer uses to: list and upload ``.mapdata``/
``.gpx`` files, read map geometry as GeoJSON, record user edits (way/node
deletion, node moves, node insertion, way splitting, tag overrides) as a
JSON "annotation store" alongside each mapdata file, export an edited map,
compute cost grids, and drive path (re)planning, including a wormhole-based
GPX handoff to a companion mobile app.

All routes share the one ``viewer`` blueprint defined in
:mod:`~map_data.viewer.routes.common`; importing the submodules below
registers their routes on it.
"""

from . import annotations, files, planning, sharing, ways
from .common import bp

__all__ = ["annotations", "bp", "files", "planning", "sharing", "ways"]
