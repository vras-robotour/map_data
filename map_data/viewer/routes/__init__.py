"""
Flask JSON/GeoJSON API of the map-data viewer/editor.

Files and export, way edits recorded in a per-map annotation store,
freehand annotations, planning, and robot handoff. Every route is on the
``viewer`` blueprint of :mod:`~map_data.viewer.routes.common`; importing the
submodules registers them.
"""

from . import annotations, files, planning, sharing, ways
from .common import bp

__all__ = ["annotations", "bp", "files", "planning", "sharing", "ways"]
