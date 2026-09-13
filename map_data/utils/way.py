from dataclasses import dataclass, field
from typing import Any

from shapely.geometry.base import BaseGeometry

FOOTWAY_VALUES = frozenset(
    [
        "living_street",
        "pedestrian",
        "footway",
        "bridleway",
        "corridor",
        "track",
        "steps",
        "cycleway",
        "path",
    ],
)

#: ``highway`` values that are classified as footways but must not be routed
#: over by a wheeled robot. Stairways are mapped as ``highway=steps`` and are
#: part of :data:`FOOTWAY_VALUES` (the viewer and the cost grid still show
#: them), so the planner and ``osm_cloud`` exclude them at use time through
#: :meth:`map_data.map_data.MapData.apply_traversability`.
NON_ROUTABLE_HIGHWAY_VALUES = frozenset({"steps"})


@dataclass
class Way:
    """
    Represents a single OSM feature with geometry and tags.

    A ``Way`` is the fundamental unit of map data. It wraps an OSM way or
    relation with its parsed geometry and tag dictionary, and provides
    helpers to classify the feature.

    Attributes
    ----------
    id : any
        OSM way ID (positive integer) or a negative integer / string for
        manually annotated ways.
    is_area : bool
        ``True`` if the geometry is a closed polygon (area), ``False`` for
        a linestring.
    nodes : list
        Ordered list of node IDs that make up this way.
    tags : dict
        OSM key-value tags (e.g. ``{"highway": "footway", "surface": "asphalt"}``).
    line : BaseGeometry or None
        Shapely geometry representing the way in UTM coordinates.
    in_out : str
        Direction hint used by some planners (``"in"``, ``"out"``, or ``""``).

    """

    id: Any = -1
    is_area: bool = False
    nodes: list[Any] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    line: BaseGeometry | None = None
    in_out: str = ""

    def is_road(self) -> bool:
        """
        Return ``True`` if this way is a vehicle road (any ``highway`` value not in footway types).
        """
        hw = self.tags.get("highway")
        return bool(hw and hw not in FOOTWAY_VALUES)

    def is_footway(self) -> bool:
        """
        Return ``True`` if this way is a pedestrian footway (``highway`` in the footway value set).
        """
        hw = self.tags.get("highway")
        return bool(hw and hw in FOOTWAY_VALUES)

    def is_barrier(
        self,
        yes_tags: dict[str, list[str]],
        not_tags: dict[str, list[str]],
        anti_tags: dict[str, list[str]],
    ) -> bool:
        """
        Return ``True`` if this way should be classified as an untraversable barrier.

        Parameters
        ----------
        yes_tags : dict
            Mapping of OSM tag key → list of values that indicate a barrier.
            A ``"*"`` value matches any tag value not in *not_tags*.
        not_tags : dict
            Exceptions to ``"*"`` wildcard matches in *yes_tags*.
        anti_tags : dict
            Tags that, if present, override a barrier match and return
            ``False`` (e.g. a gate that makes a fence passable).

        """
        has_barrier_tag = any(
            key in yes_tags
            and (
                self.tags[key] in yes_tags[key]
                or ("*" in yes_tags[key] and self.tags[key] not in not_tags.get(key, []))
            )
            for key in self.tags
        )
        has_anti_tag = any(
            key in anti_tags and self.tags[key] in anti_tags[key] for key in self.tags
        )
        return has_barrier_tag and not has_anti_tag
