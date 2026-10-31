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

#: Footway ``highway`` values a wheeled robot must not use. They stay footways
#: (the viewer shows them) and are excluded at use time through
#: :meth:`map_data.map_data.MapData.apply_traversability`.
NON_ROUTABLE_HIGHWAY_VALUES = frozenset({"steps"})


@dataclass
class Way:
    """
    One map feature (OSM way, merged relation, node obstacle or drawn path).

    Attributes
    ----------
    id : any
        OSM id; negative for synthetic ways (merged relation chains, drawn
        paths), ``"<id>:<i>"`` for segments of a split way.
    is_area : bool
        ``True`` when ``line`` is a polygon, including buffered lines.
    nodes : list
        Ordered node ids (keys of ``MapData.nodes_cache``).
    line : BaseGeometry or None
        Geometry in UTM.
    in_out : str
        Multipolygon member role, ``"outer"``, ``"inner"`` or ``""``.

    """

    id: Any = -1
    is_area: bool = False
    nodes: list[Any] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    line: BaseGeometry | None = None
    in_out: str = ""

    def is_road(self) -> bool:
        """Any ``highway`` value not in :data:`FOOTWAY_VALUES`."""
        hw = self.tags.get("highway")
        return bool(hw and hw not in FOOTWAY_VALUES)

    def is_footway(self) -> bool:
        """``highway`` in :data:`FOOTWAY_VALUES`."""
        hw = self.tags.get("highway")
        return bool(hw and hw in FOOTWAY_VALUES)

    def is_barrier(
        self,
        yes_tags: dict[str, list[str]],
        not_tags: dict[str, list[str]],
        anti_tags: dict[str, list[str]],
    ) -> bool:
        """
        Whether a tag matches *yes_tags* and none matches *anti_tags*.

        Each table maps a tag key to values; ``"*"`` in *yes_tags* matches any
        value not listed in *not_tags*.
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
