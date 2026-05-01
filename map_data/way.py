from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import numpy as np
from shapely.prepared import prep
from shapely.geometry import Point, MultiPoint
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from map_data.points_to_graph_points import get_point_line

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
    ]
)


@dataclass
class Way:
    id: int = -1
    is_area: bool = False
    nodes: List[Any] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    line: Optional[BaseGeometry] = None
    in_out: str = ""
    pcd_points: Optional[np.ndarray] = field(default=None, repr=False)

    def is_road(self) -> bool:
        hw = self.tags.get("highway")
        return bool(hw and hw not in FOOTWAY_VALUES)

    def is_footway(self) -> bool:
        hw = self.tags.get("highway")
        return bool(hw and hw in FOOTWAY_VALUES)

    def is_barrier(
        self,
        yes_tags: Dict[str, List[str]],
        not_tags: Dict[str, List[str]],
        anti_tags: Dict[str, List[str]],
    ) -> bool:
        has_barrier_tag = any(
            key in yes_tags
            and (
                self.tags[key] in yes_tags[key]
                or (
                    "*" in yes_tags[key] and self.tags[key] not in not_tags.get(key, [])
                )
            )
            for key in self.tags
        )
        has_anti_tag = any(
            key in anti_tags and self.tags[key] in anti_tags[key] for key in self.tags
        )
        return has_barrier_tag and not has_anti_tag

    def to_pcd_points(self, density: float = 2.0, filled: bool = True) -> np.ndarray:
        if self.pcd_points is not None:
            return self.pcd_points

        if not self.line:
            return np.empty((0, 2))

        if filled and isinstance(self.line, Polygon):
            xmin, ymin, xmax, ymax = self.line.bounds
            x = np.arange(
                np.floor(xmin * density) / density,
                np.ceil(xmax * density) / density,
                1 / density,
            )
            y = np.arange(
                np.floor(ymin * density) / density,
                np.ceil(ymax * density) / density,
                1 / density,
            )
            xv, yv = np.meshgrid(x, y)
            candidates = MultiPoint(np.column_stack([xv.ravel(), yv.ravel()])).geoms
            pts = list(self._mask_points(candidates, self.line))
            if not pts:
                self.pcd_points = np.empty((0, 2))
            else:
                self.pcd_points = np.array([(p.x, p.y) for p in pts])
        else:
            # For LineString or unfilled Polygon
            geom = self.line
            if hasattr(geom, "exterior"):
                coords = list(geom.exterior.coords)
            else:
                coords = list(geom.coords)

            pcd_points = np.empty((0, 2))
            for i in range(len(coords) - 1):
                p1 = Point(coords[i])
                p2 = Point(coords[i + 1])
                _, line, _ = get_point_line(p1, p2, 0.5)
                pcd_points = np.concatenate([pcd_points, line])
            self.pcd_points = pcd_points

        return self.pcd_points

    @staticmethod
    def _mask_points(points: List[Point], polygon: Polygon) -> filter:
        prepared = prep(polygon)
        return filter(prepared.contains, points)

    def mask_points(self, points: List[Point], polygon: Polygon) -> filter:
        return self._mask_points(points, polygon)
