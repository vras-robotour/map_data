import numpy as np
from shapely.prepared import prep
from shapely.geometry import Point, MultiPoint, LineString

from map_data.points_to_graph_points import get_point_line


FOOTWAY_VALUES = frozenset([
    "living_street", "pedestrian", "footway", "bridleway",
    "corridor", "track", "steps", "cycleway", "path",
])


class Way:
    def __init__(self, id=-1, is_area=False, nodes=None, tags=None, line=None, in_out=""):
        self.id = id
        self.is_area = is_area
        self.nodes = nodes if nodes is not None else []
        self.tags = tags if tags is not None else {}
        self.line = line
        self.in_out = in_out
        self.pcd_points = None

    def is_road(self):
        hw = self.tags.get("highway")
        return bool(hw and hw not in FOOTWAY_VALUES)

    def is_footway(self):
        hw = self.tags.get("highway")
        return bool(hw and hw in FOOTWAY_VALUES)

    def is_barrier(self, yes_tags, not_tags, anti_tags):
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

    def to_pcd_points(self, density=2, filled=True):
        if self.pcd_points is not None:
            return self.pcd_points

        if filled:
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
            self.pcd_points = np.array(list(LineString(pts).xy)).T
        else:
            coords = self.line.exterior.coords
            pcd_points = np.empty((0, 2))
            for i in range(len(coords)):
                p1 = Point(coords[i])
                p2 = Point(coords[(i + 1) % len(coords)])
                _, line, _ = get_point_line(p1, p2, 0.5)
                pcd_points = np.concatenate([pcd_points, line])
            self.pcd_points = pcd_points

        return self.pcd_points

    @staticmethod
    def _mask_points(points, polygon):
        prepared = prep(polygon)
        return filter(prepared.contains, points)

    def mask_points(self, points, polygon):
        return self._mask_points(points, polygon)
