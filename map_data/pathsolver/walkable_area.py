"""
Open walkable areas (pedestrian squares) for the graph planner.

A pedestrian square is mapped as a closed ``area=yes`` way, i.e. by its outline
only. Routed as a chain of nodes like any other way, it makes a route walk the
rim of the square instead of crossing it. :class:`WalkableArea` keeps the
polygon and lets the planner cross it: between two of its *entries* (network
nodes on, inside or within :data:`ENTRY_TOLERANCE` of the area) it prices the
shortest path that stays inside the polygon, holes included.

Such a path bends only at reflex corners of the outline and at hole vertices,
so a visibility graph over the entries and those corners is enough. It is built
on first use and kept with the area; the planner's global adjacency gains no
edges.
"""

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
from shapely.prepared import prep

#: A network node farther than this (metres) from an area does not enter it.
#: Mapped footways often stop just short of the square they lead onto instead
#: of sharing a node with its outline.
ENTRY_TOLERANCE = 1.0

#: Slack (metres) of the inside test, so that segments running along the
#: outline or a hole's edge still count as inside.
_EPS = 1e-3

#: A crossing: ``(cost, points)``, the points running from the first endpoint
#: to the second, both included.
Crossing = tuple[float, list[np.ndarray]]


class WalkableArea:
    """
    Shortest in-area crossings between the entries of one walkable polygon.

    Costs are the in-polygon path length times *factor*, the weight multiplier
    of the area's own edges, so a crossing is priced like walking its rim and
    the planner's straight-line A* heuristic stays admissible.
    """

    def __init__(self, polygon: Polygon, factor: float, entries: dict[int, np.ndarray]) -> None:
        """
        Parameters
        ----------
        polygon : Polygon
            The area in UTM coordinates, holes included.
        factor : float
            Weight multiplier of the area's edges (at least 1).
        entries : dict
            Node id -> ``[x, y]`` of the network nodes that enter the area. A
            node just outside is joined to its nearest point on the boundary.

        """
        self.polygon = polygon
        self.factor = factor
        self.entries = entries
        self._inside = prep(polygon.buffer(_EPS))
        self._index = {n: i for i, n in enumerate(entries)}
        anchors, self._legs = zip(*(self._anchor(p) for p in entries.values()), strict=True)
        self._points = list(anchors) + _bend_corners(polygon)
        self._weights: np.ndarray | None = None
        self._crossings: dict[int, dict[int, Crossing]] = {}

    def covers(self, point: np.ndarray) -> bool:
        """
        Return ``True`` if *point* lies in the area (boundary included, holes excluded).
        """
        return bool(self.polygon.covers(Point(point)))

    def _anchor(self, point: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Where *point* joins the area and the length of the leg to it: the point
        itself when inside, else its nearest boundary point.
        """
        point = np.asarray(point, dtype=float)
        if self.covers(point):
            return point, 0.0
        anchor = np.array(nearest_points(self.polygon, Point(point))[0].coords[0])
        return anchor, float(np.linalg.norm(point - anchor))

    def crossings(self, node: int) -> dict[int, Crossing]:
        """
        Crossings from entry *node* to every other entry it can reach inside the area.
        """
        if node not in self._crossings:
            i = self._index[node]
            dist, prev = self._search(self._points[i], [])
            self._crossings[node] = {
                other: (
                    (self._legs[i] + dist[j] + self._legs[j]) * self.factor,
                    [
                        self.entries[node],
                        *_trace(prev, self._points + [self._points[i]], j),
                        self.entries[other],
                    ],
                )
                for other, j in self._index.items()
                if other != node and np.isfinite(dist[j])
            }
        return self._crossings[node]

    def from_point(
        self, point: np.ndarray, goal: np.ndarray | None = None
    ) -> tuple[dict[int, Crossing], Crossing | None]:
        """
        Crossings from a *point* in or next to the area to its entries and, when given, to *goal*.

        A point (or goal) just outside joins the area at its nearest boundary
        point, like an entry does. The goal crossing is ``None`` when *goal* is
        not given or cannot be reached from *point* inside the area.
        """
        anchor, leg = self._anchor(point)
        extra: list[np.ndarray] = []
        goal_leg = 0.0
        if goal is not None:
            goal_anchor, goal_leg = self._anchor(goal)
            extra = [goal_anchor]
        dist, prev = self._search(anchor, extra)
        points = self._points + extra + [anchor]
        start = [np.asarray(point, dtype=float)]
        to_entries = {
            node: (
                (leg + dist[j] + self._legs[j]) * self.factor,
                [*start, *_trace(prev, points, j), self.entries[node]],
            )
            for node, j in self._index.items()
            if np.isfinite(dist[j])
        }
        to_goal = None
        g = len(self._points)
        if goal is not None and np.isfinite(dist[g]):
            to_goal = (
                (leg + dist[g] + goal_leg) * self.factor,
                [*start, *_trace(prev, points, g), np.asarray(goal, dtype=float)],
            )
        return to_entries, to_goal

    def _search(self, source: np.ndarray, extra: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Dijkstra over the visibility graph of the anchors, corners, *extra* and *source*.

        Returns distances and predecessors indexed like
        ``anchors + corners + extra + [source]``; the source's predecessor is -1.
        """
        if self._weights is None:
            self._weights = self._visibility(self._points, self._points)
        points = self._points + extra + [np.asarray(source, dtype=float)]
        n, base = len(points), len(self._points)
        weights = np.full((n, n), np.inf)
        weights[:base, :base] = self._weights
        new = self._visibility(points[base:], points)
        weights[base:, :] = new
        weights[:, base:] = new.T

        dist = np.full(n, np.inf)
        dist[-1] = 0.0
        prev = np.full(n, -1)
        done = np.zeros(n, dtype=bool)
        for _ in range(n):
            u = int(np.argmin(np.where(done, np.inf, dist)))
            if not np.isfinite(dist[u]):
                break
            done[u] = True
            better = dist[u] + weights[u] < dist
            dist[better] = dist[u] + weights[u][better]
            prev[better] = u
        return dist, prev

    def _visibility(self, rows: list[np.ndarray], cols: list[np.ndarray]) -> np.ndarray:
        """
        Length of the segment between each row and column point, ``inf`` if it leaves the area.
        """
        out = np.full((len(rows), len(cols)), np.inf)
        for i, a in enumerate(rows):
            for j, b in enumerate(cols):
                d = float(np.linalg.norm(a - b))
                if d < _EPS or self._inside.contains(LineString([a, b])):
                    out[i, j] = d
        return out


def _bend_corners(polygon: Polygon) -> list[np.ndarray]:
    """
    Vertices a shortest in-polygon path can bend at: reflex outline corners and convex hole corners.
    """
    corners: list[np.ndarray] = []
    for ring, outer in [(polygon.exterior, True), *((r, False) for r in polygon.interiors)]:
        xy = np.asarray(ring.coords)[:-1, :2]
        a = xy - np.roll(xy, 1, axis=0)
        b = np.roll(xy, -1, axis=0) - xy
        # > 0: the ring turns towards its own interior at this vertex.
        turn = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) * (1 if ring.is_ccw else -1)
        corners.extend(xy[turn < 0] if outer else xy[turn > 0])
    return corners


def _trace(prev: np.ndarray, points: list[np.ndarray], target: int) -> list[np.ndarray]:
    """
    Points of the search path from the source to *target*, both included.
    """
    path = []
    while target != -1:
        path.append(points[target])
        target = int(prev[target])
    return path[::-1]
