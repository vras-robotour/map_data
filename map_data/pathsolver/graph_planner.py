"""
Graph-based path planning on OSM ways.

This module provides the GraphPlanner class which builds a graph from
OpenStreetMap ways and finds paths using Dijkstra or A*.
"""

import itertools
import logging
import math
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

from map_data.pathsolver.astar import astar_search
from map_data.pathsolver.walkable_area import ENTRY_TOLERANCE, WalkableArea
from map_data.pathsolver.way_cost import load_cost_tables, way_cost
from map_data.traversability import TraversabilityRules, load_traversability
from map_data.utils.way import NON_ROUTABLE_HIGHWAY_VALUES, Way

logger = logging.getLogger(__name__)

#: Two consecutive route vertices closer than this (metres) are considered the
#: same point and only the first is kept. Deliberately coarse: projections of a
#: waypoint onto an edge and the edge's own nodes routinely land a few
#: centimetres apart, which would otherwise show up as a cluster of stacked
#: points in the exported route.
TOLERANCE = 0.05

#: Longest out-and-back excursion (metres) that :func:`_drop_stacked_points`
#: collapses. A waypoint snapping just past a junction makes the route leave
#: that junction, touch the projection and come straight back; such spurs are
#: sub-metre artefacts, while a genuine visit to a dead end is far longer and
#: must be preserved. Real routes show a wide gap either side of this value
#: (a 900 m park loop had spurs of 0.07-0.65 m and no other step below 7 m),
#: so the exact figure is not delicate.
MAX_SPUR_LENGTH = 1.0

#: Shortest final leg (metres) worth keeping for ``keep_goal``. A goal closer
#: than this to its projection onto the network is already on the path as far
#: as a waypoint follower is concerned, and appending it would only add a
#: stacked point (see :func:`_drop_stacked_points`).
MIN_GOAL_LEG_LENGTH = 1.0

#: Default maximum distance (in metres) between a waypoint and the nearest
#: graph edge for the waypoint to be snapped onto the network. Deliberately
#: generous — unlike annotation splicing (5 m), waypoints come from user
#: clicks or GPS fixes that can legitimately sit well off the mapped network —
#: while still rejecting waypoints that would otherwise snap to an arbitrarily
#: distant, unrelated edge.
DEFAULT_MAX_SNAP_DISTANCE = 100.0

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from map_data.map_data import MapData


class GraphPlanner:
    """
    Graph-based path planner that routes along the OSM road and footway network.

    Builds an undirected weighted graph from the ways stored in a
    :class:`~map_data.map_data.MapData` instance. Manually annotated paths
    (negative-ID ways) are spliced into the graph by projecting their
    endpoints onto the nearest OSM edge, so annotations extend the
    traversable network seamlessly.

    Walkable areas (closed ``area=yes`` or multipolygon ways, e.g. pedestrian
    squares) can be crossed rather than only walked around: A* may hop between
    any two of an area's entries along the shortest path inside it (see
    :mod:`map_data.pathsolver.walkable_area`), and a waypoint inside an area
    stays where it is instead of snapping to its rim.

    Planning is performed with A* (see :meth:`plan`). The planner operates
    entirely in UTM coordinates.
    """

    def __init__(
        self,
        map_data: "MapData",
        highway_types: list[str] | None = None,
        max_snap_distance: float = DEFAULT_MAX_SNAP_DISTANCE,
        exclude_highway: Iterable[str] = NON_ROUTABLE_HIGHWAY_VALUES,
        traversability: "TraversabilityRules | str | Path | None" = None,
        highway_costs: "Mapping[str, float] | None" = None,
        surface_costs: "Mapping[str, float] | None" = None,
        path_cost_cap: float | None = None,
    ) -> None:
        """
        Initialize the graph planner.

        Parameters
        ----------
        map_data : MapData
            Parsed map data containing the OSM ways and node cache.
        highway_types : list of str, optional
            Way categories to include in the graph. Supported values are
            ``"footway"`` and ``"road"``. Defaults to ``["footway"]``.
        max_snap_distance : float
            Maximum distance (metres) a waypoint passed to :meth:`plan` may
            be from the nearest graph edge to be snapped onto it. Waypoints
            farther than this fail the plan instead of snapping to an
            arbitrarily distant edge
            (default :data:`DEFAULT_MAX_SNAP_DISTANCE`).
        exclude_highway : iterable of str
            ``highway`` tag values never routed over, whatever their category
            (default :data:`~map_data.utils.way.NON_ROUTABLE_HIGHWAY_VALUES`,
            i.e. stairs). A map loaded through
            :func:`~map_data.annotations.load_mapdata_with_annotations` has
            them removed already; this filter also covers callers that pass a
            raw :meth:`MapData.load` map.
        traversability : TraversabilityRules or str or Path, optional
            Tag rules deciding which ways may be driven on at all and what
            extra cost they carry (``None`` = the package's
            ``config/traversability.yaml``, see
            :mod:`map_data.traversability`). ``exclude_highway`` is folded into
            them. Ways the rules reject are left out of the graph, again as a
            safety net for raw maps.
        highway_costs, surface_costs : mapping, optional
            Cost per ``highway`` / ``surface`` tag value; ``None`` takes the
            tables from ``config/planner_defaults.yaml``, the very ones the
            grid planner uses (:mod:`map_data.pathsolver.way_cost`). An edge of
            a way weighs ``length * (1 + way cost + rule cost)``, so a gravel
            detour is taken only when it is enough shorter; the *reported*
            route length stays geometric.
        path_cost_cap : float, optional
            Upper bound of the ``highway`` + ``surface`` cost (``None`` = the
            config value).

        """
        self.map_data = map_data
        self.highway_types = highway_types or ["footway"]
        self.max_snap_distance = max_snap_distance
        self.exclude_highway = frozenset(exclude_highway)
        self.traversability = load_traversability(traversability).extend(self.exclude_highway)
        self.highway_costs, self.surface_costs, self.path_cost_cap = load_cost_tables(
            highway_costs, surface_costs, path_cost_cap
        )
        self.nodes: dict[int, np.ndarray] = self.map_data.get_points()
        self.graph: dict[int, list[tuple[int, float]]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """
        Build the adjacency graph and spatial index from the allowed ways.

        Annotation ways (negative integer IDs) are connected to the OSM graph
        by projecting their endpoints onto the nearest existing edge and
        inserting a new junction node at the projection point.
        """
        self.graph = {}
        self._allowed_ways = []
        if "footway" in self.highway_types:
            self._allowed_ways.extend(self.map_data.footways_list)
        if "road" in self.highway_types:
            self._allowed_ways.extend(self.map_data.roads_list)
        # exclude_highway is part of self.traversability (extend() puts it first).
        self._allowed_ways = [
            w for w in self._allowed_ways if self.traversability.is_traversable(w)
        ]

        # Per-planner copies of the node lists. Splits are spliced into these
        # copies so the shared Way objects owned by map_data stay untouched
        # (a second GraphPlanner on the same MapData must not see synthetic IDs).
        way_nodes: list[list[int]] = [list(way.nodes) for way in self._allowed_ways]

        # First pass: identify potential splits from annotations (negative-ID
        # ways). Building the snapping index is pointless when there are none.
        has_annotations = any(isinstance(w.id, int) and w.id < 0 for w in self._allowed_ways)

        # Group splits by way and segment
        # (way_index, segment_index) -> [(proj_dist, proj_node_id, node_id, dist_to_edge)]
        splits: dict[tuple[int, int], list[tuple[float, int, int, float]]] = {}
        new_internal_id = -2000000

        if has_annotations:
            edge_segments = []
            edge_way_info = []  # (way_index, segment_index)

            for way_idx, nodes in enumerate(way_nodes):
                for i in range(len(nodes) - 1):
                    n1, n2 = nodes[i], nodes[i + 1]
                    p1 = self.nodes[n1].ravel()[:2]
                    p2 = self.nodes[n2].ravel()[:2]
                    edge_segments.append(LineString([p1, p2]))
                    edge_way_info.append((way_idx, i))

            tree = STRtree(edge_segments) if edge_segments else None
            threshold = 5.0
            if tree:
                for way in self._allowed_ways:
                    # way.id >= 0 check fails if way.id is a string (virtual ID for split ways).
                    # All split ways (strings) and OSM ways (positive ints) should be skipped here.
                    if not isinstance(way.id, int) or way.id >= 0:
                        continue
                    if not way.nodes:
                        continue

                    # Check endpoints of annotation way
                    for node_id in [way.nodes[0], way.nodes[-1]]:
                        p_node = self.nodes[node_id].ravel()[:2]
                        p_sh = Point(p_node)

                        indices = tree.query(p_sh.buffer(threshold), predicate="intersects")
                        if len(indices) == 0:
                            continue

                        # Find nearest edge that is NOT part of the same way
                        best_idx = -1
                        min_dist = float("inf")
                        for idx in indices:
                            if self._allowed_ways[edge_way_info[idx][0]].id == way.id:
                                continue
                            d = edge_segments[idx].distance(p_sh)
                            if d < min_dist:
                                min_dist = d
                                best_idx = idx

                        if best_idx != -1 and min_dist <= threshold:
                            line = edge_segments[best_idx]
                            proj_dist = line.project(p_sh)
                            p_proj = np.array(line.interpolate(proj_dist).coords[0])

                            proj_node_id = new_internal_id
                            new_internal_id -= 1
                            self.nodes[proj_node_id] = np.array(
                                [p_proj[0], p_proj[1], 0.0]
                            ).reshape(3, 1)

                            target_way_idx, segment_idx = edge_way_info[best_idx]
                            splits.setdefault((target_way_idx, segment_idx), []).append(
                                (proj_dist, proj_node_id, node_id, min_dist),
                            )

        # Group pending splits per way so they can be applied in descending
        # segment order: inserting a junction into an earlier segment would
        # otherwise shift the indices of later segments of the same way.
        splits_by_way: dict[int, list[tuple[int, list[tuple[float, int, int, float]]]]] = {}
        for (way_idx, segment_idx), s_list in splits.items():
            splits_by_way.setdefault(way_idx, []).append((segment_idx, s_list))

        # Apply splits to the per-planner node-list copies by inserting new nodes
        for way_idx, seg_splits in splits_by_way.items():
            target_nodes = way_nodes[way_idx]
            seg_splits.sort(key=lambda x: x[0], reverse=True)
            for segment_idx, s_list in seg_splits:
                # Sort splits on this segment by distance from segment start
                s_list.sort(key=lambda x: x[0], reverse=True)
                for _, proj_node_id, ann_node_id, dist_to_edge in s_list:
                    target_nodes.insert(segment_idx + 1, proj_node_id)
                    # Manually add the connection from annotation endpoint to the new junction node
                    self._add_edge(ann_node_id, proj_node_id, dist_to_edge)

        # Second pass: build final graph and tree from (possibly modified) node lists
        final_edge_segments = []
        final_edge_node_pairs = []
        final_edge_factors = []

        for way_idx, nodes in enumerate(way_nodes):
            factor = self.edge_factor(self._allowed_ways[way_idx])
            for i in range(len(nodes) - 1):
                n1, n2 = nodes[i], nodes[i + 1]
                p1 = self.nodes[n1].ravel()[:2]
                p2 = self.nodes[n2].ravel()[:2]
                dist = float(np.linalg.norm(p1 - p2))

                # Weight, not length: the route's own length is measured on its
                # geometry afterwards (map_data.pathsolver.route.path_length).
                self._add_edge(n1, n2, dist * factor)
                final_edge_segments.append(LineString([p1, p2]))
                final_edge_node_pairs.append((n1, n2))
                final_edge_factors.append(factor)

        self._edge_segments = final_edge_segments
        self._edge_node_pairs = final_edge_node_pairs
        self._edge_factors = final_edge_factors
        self._edge_tree = STRtree(final_edge_segments) if final_edge_segments else None
        self._build_areas(way_nodes)

    def _build_areas(self, way_nodes: list[list[int]]) -> None:
        """
        Collect the walkable areas among the allowed ways, with their entries.

        A walkable area is a closed way tagged ``area=yes`` or
        ``type=multipolygon``. Its entries are the graph nodes within
        :data:`~map_data.pathsolver.walkable_area.ENTRY_TOLERANCE` of it that
        belong to another way or have an edge other than the area's own
        outline: where another way (an adjacent area included) joins it, runs
        inside it or ends next to it, or an annotation is spliced onto it.
        Crossings between them are computed on demand, never added to
        :attr:`graph`.
        """
        self._areas: list[WalkableArea] = []
        self._node_areas: dict[int, list[WalkableArea]] = {}
        on_ways = Counter(n for nodes in way_nodes for n in set(nodes))
        candidates = [
            (way, nodes)
            for way, nodes in zip(self._allowed_ways, way_nodes, strict=True)
            if len(nodes) >= 4
            and nodes[0] == nodes[-1]
            and (way.tags.get("area") == "yes" or way.tags.get("type") == "multipolygon")
        ]
        if not candidates:
            return
        ids = list(self.graph)
        xy = np.array([self.nodes[n].ravel()[:2] for n in ids])
        for way, ring in candidates:
            # The outline comes from the nodes (a parsed map's way.line is buffered),
            # the holes from way.line, the only place they are kept.
            # ponytail: a parsed map's holes are shrunk by half the buffer width
            # (1.5 m for a footway); keep the unbuffered relation rings if that bites.
            lines = getattr(way.line, "geoms", [way.line])
            holes = [r.coords for g in lines if isinstance(g, Polygon) for r in g.interiors]
            polygon = Polygon([self.nodes[n].ravel()[:2] for n in ring], holes)
            if not polygon.is_valid:
                logger.warning("Area way %s has an invalid outline; not crossing it.", way.id)
                continue
            ring_nodes = set(ring)
            outline = {frozenset(e) for e in itertools.pairwise(ring)}
            lo = np.array(polygon.bounds[:2]) - ENTRY_TOLERANCE
            hi = np.array(polygon.bounds[2:]) + ENTRY_TOLERANCE
            entries = {
                ids[i]: xy[i]
                for i in np.flatnonzero(np.all((xy >= lo) & (xy <= hi), axis=1))
                if (
                    ids[i] not in ring_nodes
                    or on_ways[ids[i]] > 1
                    or any(frozenset((ids[i], v)) not in outline for v, _ in self.graph[ids[i]])
                )
                and polygon.distance(Point(xy[i])) <= ENTRY_TOLERANCE
            }
            if not entries:
                continue
            area = WalkableArea(polygon, self.edge_factor(way), entries)
            self._areas.append(area)
            for node in entries:
                self._node_areas.setdefault(node, []).append(area)

    def _area_at(self, point: np.ndarray, tolerance: float = 0.0) -> WalkableArea | None:
        """
        The walkable area nearest to *point* if within *tolerance* (0: covering it), or ``None``.
        """
        p = Point(point)
        nearest = min(self._areas, key=lambda a: a.polygon.distance(p), default=None)
        return nearest if nearest is not None and nearest.polygon.distance(p) <= tolerance else None

    def edge_factor(self, way: Way) -> float:
        """
        Weight multiplier of *way*'s edges: ``1 + way cost + rule cost``.

        The ``highway``/``surface`` cost comes from the shared tables
        (:func:`~map_data.pathsolver.way_cost.way_cost`, the grid planner's own
        prices), the extra from a ``cost:`` in the traversability rules. Never
        below 1, which keeps the straight-line A* heuristic admissible.
        """
        return (
            1.0
            + way_cost(way.tags, self.highway_costs, self.surface_costs, self.path_cost_cap)
            + self.traversability.extra_cost(way)
        )

    def _add_edge(self, u: int, v: int, d: float) -> None:
        """
        Add an undirected edge of weight *d* between nodes *u* and *v*.
        """
        self.graph.setdefault(u, []).append((v, d))
        self.graph.setdefault(v, []).append((u, d))

    def _find_closest_edge(
        self,
        point_utm: np.ndarray,
    ) -> tuple[tuple[int, int, np.ndarray, float] | None, float]:
        """
        Find the closest edge using an STRtree spatial index.

        Returns ``((node a, node b, projection, weight factor), distance)``;
        the factor is the one :meth:`edge_factor` gave that edge, so the two
        halves a waypoint splits it into are priced like the rest of the way.
        """
        if self._edge_tree is None:
            return None, float("inf")

        p_sh = Point(point_utm)
        nearest_idx = self._edge_tree.nearest(p_sh)
        if nearest_idx is None:
            return None, float("inf")

        n1, n2 = self._edge_node_pairs[nearest_idx]
        line = self._edge_segments[nearest_idx]
        min_dist = line.distance(p_sh)
        proj_dist = line.project(p_sh)
        projected_point = np.array(line.interpolate(proj_dist).coords[0])
        return (n1, n2, projected_point, self._edge_factors[nearest_idx]), min_dist

    def snap_distance(self, point_utm: np.ndarray) -> float:
        """
        Distance (m) from ``point_utm`` (``[easting, northing]``) to the nearest
        graph edge, 0 inside a walkable area, or ``inf`` when the graph has no
        edges. This is the distance :meth:`plan` compares against
        ``max_snap_distance``.
        """
        point = np.asarray(point_utm, dtype=float)[:2]
        if self._area_at(point) is not None:
            return 0.0
        _, dist = self._find_closest_edge(point)
        return float(dist)

    def _route_segment(
        self,
        id_s: int | str,
        id_g: int | str,
        positions: dict[int | str, np.ndarray],
        extra_adj: dict[int | str, list[tuple[int | str, float]]],
        extra_via: "dict[tuple[int | str, int | str], list[np.ndarray]]",
    ) -> list[np.ndarray] | None:
        """
        Run A* from *id_s* to *id_g* over the graph plus a local subgraph.

        *positions* gives the world coordinate of the two temporary snapped
        nodes (everything else resolves through :attr:`nodes`); *extra_adj*
        gives their (and their edge endpoints') extra adjacency, and
        *extra_via* the geometry of those extra hops that cross an area. Kept
        as plain dicts rather than one mixing them under string/int keys, so
        none needs a cast to satisfy the type checker. Besides the graph's own
        edges, an entry of a walkable area neighbours the area's other entries.
        """

        def get_pos(node: int | str) -> np.ndarray:
            pos = positions.get(node)
            return pos if pos is not None else self.nodes[node].ravel()[:2]  # type: ignore[index]

        def get_neighbors(u: int | str) -> list[tuple[int | str, float]]:
            neighs: list[tuple[int | str, float]] = []
            if isinstance(u, int):
                neighs.extend(self.graph.get(u, []))
                for area in self._node_areas.get(u, ()):
                    neighs.extend((v, cost) for v, (cost, _) in area.crossings(u).items())
            neighs.extend(extra_adj.get(u, []))
            return neighs

        goal_x, goal_y = get_pos(id_g)

        def heuristic(u: int | str) -> float:
            pos = get_pos(u)
            return math.hypot(pos[0] - goal_x, pos[1] - goal_y)

        node_path = astar_search(id_s, id_g, get_neighbors, heuristic)
        if node_path is None:
            return None
        route = [get_pos(node_path[0])]
        for a, b in itertools.pairwise(node_path):
            via = extra_via.get((a, b)) or self._crossing_points(a, b)
            route.extend(via[1:] if via else [get_pos(b)])
        return route

    def _crossing_points(self, a: int | str, b: int | str) -> list[np.ndarray] | None:
        """
        Points from *a* to *b* of the area crossing A* took between them, or
        ``None`` when it took the straight graph edge.

        A* only reports the node sequence, so the hop is recovered as the
        cheaper of the graph edge and the crossings joining the two nodes —
        the one A* relaxed with.
        """
        if not (isinstance(a, int) and isinstance(b, int)):
            return None
        edge = min((w for v, w in self.graph.get(a, []) if v == b), default=np.inf)
        crossings = [
            area.crossings(a)[b] for area in self._node_areas.get(a, ()) if b in area.crossings(a)
        ]
        best = min(crossings, key=lambda c: c[0], default=None)
        return best[1] if best is not None and best[0] < edge else None

    def plan(
        self,
        path_utm: np.ndarray,
        keep_start: bool = False,
        keep_goal: bool = False,
    ) -> np.ndarray | None:
        """
        Plan a path through a sequence of UTM waypoints along the graph.

        Each consecutive pair of waypoints is routed independently. The
        waypoints are snapped to the nearest graph edge before planning,
        so they do not need to lie exactly on the network; the returned
        route consists of on-network points only, the requested waypoints
        being represented by their projections rather than repeated
        verbatim (see :func:`_drop_stacked_points`).

        Parameters
        ----------
        path_utm : np.ndarray
            Array of shape ``(N, 2)`` containing ``[x, y]`` UTM coordinates
            of the desired waypoints, in order. At least two waypoints are
            required.
        keep_start : bool
            Prepend the first waypoint verbatim instead of starting the route
            at its projection. For planning from the robot's own pose, where
            the route has to begin where the robot actually is; the leading
            off-network leg is then the robot's way onto the network. Only the
            first waypoint is treated this way — doing it for a waypoint in
            the middle of the route would produce a spur out to it and back.
        keep_goal : bool
            Append the last waypoint verbatim after its projection, so the
            route ends at the requested coordinate rather than
            :attr:`max_snap_distance` metres short of it. The final,
            off-network leg is skipped when the projection is already within
            :data:`MIN_GOAL_LEG_LENGTH` of the goal.

        Returns
        -------
        np.ndarray or None
            Concatenated path as an ``(M, 2)`` UTM coordinate array with
            coincident vertices and sub-metre spurs removed, or
            ``None`` if fewer than two waypoints were given, a waypoint is
            farther than :attr:`max_snap_distance` from the network, or any
            segment could not be routed.

        """
        if len(path_utm) < 2:
            logger.warning(
                "plan() requires at least two waypoints, got %d; cannot plan.",
                len(path_utm),
            )
            return None

        full_path: list[np.ndarray] = []

        for i in range(len(path_utm) - 1):
            id_s = "temp_start"
            id_g = "temp_goal"
            ends = {
                id_s: np.asarray(path_utm[i], dtype=float)[:2],
                id_g: np.asarray(path_utm[i + 1], dtype=float)[:2],
            }

            # Positions, adjacency and crossing geometry of the two temporary
            # nodes, kept in separate dicts (rather than one mixing them under
            # string/int keys) so none needs a cast to satisfy the type checker.
            positions: dict[int | str, np.ndarray] = {}
            extra_adj: dict[int | str, list[tuple[int | str, float]]] = {}
            extra_via: dict[tuple[int | str, int | str], list[np.ndarray]] = {}
            snapped: dict[str, tuple[int, int, float]] = {}

            for tid, waypoint in ends.items():
                if self._area_at(waypoint) is not None:
                    # Inside a walkable area the waypoint itself is the node.
                    positions[tid] = waypoint
                    continue

                edge_info, dist = self._find_closest_edge(waypoint)
                if edge_info is None:
                    return None
                if dist > self.max_snap_distance:
                    logger.warning(
                        "Waypoint (%.1f, %.1f) is %.1f m from the nearest graph "
                        "edge, beyond the %.1f m snap limit; cannot plan.",
                        waypoint[0],
                        waypoint[1],
                        dist,
                        self.max_snap_distance,
                    )
                    return None
                n1, n2, proj, factor = edge_info
                positions[tid] = proj
                snapped[tid] = (n1, n2, factor)
                for node in (n1, n2):
                    # Distance from the projection to one end of its edge, priced like the way.
                    end = self.nodes[node].ravel()[:2]
                    cost = math.hypot(proj[0] - end[0], proj[1] - end[1]) * factor
                    _link(extra_adj, extra_via, tid, node, cost, None)

            # A node inside a walkable area, or snapped onto or next to one (its
            # rim, say, whose own nodes need not be entries), is joined to the
            # area's entries, and to the other node when both are in it, by
            # in-area crossings.
            areas = {tid: self._area_at(p, ENTRY_TOLERANCE) for tid, p in positions.items()}
            for tid in (id_s, id_g):
                area = areas[tid]
                if area is None:
                    continue
                shared = tid == id_s and areas[id_g] is area
                goal = positions[id_g] if shared else None
                to_entries, to_goal = area.from_point(positions[tid], goal)
                for node, (cost, points) in to_entries.items():
                    _link(extra_adj, extra_via, tid, node, cost, points)
                if to_goal is not None:
                    _link(extra_adj, extra_via, id_s, id_g, *to_goal)

            # Special case: start and goal on the same edge
            if id_s in snapped and id_g in snapped:
                n_s1, n_s2, f_s = snapped[id_s]
                if {n_s1, n_s2} == set(snapped[id_g][:2]):
                    p_s, p_g = positions[id_s], positions[id_g]
                    cost = math.hypot(p_s[0] - p_g[0], p_s[1] - p_g[1]) * f_s
                    _link(extra_adj, extra_via, id_s, id_g, cost, None)

            # Route between temporary nodes
            segment = self._route_segment(id_s, id_g, positions, extra_adj, extra_via)
            if segment is None:
                return None

            # segment is [p_proj_s, ..., p_proj_g] — entirely on the network. The
            # clicked waypoints themselves are deliberately left out: re-inserting
            # an off-network click between its own projections turns every via
            # point into a degenerate out-and-back spur of stacked points.
            full_path.extend(segment)

        if keep_start:
            full_path.insert(0, np.asarray(path_utm[0], dtype=float)[:2])
        if keep_goal:
            goal = np.asarray(path_utm[-1], dtype=float)[:2]
            if not full_path or float(np.linalg.norm(goal - full_path[-1])) > MIN_GOAL_LEG_LENGTH:
                full_path.append(goal)

        return _drop_stacked_points(full_path)


def _link(
    adj: dict[int | str, list[tuple[int | str, float]]],
    via: dict[tuple[int | str, int | str], list[np.ndarray]],
    a: int | str,
    b: int | str,
    cost: float,
    points: list[np.ndarray] | None,
) -> None:
    """
    Join nodes *a* and *b* both ways in *adj*, recording the crossing's *points* (a to b) in *via*.
    """
    adj.setdefault(a, []).append((b, cost))
    adj.setdefault(b, []).append((a, cost))
    if points is not None:
        via[(a, b)] = points
        via[(b, a)] = points[::-1]


def _drop_stacked_points(points: list[np.ndarray]) -> np.ndarray:
    """
    Remove coincident vertices and sub-metre out-and-back spurs from a route.

    Routing each waypoint pair separately makes consecutive segments meet at the
    same projected point, and a waypoint snapping just past a junction makes the
    route step off that junction and back again. Both show up in the exported
    route as several points stacked on top of each other.

    Vertices closer than :data:`TOLERANCE` to their predecessor are dropped, and
    an excursion shorter than :data:`MAX_SPUR_LENGTH` that returns to the vertex
    it started from is collapsed to that vertex. Longer detours are kept: they
    are genuine, e.g. a waypoint on a dead end that has to be visited and left
    the same way.
    """
    out: list[np.ndarray] = []
    for p in points:
        if out and math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) <= TOLERANCE:
            continue
        # p returns to out[-2] after a short hop out to out[-1]: drop the hop,
        # which also makes p coincident with the new last point.
        if (
            len(out) >= 2
            and math.hypot(p[0] - out[-2][0], p[1] - out[-2][1]) <= TOLERANCE
            and math.hypot(out[-1][0] - out[-2][0], out[-1][1] - out[-2][1]) <= MAX_SPUR_LENGTH
        ):
            out.pop()
            continue
        out.append(p)
    return np.array(out)
