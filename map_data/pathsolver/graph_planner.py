"""
Graph-based path planning on OSM ways.

This module provides the GraphPlanner class which builds a graph from
OpenStreetMap ways and finds paths using Dijkstra or A*.
"""

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from map_data.pathsolver.astar import astar_search

TOLERANCE = 1e-3

if TYPE_CHECKING:
    from map_data.map_data import MapData


class GraphPlanner:
    """
    Graph-based path planner that routes along the OSM road and footway network.

    Builds an undirected weighted graph from the ways stored in a
    :class:`~map_data.map_data.MapData` instance. Manually annotated paths
    (negative-ID ways) are spliced into the graph by projecting their
    endpoints onto the nearest OSM edge, so annotations extend the
    traversable network seamlessly.

    Planning is performed with A* (see :meth:`plan`). The planner operates
    entirely in UTM coordinates.
    """

    def __init__(self, map_data: "MapData", highway_types: list[str] | None = None) -> None:
        """
        Initialize the graph planner.

        Parameters
        ----------
        map_data : MapData
            Parsed map data containing the OSM ways and node cache.
        highway_types : list of str, optional
            Way categories to include in the graph. Supported values are
            ``"footway"`` and ``"road"``. Defaults to ``["footway"]``.

        """
        self.map_data = map_data
        self.highway_types = highway_types or ["footway"]
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

        # First pass: identify potential splits from annotations
        edge_segments = []
        edge_way_info = []  # (way_obj, segment_index)

        for way in self._allowed_ways:
            for i in range(len(way.nodes) - 1):
                n1, n2 = way.nodes[i], way.nodes[i + 1]
                p1 = self.nodes[n1].ravel()[:2]
                p2 = self.nodes[n2].ravel()[:2]
                edge_segments.append(LineString([p1, p2]))
                edge_way_info.append((way, i))

        tree = STRtree(edge_segments) if edge_segments else None

        # Group splits by way and segment
        splits = {}  # (way_id, segment_index) -> [(proj_dist, proj_node_id, node_id, dist_to_edge)]
        new_internal_id = -2000000

        if tree:
            threshold = 5.0
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
                        if edge_way_info[idx][0].id == way.id:
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
                        self.nodes[proj_node_id] = np.array([p_proj[0], p_proj[1], 0.0]).reshape(
                            3,
                            1,
                        )

                        target_way, segment_idx = edge_way_info[best_idx]
                        splits.setdefault((id(target_way), segment_idx), []).append(
                            (proj_dist, proj_node_id, node_id, min_dist),
                        )

        # Apply splits to _allowed_ways by inserting new nodes
        for (way_ptr, segment_idx), s_list in splits.items():
            # Find the way object by pointer (since we might have modified nodes)
            target_way = next(w for w in self._allowed_ways if id(w) == way_ptr)
            # Sort splits on this segment by distance from segment start
            s_list.sort(key=lambda x: x[0], reverse=True)
            for _, proj_node_id, ann_node_id, dist_to_edge in s_list:
                target_way.nodes.insert(segment_idx + 1, proj_node_id)
                # Manually add the connection from annotation endpoint to the new junction node
                self._add_edge(ann_node_id, proj_node_id, dist_to_edge)

        # Second pass: build final graph and tree from (possibly modified) ways
        final_edge_segments = []
        final_edge_node_pairs = []

        for way in self._allowed_ways:
            for i in range(len(way.nodes) - 1):
                n1, n2 = way.nodes[i], way.nodes[i + 1]
                p1 = self.nodes[n1].ravel()[:2]
                p2 = self.nodes[n2].ravel()[:2]
                dist = np.linalg.norm(p1 - p2)

                self._add_edge(n1, n2, dist)
                final_edge_segments.append(LineString([p1, p2]))
                final_edge_node_pairs.append((n1, n2))

        self._edge_segments = final_edge_segments
        self._edge_node_pairs = final_edge_node_pairs
        self._edge_tree = STRtree(final_edge_segments) if final_edge_segments else None

    def _add_edge(self, u: int, v: int, d: float) -> None:
        """
        Add an undirected edge of length *d* between nodes *u* and *v*.
        """
        self.graph.setdefault(u, []).append((v, d))
        self.graph.setdefault(v, []).append((u, d))

    def _find_closest_edge(
        self,
        point_utm: np.ndarray,
    ) -> tuple[tuple[int, int, np.ndarray] | None, float]:
        """
        Find the closest edge using an STRtree spatial index.
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
        return (n1, n2, projected_point), min_dist

    def a_star(
        self,
        start_node: int | str,
        goal_node: int | str,
        extra_nodes: dict | None = None,
    ) -> list[np.ndarray] | None:
        """
        Run A* between two graph nodes and return the path as UTM coordinates.

        Parameters
        ----------
        start_node : int or str
            ID of the start node in the graph.
        goal_node : int or str
            ID of the goal node in the graph.
        extra_nodes : dict, optional
            Temporary adjacency entries and positions for nodes not
            permanently in the graph (e.g. snapped projection points).
            Expected keys: ``"positions"`` mapping node ID → ``np.ndarray``,
            plus per-node adjacency lists.

        Returns
        -------
        list of np.ndarray or None
            Sequence of ``[x, y]`` UTM positions along the path, or
            ``None`` if no path exists.

        """

        def get_neighbors(u: int | str) -> list[tuple[int | str, float]]:
            neighs: list[tuple[int | str, float]] = []
            if isinstance(u, int):
                neighs.extend(self.graph.get(u, []))
            if extra_nodes and u in extra_nodes:
                neighs.extend(extra_nodes[u])
            return neighs

        if not get_neighbors(start_node) and start_node != goal_node:
            return None

        def heuristic(u: int | str) -> float:
            p1 = self._get_node_pos(u, extra_nodes)
            p2 = self._get_node_pos(goal_node, extra_nodes)
            return float(np.linalg.norm(p1 - p2))

        node_path = astar_search(start_node, goal_node, get_neighbors, heuristic)

        if node_path is None:
            return None

        return [self._get_node_pos(node, extra_nodes) for node in node_path]

    def _get_node_pos(self, node_id: int | str, extra_nodes_data: dict | None = None) -> np.ndarray:
        if isinstance(node_id, str) and node_id.startswith("temp_"):
            return extra_nodes_data["positions"][node_id]
        return self.nodes[node_id].ravel()[:2]

    def plan(self, path_utm: np.ndarray) -> np.ndarray | None:
        """
        Plan a path through a sequence of UTM waypoints along the graph.

        Each consecutive pair of waypoints is routed independently. The
        waypoints are snapped to the nearest graph edge before planning,
        so they do not need to lie exactly on the network.

        Parameters
        ----------
        path_utm : np.ndarray
            Array of shape ``(N, 2)`` containing ``[x, y]`` UTM coordinates
            of the desired waypoints, in order.

        Returns
        -------
        np.ndarray or None
            Concatenated path as an ``(M, 2)`` UTM coordinate array, or
            ``None`` if any segment could not be routed.

        """
        full_path = []

        for i in range(len(path_utm) - 1):
            p_start = path_utm[i]
            p_goal = path_utm[i + 1]

            # Find nearest edges and projections
            edge_start_info, _ = self._find_closest_edge(p_start)
            edge_goal_info, _ = self._find_closest_edge(p_goal)

            if not edge_start_info or not edge_goal_info:
                return None

            id_s = "temp_start"
            id_g = "temp_goal"
            n_s1, n_s2, p_proj_s = edge_start_info
            n_g1, n_g2, p_proj_g = edge_goal_info

            # Construct local subgraph for snapped points
            extra = {
                "positions": {id_s: p_proj_s, id_g: p_proj_g},
                id_s: [
                    (n_s1, np.linalg.norm(p_proj_s - self.nodes[n_s1].ravel()[:2])),
                    (n_s2, np.linalg.norm(p_proj_s - self.nodes[n_s2].ravel()[:2])),
                ],
                id_g: [
                    (n_g1, np.linalg.norm(p_proj_g - self.nodes[n_g1].ravel()[:2])),
                    (n_g2, np.linalg.norm(p_proj_g - self.nodes[n_g2].ravel()[:2])),
                ],
                n_s1: [(id_s, np.linalg.norm(p_proj_s - self.nodes[n_s1].ravel()[:2]))],
                n_s2: [(id_s, np.linalg.norm(p_proj_s - self.nodes[n_s2].ravel()[:2]))],
                n_g1: [(id_g, np.linalg.norm(p_proj_g - self.nodes[n_g1].ravel()[:2]))],
                n_g2: [(id_g, np.linalg.norm(p_proj_g - self.nodes[n_g2].ravel()[:2]))],
            }

            # Special case: start and goal on the same edge
            if (n_s1 == n_g1 and n_s2 == n_g2) or (n_s1 == n_g2 and n_s2 == n_g1):
                dist_sg = np.linalg.norm(p_proj_s - p_proj_g)
                extra[id_s].append((id_g, dist_sg))
                extra[id_g].append((id_s, dist_sg))

            # Route between temporary nodes (projections)
            segment = self.a_star(id_s, id_g, extra)
            if not segment:
                return None

            # Combine: clicked point -> projection -> graph path -> projection -> clicked point
            # segment already contains [p_proj_s, ..., p_proj_g]

            final_segment = []
            final_segment.append(p_start)

            for p in segment:
                if np.linalg.norm(p - final_segment[-1]) > TOLERANCE:
                    final_segment.append(p)

            if np.linalg.norm(p_goal - final_segment[-1]) > TOLERANCE:
                final_segment.append(p_goal)

            if i > 0:
                full_path.extend(final_segment[1:])
            else:
                full_path.extend(final_segment)

        return np.array(full_path)
