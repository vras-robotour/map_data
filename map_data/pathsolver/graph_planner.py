import heapq
import numpy as np
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree


class GraphPlanner:
    def __init__(self, map_data, highway_types=None):
        self.map_data = map_data
        self.highway_types = highway_types or ["footway"]
        self.nodes = self.map_data.get_points()  # Dict[int, np.ndarray (3,1)]
        self.graph = {}  # node_id -> [(neighbor_id, distance)]
        self._build_graph()

    def _build_graph(self):
        self._allowed_ways = []
        if "footway" in self.highway_types:
            self._allowed_ways.extend(self.map_data.footways_list)
        if "road" in self.highway_types:
            self._allowed_ways.extend(self.map_data.roads_list)

        edge_segments = []
        edge_node_pairs = []

        for way in self._allowed_ways:
            for i in range(len(way.nodes) - 1):
                n1 = way.nodes[i]
                n2 = way.nodes[i + 1]

                p1 = self.nodes[n1].ravel()[:2]
                p2 = self.nodes[n2].ravel()[:2]
                dist = np.linalg.norm(p1 - p2)

                self.graph.setdefault(n1, []).append((n2, dist))
                self.graph.setdefault(n2, []).append((n1, dist))

                edge_segments.append(LineString([p1, p2]))
                edge_node_pairs.append((n1, n2))

        self._edge_segments = edge_segments
        self._edge_node_pairs = edge_node_pairs
        self._edge_tree = STRtree(edge_segments) if edge_segments else None

    def _find_closest_edge(self, point_utm):
        """Find the closest edge using an STRtree spatial index."""
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

    def a_star(self, start_node, goal_node, extra_nodes=None):
        """
        Standard A* between two nodes.
        extra_nodes: dict for temporary nodes (e.g. snapped points)
        """
        count = 0
        q = [(0, count, start_node)]
        visited = {start_node: (0, None)}  # node -> (cost, parent)

        def get_neighbors(u):
            neighs = list(self.graph.get(u, []))
            if extra_nodes and u in extra_nodes:
                neighs.extend(extra_nodes[u])
            return neighs

        if not get_neighbors(start_node) and start_node != goal_node:
            return None

        def heuristic(u, v):
            p1 = self._get_node_pos(u, extra_nodes)
            p2 = self._get_node_pos(v, extra_nodes)
            return np.linalg.norm(p1 - p2)

        while q:
            (cost_plus_h, _, u) = heapq.heappop(q)
            cost = visited[u][0]

            if u == goal_node:
                # Path found, reconstruct
                path = []
                curr = u
                while curr is not None:
                    path.append(self._get_node_pos(curr, extra_nodes))
                    curr = visited[curr][1]
                return path[::-1]

            for v, dist in get_neighbors(u):
                new_cost = cost + dist
                if v not in visited or new_cost < visited[v][0]:
                    visited[v] = (new_cost, u)
                    h = heuristic(v, goal_node)
                    count += 1
                    heapq.heappush(q, (new_cost + h, count, v))

        return None

    def _get_node_pos(self, node_id, extra_nodes_data=None):
        if isinstance(node_id, str) and node_id.startswith("temp_"):
            return extra_nodes_data["positions"][node_id]
        return self.nodes[node_id].ravel()[:2]

    def plan(self, path_utm):
        """
        Plan path between multiple UTM points.
        Connects clicked points to the nearest point on the nearest edge.
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
                if np.linalg.norm(p - final_segment[-1]) > 1e-3:
                    final_segment.append(p)

            if np.linalg.norm(p_goal - final_segment[-1]) > 1e-3:
                final_segment.append(p_goal)

            if i > 0:
                full_path.extend(final_segment[1:])
            else:
                full_path.extend(final_segment)

        return np.array(full_path)
