import heapq
import numpy as np


class GraphPlanner:
    def __init__(self, map_data, highway_types=None):
        self.map_data = map_data
        self.highway_types = highway_types or ["footway"]
        self.nodes = self.map_data.get_points()  # Dict[int, np.ndarray (3,1)]
        self.graph = {}  # node_id -> [(neighbor_id, distance)]
        self._build_graph()

    def _build_graph(self):
        allowed_ways = []
        if "footway" in self.highway_types:
            allowed_ways.extend(self.map_data.footways_list)
        if "road" in self.highway_types:
            allowed_ways.extend(self.map_data.roads_list)

        for way in allowed_ways:
            for i in range(len(way.nodes) - 1):
                n1 = way.nodes[i]
                n2 = way.nodes[i + 1]

                p1 = self.nodes[n1].ravel()[:2]
                p2 = self.nodes[n2].ravel()[:2]
                dist = np.linalg.norm(p1 - p2)

                self.graph.setdefault(n1, []).append((n2, dist))
                self.graph.setdefault(n2, []).append((n1, dist))

    def _find_closest_node(self, point_utm):
        """
        Find the closest node in the graph to the given UTM point.
        """
        p_utm = np.array(point_utm)
        min_dist = float("inf")
        best_node = None

        for node_id in self.graph:
            pos = self.nodes[node_id].ravel()[:2]
            dist = np.linalg.norm(p_utm - pos)
            if dist < min_dist:
                min_dist = dist
                best_node = node_id
        return best_node, min_dist

    def a_star(self, start_node, goal_node, extra_nodes=None):
        """
        Standard A* between two nodes.
        extra_nodes: dict for temporary nodes (not used in current nearest-node logic but kept for future)
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
            (_, _, u) = heapq.heappop(q)
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
        Connects clicked points to the nearest graph node with a straight line.
        """
        full_path = []

        for i in range(len(path_utm) - 1):
            p_start = path_utm[i]
            p_goal = path_utm[i + 1]

            # Find nearest nodes in the graph
            node_s, _ = self._find_closest_node(p_start)
            node_g, _ = self._find_closest_node(p_goal)

            if node_s is None or node_g is None:
                return None

            # Route between topological nodes
            segment = self.a_star(node_s, node_g)
            if not segment:
                return None

            # The segment starts at node_s_pos and ends at node_g_pos.
            # We add the actual clicked points p_start and p_goal.
            # To avoid duplicate points if user clicked exactly on a node:
            segment_with_ends = []
            segment_with_ends.append(p_start)

            for p in segment:
                if np.linalg.norm(p - segment_with_ends[-1]) > 1e-3:
                    segment_with_ends.append(p)

            if np.linalg.norm(p_goal - segment_with_ends[-1]) > 1e-3:
                segment_with_ends.append(p_goal)

            if i > 0:
                full_path.extend(segment_with_ends[1:])
            else:
                full_path.extend(segment_with_ends)

        return np.array(full_path)
