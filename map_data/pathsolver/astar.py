import heapq
from collections.abc import Callable, Iterable
from typing import TypeVar

_N = TypeVar("_N")


def astar_search(
    start_node: _N,
    goal_node: _N,
    get_neighbors_func: Callable[[_N], Iterable[tuple[_N, float]]],
    heuristic_func: Callable[[_N], float],
) -> list[_N] | None:
    """
    A generic A* search implementation.

    Parameters:
    -----------
    start_node : object
        The starting node.
    goal_node : object
        The goal node.
    get_neighbors_func : callable
        A function that takes a node and returns an iterable of (neighbor, cost) tuples.
    heuristic_func : callable
        A function that takes a node and returns the estimated cost to the goal.

    Returns:
    --------
    path : list or None
        The path from start to goal as a list of nodes, or None if no path exists.
    """
    count = 0
    # Priority queue stores (f_score, count, current_node)
    # f_score = cost + heuristic
    q = [(0, count, start_node)]

    # visited maps node -> (cost, parent)
    visited = {start_node: (0, None)}
    closed = set()

    while q:
        (f_score, _, u) = heapq.heappop(q)
        if u in closed:
            continue
        closed.add(u)
        cost = visited[u][0]

        if u == goal_node:
            # Path found, reconstruct
            path = []
            curr = u
            while curr is not None:
                path.append(curr)
                curr = visited[curr][1]
            return path[::-1]

        for v, dist in get_neighbors_func(u):
            new_cost = cost + dist
            if v not in visited or new_cost < visited[v][0]:
                visited[v] = (new_cost, u)
                h = heuristic_func(v)
                count += 1
                heapq.heappush(q, (new_cost + h, count, v))

    return None
