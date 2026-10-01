from collections.abc import Sequence

import numpy as np

# Segments shorter than this (in the same units as the coordinates, i.e. metres)
# are treated as coincident points rather than subdivided further.
TOLERANCE = 1e-3


def densify_ways(
    points: dict[int, np.ndarray],
    node_lists: Sequence[Sequence[int]],
    max_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split ways (lists of node ids into *points*) into points at most *max_step* apart.

    Each way contributes its first node and then every segment's end, with
    ``ceil(length / max_step)`` equal steps per segment; ways of fewer than two
    nodes contribute nothing. Returns the ``(N, 2)`` points and, for each, the
    index into *node_lists* of the way it came from.
    """
    chunks, owner = [], []
    for k, ids in enumerate(node_lists):
        if len(ids) < 2:
            continue
        nodes = np.array([points[i].ravel()[:2] for i in ids])
        starts, ends = nodes[:-1], nodes[1:]
        dists = np.linalg.norm(ends - starts, axis=1)

        parts = [nodes[:1]]
        for point0, point1, dist in zip(starts, ends, dists, strict=True):
            if dist <= TOLERANCE:
                parts.append(point1[None])
                continue
            num = int(np.ceil(dist / max_step))
            steps = np.arange(1, num + 1) / num
            parts.append(point0 + steps[:, None] * (point1 - point0))
        way_points = np.concatenate(parts)
        chunks.append(way_points)
        owner.append(np.full(len(way_points), k))

    if not chunks:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    return np.concatenate(chunks), np.concatenate(owner)
