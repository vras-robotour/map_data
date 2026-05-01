import numpy as np
from shapely.geometry import Point, MultiPoint
from typing import Tuple


def get_point_line(
    p1: Point, p2: Point, density: float, increase: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the line between two points.
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y

    a = np.array([x1, y1])
    b = np.array([x2, y2])

    vec = (b - a).T

    # Ceil, because more points is OK, while less points could be problematic
    line = get_equidistant_points(a, b, int(np.ceil(np.linalg.norm(vec) / density)) + 1)
    dist_line = np.zeros((len(line), 1))

    if increase > 0:
        line, dist_line = increase_line(line, dist_line, b - a, increase, density)

    return vec, line, dist_line


def increase_line(
    line: np.ndarray, dist_line: np.ndarray, vec: np.ndarray, n: int, density: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Increase the line by n points in the direction of vec.
    """
    vec = vec / np.linalg.norm(vec)
    arange_increase_vec = density * np.arange(1, n + 1)
    arange_increase_vec = arange_increase_vec.reshape((-1, 1))
    increase_vec = arange_increase_vec * vec
    before = line[0] * np.ones((n, 2)) - increase_vec
    before = np.flip(before, axis=0)
    after = line[-1] * np.ones((n, 2)) + increase_vec

    line = np.concatenate((before, line, after), axis=0)
    dist_line = np.concatenate(
        (np.flip(arange_increase_vec, axis=0), dist_line, arange_increase_vec), axis=0
    )
    return line, dist_line


def get_equidistant_points(p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
    """
    Split the line between p1 and p2 into n equidistant points.
    """
    return np.concatenate(
        (
            np.expand_dims(np.linspace(p1[0], p2[0], max(n, 2)), axis=1),
            np.expand_dims(np.linspace(p1[1], p2[1], max(n, 2)), axis=1),
        ),
        axis=1,
    )


def points_to_graph_points(
    point1: Point, point2: Point, density: float = 1.0, width: float = 10.0
) -> Tuple[MultiPoint, MultiPoint, np.ndarray]:
    """
    Transform point into graph point.
    """
    perpendicular_increase = int(round(width / 2 / density))
    parallel_increase = int(round(width / 4 / density))

    if point1.bounds == point2.bounds:
        return (
            MultiPoint([[point1.x, point1.y]]),
            MultiPoint([[point1.x, point1.y]]),
            np.zeros((1, 1)),
        )
    else:
        vec, point_line, dist_line = get_point_line(
            point1, point2, density, parallel_increase
        )

    normal_vec = np.matmul(
        np.array(
            [
                [np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                [np.sin(np.pi / 2), np.cos(np.pi / 2)],
            ]
        ),
        vec,
    ) / np.linalg.norm(vec)

    points_in_line = point_line.shape[0]

    all_points = np.zeros((points_in_line * (perpendicular_increase * 2 + 1), 2))
    dist_from_line = np.zeros((points_in_line * (perpendicular_increase * 2 + 1), 1))

    line_start_index = points_in_line * perpendicular_increase
    line_end_index = points_in_line * (perpendicular_increase + 1) - 1

    all_points[line_start_index : line_end_index + 1, :] = point_line
    dist_from_line[line_start_index : line_end_index + 1] = dist_line

    for i in range(perpendicular_increase):
        pos_line_points = point_line + normal_vec * density * (i + 1)
        neg_line_points = point_line - normal_vec * density * (i + 1)
        all_points[
            points_in_line * (perpendicular_increase + i + 1) : points_in_line
            * (perpendicular_increase + i + 2),
            :,
        ] = pos_line_points
        all_points[
            points_in_line * (perpendicular_increase - i - 1) : points_in_line
            * (perpendicular_increase - i),
            :,
        ] = neg_line_points
        dist_from_line[
            points_in_line * (perpendicular_increase + i + 1) : points_in_line
            * (perpendicular_increase + i + 2)
        ] = dist_line + (i + 1) * density
        dist_from_line[
            points_in_line * (perpendicular_increase - i - 1) : points_in_line
            * (perpendicular_increase - i)
        ] = dist_line + (i + 1) * density

    all_points = MultiPoint(all_points)
    point_line = MultiPoint(point_line)

    return all_points, point_line, dist_from_line
