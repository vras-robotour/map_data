"""
How expensive a way is to drive on, from its ``highway`` and ``surface`` tags.

Both planners charge the same price for the same way: the grid planner
(:class:`~map_data.pathsolver.grid_constructor.PathGrid`) lowers a cell's cost
towards :func:`way_cost` of the nearest way, and the graph planner
(:class:`~map_data.pathsolver.graph_planner.GraphPlanner`) weighs an edge
``length * (1 + way_cost)``. The tables live in ``config/planner_defaults.yaml``
(``highway_costs``, ``surface_costs``, ``path_cost_cap``), so a route that
prefers asphalt to gravel is a config change, not a code change.

Which ways are drivable *at all* is a separate question, answered from the tags
by :mod:`map_data.traversability`; a rule there may add an extra cost on top of
the tables for something they do not know about (``informal=yes``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from map_data.utils.config import load_config

#: ``highway`` value assumed when a way carries none.
DEFAULT_HIGHWAY = "path"
#: ``surface`` value assumed when a way carries none (OSM leaves the usual
#: paved surface untagged far more often than an unpaved one).
DEFAULT_SURFACE = "asphalt"
#: Cost of a ``highway`` value that is in no table (an unknown road type is
#: assumed to be about as bad as a residential street).
UNKNOWN_HIGHWAY_COST = 0.5

#: Used when ``config/planner_defaults.yaml`` is missing or incomplete.
FALLBACK_HIGHWAY_COSTS: dict[str, float] = {
    "pedestrian": 0.0,
    "footway": 0.0,
    "path": 0.1,
    "living_street": 0.1,
    "track": 0.3,
    "service": 0.3,
    "residential": 0.5,
    "unclassified": 0.5,
    "tertiary": 0.7,
    "secondary": 0.9,
    "primary": 1.0,
}
FALLBACK_SURFACE_COSTS: dict[str, float] = {
    "asphalt": 0.0,
    "paving_stones": 0.0,
    "concrete": 0.0,
    "fine_gravel": 0.1,
    "gravel": 0.2,
    "dirt": 0.3,
    "grass": 0.5,
    "sand": 0.4,
}
FALLBACK_PATH_COST_CAP = 0.85


def load_cost_tables(
    highway_costs: Mapping[str, float] | None = None,
    surface_costs: Mapping[str, float] | None = None,
    path_cost_cap: float | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Resolve the cost tables, filling in ``config/planner_defaults.yaml``.

    Parameters
    ----------
    highway_costs, surface_costs : mapping, optional
        Explicit tables; ``None`` takes the ones from the config file (or the
        ``FALLBACK_*`` constants when it is missing).
    path_cost_cap : float, optional
        Highest cost any way may reach; ``None`` takes the config value.

    Returns
    -------
    tuple
        ``(highway_costs, surface_costs, path_cost_cap)``.

    """
    defaults: dict[str, Any] = load_config("planner_defaults.yaml")
    return (
        dict(highway_costs)
        if highway_costs is not None
        else dict(defaults.get("highway_costs", FALLBACK_HIGHWAY_COSTS)),
        dict(surface_costs)
        if surface_costs is not None
        else dict(defaults.get("surface_costs", FALLBACK_SURFACE_COSTS)),
        float(
            path_cost_cap
            if path_cost_cap is not None
            else defaults.get("path_cost_cap", FALLBACK_PATH_COST_CAP)
        ),
    )


def way_cost(
    tags: Mapping[str, str] | None,
    highway_costs: Mapping[str, float],
    surface_costs: Mapping[str, float],
    path_cost_cap: float = FALLBACK_PATH_COST_CAP,
) -> float:
    """
    Cost of driving on a way with these ``tags``, in ``[0, path_cost_cap]``.

    The ``highway`` cost and the ``surface`` cost add up and are capped: a
    gravel track is worse than a gravel footway, but nothing is worse than
    ``path_cost_cap``, which keeps a mapped way cheaper than open terrain
    (``default_off_path_cost``).

    Parameters
    ----------
    tags : mapping or None
        The way's OSM tags. A missing ``highway``/``surface`` is read as
        :data:`DEFAULT_HIGHWAY` / :data:`DEFAULT_SURFACE`.
    highway_costs, surface_costs : mapping
        The tables, e.g. from :func:`load_cost_tables`.
    path_cost_cap : float
        Upper bound of the result.

    Returns
    -------
    float

    """
    tags = tags or {}
    highway = tags.get("highway", DEFAULT_HIGHWAY)
    surface = tags.get("surface", DEFAULT_SURFACE)
    cost = highway_costs.get(highway, UNKNOWN_HIGHWAY_COST) + surface_costs.get(surface, 0.0)
    return min(path_cost_cap, cost)
