"""
Decide from OSM tags which ways the robot may drive on.

A ``.mapdata`` map is a faithful copy of OSM: it contains stairways, muddy
shortcuts, boardwalks and tunnels, all tagged but none of them judged. This
module turns those tags into a verdict — *drive here* / *never drive here* /
*drive here reluctantly* — from a YAML rule file the operator edits
(``config/traversability.yaml``), so a change of mind about, say, bridges is a
one-line edit rather than a code change.

The rules are used in three places, all of them through
:class:`TraversabilityRules`:

* :func:`~map_data.annotations.load_mapdata_with_annotations` drops the
  non-traversable ways from the map at load time, so the route planner, the
  intersection rings and the ``osm_cloud`` cost cloud all describe the same
  network;
* :class:`~map_data.pathsolver.graph_planner.GraphPlanner` filters them again
  (for callers that pass a raw map) and adds a rule's optional ``cost`` to the
  ``highway``/``surface`` cost of :mod:`map_data.pathsolver.way_cost` when it
  weighs the way's edges, so a "possible but unpleasant" way is taken only
  when the alternative is much longer;
* the ``route_planner`` and ``osm_cloud`` nodes and the ``map_data_plan`` CLI
  pass the file the operator picked.

The file format is documented in ``config/traversability.yaml`` itself and in
``docs/planning.md``. This module imports nothing from ROS.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from map_data.utils.config import find_config, load_config

if TYPE_CHECKING:
    from map_data.utils.way import Way

logger = logging.getLogger(__name__)

#: Name of the package config file used when no path is given.
DEFAULT_CONFIG_NAME = "traversability.yaml"

#: ``match`` value that matches any value of a present tag.
ANY_VALUE = "*"

_RULE_KEYS = frozenset({"match", "traversable", "cost", "reason"})
_DEFAULT_KEYS = frozenset({"traversable", "cost", "reason"})
_TOP_KEYS = frozenset({"default", "rules"})

#: Reason reported for ways no rule matched.
DEFAULT_REASON = "default"
#: Reason of the rule :meth:`TraversabilityRules.extend` prepends (the excluded
#: ``highway`` values are appended to it, e.g. ``"excluded highway type (steps)"``).
EXCLUDE_HIGHWAY_REASON = "excluded highway type"


class TraversabilityError(ValueError):
    """A traversability rule file is malformed."""


def _tag_value(value: Any) -> str:
    """
    Normalise a tag or rule value to the string OSM would carry.

    YAML parses bare ``yes``/``no`` (and ``true``/``false``) as booleans, so
    ``bridge: yes`` in a rule file and ``bridge=yes`` in the map would never
    compare equal without this. Numbers are stringified for the same reason.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


@dataclass(frozen=True)
class Verdict:
    """
    What the rules say about one way.

    Attributes
    ----------
    traversable : bool
        ``False`` removes the way from the map and from the graph.
    cost : float
        Extra cost of driving here, as a fraction of the way's length, *added*
        to the ``highway``/``surface`` cost from ``planner_defaults.yaml``:
        the graph planner weighs an edge ``(1 + way cost + cost) * length``.
    reason : str
        The matching rule's ``reason`` (:data:`DEFAULT_REASON` when no rule
        matched), for logs and route explanations.
    rule_index : int or None
        Index of the matching rule in the file, ``None`` for the default.

    """

    traversable: bool = True
    cost: float = 0.0
    reason: str = DEFAULT_REASON
    rule_index: int | None = None


@dataclass(frozen=True)
class Rule:
    """
    One ``match`` -> verdict line of the rule file.

    ``match`` is normalised to ``{tag key: tuple of accepted values}``; the
    single value :data:`ANY_VALUE` accepts any value of a present tag. All
    keys of a rule must match (AND).
    """

    match: tuple[tuple[str, tuple[str, ...]], ...]
    traversable: bool = True
    cost: float = 0.0
    reason: str = ""

    def matches(self, tags: Mapping[str, Any]) -> bool:
        """Return ``True`` when every key of :attr:`match` matches *tags*."""
        for key, values in self.match:
            if key not in tags:
                return False
            if values == (ANY_VALUE,):
                continue
            if _tag_value(tags[key]) not in values:
                return False
        return True


@dataclass(frozen=True)
class TraversabilityRules:
    """
    An ordered list of :class:`Rule` plus the verdict for unmatched ways.

    Build one with :meth:`from_file`, :meth:`from_dict` or
    :func:`load_traversability`; an instance with no rules is the "everything
    is traversable at cost 0" no-op, which is what ``--no-traversability``
    passes.
    """

    rules: tuple[Rule, ...] = ()
    default: Verdict = field(default_factory=Verdict)
    #: Where the rules came from, for log messages ("" for a built-in default).
    source: str = ""

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, source: str = "") -> TraversabilityRules:
        """
        Build the rules from the parsed YAML mapping.

        Parameters
        ----------
        data : mapping or None
            ``{"default": {...}, "rules": [...]}``; ``None`` or ``{}`` yields
            the no-op rules.
        source : str
            File the mapping came from, kept in :attr:`source` for logs.

        Raises
        ------
        TraversabilityError
            On unknown keys, a non-boolean ``traversable``, a negative
            ``cost`` or a malformed ``match``.

        """
        if not data:
            return cls(source=source)
        if not isinstance(data, Mapping):
            raise TraversabilityError(f"{source or 'traversability rules'}: expected a mapping")
        unknown = set(data) - _TOP_KEYS
        if unknown:
            raise TraversabilityError(
                f"{source or 'traversability rules'}: unknown key(s) "
                f"{', '.join(sorted(unknown))}; expected {', '.join(sorted(_TOP_KEYS))}"
            )
        default = cls._parse_default(data.get("default"), source)
        raw_rules = data.get("rules") or []
        if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, str | bytes):
            raise TraversabilityError(f"{source or 'traversability rules'}: 'rules' must be a list")
        rules = tuple(
            cls._parse_rule(raw, i, source)
            for i, raw in enumerate(raw_rules)  # type: ignore[arg-type]
        )
        return cls(rules=rules, default=default, source=source)

    @classmethod
    def from_file(cls, path: str | Path) -> TraversabilityRules:
        """Read and validate a YAML rule file."""
        p = Path(path).expanduser()
        try:
            data = yaml.safe_load(p.read_text())
        except OSError as e:
            raise TraversabilityError(f"cannot read traversability rules {p}: {e}") from e
        except yaml.YAMLError as e:
            raise TraversabilityError(f"{p} is not valid YAML: {e}") from e
        return cls.from_dict(data, source=str(p))

    @staticmethod
    def _parse_default(raw: Any, source: str) -> Verdict:
        if raw is None:
            return Verdict()
        if not isinstance(raw, Mapping):
            raise TraversabilityError(
                f"{source or 'traversability rules'}: 'default' must be a map"
            )
        unknown = set(raw) - _DEFAULT_KEYS
        if unknown:
            raise TraversabilityError(
                f"{source or 'traversability rules'}: unknown key(s) in 'default': "
                f"{', '.join(sorted(unknown))}"
            )
        return Verdict(
            traversable=_parse_traversable(raw.get("traversable", True), "default", source),
            cost=_parse_cost(raw.get("cost", 0.0), "default", source),
            reason=str(raw.get("reason", DEFAULT_REASON)),
        )

    @staticmethod
    def _parse_rule(raw: Any, index: int, source: str) -> Rule:
        where = f"{source or 'traversability rules'}: rule {index}"
        if not isinstance(raw, Mapping):
            raise TraversabilityError(f"{where} must be a map with a 'match' key")
        unknown = set(raw) - _RULE_KEYS
        if unknown:
            raise TraversabilityError(
                f"{where}: unknown key(s) {', '.join(sorted(unknown))}; "
                f"expected {', '.join(sorted(_RULE_KEYS))}"
            )
        match = raw.get("match")
        if not isinstance(match, Mapping) or not match:
            raise TraversabilityError(f"{where}: 'match' must be a non-empty map of tag: value")
        parsed: list[tuple[str, tuple[str, ...]]] = []
        for key, value in match.items():
            if isinstance(value, Sequence) and not isinstance(value, str | bytes):
                values = tuple(_tag_value(v) for v in value)
                if not values:
                    raise TraversabilityError(f"{where}: empty value list for tag '{key}'")
            else:
                values = (_tag_value(value),)
            parsed.append((str(key), values))
        return Rule(
            match=tuple(parsed),
            traversable=_parse_traversable(raw.get("traversable", True), f"rule {index}", source),
            cost=_parse_cost(raw.get("cost", 0.0), f"rule {index}", source),
            reason=str(raw.get("reason", "")),
        )

    # ------------------------------------------------------------------ use
    def evaluate(self, tags: Mapping[str, Any] | None) -> Verdict:
        """
        Verdict for a way's ``tags``; the first matching rule wins.

        Parameters
        ----------
        tags : mapping or None
            The way's OSM tags. ``None`` and ``{}`` match no rule and take
            the file's ``default``.

        Returns
        -------
        Verdict

        """
        if tags:
            for i, rule in enumerate(self.rules):
                if rule.matches(tags):
                    return Verdict(
                        traversable=rule.traversable,
                        cost=rule.cost,
                        reason=rule.reason or f"rule {i}",
                        rule_index=i,
                    )
        return self.default

    def is_traversable(self, way: Way) -> bool:
        """Return ``True`` when the robot may drive on *way*."""
        return self.evaluate(way.tags).traversable

    def extra_cost(self, way: Way) -> float:
        """
        The matching rule's ``cost`` for *way* (0 when none asks for one).

        This is an *extra* on top of the ``highway``/``surface`` cost the
        planners take from ``config/planner_defaults.yaml``
        (:func:`~map_data.pathsolver.way_cost.way_cost`), for tags those tables
        know nothing about.
        """
        return self.evaluate(way.tags).cost

    def edge_factor(self, way: Way) -> float:
        """
        Multiplier the rules alone put on the graph edges of *way*
        (``1 + extra cost``; the graph planner adds the table cost on top).

        Always ``>= 1``, so the straight-line A* heuristic stays admissible
        and a route's weight never falls below its geometric length.
        """
        return 1.0 + self.extra_cost(way)

    def summary(self, ways: Iterable[Way]) -> dict[str, int]:
        """
        Count the non-traversable ways of *ways* per reason.

        Returns
        -------
        dict
            ``{reason: number of ways removed for it}``, in rule order.

        """
        counts: dict[str, int] = {}
        for way in ways:
            verdict = self.evaluate(way.tags)
            if not verdict.traversable:
                counts[verdict.reason] = counts.get(verdict.reason, 0) + 1
        return counts

    def extend(self, exclude_highway: Iterable[str]) -> TraversabilityRules:
        """
        Copy of these rules with ``highway`` values that are never traversable.

        This is how the older ``exclude_highway`` parameter of the nodes, the
        loader and the planner keeps working: its values become one deny rule
        in front of everything else, so they win over any allow rule in the
        file.

        Parameters
        ----------
        exclude_highway : iterable of str
            ``highway`` tag values to remove. Empty returns ``self``.

        """
        values = tuple(sorted({str(v) for v in exclude_highway}))
        if not values:
            return self
        rule = Rule(
            match=(("highway", values),),
            traversable=False,
            reason=f"{EXCLUDE_HIGHWAY_REASON} ({'/'.join(values)})",
        )
        return replace(self, rules=(rule, *self.rules))

    def describe(self) -> str:
        """One-line description of the rule set, for the node logs."""
        where = self.source or "built-in defaults"
        return f"{len(self.rules)} rule(s) from {where}"


def _parse_traversable(value: Any, where: str, source: str) -> bool:
    if not isinstance(value, bool):
        raise TraversabilityError(
            f"{source or 'traversability rules'}: {where}: 'traversable' must be true or false, "
            f"got {value!r}"
        )
    return value


def _parse_cost(value: Any, where: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TraversabilityError(
            f"{source or 'traversability rules'}: {where}: 'cost' must be a number, got {value!r}"
        )
    if value < 0:
        raise TraversabilityError(
            f"{source or 'traversability rules'}: {where}: 'cost' must be >= 0, got {value}"
        )
    return float(value)


def resolve_traversability_path(path: str | Path | None = None) -> Path | None:
    """
    The rule file a ``path`` of ``None``/``""`` (the package default) resolves to.

    Returns ``None`` when the file does not exist, which is how the nodes tell
    "no file to watch for changes" from a real path.
    """
    if path:
        p = Path(path).expanduser()
        return p if p.is_file() else None
    return find_config(DEFAULT_CONFIG_NAME)


def load_traversability(
    path: TraversabilityRules | str | Path | None = None,
) -> TraversabilityRules:
    """
    Rules from *path*, the package default file, or a ready-made rule set.

    Parameters
    ----------
    path : TraversabilityRules or str or Path or None
        A :class:`TraversabilityRules` is returned unchanged (so every caller
        can accept "rules or a path"); ``None`` and ``""`` load
        ``config/traversability.yaml`` from the installed package; anything
        else is a path to a rule file.

    Returns
    -------
    TraversabilityRules

    Raises
    ------
    TraversabilityError
        If a rule file is given and is missing or malformed. A missing
        *package* default is not an error: it yields the no-op rules.

    """
    if isinstance(path, TraversabilityRules):
        return path
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise TraversabilityError(f"traversability rules {p} not found")
        return TraversabilityRules.from_file(p)
    found = resolve_traversability_path()
    if found is None:
        logger.debug("No %s in the package config; every way is traversable", DEFAULT_CONFIG_NAME)
        return TraversabilityRules()
    return TraversabilityRules.from_dict(load_config(DEFAULT_CONFIG_NAME), source=str(found))
