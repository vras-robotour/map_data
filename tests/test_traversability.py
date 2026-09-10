"""Tag rules deciding which ways the robot may drive on."""

from pathlib import Path

import pytest
import yaml

from map_data.traversability import (
    DEFAULT_CONFIG_NAME,
    TraversabilityError,
    TraversabilityRules,
    load_traversability,
    resolve_traversability_path,
)
from map_data.utils.way import Way

PKG = Path(__file__).resolve().parents[1]
DEFAULT_RULES_FILE = PKG / "config" / DEFAULT_CONFIG_NAME


def _way(**tags):
    return Way(id=1, tags=tags)


def _rules(*rules, default=None):
    data = {"rules": list(rules)}
    if default is not None:
        data["default"] = default
    return TraversabilityRules.from_dict(data, source="test")


# ── matching ────────────────────────────────────────────────────────────────


def test_exact_value_match():
    rules = _rules({"match": {"highway": "steps"}, "traversable": False, "reason": "stairs"})

    verdict = rules.evaluate({"highway": "steps"})

    assert verdict.traversable is False
    assert verdict.reason == "stairs"
    assert verdict.rule_index == 0
    assert rules.evaluate({"highway": "footway"}).traversable is True


def test_any_of_a_list_matches():
    rules = _rules({"match": {"surface": ["grass", "mud"]}, "traversable": False})

    assert not rules.is_traversable(_way(surface="grass"))
    assert not rules.is_traversable(_way(surface="mud"))
    assert rules.is_traversable(_way(surface="asphalt"))


def test_wildcard_matches_any_value_of_a_present_tag():
    rules = _rules({"match": {"bridge": "*"}, "traversable": False})

    assert not rules.is_traversable(_way(bridge="yes"))
    assert not rules.is_traversable(_way(bridge="boardwalk"))
    assert rules.is_traversable(_way(highway="footway"))


def test_several_keys_are_anded():
    rules = _rules({"match": {"highway": "path", "surface": ["ground", "dirt"]}, "cost": 0.5})

    assert rules.evaluate({"highway": "path", "surface": "dirt"}).cost == 0.5
    assert rules.evaluate({"highway": "path", "surface": "asphalt"}).cost == 0.0
    assert rules.evaluate({"highway": "footway", "surface": "dirt"}).cost == 0.0


def test_first_match_wins_so_an_allow_rule_can_precede_a_deny_rule():
    rules = _rules(
        {"match": {"bridge": "boardwalk"}, "traversable": True, "reason": "the good boardwalk"},
        {"match": {"bridge": "*"}, "traversable": False, "reason": "bridge"},
    )

    good = rules.evaluate({"bridge": "boardwalk"})
    assert good.traversable is True
    assert good.rule_index == 0
    assert rules.evaluate({"bridge": "yes"}).reason == "bridge"


def test_unmatched_way_takes_the_default():
    rules = _rules(
        {"match": {"highway": "footway"}, "traversable": True},
        default={"traversable": False, "cost": 0.0},
    )

    assert rules.is_traversable(_way(highway="footway"))
    assert not rules.is_traversable(_way(highway="track"))
    assert rules.evaluate({}).rule_index is None
    assert rules.evaluate(None).traversable is False


def test_yes_and_no_are_normalised_on_both_sides():
    """YAML turns bare yes/no into booleans; OSM only ever has the strings."""
    data = yaml.safe_load(
        "rules:\n"
        "  - match: {informal: yes}\n"
        "    cost: 1.0\n"
        "  - match: {access: no}\n"
        "    traversable: false\n"
    )
    rules = TraversabilityRules.from_dict(data, source="test")

    assert rules.evaluate({"informal": "yes"}).cost == 1.0
    assert rules.evaluate({"informal": "no"}).cost == 0.0
    assert not rules.is_traversable(_way(access="no"))
    # ... and a tag value that somehow arrives as a bool matches too
    assert rules.evaluate({"informal": True}).cost == 1.0


def test_empty_rules_are_a_no_op():
    rules = TraversabilityRules()

    assert rules.is_traversable(_way(highway="steps", surface="grass"))
    assert rules.edge_factor(_way(highway="steps")) == 1.0
    assert rules.summary([_way(highway="steps")]) == {}


# ── costs, summary, extend ──────────────────────────────────────────────────


def test_edge_factor_is_one_plus_the_extra_cost():
    rules = _rules({"match": {"informal": "yes"}, "cost": 1.0})

    assert rules.extra_cost(_way(informal="yes")) == 1.0
    assert rules.edge_factor(_way(informal="yes")) == 2.0
    assert rules.edge_factor(_way(highway="footway")) == 1.0


def test_summary_counts_removed_ways_per_reason():
    rules = _rules(
        {"match": {"highway": "steps"}, "traversable": False, "reason": "stairs"},
        {"match": {"surface": "grass"}, "traversable": False, "reason": "soft surface"},
        {"match": {"informal": "yes"}, "cost": 1.0, "reason": "informal path"},
    )
    ways = [
        _way(highway="steps"),
        _way(highway="steps"),
        _way(surface="grass"),
        _way(informal="yes"),  # costly, not removed
        _way(highway="footway"),
    ]

    assert rules.summary(ways) == {"stairs": 2, "soft surface": 1}


def test_extend_prepends_a_deny_rule_for_the_excluded_highway_values():
    rules = _rules({"match": {"highway": "steps"}, "traversable": True, "reason": "allowed"})

    extended = rules.extend(["steps", "track"])

    assert not extended.is_traversable(_way(highway="steps"))  # wins over the allow rule
    assert not extended.is_traversable(_way(highway="track"))
    assert extended.is_traversable(_way(highway="footway"))
    assert rules.is_traversable(_way(highway="steps")), "the original must not change"


def test_extend_with_nothing_to_exclude_returns_the_same_rules():
    rules = _rules({"match": {"highway": "steps"}, "traversable": False})

    assert rules.extend(()) is rules


# ── validation ──────────────────────────────────────────────────────────────


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(TraversabilityError, match="unknown key"):
        TraversabilityRules.from_dict({"rulez": []}, source="test")


def test_unknown_rule_key_is_rejected():
    with pytest.raises(TraversabilityError, match="unknown key"):
        _rules({"match": {"highway": "steps"}, "traversible": False})


def test_negative_cost_is_rejected():
    with pytest.raises(TraversabilityError, match="must be >= 0"):
        _rules({"match": {"highway": "steps"}, "cost": -1.0})


def test_non_boolean_traversable_is_rejected():
    with pytest.raises(TraversabilityError, match="must be true or false"):
        _rules({"match": {"highway": "steps"}, "traversable": "false"})


def test_non_numeric_cost_is_rejected():
    with pytest.raises(TraversabilityError, match="must be a number"):
        _rules({"match": {"highway": "steps"}, "cost": "high"})


def test_rule_without_a_match_is_rejected():
    with pytest.raises(TraversabilityError, match="non-empty map"):
        _rules({"traversable": False})


def test_empty_value_list_is_rejected():
    with pytest.raises(TraversabilityError, match="empty value list"):
        _rules({"match": {"surface": []}})


def test_rules_must_be_a_list():
    with pytest.raises(TraversabilityError, match="must be a list"):
        TraversabilityRules.from_dict({"rules": {"match": {}}}, source="test")


def test_missing_file_is_an_error():
    with pytest.raises(TraversabilityError, match="not found"):
        load_traversability("/nonexistent/traversability.yaml")


def test_malformed_file_names_itself(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("rules:\n  - match: {highway: steps}\n    cost: -2\n")

    with pytest.raises(TraversabilityError, match=str(path)):
        load_traversability(path)


# ── loading ─────────────────────────────────────────────────────────────────


def test_from_file_round_trip(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        "default: {traversable: true}\n"
        "rules:\n"
        "  - match: {highway: steps}\n"
        "    traversable: false\n"
        "    reason: stairs\n"
    )

    rules = TraversabilityRules.from_file(path)

    assert rules.source == str(path)
    assert not rules.is_traversable(_way(highway="steps"))


def test_load_traversability_passes_a_rule_set_through():
    rules = TraversabilityRules()

    assert load_traversability(rules) is rules


def test_load_traversability_without_a_path_reads_the_package_file():
    rules = load_traversability()

    assert rules.source == str(DEFAULT_RULES_FILE)
    assert resolve_traversability_path() == DEFAULT_RULES_FILE
    assert resolve_traversability_path(DEFAULT_RULES_FILE) == DEFAULT_RULES_FILE


# ── the shipped defaults ────────────────────────────────────────────────────


def test_shipped_defaults_refuse_what_the_robot_cannot_drive():
    rules = load_traversability()

    assert not rules.is_traversable(_way(highway="steps"))
    assert not rules.is_traversable(_way(highway="footway", surface="grass"))
    assert not rules.is_traversable(_way(highway="footway", bridge="yes"))
    assert not rules.is_traversable(_way(highway="footway", smoothness="very_bad"))
    assert not rules.is_traversable(_way(highway="footway", access="private"))
    assert rules.is_traversable(_way(highway="footway", surface="asphalt"))
    # The tunnel rule is shipped commented out: it cuts Stromovka in two.
    assert rules.is_traversable(_way(highway="footway", tunnel="yes"))
    # Surfaces are priced by planner_defaults.yaml, not here.
    assert rules.extra_cost(_way(highway="path", surface="dirt")) == 0.0
    assert rules.extra_cost(_way(highway="path", informal="yes")) == 1.0
