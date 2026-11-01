"""Tests for the ``map_data_plan`` command line tool."""

import argparse
import json
from pathlib import Path

import pytest
import utm

from map_data.map_data import MapData
from map_data.plan_route_cli import main, parse_latlon


def _latlon(lat0, lon0, dx, dy):
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    lat, lon = utm.to_latlon(e0 + dx, n0 + dy, zn, zl)
    return f"{lat:.8f},{lon:.8f}"


def test_parse_latlon_accepts_geo_uri():
    assert parse_latlon("geo:48.8016394,16.8011145") == (48.8016394, 16.8011145)
    assert parse_latlon(" 50.1, 14.4 ") == (50.1, 14.4)
    with pytest.raises(argparse.ArgumentTypeError, match="expected lat,lon"):
        parse_latlon("nonsense")


def test_cli_plans_and_saves_gpx(footway_network_mapdata, tmp_path, capsys):
    path, lat0, lon0 = footway_network_mapdata
    out = tmp_path / "route.gpx"
    rc = main(
        [
            "-f",
            str(path),
            "--start",
            _latlon(lat0, lon0, 0.0, 0.0),
            "--goal",
            _latlon(lat0, lon0, 100.0, 95.0),
            "--spacing",
            "3",
            "--save",
            str(out),
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert data["success"]
    assert data["waypoints"] > 50
    assert data["length_m"] == pytest.approx(190.0, abs=10.0)
    assert "<trk>" in out.read_text()


def test_cli_reports_failure(footway_network_mapdata, capsys):
    path, lat0, lon0 = footway_network_mapdata
    rc = main(
        [
            "-f",
            str(path),
            "--start",
            _latlon(lat0, lon0, 0.0, 0.0),
            "--goal",
            _latlon(lat0, lon0, 450.0, 0.0),
            "--json",
        ]
    )
    assert rc == 1
    data = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert data == {"success": False, "reason": "unreachable", "message": data["message"]}


def test_cli_reports_an_unprojectable_point(footway_network_mapdata, caplog):
    path, lat0, lon0 = footway_network_mapdata
    rc = main(["-f", str(path), "--start", f"{lat0},{lon0}", "-p", "91,14"])
    assert rc == 1
    assert "bad_parameter" in caplog.text


def test_cli_needs_two_points(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    assert main(["-f", str(path), "--goal", f"{lat0},{lon0}"]) == 2


def test_cli_annotations_switch(footway_network_mapdata, capsys):
    """``--annotations none`` ignores a store that would delete the needed way."""
    path, lat0, lon0 = footway_network_mapdata
    path.with_name(path.stem + ".annotations.json").write_text(
        json.dumps({"version": 1, "annotations": [], "deleted_ways": [{"id": 2}]})
    )
    args = [
        "-f",
        str(path),
        "--start",
        _latlon(lat0, lon0, 0.0, 0.0),
        "--goal",
        _latlon(lat0, lon0, 100.0, 95.0),
        "--max-snap-distance",
        "50",
        "--json",
    ]
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["reason"] == "snap_too_far"
    assert main([*args, "--annotations", "none"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["success"]


def test_cli_traversability_switches(footway_network_mapdata, tmp_path, capsys):
    """A rule file that removes the branch fails; ``--no-traversability`` ignores it."""
    path, lat0, lon0 = footway_network_mapdata
    md = MapData.load(str(path))
    for way in md.footways_list:
        if way.id == 2:  # the branch up to the goal
            way.tags = {"highway": "footway", "surface": "grass"}
    md.save(str(path))
    rules = tmp_path / "rules.yaml"
    rules.write_text("rules:\n  - match: {surface: grass}\n    traversable: false\n")
    args = [
        "-f",
        str(path),
        "--start",
        _latlon(lat0, lon0, 0.0, 0.0),
        "--goal",
        _latlon(lat0, lon0, 100.0, 95.0),
        "--max-snap-distance",
        "50",
        "--json",
    ]

    assert main([*args, "--traversability", str(rules)]) == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["reason"] == "snap_too_far"
    assert main([*args, "--no-traversability"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["success"]


def test_cli_rejects_a_missing_rule_file(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata

    with pytest.raises(SystemExit, match="not found"):
        main(
            [
                "-f",
                str(path),
                "--start",
                _latlon(lat0, lon0, 0.0, 0.0),
                "--goal",
                _latlon(lat0, lon0, 200.0, 0.0),
                "--traversability",
                "/nonexistent/rules.yaml",
            ]
        )


@pytest.mark.parametrize(
    "name",
    [
        str(Path(__file__).resolve().parents[1] / "demo" / "buchlovice.mapdata"),
        "buchlovice.mapdata",
    ],
)
def test_cli_plans_the_documented_demo_route(name, capsys, tmp_path, monkeypatch):
    """The demo command in README.md and docs/planning.md, on the default footway graph."""
    monkeypatch.chdir(tmp_path)  # a bare name is found in the package, not the cwd
    rc = main(
        [
            "-f",
            name,
            "--start",
            "49.084416,17.336309",
            "--goal",
            "49.083106,17.338527",
            "--json",
        ]
    )

    assert rc == 0
    data = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert data["algorithm"] == "graph"
    assert data["length_m"] > 300


@pytest.mark.parametrize(
    "flags",
    [
        ["--cell-size", "0"],
        ["--cell-size", "-1"],
        ["--cell-size", "nan"],
        ["--inflate-obstacles", "-0.1"],
        ["--max-snap-distance", "0"],
        ["--ways", "footpath"],
    ],
)
def test_cli_rejects_bad_parameters(flags):
    from map_data.plan_route_cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["-f", "x.mapdata", *flags])
