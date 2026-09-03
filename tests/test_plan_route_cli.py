"""Tests for the ``map_data_plan`` command line tool."""

import json

import pytest
import utm

from map_data.plan_route_cli import main, parse_latlon


def _latlon(lat0, lon0, dx, dy):
    e0, n0, zn, zl = utm.from_latlon(lat0, lon0)
    lat, lon = utm.to_latlon(e0 + dx, n0 + dy, zn, zl)
    return f"{lat:.8f},{lon:.8f}"


def test_parse_latlon_accepts_geo_uri():
    assert parse_latlon("geo:48.8016394,16.8011145") == (48.8016394, 16.8011145)
    assert parse_latlon(" 50.1, 14.4 ") == (50.1, 14.4)
    with pytest.raises(Exception):  # noqa: B017 - argparse type error
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
    assert data["success"] and data["waypoints"] > 50
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


def test_cli_needs_two_points(footway_network_mapdata):
    path, lat0, lon0 = footway_network_mapdata
    assert main(["-f", str(path), "--goal", f"{lat0},{lon0}"]) == 2
