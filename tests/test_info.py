import json
from types import SimpleNamespace

import pytest
import utm
from shapely.geometry import LineString

from map_data import info
from map_data.utils.way import Way


def _make_mapdata(
    roads=None,
    footways=None,
    barriers=None,
    coords_file="coords.gpx",
):
    return SimpleNamespace(
        coords_file=coords_file,
        zone_number=33,
        zone_letter="U",
        min_x=0.0,
        max_x=100.0,
        min_y=0.0,
        max_y=50.0,
        roads_list=roads or [],
        footways_list=footways or [],
        barriers_list=barriers or [],
    )


def _run_stats(tmp_path, monkeypatch, capsys, md, annotations=None):
    """
    Run get_stats against a fake MapData and return the captured stdout.
    """
    p = tmp_path / "site.mapdata"
    p.write_text("{}")
    if annotations is not None:
        sidecar = tmp_path / "site.annotations.json"
        sidecar.write_text(json.dumps({"annotations": annotations}))
    monkeypatch.setattr(info, "MapData", SimpleNamespace(load=lambda path: md))
    info.get_stats(str(p))
    return capsys.readouterr().out


def _stat(out, label):
    """
    Extract the value printed after 'label:' in the stats output.
    """
    for line in out.splitlines():
        if line.startswith(label):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"stat {label!r} not found in output:\n{out}")


def test_get_stats_counts_and_metadata(tmp_path, monkeypatch, capsys):
    roads = [
        Way(id=1, tags={"highway": "primary"}, line=LineString([(0, 0), (1, 0)])),
        Way(id=2, tags={"highway": "service"}, line=LineString([(0, 1), (1, 1)])),
    ]
    footways = [Way(id=3, tags={"highway": "footway"}, line=LineString([(0, 2), (1, 2)]))]
    barriers = [
        Way(id=4, tags={"barrier": "wall"}, line=LineString([(0, 3), (1, 3)])),
        Way(id=5, tags={"barrier": "fence"}, line=LineString([(0, 4), (1, 4)])),
        Way(id=6, tags={"barrier": "hedge"}, line=LineString([(0, 5), (1, 5)])),
    ]
    md = _make_mapdata(roads=roads, footways=footways, barriers=barriers)

    out = _run_stats(tmp_path, monkeypatch, capsys, md)

    assert "site.mapdata" in out
    assert _stat(out, "Source") == "File: coords.gpx"
    assert _stat(out, "UTM Zone") == "33U"
    assert _stat(out, "Bounds X") == "[0.0, 100.0]"
    assert _stat(out, "Bounds Y") == "[0.0, 50.0]"
    # (100 - 0) * (50 - 0), thousands-separated
    assert _stat(out, "Total Area") == "5,000 m²"
    assert _stat(out, "Roads") == "2"
    assert _stat(out, "Footways") == "1"
    assert _stat(out, "Barriers") == "3"
    # No sidecar file was created
    assert "Annotations" not in out


def test_get_stats_array_source_and_geometryless_footway(tmp_path, monkeypatch, capsys):
    footways = [
        Way(id=1, tags={"highway": "footway"}, line=LineString([(0, 0), (10, 0)])),
        Way(id=2, tags={"highway": "footway"}, line=None),
    ]
    md = _make_mapdata(footways=footways, coords_file=None)

    out = _run_stats(tmp_path, monkeypatch, capsys, md)

    assert _stat(out, "Source") == "Array"
    # Both footways are counted, but the one without geometry contributes no length
    assert _stat(out, "Footways") == "2"
    assert float(_stat(out, "Total Footway Distance").removesuffix(" m")) == pytest.approx(10.0)


def test_get_stats_reports_annotations_sidecar(tmp_path, monkeypatch, capsys):
    md = _make_mapdata()

    out = _run_stats(tmp_path, monkeypatch, capsys, md, annotations=[{}, {}])

    assert _stat(out, "Annotations") == "2 (manual edits)"


def test_get_stats_missing_file_prints_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        info,
        "MapData",
        SimpleNamespace(load=lambda path: pytest.fail("load must not be called")),
    )
    info.get_stats(str(tmp_path / "does_not_exist.mapdata"))
    assert capsys.readouterr().out == ""


def test_get_stats_footway_distance_is_centerline_length(tmp_path, monkeypatch, capsys):
    """
    In a loaded MapData, footways have been buffered into Polygons, so
    ``w.line.length`` is the buffer's perimeter (~2*L + pi*width), not the
    walked distance. The stat should report the centerline length instead.
    """
    centerline = LineString([(0.0, 0.0), (10.0, 0.0)])
    footway = Way(
        id=1,
        is_area=True,
        nodes=[100, 101],
        tags={"highway": "footway"},
        line=centerline.buffer(1.5),  # what buffer_line produces for footways
    )
    md = _make_mapdata(footways=[footway])

    out = _run_stats(tmp_path, monkeypatch, capsys, md)

    reported = float(_stat(out, "Total Footway Distance").removesuffix(" m"))
    assert reported == pytest.approx(centerline.length, abs=0.1)


def test_get_stats_footway_distance_uses_node_coordinates(tmp_path, monkeypatch, capsys):
    """
    When the loaded MapData has a nodes_cache covering the footway's nodes,
    the centerline length is reconstructed from the node coordinates rather
    than estimated from the buffer polygon.
    """
    e, n, zn, zl = utm.from_latlon(50.0, 14.0)
    lat2, lon2 = utm.to_latlon(e + 10.0, n, zn, zl)
    centerline = LineString([(e, n), (e + 10.0, n)])
    footway = Way(
        id=1,
        is_area=True,
        nodes=[100, 101],
        tags={"highway": "footway"},
        line=centerline.buffer(1.5),
    )
    md = _make_mapdata(footways=[footway])
    md.zone_number, md.zone_letter = zn, zl
    md.nodes_cache = {
        100: {"lat": 50.0, "lon": 14.0, "tags": {}},
        101: {"lat": lat2, "lon": lon2, "tags": {}},
    }

    out = _run_stats(tmp_path, monkeypatch, capsys, md)

    reported = float(_stat(out, "Total Footway Distance").removesuffix(" m"))
    assert reported == pytest.approx(10.0, abs=0.01)
