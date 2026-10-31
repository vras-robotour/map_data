#!/usr/bin/env python3
"""Benchmark map loading, the osm_cloud grid, and OSM parsing."""

from __future__ import annotations

import argparse
import json
import logging
import resource
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import overpy

from map_data.annotations import load_mapdata_with_annotations
from map_data.map_data import MapData

ROOT = Path(__file__).resolve().parents[2]
TRACKED_MAPS = (ROOT / "tests/data/stromovka.mapdata", ROOT / "demo/buchlovice.mapdata")
_ROS_STUBS = (
    "geometry_msgs",
    "geometry_msgs.msg",
    "rclpy",
    "rclpy.executors",
    "rclpy.node",
    "rclpy.qos",
    "rcl_interfaces",
    "rcl_interfaces.msg",
    "ros2_numpy",
    "sensor_msgs",
    "sensor_msgs.msg",
    "tf2_msgs",
    "tf2_msgs.msg",
    "tf2_ros",
    "visualization_msgs",
    "visualization_msgs.msg",
)


def _stats(samples: list[float]) -> tuple[float, float]:
    """Return median and median absolute deviation, a robust run-to-run spread."""
    median = statistics.median(samples)
    return median, statistics.median(abs(sample - median) for sample in samples)


def _peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _benchmark(label: str, call: Callable[[], Any], repeat: int) -> tuple[float, float]:
    samples: list[float] = []
    for i in range(repeat):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
        logging.info("%s run %d/%d: %.6f s", label, i + 1, repeat, samples[-1])
        del result
    return _stats(samples)


def _load(path: Path) -> None:
    load_mapdata_with_annotations(path)


def _osm_cloud_functions() -> Any:
    try:
        from map_data import osm_cloud
    except ModuleNotFoundError as e:
        if e.name not in _ROS_STUBS:
            raise
        sys.modules.update({name: MagicMock() for name in _ROS_STUBS})
        sys.modules["rclpy.node"].Node = object
        from map_data import osm_cloud
    return osm_cloud


def _grid_call(map_data: MapData) -> Callable[[], Any]:
    osm_cloud = _osm_cloud_functions()
    node = SimpleNamespace(
        map_data=map_data,
        grid_min=[map_data.min_x, map_data.min_y],
        grid_max=[map_data.max_x, map_data.max_y],
        grid_res=0.25,
        local_frame="utm",
        max_path_dist=1.0,
        neighbor_cost="linear",
        highway_types=["footway"],
        _to_local=lambda points: osm_cloud.transform_points(points, np.eye(4), 0.0),
        get_logger=lambda: logging.getLogger("bench.osm_cloud"),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: None)),
    )
    return lambda: osm_cloud.OSMCloud.get_cloud(node)


def _parse_call(raw_response: str, gpx_path: Path) -> Callable[[], int]:
    result = overpy.Overpass().parse_json(raw_response)

    def call() -> int:
        map_data = MapData(gpx_path)
        map_data.osm_data = result
        return map_data.run_parse()

    return call


def _print_result(label: str, result: tuple[float, float]) -> None:
    median, spread = result
    print(f"{label}: {median:.6f} s ± {spread:.6f} s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5, help="repetitions per measurement")
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        help="mapdata files for load/grid benchmarks (defaults to tracked maps)",
    )
    parser.add_argument(
        "--osm-cache",
        type=Path,
        help="a real .osm_cache.json for the run_parse benchmark",
    )
    parser.add_argument("--gpx", type=Path, help="GPX file matching --osm-cache")
    parser.add_argument("--quick", action="store_true", help="one repetition on tracked maps")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.quick:
        args.repeat = 1
        args.inputs = list(TRACKED_MAPS)
    elif args.inputs is None:
        args.inputs = list(TRACKED_MAPS)
    if bool(args.osm_cache) != bool(args.gpx):
        parser.error("--osm-cache and --gpx must be given together")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args(argv)

    maps: dict[Path, MapData] = {}
    for path in args.inputs:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        median, spread = _benchmark(f"load {path.name}", partial(_load, path), args.repeat)
        maps[path] = load_mapdata_with_annotations(path)[0]
        _print_result(f"load {path.name}", (median, spread))

    for path, loaded_map in maps.items():
        grid = _grid_call(loaded_map)
        median, spread = _benchmark(f"grid {path.name}", grid, args.repeat)
        _print_result(f"grid {path.name}", (median, spread))

    if args.osm_cache and args.gpx:
        cache = json.loads(args.osm_cache.read_text())
        median, spread = _benchmark(
            f"parse {args.osm_cache.name}",
            _parse_call(cache["raw"], args.gpx),
            args.repeat,
        )
        _print_result(f"parse {args.osm_cache.name}", (median, spread))

    print(f"peak RSS: {_peak_rss_mb():.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
