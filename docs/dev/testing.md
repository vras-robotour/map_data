# Testing

## Running the tests

On the host, after `uv sync` (see [Contributing](contributing.md)):

```bash
uv run python -m pytest                      # whole suite
uv run python -m pytest tests/test_route.py  # one file
uv run python -m pytest -k overpass          # by keyword
uv run python -m pytest --cov=map_data --cov-report=term-missing   # with coverage
```

In a ROS 2 workspace (built and sourced, see [Installation](../index.md#with-ros-2-colcon)), the same suite also runs the ROS-only tests:

```bash
python3 -m pytest -q          # from src/map_data
bash scripts/ros_smoke.sh     # route_planner, osm_cloud and the viewer on the demo map
```

`ros_smoke.sh` starts the three executables in a private ROS domain and checks the node list, the `/grid` cloud, one PlanRoute goal, the viewer's HTTP endpoints and a clean SIGINT exit.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests to `master`:

| Job | What |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `typecheck` | `mypy map_data/` |
| `test` | The suite on Python 3.12 and 3.13 with `--cov-fail-under=72` |
| `ros` | `colcon build` in a `ros:jazzy` container, the suite, `ros_smoke.sh` |

`docs.yml` builds the documentation with `mkdocs build --strict`.

The coverage gate lives in CI only, not in the pytest config, so single-file runs and `colcon test` work without `pytest-cov`.

## Principles

- **No network.** Tests that would reach Overpass patch `OverpassClient` (the `mock_overpass_client` fixture in `conftest.py`) or `OverpassClient._http`, with `time.sleep` patched so retries are instant.
- **No ROS required.** Tests of ROS code skip when ROS 2 is missing (`importorskip` of `map_data.route_planner` and `launch`, `ROS_AVAILABLE` for the tracker node); the QR tests skip without OpenCV's `QRCodeEncoder`. The viewer under test is `create_app(data_dir=...)`.
- **Isolated filesystem.** Every test that writes uses `tmp_path`; nothing touches `data/`.
- **ROS isolation in `conftest.py`.** A session fixture stops the viewer's tracker node at exit (its executor thread would otherwise block shutdown). The logger class is reset, since ROS's `launch_testing` pytest plugin makes `map_data` loggers stop propagating to `caplog`.
- **Small shared fixtures.** `conftest.py` holds the canned Overpass responses, `footway_network_mapdata` (a three-way footway network) and `STROMOVKA_FIXTURE`.

## Real-map fixture

`tests/data/stromovka.mapdata` is a small crop (< 300 KB) of the Stromovka park map in Prague, with a hand-written `stromovka.annotations.json`. It covers footways, roads, stairs, boardwalks and a doubly mapped corridor; `tests/data/README.md` describes how it was cut.

Map data © OpenStreetMap contributors, under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/). Keep the attribution in `tests/data/README.md` when changing the fixture.
