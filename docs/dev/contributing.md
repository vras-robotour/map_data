# Contributing

## Prerequisites

- Python 3.12 (pinned in `.python-version`; ROS 2 Jazzy's Python)
- [`uv`](https://docs.astral.sh/uv/)
- ROS 2 Jazzy, only for the ROS nodes (`osm_cloud`, `route_planner`, the viewer's Tracker)

## Setup

```bash
git clone https://github.com/vras-robotour/map_data.git
cd map_data
uv sync                  # runtime + the dev dependency group, from uv.lock
uv run pre-commit install
```

`uv sync` installs the locked versions in `uv.lock`. After changing dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock`.

For a ROS 2 workspace, clone into `<ws>/src/map_data` and run `scripts/setup_ws.sh`; it symlinks the nested `map_data_interfaces` package into `<ws>/src` so `colcon` finds it. See [Installation](../index.md#with-ros-2-colcon).

## Checks

```bash
uv run python -m pytest       # tests, see Testing
uv run ruff check --fix .
uv run ruff format .
uv run mypy map_data/
uv run mkdocs build --strict  # docs; mkdocs serve for a live preview
```

The pre-commit hook runs ruff and mypy on every commit. CI runs the same checks, the test suite with a 72 % coverage gate and a ROS 2 build with the smoke test; see [Testing](testing.md#ci).

Tool versions are pinned and must match: ruff in `pyproject.toml` (dev group), `.pre-commit-config.yaml` and `RUFF_VERSION` in `.github/workflows/ci.yml`; mypy in `pyproject.toml` and `.pre-commit-config.yaml`.

## Code style

Configuration lives in `pyproject.toml`.

- **Line length**: 100.
- **Types**: PEP 585/604 annotations (`list[str]`, `X | None`); return annotations on new functions. Third-party modules without stubs are in mypy's `ignore_missing_imports`; a remaining `# type: ignore[<code>]` carries a short reason.
- **Docstrings**: NumPy style on public API (rendered by mkdocstrings); state what the code does, not its history.
- **Comments**: only for a non-obvious *why*.

## Submitting changes

1. Branch from `master`.
2. Add or update tests and docs with the change; add a line to the *Unreleased* section of the [changelog](changelog.md).
3. Make sure the checks above pass.
4. Open a pull request against `master`.
