# Contributing

## Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (preferred) or plain `pip`
- ROS2 Humble or later — **optional**; only needed for the `create_mapdata` and `osm_cloud` nodes

## Getting the code

```bash
git clone https://github.com/vras-robotour/map_data.git
cd map_data
```

## Setting up the development environment

Install the package in editable mode together with the documentation dependencies:

```bash
uv pip install -e .
uv pip install mkdocs mkdocs-material "mkdocstrings[python]"
```

!!! warning "Use `uv pip`, not plain `pip`"
    The project uses `uv` for package management. Plain `pip` may install into the wrong interpreter on machines with multiple Python environments.

## Running the tests

```bash
pytest tests/
```

All tests are standalone — no ROS2 context or network access is required.

To run a specific test file:

```bash
pytest tests/test_astar.py -v
```

## Running the docs locally

```bash
mkdocs serve
```

The site is served at `http://127.0.0.1:8000` with live reload.

To build a static copy:

```bash
mkdocs build
```

## Code style

- **Formatting** — no formatter is currently enforced; follow the existing style (4-space indentation, ~100-character lines).
- **Type annotations** — add return type annotations to any new function. Parameter annotations are required for public API methods.
- **Docstrings** — public methods use NumPy-style docstrings (matches the `mkdocstrings` configuration). Internal helpers can omit docstrings when the name is self-explanatory.
- **Comments** — only add a comment when the *why* is non-obvious. Do not annotate what the code does.

## Submitting changes

1. Create a feature branch from `master`.
2. Make your changes and add or update tests where relevant.
3. Run `pytest tests/` and ensure all tests pass.
4. Open a pull request against `master` with a clear description of what changed and why.
