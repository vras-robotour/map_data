# Contributing

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (preferred) or plain `pip`
- ROS2 Jazzy or later — **optional**; only needed for the `osm_cloud` node

## Getting the code

```bash
git clone https://github.com/vras-robotour/map_data.git
cd map_data
```

## Setting up the development environment

Install the package in editable mode together with the documentation and tooling dependencies:

```bash
uv pip install -e ".[dev]"
```

!!! warning "Use `uv pip`, not plain `pip`"
    The project uses `uv` for package management. Plain `pip` may install into the wrong interpreter on machines with multiple Python environments.

## Running the tests

```bash
pytest tests/ -v
```

All tests are standalone — no ROS2 context or network access is required. See the [Testing](testing.md) page for a full breakdown of test files, design principles, and guidance on adding new tests.

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

The project uses [Ruff](https://docs.astral.sh/ruff/) for both formatting and linting. Configuration lives in `pyproject.toml`.

Before submitting a pull request, run:

```bash
ruff check --fix .
ruff format .
```

Key style rules enforced:

- **Line length** — 100 characters.
- **Import order** — isort-compatible, `map_data` is treated as first-party.
- **Type annotations** — modern PEP 585/604 style (`list[str]`, `X | None`). Add return annotations to all new functions; parameter annotations are required for public API methods.
- **Docstrings** — public methods use NumPy-style docstrings (matches the `mkdocstrings` configuration). Internal helpers can omit docstrings when the name is self-explanatory.
- **Comments** — only add a comment when the *why* is non-obvious. Do not annotate what the code does.

!!! tip "Editor integration"
    If you use Neovim with LazyVim, enable the `lazyvim.plugins.extras.lang.python` extra and add a `conform.nvim` plugin spec with `ruff_format` to get format-on-save automatically.

## Submitting changes

1. Create a feature branch from `master`.
2. Make your changes and add or update tests where relevant.
3. Run `pytest tests/` and ensure all tests pass.
4. Open a pull request against `master` with a clear description of what changed and why.
