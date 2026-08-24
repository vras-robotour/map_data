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

### The lockfile

`uv.lock` is committed, pinning an exact, fully resolved set of 78 packages so a build can
be reproduced later. Note that the command above — `uv pip install`, the pip-compatible
interface — resolves from `pyproject.toml` and **ignores the lockfile**. To install the
locked versions exactly, use `uv sync` instead:

```bash
uv sync --extra dev
```

!!! warning "Regenerate with an explicit `--python 3.12`"
    `pyproject.toml` intentionally omits `requires-python` (see the comment there: colcon's
    `ament_python` build chokes on the `SpecifierSet` it produces). Without it, `uv` infers
    the floor from whichever interpreter you happen to run it with — so a plain `uv lock` on
    a 3.13 machine silently rewrites the floor to `>=3.13` and locks out 3.12 contributors.
    Always regenerate with:

    ```bash
    uv lock --python 3.12
    ```

    For the same reason, `uv sync --locked` fails on 3.13 against the `>=3.12` lockfile.
    Drop `--locked` (or use `uv pip install -e ".[dev]"`) on newer interpreters. CI does not
    consume the lockfile at all — it installs with plain `pip` across a 3.12/3.13 matrix.

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

### The static viewer demo

The [live demo](https://vras-robotour.github.io/map_data/demo/) linked from the site nav is
a read-only scrape of the real viewer, not a separate implementation. `build_static_demo.sh`
boots `map_data_viewer` against the committed `demo/` dataset, dumps the rendered page and
every JSON endpoint the frontend needs, and rewrites the page to run in static mode:

```bash
bash scripts/build_static_demo.sh    # writes _demo_out/
```

`PORT` (default `5017`) and `OUT` (default `_demo_out`) can be overridden via the
environment. The `Demo` workflow runs this and publishes `_demo_out/` to `gh-pages` whenever
`map_data/viewer/**`, `demo/**`, or the script itself changes — so **frontend changes that
alter which endpoints the page calls need the script updated too**, or the demo will 404 on
the missing JSON.

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

## Type checking

The project uses [mypy](https://mypy-lang.org/) for static type checking. Configuration lives in `pyproject.toml` under `[tool.mypy]`.

```bash
mypy map_data/
```

Third-party libraries without complete type stubs (numpy, shapely, overpy, flask_socketio, rclpy, etc.) are covered by `ignore_missing_imports`. A small number of remaining false positives — mostly from numpy/shapely returning loosely-typed values, or from patterns mypy can't narrow (e.g. Flask's `nonlocal`-captured variables) — are silenced with targeted `# type: ignore[<code>]` comments that include a short reason. New code should be fully typed rather than relying on ignores.

## Submitting changes

1. Create a feature branch from `master`.
2. Make your changes and add or update tests where relevant.
3. Run `pytest tests/` and ensure all tests pass.
4. Run `ruff check . && ruff format --check .` and `mypy map_data/` and ensure both are clean.
5. Open a pull request against `master` with a clear description of what changed and why.
