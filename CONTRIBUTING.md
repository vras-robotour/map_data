# Contributing to map_data

Thanks for your interest in contributing!

The full contributor guide — development environment setup, code style, type checking,
running the test suite, and the pull request checklist — lives in the documentation:

**[Contributing guide](https://vras-robotour.github.io/map_data/dev/contributing/)**
&nbsp;·&nbsp;
[Testing guide](https://vras-robotour.github.io/map_data/dev/testing/)
&nbsp;·&nbsp;
[Architecture overview](https://vras-robotour.github.io/map_data/dev/architecture/)

The source for those pages is in [`docs/dev/`](docs/dev/), so you can also read them
offline or with `mkdocs serve`.

## The short version

```bash
git clone https://github.com/vras-robotour/map_data.git
cd map_data
uv pip install -e ".[dev]"
```

Before opening a pull request against `master`:

```bash
ruff check . && ruff format --check .
mypy map_data/
pytest
```

## Reporting issues

- **Bugs** — open a [bug report](https://github.com/vras-robotour/map_data/issues/new?template=bug_report.md)
- **Ideas and improvements** — open a [feature request](https://github.com/vras-robotour/map_data/issues/new?template=feature_request.md)
- **Security vulnerabilities** — do *not* open a public issue; follow [SECURITY.md](SECURITY.md)

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating you
are expected to uphold it.

## License

By contributing, you agree that your contributions will be licensed under the
[BSD 3-Clause License](LICENSE) that covers this project.
