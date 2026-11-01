## Description

What changed and why.

## Related issue

Fixes #

## Testing

How you tested it (`uv run python -m pytest`, the viewer, a ROS 2 launch, `scripts/ros_smoke.sh`).

## Checklist

- [ ] `uv run pre-commit run --all-files` passes (ruff, mypy)
- [ ] `uv run python -m pytest` passes; new code has tests (CI enforces 72 % coverage)
- [ ] `uv.lock` is updated if dependencies changed
- [ ] Docs (`docs/`) updated; `uv run mkdocs build --strict` passes
- [ ] Changelog *Unreleased* section updated
