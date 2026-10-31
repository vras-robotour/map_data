"""Locate package data (ament share or source tree) and load YAML configs."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def package_share(subdir: str) -> Path:
    """The installed ``share/map_data/<subdir>``, or the source tree's ``<subdir>``."""
    try:
        from ament_index_python.resources import get_resource

        _, prefix = get_resource("packages", "map_data")
        return Path(prefix) / "share" / "map_data" / subdir
    except (ImportError, LookupError):
        return (Path(__file__).parent / ".." / ".." / subdir).resolve()


def resolve_data_file(name: str, data_dir: str | Path | None = None) -> Path | None:
    """
    Find the data file *name*, or ``None`` when there is none.

    *name* as a path (absolute, or relative to the cwd), else in *data_dir*
    (default: the package ``data/``), else in the package ``demo/`` (a source
    checkout; an install copies the demo into ``data/``).
    """
    p = Path(name).expanduser()
    if p.is_absolute() or p.is_file():
        return p.resolve() if p.is_file() else None
    base = Path(data_dir).expanduser() if data_dir else package_share("data")
    return next((c for c in (base / name, package_share("demo") / name) if c.is_file()), None)


def resolve_client_data_file(name: str, data_dir: str | Path | None = None) -> Path | None:
    """
    :func:`resolve_data_file` for a *name* from a remote client, which may not leave the
    data directories: only a relative path without ``..`` that resolves inside *data_dir*
    (default: the package ``data/``) or the package ``demo/``.
    """
    if Path(name).is_absolute() or ".." in Path(name).parts:
        return None
    base = Path(data_dir).expanduser() if data_dir else package_share("data")
    for d in (base.resolve(), package_share("demo").resolve()):
        p = (d / name).resolve()
        if p.is_relative_to(d) and p.is_file():
            return p
    return None


def config_path(filename: str) -> Path:
    """Where a config file of the package would live (need not exist)."""
    return package_share("config") / filename


def load_config(filename: str) -> dict[str, Any]:
    """Load ``config/<filename>``; ``{}`` when it is missing or unreadable."""
    config_path_ = config_path(filename)

    if config_path_.exists():
        try:
            with config_path_.open() as f:
                return yaml.safe_load(f) or {}
        except Exception:
            logger.exception("Error loading config file %s", config_path_)
            return {}

    logger.debug("Config file not found: %s", config_path_)
    return {}
