"""
Configuration loading utilities for map_data.

This module provides helpers for loading YAML configuration files from
the package's config directory, with ROS2 integration.
"""

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


def package_share(subdir: str, package: str = "map_data") -> Path:
    """The installed ``share/<package>/<subdir>``, or the source tree's ``<subdir>``."""
    try:
        from ament_index_python.resources import get_resource

        _, prefix = get_resource("packages", package)
        return Path(prefix) / "share" / package / subdir
    except (ImportError, LookupError):
        return (Path(__file__).parent / ".." / ".." / subdir).resolve()


def config_path(filename: str) -> Path:
    """Where a config file of the package would live (need not exist; see :func:`find_config`)."""
    return package_share("config") / filename


def find_config(filename: str) -> Path | None:
    """The package's config file of that name, or ``None`` if it is not installed."""
    path = config_path(filename)
    return path if path.is_file() else None


def load_config(filename: str) -> dict[str, Any]:
    """
    Load a YAML configuration file from the package's config directory.

    Attempts to find the file via ROS2 resource index, falling back to
    relative path from this file.
    """
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
