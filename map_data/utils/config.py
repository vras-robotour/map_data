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


def config_path(filename: str) -> Path:
    """
    Where a config file of the package would live.

    Attempts the ROS2 resource index, falling back to a path relative to this
    file (a source checkout). The file need not exist; see :func:`find_config`.
    """
    try:
        from ament_index_python.resources import get_resource

        _, package_path = get_resource("packages", "map_data")
        return Path(package_path) / "share" / "map_data" / "config" / filename
    except (ImportError, LookupError):
        # Fallback for non-ROS2 environments
        return (Path(__file__).parent / ".." / ".." / "config" / filename).resolve()


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
