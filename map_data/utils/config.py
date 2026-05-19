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


def load_config(filename: str) -> dict[str, Any]:
    """
    Load a YAML configuration file from the package's config directory.

    Attempts to find the file via ROS2 resource index, falling back to
    relative path from this file.
    """
    try:
        from ament_index_python.resources import get_resource

        _, package_path = get_resource("packages", "map_data")
        config_path = Path(package_path) / "share" / "map_data" / "config" / filename
    except (ImportError, LookupError):
        # Fallback for non-ROS2 environments
        config_path = (Path(__file__).parent / ".." / ".." / "config" / filename).resolve()

    if config_path.exists():
        try:
            with config_path.open() as f:
                return yaml.safe_load(f) or {}
        except Exception:
            logger.exception("Error loading config file %s", config_path)
            return {}

    logger.debug("Config file not found: %s", config_path)
    return {}
