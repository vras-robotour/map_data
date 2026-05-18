import logging
import os
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
        config_path = os.path.join(package_path, "share", "map_data", "config", filename)
    except (ImportError, LookupError):
        # Fallback for non-ROS2 environments
        config_path = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "config", filename)
        )

    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            return {}

    logger.debug(f"Config file not found: {config_path}")
    return {}
