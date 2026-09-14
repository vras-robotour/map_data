"""
Small helpers shared by the package's launch files.

Launch arguments are strings, node parameters are typed, and a config file name may be
either an absolute path or a name in the package's ``config/`` directory. These live in
the package (not in ``launch/``) so they can be unit-tested without a running launch.
"""

from __future__ import annotations

import re
from pathlib import Path

from map_data.utils.config import package_share


def resolve_config_file(name: str, package: str = "map_data") -> str:
    """An absolute path is taken as is; a bare name is looked up in ``config/``."""
    path = Path(name).expanduser()
    if not path.is_absolute():
        in_package = package_share("config", package) / name
        if in_package.exists():
            return str(in_package)
    return str(path)


def way_types(value: str) -> list[str]:
    """``"footway,road"`` or ``"footway road"`` -> ``["footway", "road"]``."""
    return [w for w in re.split(r"[,\s]+", value) if w]


def flag(value: str) -> bool:
    """Launch-argument spelling of a boolean."""
    return value.strip().lower() in ("1", "true", "yes", "on")
