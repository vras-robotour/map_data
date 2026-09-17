import os
from functools import lru_cache

from map_data.map_data import MapData

# IMPORTANT -- shared-object cache: `load_mapdata_cached` returns the *same*
# MapData instance to every caller for a given (path, mtime), including
# concurrent Flask requests. Callers MUST NOT mutate the returned object (or
# any Way inside its lists) in place; see "MapData copy semantics" in the
# module docstring of `viewer/routes.py`.


@lru_cache(maxsize=3)
def _load(path: str, mtime: float) -> MapData:
    return MapData.load(path)


def load_mapdata_cached(path: str) -> MapData:
    """Load and cache a ``MapData`` file, keyed by path and modification time (3 files kept)."""
    return _load(path, os.stat(path).st_mtime)
