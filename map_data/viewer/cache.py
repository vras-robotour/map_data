from pathlib import Path

from map_data.map_data import MapData

# ------------------------------------------------------------------
# Mapdata cache  (keyed by (abs_path, mtime); holds at most 3 files)
# ------------------------------------------------------------------
#
# IMPORTANT -- shared-object cache: `load_mapdata_cached` returns the *same*
# MapData instance to every caller for a given (path, mtime), including
# concurrent Flask requests. Callers in `viewer/routes.py` MUST NOT mutate
# the returned object (or any Way inside its roads_list/footways_list/
# barriers_list) in place -- that would corrupt the cache for every other
# request touching this file. Instead they `copy.copy` the MapData (or an
# individual Way) before applying edits, and every edit helper in
# `viewer/helpers.py` returns a fresh copy rather than mutating its
# argument. See the module docstring in `viewer/routes.py` ("MapData copy
# semantics") for the full write-up of which copy strategy is used where
# and why.

_mapdata_cache: dict = {}
CACHE_CAPACITY = 3


def load_mapdata_cached(path: str) -> MapData:
    """
    Load and cache a ``MapData`` file, keyed by path and modification time.

    Returns a shared, process-wide cached instance -- callers must not
    mutate it in place (see the module-level cache note above). At most
    :data:`CACHE_CAPACITY` files are kept; the oldest entry is evicted
    (insertion order, not LRU) when a new file would exceed that. Any
    stale entries for the same *path* (i.e. an older ``mtime``, meaning the
    file changed on disk since it was cached) are evicted first, so an
    edited file is always re-read rather than served from a stale cache.

    Parameters
    ----------
    path : str
        Filesystem path to a ``.mapdata`` file.

    Returns
    -------
    MapData
        The cached (or freshly loaded) ``MapData`` instance.

    """
    p = Path(path)
    mtime = p.stat().st_mtime
    key = (path, mtime)
    if key not in _mapdata_cache:
        # Evict stale entries for the same path
        for k in [k for k in _mapdata_cache if k[0] == path]:
            del _mapdata_cache[k]
        # Evict oldest entry if over capacity
        while len(_mapdata_cache) >= CACHE_CAPACITY:
            del _mapdata_cache[next(iter(_mapdata_cache))]

        _mapdata_cache[key] = MapData.load(path)
    return _mapdata_cache[key]
