from pathlib import Path

from map_data.map_data import MapData

# ------------------------------------------------------------------
# Mapdata cache  (keyed by (abs_path, mtime); holds at most 3 files)
# ------------------------------------------------------------------

_mapdata_cache: dict = {}
CACHE_CAPACITY = 3


def load_mapdata_cached(path: str) -> MapData:
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
