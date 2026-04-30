import os
import pickle

# ------------------------------------------------------------------
# Mapdata cache  (keyed by (abs_path, mtime); holds at most 3 files)
# ------------------------------------------------------------------

_mapdata_cache: dict = {}


def load_mapdata_cached(path):
    mtime = os.path.getmtime(path)
    key = (path, mtime)
    if key not in _mapdata_cache:
        # Evict stale entries for the same path
        for k in [k for k in _mapdata_cache if k[0] == path]:
            del _mapdata_cache[k]
        # Evict oldest entry if over capacity
        while len(_mapdata_cache) >= 3:
            del _mapdata_cache[next(iter(_mapdata_cache))]
        with open(path, "rb") as f:
            _mapdata_cache[key] = pickle.load(f)
    return _mapdata_cache[key]
