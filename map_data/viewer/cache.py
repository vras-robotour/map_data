"""
The viewer's process-wide read caches.

Two layers, both keyed on what the cached answer was computed from rather than
on a clock: :func:`load_mapdata_cached` keeps parsed
:class:`~map_data.map_data.MapData` objects, and :func:`mapdata_geojson_cached`
keeps the serialized GeoJSON ``/api/mapdata`` hands out, so a repeat request
skips the shallow copy, the re-applied edits (``parse_intersections`` above
all) and the FeatureCollection rebuild altogether.
"""

import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from map_data.map_data import MapData

# IMPORTANT -- shared-object cache: `load_mapdata_cached` returns the *same*
# MapData instance to every caller for a given file signature, including
# concurrent Flask requests. Callers MUST NOT mutate the returned object (or
# any Way inside its lists) in place; see "MapData copy semantics" in the
# module docstring of `viewer/routes.py`.


def _file_signature(path: str) -> tuple[int, int, int]:
    """
    ``(mtime_ns, size, inode)`` -- the identity of the bytes a cached answer came from.

    The modification time alone would not do: a filesystem with a coarse
    timestamp clock can hand a quick rewrite the mtime the file already had, so
    the size and the inode ride along. An atomic rewrite (tempfile plus
    ``os.replace``, which is how a mapdata file and an annotation store are
    saved) always lands on a fresh inode, and an in-place rewrite that keeps
    both the inode and the timestamp still has to keep the byte count to slip
    through unnoticed.
    """
    st = os.stat(path)
    return st.st_mtime_ns, st.st_size, st.st_ino


@lru_cache(maxsize=3)
def _load(path: str, signature: tuple[int, int, int]) -> MapData:
    return MapData.load(path)


def load_mapdata_cached(path: str) -> MapData:
    """Load and cache a ``MapData`` file, keyed by path and file signature (3 files kept)."""
    return _load(path, _file_signature(path))


#: Merged GeoJSON documents kept, one per mapdata file like :func:`_load`.
_GEOJSON_CACHE_SIZE = 3

#: cache key -> ``(serialized FeatureCollection, ETag)``, least recently used first.
_geojson_cache: OrderedDict[tuple[Any, ...], tuple[str, str]] = OrderedDict()
_geojson_lock = threading.Lock()


def _store_digest(store: dict[str, Any]) -> str:
    """
    Digest of an annotation store's contents.

    Taken from the store as it was loaded, not from the file it came from: the
    caller has it in hand anyway, so no second read is needed, a store rewritten
    to the same edits keeps its cache entry, and an edit added, changed or
    removed by any route changes the digest the moment it is saved.
    """
    payload = json.dumps(store, sort_keys=True, default=repr).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def mapdata_geojson_cached(
    path: str,
    store: dict[str, Any],
    build: Callable[[], str],
) -> tuple[str, str]:
    """
    Return ``(serialized GeoJSON, ETag)`` for the mapdata at *path* edited by *store*.

    *build* is called only on a miss; the key is the mapdata file's
    :func:`_file_signature` plus the store's :func:`_store_digest`, which is
    exactly what the built document depends on, so it survives nothing else
    than an unchanged map and an unchanged set of edits.

    The returned text is shared between requests -- like
    :func:`load_mapdata_cached`'s ``MapData``, treat it as immutable.
    """
    key = (path, _file_signature(path), _store_digest(store))
    with _geojson_lock:
        hit = _geojson_cache.get(key)
        if hit is not None:
            _geojson_cache.move_to_end(key)
            return hit
    # Built outside the lock: two requests racing on a cold cache do the work
    # twice instead of queueing up behind each other, for the same result.
    # The ETag comes from the key, not the (potentially megabytes of) body --
    # equal keys are what "the same document" means here.
    entry = (build(), hashlib.blake2b(repr(key).encode(), digest_size=16).hexdigest())
    with _geojson_lock:
        _geojson_cache[key] = entry
        _geojson_cache.move_to_end(key)
        while len(_geojson_cache) > _GEOJSON_CACHE_SIZE:
            _geojson_cache.popitem(last=False)
    return entry
