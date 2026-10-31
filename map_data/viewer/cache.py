"""
The viewer's process-wide read caches, keyed on what each answer was computed from
(file signature, annotation-store digest), never on a clock.

:func:`load_mapdata_cached` keeps parsed ``MapData``, :func:`mapdata_geojson_cached` the
GeoJSON ``/api/mapdata`` serves, :func:`merged_mapdata_cached` the edited ``MapData``
that export and planning use.
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

# Shared objects: every cache here hands the *same* instance to all callers, concurrent
# requests included. Never mutate a returned object (or any Way inside it) in place.


def _file_signature(path: str) -> tuple[int, int, int]:
    """
    ``(mtime_ns, size, inode)``: the identity of the bytes a cached answer came from.

    mtime alone is not enough on a coarse-clock filesystem. Atomic saves (tempfile plus
    ``os.replace``) get a fresh inode; an in-place rewrite would also have to keep the
    size to go unnoticed.
    """
    st = os.stat(path)
    return st.st_mtime_ns, st.st_size, st.st_ino


@lru_cache(maxsize=3)
def _load(path: str, signature: tuple[int, int, int]) -> MapData:
    return MapData.load(path)


def load_mapdata_cached(path: str) -> MapData:
    """Load and cache a ``MapData`` file, keyed by path and file signature (3 files kept)."""
    return _load(path, _file_signature(path))


#: Entries kept per cache below, one per mapdata file like :func:`_load`.
_CACHE_SIZE = 3

#: cache key -> ``(serialized FeatureCollection, ETag)``, least recently used first.
_geojson_cache: OrderedDict[tuple[Any, ...], tuple[str, str]] = OrderedDict()
#: cache key -> fully edited ``MapData``, least recently used first.
_merged_cache: OrderedDict[tuple[Any, ...], MapData] = OrderedDict()
_lock = threading.Lock()


def _store_digest(store: dict[str, Any]) -> str:
    """
    Digest of an annotation store's loaded contents.

    Taken from the store, not its file: no second read, and a rewrite with the same
    edits keeps the cache entry.
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

    *build* runs only on a miss. Keyed on the file signature and the store digest.
    """
    key = _edited_key(path, store)
    # ETag from the key, not the (megabytes of) body: equal keys mean the same document.
    return _lru(
        _geojson_cache,
        key,
        lambda: (build(), hashlib.blake2b(repr(key).encode(), digest_size=16).hexdigest()),
    )


def merged_mapdata_cached(
    path: str, store: dict[str, Any], build: Callable[[], MapData]
) -> MapData:
    """Return the fully edited ``MapData`` for *path* under *store*, built by *build* on a miss."""
    return _lru(_merged_cache, _edited_key(path, store), build)


def _edited_key(path: str, store: dict[str, Any]) -> tuple[Any, ...]:
    """What a document built from the mapdata at *path* edited by *store* depends on."""
    return (path, _file_signature(path), _store_digest(store))


def _lru[V](
    cache: OrderedDict[tuple[Any, ...], V], key: tuple[Any, ...], build: Callable[[], V]
) -> V:
    """Look *key* up in *cache*, building the value on a miss (LRU of :data:`_CACHE_SIZE`)."""
    with _lock:
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return hit
    # Built outside the lock: racing requests on a cold cache may both build (same
    # result) rather than queue behind each other.
    value = build()
    with _lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_SIZE:
            cache.popitem(last=False)
    return value
