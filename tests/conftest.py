"""Shared fixtures and canned data for the test suite."""

import json
from unittest.mock import MagicMock, patch

import overpy
import pytest

# One footway way between two nodes — the canonical mocked Overpass response.
# The nodes sit just inside a 50.000-50.001 / 14.000-14.001 bbox so the same
# payload works for both MapData integration tests and the viewer's
# fetch_area tests (whose bbox is that tight).
FOOTWAY_WAYS_JSON = json.dumps(
    {
        "version": 0.6,
        "elements": [
            {"type": "node", "id": 1, "lat": 50.0005, "lon": 14.0005},
            {"type": "node", "id": 2, "lat": 50.0006, "lon": 14.0006},
            {"type": "way", "id": 101, "nodes": [1, 2], "tags": {"highway": "footway"}},
        ],
    },
)

EMPTY_OSM_JSON = json.dumps({"version": 0.6, "elements": []})


@pytest.fixture
def mock_overpass_client():
    """
    Patch ``map_data.map_data.OverpassClient`` with a canned footway response.

    Yields the mock instance so tests can override ``query_raw.return_value``
    or inspect calls. ``api`` is a real ``overpy.Overpass`` so ``parse_json``
    behaves exactly as in production.
    """
    with patch("map_data.map_data.OverpassClient") as mock_client:
        instance = MagicMock()
        instance.query_raw.return_value = FOOTWAY_WAYS_JSON
        instance.api = overpy.Overpass()
        mock_client.return_value = instance
        yield instance
