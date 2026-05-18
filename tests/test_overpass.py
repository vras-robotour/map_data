import json
from unittest.mock import MagicMock, patch

import overpy
import requests

from map_data.utils.overpass import OverpassClient

MINIMAL_OVERPASS_JSON = json.dumps({"version": 0.6, "elements": []})


def _resp(status_code, text=""):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    return r


def test_query_raw_success():
    client = OverpassClient()
    with (
        patch.object(client.session, "get", return_value=_resp(200, "")),
        patch.object(client.session, "post", return_value=_resp(200, MINIMAL_OVERPASS_JSON)),
        patch("map_data.utils.overpass.time.sleep"),
    ):
        result = client.query_raw("test query")
    assert result == MINIMAL_OVERPASS_JSON


def test_query_returns_overpy_result():
    client = OverpassClient()
    with (
        patch.object(client.session, "get", return_value=_resp(200, "")),
        patch.object(client.session, "post", return_value=_resp(200, MINIMAL_OVERPASS_JSON)),
        patch("map_data.utils.overpass.time.sleep"),
    ):
        result = client.query("test query")
    assert isinstance(result, overpy.Result)


def test_query_raw_rate_limited_rotates_endpoint():
    client = OverpassClient()
    side_effects = [_resp(429), _resp(200, MINIMAL_OVERPASS_JSON)]
    with (
        patch.object(client.session, "get", return_value=_resp(200, "")),
        patch.object(client.session, "post", side_effect=side_effects),
        patch("map_data.utils.overpass.time.sleep"),
    ):
        result = client.query_raw("test query", retries=2)
    assert result == MINIMAL_OVERPASS_JSON
    assert client._endpoint_index > 0


def test_query_raw_server_error_retries():
    client = OverpassClient()
    side_effects = [_resp(500, "error"), _resp(200, MINIMAL_OVERPASS_JSON)]
    with (
        patch.object(client.session, "get", return_value=_resp(200, "")),
        patch.object(client.session, "post", side_effect=side_effects),
        patch("map_data.utils.overpass.time.sleep"),
    ):
        result = client.query_raw("test query", retries=2)
    assert result == MINIMAL_OVERPASS_JSON


def test_query_raw_exhausts_retries():
    client = OverpassClient()
    with (
        patch.object(client.session, "get", return_value=_resp(200, "")),
        patch.object(client.session, "post", return_value=_resp(500, "error")),
        patch("map_data.utils.overpass.time.sleep"),
    ):
        result = client.query_raw("test query", retries=3)
    assert result is None


def test_query_raw_request_exception():
    client = OverpassClient()
    with (
        patch.object(client.session, "get", return_value=_resp(200, "")),
        patch.object(client.session, "post", side_effect=requests.Timeout("timeout")),
        patch("map_data.utils.overpass.time.sleep"),
    ):
        result = client.query_raw("test query", retries=3)
    assert result is None


def test_wait_for_slot_skips_non_overpass_endpoint():
    client = OverpassClient()
    with (
        patch.object(client.session, "get") as mock_get,
        patch("map_data.utils.overpass.time.sleep") as mock_sleep,
    ):
        client._wait_for_slot("https://overpass.private.coffee/api/interpreter")
    mock_get.assert_not_called()
    mock_sleep.assert_not_called()


def test_wait_for_slot_returns_immediately_if_slots_available():
    client = OverpassClient()
    status_text = "2 slots available now\n"
    with (
        patch.object(client.session, "get", return_value=_resp(200, status_text)),
        patch("map_data.utils.overpass.time.sleep") as mock_sleep,
    ):
        client._wait_for_slot("https://overpass-api.de/api/interpreter")
    mock_sleep.assert_not_called()


def test_wait_for_slot_sleeps_when_no_slots():
    client = OverpassClient()
    status_text = "0 slots available now\nin 30 seconds\n"
    with (
        patch.object(client.session, "get", return_value=_resp(200, status_text)),
        patch("map_data.utils.overpass.time.sleep") as mock_sleep,
    ):
        client._wait_for_slot("https://overpass-api.de/api/interpreter")
    mock_sleep.assert_called_once()
    assert mock_sleep.call_args[0][0] <= 32
