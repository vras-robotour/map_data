import http
import json
import logging
import re
import time
from collections.abc import Callable

import overpy
import requests

from map_data import __version__

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# HTTP timeout for a single request. Overpass servers apply their own,
# usually much shorter, default query timeout (e.g. 25s) unless the query
# itself carries a `[timeout:N]` directive, so callers building queries
# should request a server-side timeout comfortably below this value.
REQUEST_TIMEOUT = 180


class OverpassClient:
    def __init__(self, endpoints: list[str] | None = None) -> None:
        self.endpoints = endpoints or OVERPASS_ENDPOINTS
        self._endpoint_index = 0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": f"map_data/{__version__} (research; +https://github.com/vras-robotour/map_data)"
            },
        )
        self.api = overpy.Overpass()

    def query(self, query_str: str, retries: int | None = None) -> overpy.Result | None:
        raw_text = self.query_raw(query_str, retries)
        if not raw_text:
            return None
        try:
            return self.api.parse_json(raw_text)
        except (overpy.exception.OverPyException, json.JSONDecodeError):
            logger.exception("Could not parse Overpass response")
            return None

    def query_raw(
        self,
        query_str: str,
        retries: int | None = None,
        on_attempt: Callable[[str, int, int], None] | None = None,
    ) -> str | None:
        # Give every endpoint two shots by default rather than hardcoding a
        # count that has to be kept in sync by hand as endpoints are added.
        if retries is None:
            retries = 2 * len(self.endpoints)

        for attempt in range(1, retries + 1):
            endpoint = self.endpoints[self._endpoint_index % len(self.endpoints)]
            self._wait_for_slot(endpoint)

            logger.info("Querying Overpass via %s (attempt %s/%s)", endpoint, attempt, retries)
            logger.debug("Query string: %s", query_str)
            if on_attempt is not None:
                on_attempt(endpoint, attempt, retries)
            try:
                response = self.session.post(
                    endpoint, data={"data": query_str}, timeout=REQUEST_TIMEOUT
                )
                if response.status_code == http.HTTPStatus.OK:
                    body_error = self._body_error(response.text)
                    if body_error is None:
                        return response.text
                    # Overpass reports query timeouts / memory exhaustion as
                    # HTTP 200 with a "remark" in the JSON body, and a busy
                    # mirror may answer 200 with an HTML page. Treat both
                    # exactly like a retryable server error.
                    logger.warning(
                        "Overpass error body (HTTP 200) on %s: %s",
                        endpoint,
                        body_error,
                    )
                    self._endpoint_index += 1
                    time.sleep(2 * attempt)
                elif response.status_code in (429, 406):
                    logger.warning(
                        "Rate limited (HTTP %s) on %s. Switching endpoint...",
                        response.status_code,
                        endpoint,
                    )
                    self._endpoint_index += 1
                    time.sleep(5 * attempt)  # Backoff before trying next endpoint
                else:
                    logger.warning(
                        "HTTP %s on %s. Response: %s",
                        response.status_code,
                        endpoint,
                        response.text[:200],
                    )
                    # For other errors, also try next endpoint
                    self._endpoint_index += 1
                    time.sleep(2 * attempt)

            except requests.RequestException as e:
                logger.warning("Request failed on %s: %s", endpoint, e)
                self._endpoint_index += 1
                if attempt < retries:
                    time.sleep(2 * attempt)

        return None

    @staticmethod
    def _body_error(text: str) -> str | None:
        """
        Detect an error hidden in an HTTP 200 body.

        Returns a short description of the problem (an Overpass ``remark``
        such as ``runtime error: Query timed out ...``, or a non-JSON body),
        or ``None`` if the body looks like a valid result.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return f"non-JSON body: {text[:200]!r}"
        remark = data.get("remark") if isinstance(data, dict) else None
        if remark:
            return str(remark)[:200]
        return None

    def _wait_for_slot(self, endpoint: str, max_wait: int = 300) -> None:
        if "overpass-api.de" not in endpoint:
            return
        status_url = endpoint.replace("/api/interpreter", "/api/status")
        try:
            resp = self.session.get(status_url, timeout=10)
            if resp.status_code == http.HTTPStatus.OK:
                text = resp.text
                if "slots available now" in text:
                    m = re.search(r"(\d+) slots available now", text)
                    if m and int(m.group(1)) > 0:
                        return
                    m_wait = re.search(r"in (\d+) seconds", text)
                    wait_secs = min(int(m_wait.group(1)) + 2 if m_wait else 60, max_wait)
                    logger.info("Overpass busy, waiting %ss...", wait_secs)
                    time.sleep(wait_secs)
                elif "Connected as:" in text and "Rate limit:" in text:
                    # Alternative status format sometimes seen
                    if "Available slots: 0" in text:
                        logger.info("Overpass busy (0 slots), waiting 15s...")
                        time.sleep(15)
        except requests.RequestException as e:
            logger.debug("Could not check Overpass status at %s: %s", status_url, e)
