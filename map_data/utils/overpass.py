import http
import json
import logging
import re
import time
from collections.abc import Callable
from urllib import error, parse, request

import overpy

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
USER_AGENT = f"map_data/{__version__} (research; +https://github.com/vras-robotour/map_data)"


class OverpassClient:
    def __init__(self, endpoints: list[str] | None = None) -> None:
        self.endpoints = endpoints or OVERPASS_ENDPOINTS
        self._endpoint_index = 0
        self.api = overpy.Overpass()

    @staticmethod
    def _http(
        url: str, data: bytes | None = None, timeout: float = REQUEST_TIMEOUT
    ) -> tuple[int, str]:
        """``(status, body)`` of one GET (``data=None``) or POST; error statuses are returned."""
        req = request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

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
        body = parse.urlencode({"data": query_str}).encode()

        for attempt in range(1, retries + 1):
            endpoint = self.endpoints[self._endpoint_index % len(self.endpoints)]
            self._wait_for_slot(endpoint)

            logger.info("Querying Overpass via %s (attempt %s/%s)", endpoint, attempt, retries)
            logger.debug("Query string: %s", query_str)
            if on_attempt is not None:
                on_attempt(endpoint, attempt, retries)
            try:
                status, text = self._http(endpoint, body)
                if status == http.HTTPStatus.OK:
                    # Overpass reports query timeouts / memory exhaustion as
                    # HTTP 200 with a "remark" in the JSON body, and a busy
                    # mirror may answer 200 with an HTML page.
                    problem = self._body_error(text)
                    if problem is None:
                        return text
                else:
                    problem = f"HTTP {status}: {text[:200]}"
            except OSError as e:  # URLError, timeouts, connection resets
                problem = str(e)
            logger.warning("Overpass request failed on %s: %s", endpoint, problem)
            self._endpoint_index += 1  # try the next endpoint
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
            status, text = self._http(status_url, timeout=10)
            if status == http.HTTPStatus.OK and "slots available now" in text:
                m = re.search(r"(\d+) slots available now", text)
                if m and int(m.group(1)) > 0:
                    return
                m_wait = re.search(r"in (\d+) seconds", text)
                wait_secs = min(int(m_wait.group(1)) + 2 if m_wait else 60, max_wait)
                logger.info("Overpass busy, waiting %ss...", wait_secs)
                time.sleep(wait_secs)
        except OSError as e:
            logger.debug("Could not check Overpass status at %s: %s", status_url, e)
