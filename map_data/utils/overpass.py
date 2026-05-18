import logging
import re
import time

import overpy
import requests

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


class OverpassClient:
    def __init__(self, endpoints: list | None = None):
        self.endpoints = endpoints or OVERPASS_ENDPOINTS
        self._endpoint_index = 0
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "map_data/2.0 (https://github.com/vras-robotour/map_data)"}
        )
        self.api = overpy.Overpass()

    def query(self, query_str: str, retries: int = 3) -> overpy.Result | None:
        raw_text = self.query_raw(query_str, retries)
        if raw_text:
            return self.api.parse_json(raw_text)
        return None

    def query_raw(self, query_str: str, retries: int = 3) -> str | None:
        for attempt in range(1, retries + 1):
            endpoint = self.endpoints[self._endpoint_index % len(self.endpoints)]
            self._wait_for_slot(endpoint)

            logger.info(f"Querying Overpass via {endpoint} (attempt {attempt}/{retries})")
            logger.debug(f"Query string: {query_str}")
            try:
                response = self.session.post(endpoint, data={"data": query_str}, timeout=180)
                if response.status_code == 200:
                    return response.text

                if response.status_code in (429, 406):
                    logger.warning(
                        f"Rate limited (HTTP {response.status_code}) on {endpoint}. Switching endpoint..."
                    )
                    self._endpoint_index += 1
                    time.sleep(5 * attempt)  # Backoff before trying next endpoint
                else:
                    logger.warning(
                        f"HTTP {response.status_code} on {endpoint}. Response: {response.text[:200]}"
                    )
                    # For other errors, also try next endpoint
                    self._endpoint_index += 1
                    time.sleep(2 * attempt)

            except requests.RequestException as e:
                logger.warning(f"Request failed on {endpoint}: {e}")
                self._endpoint_index += 1
                if attempt < retries:
                    time.sleep(2 * attempt)

        return None

    def _wait_for_slot(self, endpoint: str, max_wait: int = 300) -> None:
        if "overpass-api.de" not in endpoint:
            return
        status_url = endpoint.replace("/api/interpreter", "/api/status")
        try:
            resp = self.session.get(status_url, timeout=10)
            if resp.status_code == 200:
                text = resp.text
                if "slots available now" in text:
                    m = re.search(r"(\d+) slots available now", text)
                    if m and int(m.group(1)) > 0:
                        return
                    m_wait = re.search(r"in (\d+) seconds", text)
                    wait_secs = min(int(m_wait.group(1)) + 2 if m_wait else 60, max_wait)
                    logger.info(f"Overpass busy, waiting {wait_secs}s...")
                    time.sleep(wait_secs)
                elif "Connected as:" in text and "Rate limit:" in text:
                    # Alternative status format sometimes seen
                    if "Available slots: 0" in text:
                        logger.info("Overpass busy (0 slots), waiting 15s...")
                        time.sleep(15)
        except Exception as e:
            logger.debug(f"Could not check Overpass status at {status_url}: {e}")
