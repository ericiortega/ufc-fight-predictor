"""Polite HTTP client: per-host delays with jitter, retries, and loud failures."""

from __future__ import annotations

import logging
import random
import time
from urllib.parse import urlparse

import requests

from ufc_model.config import HttpConfig

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_CHALLENGE_MARKERS = ("Checking your browser", "This site requires JavaScript")


class FetchError(RuntimeError):
    """A request failed after all retries, or returned a non-retryable error."""


class BlockedError(FetchError):
    """The server returned a bot challenge instead of content. Never bypassed."""


class PoliteClient:
    """``requests`` wrapper that spaces out requests to each host.

    Failures raise instead of returning partial data, so a scheduled job that
    gets blocked fails loudly rather than silently skipping work.
    """

    def __init__(self, cfg: HttpConfig, user_agent: str, seed: int | None = None) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._last_request: dict[str, float] = {}
        self._rng = random.Random(seed)

    def _wait_for_host(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            delay = self._rng.uniform(self.cfg.min_delay_s, self.cfg.max_delay_s)
            remaining = delay - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request[host] = time.monotonic()

    def get(
        self, url: str, params: dict | None = None, headers: dict | None = None
    ) -> requests.Response:
        """GET ``url``, retrying transient errors with exponential backoff."""
        host = urlparse(url).netloc
        for attempt in range(1, self.cfg.max_retries + 1):
            self._wait_for_host(host)
            try:
                resp = self.session.get(
                    url, params=params, headers=headers, timeout=self.cfg.timeout_s
                )
            except requests.RequestException as exc:
                log.warning("GET %s failed (attempt %d): %s", url, attempt, exc)
                if attempt == self.cfg.max_retries:
                    raise FetchError(f"GET {url} failed: {exc}") from exc
            else:
                if resp.status_code == 200:
                    if any(m in resp.text[:2000] for m in _CHALLENGE_MARKERS):
                        raise BlockedError(f"Bot challenge served by {host}; not bypassing.")
                    return resp
                if resp.status_code not in _RETRYABLE_STATUS or attempt == self.cfg.max_retries:
                    raise FetchError(f"GET {url} returned HTTP {resp.status_code}")
                log.warning("GET %s -> HTTP %d (attempt %d)", url, resp.status_code, attempt)
            time.sleep(2**attempt)
        raise FetchError(f"GET {url} exhausted retries")  # pragma: no cover
