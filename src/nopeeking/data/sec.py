"""Polite SEC EDGAR client: declared User-Agent, global rate limit (< 10 req/s), retries, gzip disk cache.

SEC fair-access policy: https://www.sec.gov/os/accessing-edgar-data
Set ``NOPEEKING_SEC_USER_AGENT`` to "<name or org> <contact email>" for large or recurring runs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "nopeeking-research-client/0.1"
ENV_KEY = "NOPEEKING_SEC_USER_AGENT"


def _from_env_files() -> str | None:
    """Look for ``NOPEEKING_SEC_USER_AGENT=...`` in ./.env, then ~/.nopeeking.env (never committed)."""
    for path in (Path(".env"), Path.home() / ".nopeeking.env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == ENV_KEY and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def user_agent() -> str:
    """SEC asks automated clients to declare "<name> <contact email>"; www.sec.gov returns 403 otherwise."""
    return os.environ.get(ENV_KEY) or _from_env_files() or DEFAULT_USER_AGENT


class RateLimiter:
    """Thread-safe spacing of calls: at most ``max_per_second`` calls start per second."""

    def __init__(self, max_per_second: float) -> None:
        if max_per_second <= 0 or max_per_second > 10:
            raise ValueError("SEC allows at most 10 requests/second")
        self.interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)


class SecClient:
    def __init__(
        self,
        cache_dir: str | Path,
        max_per_second: float = 8.0,
        timeout: float = 30.0,
        retries: int = 5,
    ) -> None:
        self.cache_dir = Path(cache_dir) / "sec"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"})
        self.limiter = RateLimiter(max_per_second)
        self.timeout = timeout
        self.retries = retries
        self.n_requests = 0
        self._count_lock = threading.Lock()

    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.gz"

    def get(self, url: str, use_cache: bool = True, max_age_days: float | None = None) -> bytes:
        path = self._cache_path(url)
        if use_cache and path.exists():
            age_days = (time.time() - path.stat().st_mtime) / 86400
            if max_age_days is None or age_days <= max_age_days:
                return gzip.decompress(path.read_bytes())
        content = self._fetch(url)
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(content))
            tmp.replace(path)
        return content

    def get_json(self, url: str, use_cache: bool = True, max_age_days: float | None = None) -> Any:
        return json.loads(self.get(url, use_cache=use_cache, max_age_days=max_age_days))

    def _fetch(self, url: str) -> bytes:
        last_err: str = ""
        for attempt in range(self.retries):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as err:
                last_err = repr(err)
                time.sleep(min(30, 2**attempt))
                continue
            with self._count_lock:
                self.n_requests += 1
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                raise FileNotFoundError(url)
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}"
                wait = min(120, 5 * 2**attempt)
                log.warning("SEC %s on %s, backing off %ss", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"SEC request failed after {self.retries} attempts ({last_err}): {url}")
