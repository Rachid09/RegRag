"""Disk cache and 429 retry for every hosted-API call.

Every request is keyed by a SHA-256 of (namespace, payload) and stored as JSON in one SQLite
file, so a rerun of ingestion or evaluation never pays for the same call twice.
"""

import hashlib
import json
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from regrag.config import CACHE_PATH


def cache_key(namespace: str, payload: Any) -> str:
    blob = json.dumps([namespace, payload], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class DiskCache:
    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str) -> Any | None:
        row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def cached(self, namespace: str, payload: Any, compute: Callable[[], Any]) -> Any:
        """Return the cached value for (namespace, payload), or compute, store and return it."""
        key = cache_key(namespace, payload)
        hit = self.get(key)
        if hit is not None:
            return hit
        value = compute()
        self.set(key, value)
        return value


def is_rate_limited(exc: BaseException) -> bool:
    """True for HTTP 429 (LiteLLM's RateLimitError carries status_code=429)."""
    return getattr(exc, "status_code", None) == 429


def _log_backoff(state) -> None:
    wait = state.next_action.sleep if state.next_action else 0.0
    exc = state.outcome.exception() if state.outcome else None
    detail = " ".join(str(exc).split()) if exc else ""
    if "Quota exceeded for metric" in detail:  # Gemini names the exhausted quota here
        detail = detail[detail.index("Quota exceeded for metric") :]
    detail = detail[:300]
    print(
        f"  429 rate limited; retry {state.attempt_number} in {wait:.0f}s: {detail}",
        file=sys.stderr,
    )


def retry_on_429(attempts: int = 8, max_wait: float = 120.0):
    """Exponential backoff with jitter on 429s; other errors propagate immediately."""
    return retry(
        retry=retry_if_exception(is_rate_limited),
        wait=wait_random_exponential(multiplier=2, max=max_wait),
        stop=stop_after_attempt(attempts),
        reraise=True,
        before_sleep=_log_backoff,
    )
