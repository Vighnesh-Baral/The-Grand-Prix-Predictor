"""Small, rate-limited OpenF1 historical API client (free historical tier)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from src.config import CACHE_DIR


class OpenF1APIError(RuntimeError):
    """Raised when OpenF1 cannot provide a usable response."""


class OpenF1Client:
    BASE_URL = "https://api.openf1.org/v1"

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        cache_dir: Path | str = CACHE_DIR / "openf1_http",
        min_interval_seconds: float = 2.05,
        timeout_seconds: float = 45.0,
        sleep=time.sleep,
    ) -> None:
        self.session = session or requests.Session()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.timeout_seconds = float(timeout_seconds)
        self._sleep = sleep
        self._last_request = 0.0

    def get(self, endpoint: str, params: dict[str, Any] | None = None, *, refresh: bool = False) -> list[dict[str, Any]]:
        endpoint = endpoint.strip("/")
        if not endpoint or "/" in endpoint:
            raise ValueError("endpoint must be a single OpenF1 endpoint name")
        query = sorted((str(k), str(v)) for k, v in (params or {}).items() if v is not None)
        cache_key = hashlib.sha256(json.dumps([endpoint, query], separators=(",", ":")).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists() and not refresh:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        url = f"{self.BASE_URL}/{endpoint}"
        last_error: Exception | None = None
        for attempt in range(5):
            wait = self.min_interval_seconds - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            try:
                response = self.session.get(url, params=params or {}, timeout=self.timeout_seconds)
                self._last_request = time.monotonic()
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    backoff = float(retry_after) if retry_after else min(60.0, 2.0 ** (attempt + 2))
                    self._sleep(backoff)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise OpenF1APIError(f"OpenF1 /{endpoint} returned {type(payload).__name__}, expected a JSON array")
                cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return payload
            except (requests.RequestException, ValueError, OpenF1APIError) as exc:
                last_error = exc
                if attempt < 4:
                    self._sleep(min(30.0, 2.0 ** attempt))
        raise OpenF1APIError(f"OpenF1 request failed for /{endpoint}: {last_error}") from last_error

    def sessions(self, year: int, *, refresh: bool = False) -> list[dict[str, Any]]:
        return self.get("sessions", {"year": int(year)}, refresh=refresh)

    def session_endpoint(self, endpoint: str, session_key: int | str, *, refresh: bool = False) -> list[dict[str, Any]]:
        return self.get(endpoint, {"session_key": session_key}, refresh=refresh)
