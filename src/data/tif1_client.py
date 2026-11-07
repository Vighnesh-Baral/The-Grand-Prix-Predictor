"""Unauthenticated historical F1 timing-data access through TIF1 public archives.

TIF1 serves post-session, FastF1-derived timing files from public CDNs. It does
not sign in to Formula 1 accounts and does not provide live timing.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

SESSION_NAMES = {
    "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3",
    "Q": "Qualifying", "SQ": "Sprint Qualifying", "S": "Sprint", "R": "Race",
}


class TIF1DataUnavailable(RuntimeError):
    """Raised when a public TIF1 session archive is not available."""


def normalize_session_code(session_code: str) -> str:
    code = str(session_code).strip().upper()
    if code not in SESSION_NAMES:
        raise ValueError(f"Unsupported session code: {session_code}")
    return SESSION_NAMES[code]


def load_session(year: int, event: str | int, session_code: str) -> Any:
    """Load a historical session from TIF1's public CDN archives; never prompts."""
    try:
        import tif1
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("TIF1 is not installed. activate `.venv-tif1` and run `pip install -r requirements-tif1.txt`.") from exc

    session_name = normalize_session_code(session_code)
    try:
        # TIF1 lazily downloads the requested files from public CDNs. Accessing
        # laps materializes session data; no FastF1 authentication is used.
        session = tif1.get_session(int(year), event, session_name)
        _ = session.laps
        return session
    except Exception as exc:
        raise TIF1DataUnavailable(
            f"TIF1 archive unavailable for {year} {event} {session_name}: {type(exc).__name__}: {exc}"
        ) from exc


def session_driver_metadata(session: Any) -> pd.DataFrame:
    """Combine TIF1 driver metadata with available session result metadata."""
    frames: list[pd.DataFrame] = []
    for attr in ("drivers_df", "results"):
        try:
            value = getattr(session, attr, None)
            frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
        except Exception:
            frame = pd.DataFrame()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "Abbreviation" not in combined and "Driver" in combined:
        combined["Abbreviation"] = combined["Driver"]
    if "FullName" not in combined:
        first = combined.get("FirstName", pd.Series("", index=combined.index)).fillna("").astype(str)
        last = combined.get("LastName", pd.Series("", index=combined.index)).fillna("").astype(str)
        combined["FullName"] = (first.str.strip() + " " + last.str.strip()).str.strip()
    if "TeamName" not in combined and "Team" in combined:
        combined["TeamName"] = combined["Team"]
    if "Team" not in combined and "TeamName" in combined:
        combined["Team"] = combined["TeamName"]
    return combined


def configure_cache() -> None:
    """Initialize TIF1's local cache (TIF1 enables caching by default)."""
    try:
        import tif1
        tif1.get_cache()
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("TIF1 is not installed. activate `.venv-tif1` and run `pip install -r requirements-tif1.txt`.") from exc
