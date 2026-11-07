"""Historical session helpers backed by TIF1's no-login public archives.

The module name is retained for compatibility with existing project commands.
All completed-session collection here uses TIF1 rather than direct FastF1 3.9+
F1TV-authenticated API access. TIF1 serves post-session FastF1-derived data from
public CDNs and does not provide live timing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
from src.data.tif1_client import configure_cache, load_session as load_tif1_session


def configure_fastf1_cache() -> None:
    """Backward-compatible alias; initializes TIF1's local cache without login."""
    configure_cache()


def load_session(year: int, event: str | int, session_code: str, load_telemetry: bool = False) -> Any:
    """Load a completed session from TIF1's public archives, without credentials."""
    del load_telemetry  # accepted for compatibility; session.laps is loaded lazily
    return load_tif1_session(year, event, session_code)


def _timedelta_seconds(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        if hasattr(value, "total_seconds"):
            return float(value.total_seconds())
        return float(pd.to_timedelta(value).total_seconds())
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def extract_session_laps(year: int, round_number: int, session_code: str) -> pd.DataFrame:
    """Return a normalized one-row-per-driver-lap table for one completed session."""
    session = load_session(year, round_number, session_code, load_telemetry=False)
    laps = session.laps.copy()
    if laps.empty:
        return pd.DataFrame()
    mappings = {
        "Driver": "driver_code", "Team": "constructor_name", "LapNumber": "lap_number",
        "Stint": "stint", "LapTime": "lap_time", "Compound": "compound",
        "TyreLife": "tyre_life", "FreshTyre": "fresh_tyre", "TrackStatus": "track_status",
        "IsAccurate": "is_accurate", "Deleted": "deleted", "DeletedReason": "deleted_reason",
        "FastF1Generated": "fastf1_generated", "PitInTime": "pit_in", "PitOutTime": "pit_out",
        "Sector1Time": "sector1_time", "Sector2Time": "sector2_time", "Sector3Time": "sector3_time",
        "SpeedI1": "speed_i1", "SpeedI2": "speed_i2", "SpeedFL": "speed_fl", "SpeedST": "speed_st",
    }
    frame = pd.DataFrame(index=laps.index)
    for source, target in mappings.items():
        frame[target] = laps[source] if source in laps.columns else None
    frame["season"] = int(year)
    frame["round"] = int(round_number)
    frame["session"] = str(session_code).upper()
    frame["data_source"] = "tif1_public_archive"
    frame["lap_time_seconds"] = frame["lap_time"].map(_timedelta_seconds)
    for col in ("lap_time", "pit_in", "pit_out", "sector1_time", "sector2_time", "sector3_time"):
        frame[col] = frame[col].map(lambda value: str(value) if value is not None and not pd.isna(value) else None)
    return frame.reset_index(drop=True)


def _merge_lap_rows(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    keys = ["season", "round", "session", "driver_code", "lap_number"]
    if set(keys).issubset(combined.columns):
        combined = combined.drop_duplicates(keys, keep="last")
    return combined.sort_values(keys, na_position="last").reset_index(drop=True)


def collect_fastf1_sessions(
    season: int, round_number: int, sessions: list[str] | None = None,
    destination: Path | str = PROCESSED_DIR / "session_laps.csv",
) -> pd.DataFrame:
    """Download completed sessions from public TIF1 archives; never requests credentials."""
    sessions = [str(code).upper() for code in (sessions or ["FP1", "FP2", "FP3"])]
    valid = {"FP1", "FP2", "FP3", "Q", "SQ", "S", "R"}
    invalid = sorted(set(sessions) - valid)
    if invalid:
        raise ValueError(f"Unsupported session codes: {', '.join(invalid)}")
    path = Path(destination)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    collected: list[pd.DataFrame] = []
    errors: list[str] = []
    for code in sessions:
        try:
            data = extract_session_laps(season, round_number, code)
            if not data.empty:
                collected.append(data)
        except Exception as exc:
            errors.append(f"{code}: {exc}")
    new = pd.concat(collected, ignore_index=True, sort=False) if collected else pd.DataFrame()
    if new.empty and existing.empty:
        if errors:
            raise RuntimeError("No TIF1 lap data collected. " + " | ".join(errors))
        return new
    merged = _merge_lap_rows(existing, new)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    merged.to_csv(temp, index=False)
    temp.replace(path)
    raw_path = RAW_DIR / "session_laps.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(raw_path, index=False)
    if errors:
        print("Some sessions could not be collected from the public archive: " + " | ".join(errors))
    return merged
