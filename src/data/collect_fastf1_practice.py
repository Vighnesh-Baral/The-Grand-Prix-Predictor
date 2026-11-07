"""Backfill FP1/FP2/FP3 fastest valid-lap records without an F1 login.

Uses TIF1 public, post-session FastF1-derived CDN archives. This collector does
not call FastF1's authenticated F1 data API and never prompts for an email/password.

Examples:
  python -m src.data.collect_fastf1_practice --start-year 2018 --end-year 2026
  python -m src.data.collect_fastf1_practice --start-year 2024 --end-year 2026 --max-events 5

The collector checkpoints after each session and skips sessions already covered by
at least 15 drivers unless --force is supplied. Sprint weekends may not have all
three practice sessions. TIF1 is not a live-timing source.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
from src.data.tif1_client import configure_cache as configure_tif1_cache, load_session as load_tif1_session, session_driver_metadata
from src.data.identity import add_canonical_identities, normalize_identity_name, resolve_season_code_metadata

LOGGER = logging.getLogger("f1_tif1_practice")
SESSION_CODES = ("FP1", "FP2", "FP3")
OUTPUT_COLUMNS = [
    "season", "round", "race_name", "circuit_id", "session", "position",
    "driver_id", "driver_code", "driver_name", "constructor_id",
    "constructor_name", "time", "time_seconds", "laps", "status", "source",
    "driver_key", "driver_identity_status", "constructor_key", "constructor_identity_status",
]


def _seconds(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        if hasattr(value, "total_seconds"):
            return float(value.total_seconds())
        # TIF1 exposes LapTime as Timedelta and LapTimeSeconds as a numeric
        # convenience field. Numeric lap-time inputs are already in seconds.
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return float(pd.to_timedelta(value).total_seconds())
    except (TypeError, ValueError):
        return None


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text and normalize_identity_name(text):
            return value
    return None


def _format_time(seconds: float | None) -> str | None:
    if seconds is None or not np.isfinite(seconds) or seconds <= 0:
        return None
    mins = int(seconds // 60)
    return f"{mins}:{seconds - mins * 60:06.3f}"


def build_classification_rows(
    season: int,
    round_number: int,
    session_code: str,
    laps: pd.DataFrame,
    session_results: pd.DataFrame | None = None,
    race_entries: pd.DataFrame | None = None,
    race_name: str | None = None,
    circuit_id: str | None = None,
) -> pd.DataFrame:
    """Build one fastest-lap record per driver, without needing a race result."""
    if laps is None or laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = laps.copy()
    lap_time_col = "LapTime" if "LapTime" in data.columns else "LapTimeSeconds" if "LapTimeSeconds" in data.columns else None
    if lap_time_col is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data["_seconds"] = data[lap_time_col].map(_seconds)
    data = data.dropna(subset=["Driver", "_seconds"])
    data = data[data["_seconds"].between(40, 200)]
    def truthy(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})

    if "IsAccurate" in data:
        data = data[truthy(data["IsAccurate"])]
    if "Deleted" in data:
        data = data[~truthy(data["Deleted"])]
    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    best = data.sort_values("_seconds").groupby("Driver", as_index=False).first()
    best = best.sort_values("_seconds").reset_index(drop=True)
    best["position"] = np.arange(1, len(best) + 1)

    result_map: dict[str, dict[str, Any]] = {}
    if session_results is not None and not session_results.empty:
        for _, row in session_results.iterrows():
            code = _first_value(row.get("Abbreviation"), row.get("Driver"))
            if code is not None:
                result_map[str(code).upper()] = row.to_dict()
    race_map: dict[str, dict[str, Any]] = {}
    race_name_map: dict[str, dict[str, Any]] = {}
    if race_entries is not None and not race_entries.empty:
        if "driver_code" in race_entries:
            for _, row in race_entries.dropna(subset=["driver_code"]).drop_duplicates("driver_code").iterrows():
                race_map[str(row["driver_code"]).upper()] = row.to_dict()
        if "driver_name" in race_entries:
            for _, row in race_entries.dropna(subset=["driver_name"]).drop_duplicates("driver_name").iterrows():
                race_name_map[normalize_identity_name(row["driver_name"])] = row.to_dict()

    rows: list[dict[str, Any]] = []
    for _, item in best.iterrows():
        code = str(item["Driver"]).upper()
        fastf1_row = result_map.get(code, {})
        seconds = float(item["_seconds"])
        metadata_name = _first_value(fastf1_row.get("FullName"), fastf1_row.get("BroadcastName"))
        race_row = race_map.get(code, {})
        # Only use a code match if its full name does not conflict with session metadata.
        if metadata_name is not None and race_row and normalize_identity_name(metadata_name) != normalize_identity_name(race_row.get("driver_name")):
            race_row = {}
        if not race_row and metadata_name is not None:
            race_row = race_name_map.get(normalize_identity_name(metadata_name), {})
        full_name = _first_value(metadata_name, race_row.get("driver_name"), code)
        team = _first_value(item.get("Team"), item.get("TeamName"), fastf1_row.get("TeamName"), fastf1_row.get("Team"), race_row.get("constructor_name"))
        rows.append({
            "season": int(season), "round": int(round_number), "race_name": race_name,
            "circuit_id": circuit_id, "session": str(session_code).upper(),
            "position": int(item["position"]), "driver_id": race_row.get("driver_id"),
            "driver_code": code, "driver_name": full_name,
            "constructor_id": race_row.get("constructor_id"), "constructor_name": team,
            "time": _format_time(seconds), "time_seconds": seconds,
            "laps": int(data[data["Driver"].astype(str).str.upper().eq(code)].shape[0]),
            "status": "classified", "source": "TIF1 public archive (FastF1-derived lap classification)",
        })
    return add_canonical_identities(pd.DataFrame(rows, columns=OUTPUT_COLUMNS))


def _merge_practice(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (existing, incoming) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for col in OUTPUT_COLUMNS:
        if col not in combined:
            combined[col] = np.nan
    combined["season"] = pd.to_numeric(combined["season"], errors="coerce")
    combined["round"] = pd.to_numeric(combined["round"], errors="coerce")
    combined["session"] = combined["session"].astype(str).str.upper()
    # Resolve placeholder/missing names only from unambiguous same-season code history.
    combined = resolve_season_code_metadata(combined)
    combined = add_canonical_identities(combined)
    # Prefer the TIF1/FastF1-derived lap reconstruction over legacy Jolpica rows
    # when both sources have the same driver-session row. Some
    # legacy Jolpica exports do not carry driver_code; use stable driver_id/name
    # fallbacks so unrelated rows are never collapsed into one null-key record.
    # Merge on normalized full name when available so a Jolpica row carrying an
    # upstream ID can still match its TIF1 counterpart when that source lacks IDs.
    # This is a deduplication key only; it does not overwrite either source ID.
    normalized_names = combined.get("driver_name", pd.Series(index=combined.index, dtype=object)).map(normalize_identity_name)
    combined["_driver_key"] = pd.Series(pd.NA, index=combined.index, dtype="string")
    named = normalized_names.ne("") & normalized_names.ne(
        combined.get("driver_code", pd.Series(index=combined.index, dtype=object)).map(normalize_identity_name)
    )
    combined.loc[named, "_driver_key"] = "name:" + normalized_names[named]
    missing_key = combined["_driver_key"].isna()
    ids = combined.get("driver_id", pd.Series(index=combined.index, dtype=object)).astype("string").str.strip()
    has_id = missing_key & ids.notna() & ids.ne("")
    combined.loc[has_id, "_driver_key"] = "id:" + ids[has_id]
    missing_key = combined["_driver_key"].isna()
    codes = combined.get("driver_code", pd.Series(index=combined.index, dtype=object)).astype("string").str.upper()
    has_code = missing_key & codes.notna() & codes.ne("")
    combined.loc[has_code, "_driver_key"] = "code:" + codes[has_code]
    combined.loc[combined["_driver_key"].isna(), "_driver_key"] = "row:" + combined.index[combined["_driver_key"].isna()].astype(str)
    combined["_source_priority"] = combined["source"].astype(str).str.lower().str.contains("fastf1", regex=False).astype(int)
    keys = ["season", "round", "session", "_driver_key"]
    # Preserve verified upstream IDs from either source while allowing the
    # preferred TIF1 lap-time row to supply the session performance fields.
    for col in ("driver_id", "constructor_id"):
        if col in combined:
            combined[col] = combined.groupby(keys, dropna=False)[col].transform(lambda values: values.bfill().ffill())
    combined = combined.sort_values("_source_priority").drop_duplicates(keys, keep="last")
    combined = add_canonical_identities(combined)
    return combined.drop(columns=["_source_priority", "_driver_key"]).sort_values(keys[:3]).reset_index(drop=True)


def collect_range(
    start_year: int, end_year: int, calendar_path: Path | str = PROCESSED_DIR / "races.csv",
    practice_path: Path | str = PROCESSED_DIR / "practice_results.csv", *,
    max_events: int | None = None, force: bool = False,
) -> pd.DataFrame:
    calendar_file = Path(calendar_path)
    if not calendar_file.exists():
        raise FileNotFoundError(f"Calendar not found: {calendar_file}")
    calendar = pd.read_csv(calendar_file)
    calendar["season"] = pd.to_numeric(calendar["season"], errors="coerce")
    calendar["round"] = pd.to_numeric(calendar["round"], errors="coerce")
    calendar = calendar[calendar["season"].between(start_year, end_year)].dropna(subset=["season", "round"])
    calendar = calendar.drop_duplicates(["season", "round"]).sort_values(["season", "round"])
    if max_events:
        calendar = calendar.head(max_events)
    output = Path(practice_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    race_path = PROCESSED_DIR / "race_results.csv"
    race_results = pd.read_csv(race_path) if race_path.exists() else pd.DataFrame()
    configure_tif1_cache()
    success = 0
    skipped = 0
    errors: list[str] = []
    total = len(calendar) * len(SESSION_CODES)
    done = 0

    for event in calendar.itertuples(index=False):
        season, rnd = int(event.season), int(event.round)
        name = str(getattr(event, "race_name", f"Round {rnd}"))
        circuit = getattr(event, "circuit_id", None)
        entries = race_results[(pd.to_numeric(race_results.get("season"), errors="coerce") == season) &
                               (pd.to_numeric(race_results.get("round"), errors="coerce") == rnd)] if not race_results.empty else pd.DataFrame()
        for session_code in SESSION_CODES:
            done += 1
            existing_session = existing[
                (pd.to_numeric(existing.get("season"), errors="coerce") == season) &
                (pd.to_numeric(existing.get("round"), errors="coerce") == rnd) &
                (existing.get("session", pd.Series(index=existing.index, dtype=object)).astype(str).str.upper() == session_code)
            ] if not existing.empty else pd.DataFrame()
            existing_time_rows = int(pd.to_numeric(existing_session.get("time_seconds", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not existing_session.empty else 0
            if not force and len(existing_session) >= 15 and existing_time_rows >= 15:
                skipped += 1
                LOGGER.info("[%s/%s] Skip %s %s: %s existing driver rows with usable times", done, total, season, name, existing_time_rows)
                continue
            LOGGER.info("[%s/%s] Fetching %s %s", done, total, name, session_code)
            try:
                # Read completed-session files from public CDNs. This avoids the
                # interactive F1TV login introduced in direct FastF1 3.9+ access.
                session = load_tif1_session(season, name, session_code)
                built = build_classification_rows(
                    season, rnd, session_code, session.laps.copy(),
                    session_driver_metadata(session), entries, name, circuit,
                )
                if built.empty:
                    LOGGER.warning("No usable fastest-lap classification for %s %s", name, session_code)
                    continue
                existing = _merge_practice(existing, built)
                temp = output.with_suffix(output.suffix + ".tmp")
                existing.to_csv(temp, index=False)
                temp.replace(output)
                raw_path = RAW_DIR / "practice_results.csv"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                existing.to_csv(raw_path, index=False)
                success += 1
                LOGGER.info("Saved %s rows for %s %s", len(built), name, session_code)
            except Exception as exc:
                errors.append(f"{season} {name} {session_code}: {type(exc).__name__}: {exc}")
                LOGGER.warning("Unavailable %s %s: %s", name, session_code, exc)
    print(f"Unauthenticated TIF1 practice collection complete: {success} sessions collected, {skipped} already covered, {len(errors)} session errors.")
    print(f"Practice results: {output}")
    if errors:
        print("Unavailable sessions (some sprint weekends do not have all three FP sessions):")
        for error in errors:
            print(f" - {error}")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog="No F1TV login is used. TIF1 serves completed sessions from public CDNs.")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--calendar", default=str(PROCESSED_DIR / "races.csv"))
    parser.add_argument("--output", default=str(PROCESSED_DIR / "practice_results.csv"))
    parser.add_argument("--max-events", type=int, default=None, help="Limit events for a small test run")
    parser.add_argument("--force", action="store_true", help="Refresh sessions even when 15+ driver rows already exist")
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year must be >= --start-year")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    collect_range(args.start_year, args.end_year, args.calendar, args.output, max_events=args.max_events, force=args.force)


if __name__ == "__main__":
    main()
