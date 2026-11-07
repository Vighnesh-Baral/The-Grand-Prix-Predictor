"""Incremental OpenF1 historical sync and optional periodic scheduler.

Free-tier data are fetched only after the documented live-access window closes.
The sync is idempotent and stores raw endpoint tables plus mapped weather rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.config import PROCESSED_DIR, RAW_DIR, CACHE_DIR
from src.data.openf1_client import OpenF1Client

LOGGER = logging.getLogger("f1_openf1_sync")
SESSION_CODES = {
    "PRACTICE 1": "FP1", "PRACTICE 2": "FP2", "PRACTICE 3": "FP3",
    "QUALIFYING": "Q", "SPRINT QUALIFYING": "SQ", "SPRINT SHOOTOUT": "SQ",
    "SPRINT": "S", "RACE": "R",
}
ENDPOINT_POLICY = {
    "FP1": ("drivers", "laps", "stints", "weather", "race_control"),
    "FP2": ("drivers", "laps", "stints", "weather", "race_control"),
    "FP3": ("drivers", "laps", "stints", "weather", "race_control"),
    "SQ": ("drivers", "laps", "stints", "weather", "race_control", "session_result"),
    "Q": ("drivers", "laps", "stints", "weather", "race_control", "session_result"),
    "S": ("drivers", "laps", "stints", "weather", "race_control", "session_result"),
    "R": ("drivers", "laps", "stints", "weather", "pit", "race_control", "session_result", "starting_grid"),
}
RETRY_HOURS = (1, 3, 6)

# OpenF1's short circuit labels and Jolpica's Ergast-era identifiers are not
# always textually identical (e.g. Austin vs circuit_of_the_americas).
CIRCUIT_ALIASES = {
    "albertpark": ("albert_park", "australian"),
    "shanghai": ("shanghai", "chinese"),
    "suzuka": ("suzuka", "japanese"),
    "sakhir": ("bahrain",),
    "bahrain": ("bahrain",),
    "jeddah": ("jeddah", "saudi"),
    "miami": ("miami",),
    "imola": ("imola", "emilia"),
    "monaco": ("monaco",),
    "barcelona": ("catalunya", "spanish"),
    "montmelo": ("catalunya", "spanish"),
    "montreal": ("villeneuve", "canadian"),
    "spielberg": ("red_bull_ring", "austrian"),
    "silverstone": ("silverstone", "british"),
    "hungaroring": ("hungaroring", "hungarian"),
    "spa": ("spa", "belgian"),
    "zandvoort": ("zandvoort", "dutch"),
    "monza": ("monza", "italian"),
    "baku": ("baku", "azerbaijan"),
    "marinabay": ("marina_bay", "singapore"),
    "austin": ("circuit_of_the_americas", "united_states"),
    "mexicocity": ("rodriguez", "mexico"),
    "interlagos": ("interlagos", "sao_paulo"),
    "saopaulo": ("interlagos", "sao_paulo"),
    "lasvegas": ("las_vegas",),
    "lusail": ("losail", "qatar"),
    "yasmarina": ("yas_marina", "abu_dhabi"),
    "ricardo tormo": ("ricardo_tormo",),
}


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _session_code(value: object) -> str | None:
    return SESSION_CODES.get(str(value or "").strip().upper())


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _map_sessions_to_calendar(sessions: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Match OpenF1 meeting metadata to Jolpica GP rows using circuit name + date."""
    if sessions.empty:
        return sessions
    result = sessions.copy()
    result["season"] = pd.to_numeric(result.get("year"), errors="coerce")
    result["round"] = pd.NA
    result["circuit_id"] = None
    result["race_name"] = None
    if calendar is None or calendar.empty:
        return result
    races = calendar.copy()
    races["season"] = pd.to_numeric(races.get("season"), errors="coerce")
    races["round"] = pd.to_numeric(races.get("round"), errors="coerce")
    races["_race_date"] = pd.to_datetime(races.get("date"), errors="coerce", utc=True)
    for idx, session in result.iterrows():
        year = pd.to_numeric(session.get("year"), errors="coerce")
        start = pd.to_datetime(session.get("date_start"), errors="coerce", utc=True)
        if pd.isna(year) or pd.isna(start):
            continue
        candidates = races[races["season"].eq(year)].copy()
        if candidates.empty:
            continue
        circuit = _norm(session.get("circuit_short_name"))
        race_names = candidates.get("race_name", pd.Series(index=candidates.index, dtype=object)).map(_norm)
        circuit_ids = candidates.get("circuit_id", pd.Series(index=candidates.index, dtype=object)).map(_norm)
        aliases = CIRCUIT_ALIASES.get(circuit, ())
        tokens = tuple(_norm(token) for token in (circuit, *aliases) if token)
        name_match = pd.Series(False, index=candidates.index)
        for token in tokens:
            name_match |= race_names.str.contains(token, regex=False) | circuit_ids.str.contains(token, regex=False)
        candidates = candidates[name_match].copy()
        if candidates.empty:
            continue
        candidates["_distance_days"] = (candidates["_race_date"] - start).abs().dt.total_seconds() / 86400
        candidates = candidates[candidates["_distance_days"].le(5)]
        if candidates.empty:
            continue
        match = candidates.sort_values("_distance_days").iloc[0]
        result.loc[idx, "round"] = int(match["round"])
        result.loc[idx, "circuit_id"] = match.get("circuit_id")
        result.loc[idx, "race_name"] = match.get("race_name")
    return result


class OpenF1Sync:
    def __init__(
        self,
        client: OpenF1Client | None = None,
        *,
        processed_dir: Path | str = PROCESSED_DIR,
        raw_dir: Path | str = RAW_DIR,
        state_path: Path | str | None = None,
        free_delay_minutes: int = 35,
        now_fn=lambda: datetime.now(timezone.utc),
    ) -> None:
        self.client = client or OpenF1Client()
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.state_path = Path(state_path) if state_path else CACHE_DIR / "openf1_sync_state.json"
        self.free_delay = timedelta(minutes=free_delay_minutes)
        self.now_fn = now_fn
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp.replace(self.state_path)

    def _read_endpoint(self, endpoint: str) -> pd.DataFrame:
        """Merge raw and processed endpoint exports, keeping the newest duplicate.

        Raw exports can contain rows that were not present when an older processed
        copy was built. Returning the processed file alone would silently hide them.
        """
        processed = _read_csv(self.processed_dir / f"openf1_{endpoint}.csv")
        raw = _read_csv(self.raw_dir / f"openf1_{endpoint}.csv")
        frames = [frame for frame in (raw, processed) if frame is not None and not frame.empty]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True, sort=False)
        dedup_keys = {
            "sessions": ["session_key"],
            "drivers": ["session_key", "driver_number"],
            "laps": ["session_key", "driver_number", "lap_number"],
            "stints": ["session_key", "driver_number", "stint_number"],
            "weather": ["session_key", "date"],
            "pit": ["session_key", "driver_number", "lap_number"],
            "race_control": ["session_key", "date", "message"],
            "session_result": ["session_key", "driver_number"],
            "starting_grid": ["session_key", "driver_number"],
        }.get(endpoint, [])
        keys = [column for column in dedup_keys if column in combined.columns]
        if keys:
            combined = combined.drop_duplicates(keys, keep="last")
        return combined.reset_index(drop=True)

    def rebuild_local_tables(self) -> None:
        """Rebuild canonical local tables from existing OpenF1 files without network calls.

        This also supports projects where OpenF1 exports were copied into data/raw
        but the older synchroniser never generated processed/openf1_* tables.
        """
        for endpoint in ("sessions", "drivers", "laps", "stints", "weather", "pit", "race_control", "session_result", "starting_grid"):
            frame = self._read_endpoint(endpoint)
            target = self.processed_dir / f"openf1_{endpoint}.csv"
            if not frame.empty:
                _atomic_csv(frame, target)
        self._rebuild_weather_table()
        self._rebuild_openf1_lap_table()
        self._rebuild_starting_grid_table()

    def _persist(self, endpoint: str, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> int:
        enriched = [{**row, **metadata} for row in rows]
        processed = self.processed_dir / f"openf1_{endpoint}.csv"
        raw = self.raw_dir / f"openf1_{endpoint}.csv"
        old = _read_csv(processed)
        new = pd.DataFrame(enriched)
        combined = pd.concat([old, new], ignore_index=True, sort=False) if not old.empty or not new.empty else pd.DataFrame()
        dedup_keys = {
            "laps": ["session_key", "driver_number", "lap_number"],
            "stints": ["session_key", "driver_number", "stint_number"],
            "weather": ["session_key", "date"],
            "drivers": ["session_key", "driver_number"],
            "pit": ["session_key", "driver_number", "lap_number"],
            "race_control": ["session_key", "date", "message"],
            "session_result": ["session_key", "driver_number"],
            "starting_grid": ["session_key", "driver_number"],
        }.get(endpoint, ["session_key"])
        if not combined.empty:
            keys = [k for k in dedup_keys if k in combined.columns]
            if keys:
                combined = combined.drop_duplicates(keys, keep="last")
            _atomic_csv(combined, processed)
            _atomic_csv(combined, raw)
        return int(len(new))

    def sync(self, start_year: int = 2023, end_year: int | None = None, *, refresh_sessions: bool = False) -> dict[str, Any]:
        if start_year < 2023:
            raise ValueError("OpenF1 historical API coverage starts in 2023; use FastF1/Jolpica for earlier seasons.")
        end_year = int(end_year or self.now_fn().year)
        if end_year < start_year:
            raise ValueError("end_year must be >= start_year")
        calendar = _read_csv(self.processed_dir / "races.csv")
        state = self._state()
        session_frames: list[pd.DataFrame] = []
        api_errors: list[str] = []
        for year in range(start_year, end_year + 1):
            try:
                rows = self.client.sessions(year, refresh=refresh_sessions)
                if rows:
                    session_frames.append(pd.DataFrame(rows))
            except Exception as exc:
                api_errors.append(f"sessions/{year}: {exc}")
        all_sessions = pd.concat(session_frames, ignore_index=True, sort=False) if session_frames else pd.DataFrame()
        if all_sessions.empty:
            return {"sessions_discovered": 0, "sessions_eligible": 0, "endpoints_fetched": 0, "rows_written": {}, "errors": api_errors}

        all_sessions = _map_sessions_to_calendar(all_sessions, calendar)
        all_sessions["session_code"] = all_sessions.get("session_name", pd.Series(index=all_sessions.index, dtype=object)).map(_session_code)
        _atomic_csv(all_sessions, self.processed_dir / "openf1_sessions.csv")
        _atomic_csv(all_sessions, self.raw_dir / "openf1_sessions.csv")

        now = self.now_fn()
        end_times = pd.to_datetime(all_sessions.get("date_end"), errors="coerce", utc=True)
        row_counts: dict[str, int] = {}
        fetched = 0
        # The race-session starting-grid endpoint can be published after
        # qualifying, before the race itself. Fetch it in that window so race
        # predictions can use penalties/official grid rather than qualifying P.
        q_sessions = all_sessions[all_sessions["session_code"].eq("Q")].copy()
        race_sessions = all_sessions[all_sessions["session_code"].eq("R")].copy()
        for race_session in race_sessions.to_dict(orient="records"):
            race_start = pd.to_datetime(race_session.get("date_start"), errors="coerce", utc=True)
            if pd.isna(race_start) or race_start <= now + timedelta(minutes=30):
                continue
            year = pd.to_numeric(race_session.get("season"), errors="coerce")
            rnd = pd.to_numeric(race_session.get("round"), errors="coerce")
            q_match = q_sessions[
                pd.to_numeric(q_sessions["season"], errors="coerce").eq(year)
                & pd.to_numeric(q_sessions["round"], errors="coerce").eq(rnd)
            ].copy()
            if q_match.empty:
                continue
            q_ends = pd.to_datetime(q_match.get("date_end"), errors="coerce", utc=True)
            q_match = q_match[q_ends.notna() & (q_ends <= now - self.free_delay)]
            if q_match.empty:
                continue
            session_key = race_session.get("session_key")
            if session_key is None:
                continue
            state_key = f"{session_key}:starting_grid"
            previous = state.get(state_key, {})
            last_attempt = pd.to_datetime(previous.get("last_attempt"), errors="coerce", utc=True)
            attempts = int(previous.get("attempts", 0))
            if pd.notna(last_attempt) and attempts > 0:
                elapsed = now - last_attempt.to_pydatetime()
                if previous.get("complete") and elapsed < timedelta(hours=24):
                    continue
                if not previous.get("complete"):
                    retry_hours = RETRY_HOURS[min(attempts - 1, len(RETRY_HOURS) - 1)]
                    if elapsed < timedelta(hours=retry_hours):
                        continue
            metadata = {
                "season": race_session.get("season"), "round": race_session.get("round"),
                "session": "R", "session_key": session_key,
                "meeting_key": race_session.get("meeting_key"),
                "circuit_id": race_session.get("circuit_id"),
                "race_name": race_session.get("race_name"),
                "circuit_short_name": race_session.get("circuit_short_name"),
            }
            try:
                grid_data = self.client.session_endpoint("starting_grid", session_key)
                count = self._persist("starting_grid", grid_data, metadata)
                row_counts["starting_grid"] = row_counts.get("starting_grid", 0) + count
                fetched += 1
                state[state_key] = {
                    "last_attempt": now.isoformat(), "attempts": attempts + 1,
                    "rows": count, "complete": bool(count > 0),
                }
            except Exception as exc:
                api_errors.append(f"{session_key}/starting_grid: {exc}")
                state[state_key] = {
                    "last_attempt": now.isoformat(), "attempts": attempts + 1,
                    "rows": 0, "complete": False, "last_error": str(exc),
                }
            self._save_state(state)

        eligible = all_sessions[all_sessions["session_code"].isin(ENDPOINT_POLICY)].copy()
        eligible = eligible[end_times.loc[eligible.index].notna() & (end_times.loc[eligible.index] <= now - self.free_delay)]
        eligible = eligible[eligible.get("is_cancelled", pd.Series(False, index=eligible.index)).fillna(False).astype(str).str.lower().isin(["false", "0", "no"])]

        for session in eligible.to_dict(orient="records"):
            session_key = session.get("session_key")
            code = session.get("session_code")
            if session_key is None or code not in ENDPOINT_POLICY:
                continue
            metadata = {
                "season": session.get("year"), "round": session.get("round"),
                "session": code, "session_key": session_key,
                "meeting_key": session.get("meeting_key"),
                "circuit_id": session.get("circuit_id"),
                "race_name": session.get("race_name"),
                "circuit_short_name": session.get("circuit_short_name"),
            }
            for endpoint in ENDPOINT_POLICY[code]:
                state_key = f"{session_key}:{endpoint}"
                previous = state.get(state_key, {})
                last_attempt = pd.to_datetime(previous.get("last_attempt"), errors="coerce", utc=True)
                attempts = int(previous.get("attempts", 0))
                if pd.notna(last_attempt) and attempts > 0:
                    elapsed = now - last_attempt.to_pydatetime()
                    if previous.get("complete") and elapsed < timedelta(hours=24):
                        continue
                    if not previous.get("complete"):
                        retry_hours = RETRY_HOURS[min(attempts - 1, len(RETRY_HOURS) - 1)]
                        if elapsed < timedelta(hours=retry_hours):
                            continue
                try:
                    data = self.client.session_endpoint(endpoint, session_key)
                    count = self._persist(endpoint, data, metadata)
                    previous = {
                        "last_attempt": now.isoformat(), "attempts": attempts + 1,
                        "rows": count, "complete": bool(count > 0),
                    }
                    state[state_key] = previous
                    fetched += 1
                    row_counts[endpoint] = row_counts.get(endpoint, 0) + count
                except Exception as exc:
                    api_errors.append(f"{session_key}/{endpoint}: {exc}")
                    state[state_key] = {
                        "last_attempt": now.isoformat(), "attempts": attempts + 1,
                        "rows": 0, "complete": False, "last_error": str(exc),
                    }
            self._save_state(state)

        self._rebuild_weather_table()
        self._rebuild_openf1_lap_table()
        self._rebuild_starting_grid_table()
        return {
            "sessions_discovered": int(len(all_sessions)),
            "sessions_mapped_to_jolpica": int(all_sessions["round"].notna().sum()),
            "sessions_eligible": int(len(eligible)),
            "endpoints_fetched": fetched,
            "rows_written": row_counts,
            "errors": api_errors,
            "free_access_delay_minutes": int(self.free_delay.total_seconds() / 60),
        }

    def _rebuild_weather_table(self) -> None:
        weather = self._read_endpoint("weather")
        if weather.empty:
            return
        required = {"season", "round", "session", "session_key"}
        if not required.issubset(weather.columns):
            return
        keep = [c for c in ["season", "round", "session", "session_key", "meeting_key", "date", "air_temperature", "track_temperature", "humidity", "wind_speed", "wind_direction", "rainfall", "pressure", "circuit_id"] if c in weather.columns]
        normalized = weather[keep].copy()
        _atomic_csv(normalized, self.processed_dir / "weather_observations.csv")

    def _rebuild_openf1_lap_table(self) -> None:
        laps = self._read_endpoint("laps")
        if laps.empty:
            return
        drivers = self._read_endpoint("drivers")
        stints = self._read_endpoint("stints")
        driver_cols = [c for c in ["session_key", "driver_number", "name_acronym", "full_name", "team_name"] if c in drivers.columns]
        if driver_cols:
            laps = laps.merge(drivers[driver_cols].drop_duplicates(["session_key", "driver_number"]), on=["session_key", "driver_number"], how="left")
        for col in ("lap_number", "driver_number", "lap_duration"):
            laps[col] = pd.to_numeric(laps.get(col), errors="coerce")
        if not stints.empty and {"session_key", "driver_number", "lap_start", "lap_end"}.issubset(stints.columns):
            stints = stints.copy()
            for col in ("driver_number", "lap_start", "lap_end", "stint_number", "tyre_age_at_start"):
                if col in stints:
                    stints[col] = pd.to_numeric(stints[col], errors="coerce")
            merged_rows: list[pd.DataFrame] = []
            for session_key, part in laps.groupby("session_key"):
                s = stints[stints["session_key"].eq(session_key)]
                if s.empty:
                    merged_rows.append(part)
                    continue
                p = part.merge(s, on=["session_key", "driver_number"], how="left", suffixes=("", "_stint"))
                in_range = p["lap_number"].between(p["lap_start"], p["lap_end"], inclusive="both")
                p = p[in_range | p["lap_start"].isna()].copy()
                merged_rows.append(p)
            laps = pd.concat(merged_rows, ignore_index=True) if merged_rows else laps
        laps["lap_time_seconds"] = pd.to_numeric(laps.get("lap_duration"), errors="coerce")
        driver_codes = laps.get("name_acronym", pd.Series(index=laps.index, dtype="string")).astype("string").str.upper()
        laps["driver_code"] = driver_codes.replace({"NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA, "": pd.NA})
        driver_names = laps.get("full_name", pd.Series(index=laps.index, dtype=object))
        laps["driver_name"] = driver_names.where(driver_names.notna(), laps["driver_code"])
        if "driver_number" in laps:
            laps["driver_name"] = laps["driver_name"].fillna("#" + laps["driver_number"].astype(str))
        laps["constructor_name"] = laps.get("team_name")
        laps["data_source"] = "openf1"
        laps["compound"] = laps.get("compound")
        laps["stint"] = pd.to_numeric(laps.get("stint_number"), errors="coerce")
        tyre_age_start = pd.to_numeric(laps.get("tyre_age_at_start", pd.Series(np.nan, index=laps.index)), errors="coerce")
        lap_number = pd.to_numeric(laps.get("lap_number", pd.Series(np.nan, index=laps.index)), errors="coerce")
        lap_start = pd.to_numeric(laps.get("lap_start", pd.Series(np.nan, index=laps.index)), errors="coerce")
        laps["tyre_life"] = tyre_age_start + (lap_number - lap_start)
        laps["is_accurate"] = laps["lap_time_seconds"].between(40, 200)
        laps["pit_out"] = laps.get("is_pit_out_lap", pd.Series(False, index=laps.index))
        # OpenF1 does not expose FastF1's complete track-status/timing-quality model.
        laps["track_status"] = None
        cols = [c for c in ["season", "round", "session", "driver_code", "driver_name", "constructor_name", "lap_number", "stint", "compound", "tyre_life", "lap_time_seconds", "is_accurate", "pit_out", "track_status", "session_key", "driver_number", "data_source"] if c in laps]
        normalized = laps[cols].copy()
        # FastF1 may identify drivers by abbreviation while OpenF1's driver
        # endpoint can be incomplete for a few rows. Use car number as a fallback
        # identity instead of collapsing every missing abbreviation together.
        normalized_codes = normalized.get("driver_code", pd.Series(pd.NA, index=normalized.index)).astype("string")
        normalized_numbers = normalized.get("driver_number", pd.Series(pd.NA, index=normalized.index)).astype("string")
        normalized["_driver_identity"] = normalized_codes.fillna("NUMBER:" + normalized_numbers.fillna("unknown"))
        # FastF1 remains the preferred table; OpenF1 fills missing event/session/
        # driver/lap keys where an identity is shared.
        canonical_path = self.processed_dir / "session_laps.csv"
        fastf1 = _read_csv(canonical_path)
        if not fastf1.empty and "data_source" not in fastf1.columns:
            fastf1["data_source"] = "fastf1"
        if not fastf1.empty:
            fast_codes = fastf1.get("driver_code", pd.Series(pd.NA, index=fastf1.index)).astype("string")
            fast_numbers = fastf1.get("driver_number", pd.Series(pd.NA, index=fastf1.index)).astype("string")
            fastf1["_driver_identity"] = fast_codes.fillna("NUMBER:" + fast_numbers.fillna("unknown"))
        combined = pd.concat([fastf1, normalized], ignore_index=True, sort=False)
        keys = [c for c in ["season", "round", "session", "_driver_identity", "lap_number"] if c in combined]
        if keys:
            combined = combined.drop_duplicates(keys, keep="first")
        combined = combined.drop(columns=["_driver_identity"], errors="ignore")
        if not combined.empty:
            _atomic_csv(combined, canonical_path)



    def _rebuild_starting_grid_table(self) -> None:
        """Join OpenF1 race grids to meeting drivers and Jolpica driver IDs."""
        grid = self._read_endpoint("starting_grid")
        if grid.empty or not {"season", "round", "meeting_key", "driver_number", "position"}.issubset(grid.columns):
            return
        drivers = self._read_endpoint("drivers")
        qualifying = _read_csv(self.processed_dir / "qualifying_results.csv")
        result = grid.copy()
        result["driver_number"] = pd.to_numeric(result["driver_number"], errors="coerce")
        result["grid"] = pd.to_numeric(result["position"], errors="coerce")
        if not drivers.empty and {"meeting_key", "driver_number", "name_acronym"}.issubset(drivers.columns):
            lookup = drivers[[c for c in ["meeting_key", "driver_number", "name_acronym", "full_name", "team_name"] if c in drivers.columns]].copy()
            lookup["driver_number"] = pd.to_numeric(lookup["driver_number"], errors="coerce")
            lookup["driver_code"] = lookup["name_acronym"].astype(str).str.upper()
            lookup = lookup.drop_duplicates(["meeting_key", "driver_number"], keep="last")
            result = result.merge(lookup.drop(columns=["name_acronym"], errors="ignore"), on=["meeting_key", "driver_number"], how="left")
        if "driver_code" not in result:
            result["driver_code"] = None
        result["driver_code"] = result["driver_code"].astype(str).str.upper().replace({"NAN": None, "NONE": None})
        if not qualifying.empty and {"season", "round", "driver_code", "driver_id"}.issubset(qualifying.columns):
            q = qualifying[["season", "round", "driver_code", "driver_id"]].copy()
            q["season"] = pd.to_numeric(q["season"], errors="coerce")
            q["round"] = pd.to_numeric(q["round"], errors="coerce")
            q["driver_code"] = q["driver_code"].astype(str).str.upper()
            result = result.merge(q.drop_duplicates(["season", "round", "driver_code"]), on=["season", "round", "driver_code"], how="left")
        if "driver_id" not in result:
            result["driver_id"] = None
        result["grid_source"] = "openf1_starting_grid"
        keep = [c for c in ["season", "round", "driver_id", "driver_code", "driver_number", "grid", "grid_source", "session_key", "meeting_key"] if c in result.columns]
        result = result[keep].dropna(subset=["season", "round", "grid"])
        result = result.drop_duplicates(["season", "round", "driver_code"], keep="last")
        if not result.empty:
            _atomic_csv(result, self.processed_dir / "starting_grid.csv")


def status(processed_dir: Path | str = PROCESSED_DIR) -> dict[str, Any]:
    root = Path(processed_dir)
    files = ["openf1_sessions.csv", "openf1_laps.csv", "openf1_stints.csv", "openf1_weather.csv", "openf1_starting_grid.csv", "starting_grid.csv", "weather_observations.csv", "session_laps.csv"]
    counts = {name: len(_read_csv(root / name)) for name in files}
    sessions = _read_csv(root / "openf1_sessions.csv")
    mapped = int(pd.to_numeric(sessions.get("round", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not sessions.empty else 0
    return {"available": counts["openf1_sessions.csv"] > 0, "mapped_sessions": mapped, "rows": counts, "historical_start_year": 2023, "access": "free historical tier; data fetched after the live window"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--refresh-sessions", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Poll OpenF1 periodically; intended as a separate process.")
    parser.add_argument("--interval-minutes", type=int, default=15)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    syncer = OpenF1Sync()
    if not args.watch:
        print(json.dumps(syncer.sync(args.start_year, args.end_year, refresh_sessions=args.refresh_sessions), indent=2))
        return
    if args.interval_minutes < 5:
        raise SystemExit("Use an interval of at least 5 minutes to avoid unnecessary API requests.")
    while True:
        try:
            result = syncer.sync(args.start_year, args.end_year, refresh_sessions=True)
            LOGGER.info("OpenF1 sync complete: %s", result)
        except Exception:
            LOGGER.exception("OpenF1 periodic sync failed")
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
