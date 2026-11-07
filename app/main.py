"""FastAPI application for F1 Race Intelligence 0.9.3.

Historical practice views rebuild missing canonical lap rows from local OpenF1
exports; prediction endpoints enforce strict session cutoffs and model metadata.
"""
from __future__ import annotations

import json
from datetime import datetime, time, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import ARTIFACTS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.features.build_features import build_event_prediction_frame
from src.historical.gp_history import build_gp_history
from src.historical.standings import local_constructor_standings, local_driver_standings
from src.models.predict import ModelBundleError, load_model_bundle, predict_pole, predict_race
from src.features.weekend_pace import summarise_constructor_pace
from src.data.openf1_sync import OpenF1Sync, status as openf1_status
from src.data.open_meteo_client import OpenMeteoClient
from src.data.normalize import parse_lap_time_seconds

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"
TABLE_FILES = {
    "races": "races.csv",
    "race_results": "race_results.csv",
    "qualifying": "qualifying_results.csv",
    "practice": "practice_results.csv",
    "sprints": "sprint_results.csv",
    "session_laps": "session_laps.csv",
    "driver_roster": "driver_roster.csv",
    "weather": "weather_observations.csv",
    "race_weather_forecasts": "race_weather_forecasts.csv",
    "starting_grid": "starting_grid.csv",
    "race_control": "openf1_race_control.csv",
    "pit_stops": "openf1_pit.csv",
    "openf1_drivers": "openf1_drivers.csv",
}

app = FastAPI(
    title="The Grand Prix Predictor API",
    version="0.9.3",
    description="Upcoming-event F1 intelligence workspace. Prediction readiness and feature coverage are reported explicitly.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_tables() -> dict[str, pd.DataFrame]:
    # Rebuild canonical tables from locally cached OpenF1 exports when needed.
    # This is strictly local I/O; it does not make network requests.
    try:
        OpenF1Sync().rebuild_local_tables()
    except Exception:
        # Historical API availability should not prevent the rest of the app loading.
        pass
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILES.items():
        path = PROCESSED_DIR / filename
        if path.exists():
            try:
                tables[key] = pd.read_csv(path)
            except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                tables[key] = pd.DataFrame()
        else:
            tables[key] = pd.DataFrame()
    return tables


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fastf1_derived_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify direct FastF1 and TIF1 public-archive (FastF1-derived) lap rows."""
    if frame is None or frame.empty:
        return pd.Series(False, index=frame.index if frame is not None else None, dtype=bool)
    if "data_source" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    source = frame["data_source"].fillna("fastf1").astype(str).str.lower()
    return source.eq("fastf1") | source.str.startswith("tif1")


def _tif1_source_mask(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or "data_source" not in frame.columns:
        return pd.Series(False, index=frame.index if frame is not None else None, dtype=bool)
    return frame["data_source"].fillna("").astype(str).str.lower().str.startswith("tif1")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    clean = frame.replace({np.nan: None})
    return [_json_safe(row) for row in clean.to_dict(orient="records")]


def _season_round_keys(frame: pd.DataFrame) -> set[tuple[int, int]]:
    if frame is None or frame.empty or not {"season", "round"}.issubset(frame.columns):
        return set()
    season = _number(frame["season"])
    rnd = _number(frame["round"])
    return {
        (int(s), int(r)) for s, r in zip(season, rnd)
        if pd.notna(s) and pd.notna(r)
    }


def _event_start(row: pd.Series) -> datetime | None:
    date_value = row.get("date")
    if date_value is None or pd.isna(date_value) or str(date_value).strip() == "":
        return None
    time_value = row.get("time")
    text = str(date_value).strip()
    if time_value is not None and not pd.isna(time_value) and str(time_value).strip():
        if "T" not in text and " " not in text:
            text = f"{text}T{str(time_value).strip()}"
    elif "T" not in text and " " not in text:
        # Never invent a race start time. A date-only event may appear in the
        # upcoming calendar, but live predictions stay disabled until timing is known.
        return None
    try:
        parsed = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except (TypeError, ValueError):
        return None


def _event_catalog() -> list[dict[str, Any]]:
    tables = load_tables()
    races = tables["races"].copy()
    if races.empty:
        return []
    for col in ("season", "round"):
        races[col] = _number(races.get(col, pd.Series(dtype=float)))
    race_keys = _season_round_keys(tables["race_results"])
    quali_keys = _season_round_keys(tables["qualifying"])
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for _, row in races.dropna(subset=["season", "round"]).iterrows():
        season, rnd = int(row["season"]), int(row["round"])
        key = (season, rnd)
        start = _event_start(row)
        completed = key in race_keys
        qualifying_done = key in quali_keys
        event_date = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
        date_is_future = bool(pd.notna(event_date) and event_date.date() > now.date())
        if completed:
            status = "completed"
        elif qualifying_done:
            status = "race_prediction_available" if start and start > now else "qualifying_complete_race_window_unclear"
        elif start and start > now:
            status = "pole_prediction_window_candidate"
        elif date_is_future and start is None:
            status = "scheduled_time_unknown"
        else:
            status = "in_progress_or_source_data_incomplete"
        record = row.to_dict()
        record.update({
            "season": season,
            "round": rnd,
            "start_utc": start.isoformat() if start else None,
            "scheduled_date_is_future": date_is_future,
            "status": status,
            "has_race_result": completed,
            "has_qualifying_result": qualifying_done,
            "pole_prediction_available": bool(not completed and not qualifying_done and start and start > now),
            "race_prediction_available": bool(not completed and qualifying_done and start and start > now),
            "race_prediction_before_qualifying_available": bool(not completed and not qualifying_done and start and start > now),
        })
        rows.append(_json_safe(record))
    return sorted(rows, key=lambda x: (x["season"], x["round"]))


def _is_future_event(event: dict[str, Any]) -> bool:
    """True only while the scheduled race start is still in the future."""
    value = event.get("start_utc")
    if value:
        try:
            start = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return start > datetime.now(timezone.utc)
        except (TypeError, ValueError):
            return False
    # Date-only future calendar rows remain selectable, but prediction endpoints
    # still reject them because the actual race start cannot be verified.
    return bool(event.get("scheduled_date_is_future"))


def _event_or_404(season: int, round_number: int) -> dict[str, Any]:
    for event in _event_catalog():
        if event["season"] == season and event["round"] == round_number:
            return event
    raise HTTPException(status_code=404, detail="The selected event is not present in the local race calendar.")


def _event_frame(
    season: int,
    round_number: int,
    stage: str,
    as_of_session: str = "FP3",
    weather_as_of_session: str | None = None,
) -> pd.DataFrame:
    t = load_tables()
    if t["race_results"].empty:
        raise HTTPException(status_code=503, detail="Race-history data are missing. Run the dataset ingestion command first.")
    try:
        before_free_practice = str(as_of_session).upper() == "PRE_FP"
        effective_as_of_session = "FP1" if before_free_practice else as_of_session
        effective_weather_cutoff = "FP1" if before_free_practice and str(weather_as_of_session or "PRE_FP").upper() == "PRE_FP" else weather_as_of_session
        practice_source = pd.DataFrame() if before_free_practice else t["practice"]
        lap_source = pd.DataFrame() if before_free_practice else t["session_laps"]
        weather_source = pd.DataFrame() if before_free_practice else t.get("weather", pd.DataFrame())
        race_weather_forecasts = t.get("race_weather_forecasts", pd.DataFrame()).copy()
        if stage == "race" and not t["races"].empty:
            schedule = t["races"].copy()
            schedule["season"] = _number(schedule.get("season", pd.Series(dtype=float)))
            schedule["round"] = _number(schedule.get("round", pd.Series(dtype=float)))
            event_rows = schedule[(schedule["season"] == season) & (schedule["round"] == round_number)]
            if not event_rows.empty:
                # A pre-race forecast is eligible in either prediction stage.
                # Session weather is separately cut off at FP3 or qualifying.
                live_forecast = OpenMeteoClient().live_event_forecast(event_rows.iloc[0])
                if live_forecast:
                    race_weather_forecasts = pd.concat(
                        [race_weather_forecasts, pd.DataFrame([live_forecast])], ignore_index=True, sort=False
                    )
        return build_event_prediction_frame(
            season, round_number, t["race_results"], t["qualifying"], practice_source,
            t["races"], t["sprints"], stage=stage,
            session_laps=lap_source, as_of_session=effective_as_of_session,
            driver_roster=t.get("driver_roster", pd.DataFrame()),
            weather_observations=weather_source,
            weather_as_of_session=effective_weather_cutoff or ("Q" if stage == "race" else effective_as_of_session),
            race_weather_forecasts=race_weather_forecasts,
            starting_grid=t.get("starting_grid", pd.DataFrame()),
            race_control=t.get("race_control", pd.DataFrame()),
            pit_stops=t.get("pit_stops", pd.DataFrame()),
            openf1_drivers=t.get("openf1_drivers", pd.DataFrame()),
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not construct event features: {exc}") from exc


def _load_model_bundle() -> dict[str, Any]:
    """Load the installed local prediction bundle (legacy or pace-aware)."""
    try:
        return load_model_bundle(ARTIFACTS_DIR / "f1_models.joblib")
    except ModelBundleError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load the local model artifact: {exc}") from exc


def _task_model_metadata(bundle: dict[str, Any], task: str) -> dict[str, Any]:
    metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
    tasks = metadata.get("tasks", {}) if isinstance(metadata, dict) else {}
    value = tasks.get(task, {}) if isinstance(tasks, dict) else {}
    return value if isinstance(value, dict) else {}


def _feature_coverage(bundle: dict[str, Any], task: str, feature: str) -> float:
    task_meta = _task_model_metadata(bundle, task)
    coverage = task_meta.get("feature_coverage_fraction", {})
    try:
        return float(coverage.get(feature, 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _model_readiness() -> dict[str, Any]:
    """Describe whether the installed bundle learned the newer pace features."""
    payload = _metrics()
    metrics = payload.get("metrics", {}) if payload.get("available") else {}
    tasks = metrics.get("tasks", {}) if isinstance(metrics, dict) else {}
    race_meta = tasks.get("winner", {}) if isinstance(tasks, dict) else {}
    race_cov = race_meta.get("feature_coverage_fraction", {}) if isinstance(race_meta, dict) else {}

    def safe_coverage(source: dict[str, Any], key: str) -> float:
        try:
            value = float(source.get(key, 0.0) or 0.0)
            return value if np.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    pole_stages: dict[str, dict[str, Any]] = {}
    for stage in ("FP1", "FP2", "FP3"):
        task = tasks.get(f"pole_{stage.lower()}", tasks.get("pole", {}) if stage == "FP3" else {}) if isinstance(tasks, dict) else {}
        coverage = task.get("feature_coverage_fraction", {}) if isinstance(task, dict) else {}
        fp_coverage = safe_coverage(coverage, "fp_best_gap_pct")
        constructor_coverage = safe_coverage(coverage, "constructor_quali_gap_recent3_pct")
        pole_stages[stage] = {
            "fp_lap_feature_coverage": fp_coverage,
            "constructor_qualifying_feature_coverage": constructor_coverage,
            "pace_aware": fp_coverage > 0.05 and constructor_coverage > 0.05,
        }
    race_coverage = safe_coverage(race_cov, "constructor_race_longrun_gap_recent3_pct")
    constructor_practice_coverage = safe_coverage(race_cov, "constructor_practice_longrun_gap_pct")
    reliability_coverage = safe_coverage(race_cov, "constructor_dnf_rate_recent5")
    pole_all_stages_pace_aware = all(item["pace_aware"] for item in pole_stages.values())
    race_pace_aware = race_coverage > 0.05
    fp3_coverage = pole_stages["FP3"]["fp_lap_feature_coverage"]
    fp3_constructor_coverage = pole_stages["FP3"]["constructor_qualifying_feature_coverage"]
    return {
        "metrics_available": bool(payload.get("available")),
        "pole_models_by_stage": pole_stages,
        # Compatibility aliases retained for the existing React frontend.
        "pole_pace_feature_coverage": fp3_coverage,
        "constructor_qualifying_feature_coverage": fp3_constructor_coverage,
        "pole_model_pace_aware": pole_stages["FP3"]["pace_aware"],
        "race_longrun_feature_coverage": race_coverage,
        "pole_model_pace_aware_for_all_stages": pole_all_stages_pace_aware,
        "race_model_pace_aware": race_pace_aware,
        "constructor_practice_longrun_feature_coverage": constructor_practice_coverage,
        "constructor_reliability_feature_coverage": reliability_coverage,
        "race_model_full_constructor_intelligence": constructor_practice_coverage > 0.05 and reliability_coverage > 0.05,
        "overall_status": "pace_aware_models" if pole_all_stages_pace_aware and race_pace_aware else "experimental_baseline_or_partial_pace_coverage",
    }


def _metrics() -> dict[str, Any]:
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.exists():
        path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        return {"available": False, "detail": "No local metrics.json file was found."}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return {"available": True, "metrics": json.load(handle)}
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "detail": f"Could not read model metrics: {exc}"}


class EventPredictionRequest(BaseModel):
    season: int = Field(ge=2018, le=2100)
    round: int = Field(ge=1, le=100)
    as_of_session: Literal["PRE_FP", "FP1", "FP2", "FP3"] = "FP3"
    race_stage: Literal["before_qualifying", "after_qualifying"] = "after_qualifying"


@app.get("/api/health")
def health() -> dict[str, Any]:
    t = load_tables()
    missing = [key for key in ("races", "race_results") if t[key].empty]
    artifact_exists = (ARTIFACTS_DIR / "f1_models.joblib").exists()
    laps = t["session_laps"]
    fastf1_derived = _fastf1_derived_mask(laps)
    tif1_rows = _tif1_source_mask(laps)
    openf1_rows = laps.get("data_source", pd.Series("", index=laps.index)).astype(str).str.lower().eq("openf1") if not laps.empty else pd.Series(dtype=bool)
    return {
        "status": "ok" if not missing else "data_missing",
        "data_ready": not missing,
        "missing_tables": missing,
        "model_artifact_present": artifact_exists,
        "fastf1_session_lap_data_present": bool(fastf1_derived.any()),
        "session_lap_rows": int(len(laps)),
        "fastf1_session_lap_rows": int(fastf1_derived.sum()),
        "tif1_session_lap_rows": int(tif1_rows.sum()),
        "openf1_lap_rows": int(openf1_rows.sum()),
        "openf1_weather_rows": int(len(t.get("weather", pd.DataFrame()))),
        "frontend_built": (STATIC_DIR / "index.html").exists(),
        "prediction_engine": _model_readiness()["overall_status"],
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    t = load_tables()
    events = _event_catalog()
    upcoming = [e for e in events if not e["has_race_result"] and _is_future_event(e)]
    upcoming.sort(key=lambda x: x["start_utc"] or f"{x.get('date', '9999')}T23:59:59Z")
    seasons = sorted({e["season"] for e in events})
    current_year = datetime.now(timezone.utc).year
    current_season = current_year if current_year in seasons else max((year for year in seasons if year <= current_year), default=max(seasons) if seasons else None)
    metrics = _metrics()
    return {
        "project": "The Grand Prix Predictor",
        "current_season": current_season,
        "upcoming_event": upcoming[0] if upcoming else None,
        "upcoming_event_count": len(upcoming),
        "race_events": len(t["races"]),
        "race_result_rows": len(t["race_results"]),
        "qualifying_rows": len(t["qualifying"]),
        "practice_rows": len(t["practice"]),
        "sprint_rows": len(t["sprints"]),
        "session_lap_rows": len(t["session_laps"]),
        "fastf1_session_lap_rows": int(_fastf1_derived_mask(t["session_laps"]).sum()),
        "tif1_session_lap_rows": int(_tif1_source_mask(t["session_laps"]).sum()),
        "openf1_lap_rows": int(t["session_laps"][t["session_laps"]["data_source"].astype(str).str.lower().eq("openf1")].shape[0]) if not t["session_laps"].empty and "data_source" in t["session_laps"] else 0,
        "weather_observation_rows": len(t.get("weather", pd.DataFrame())),
        "openf1": openf1_status(PROCESSED_DIR),
        "seasons": seasons,
        "season_range": f"{min(seasons)}–{max(seasons)}" if seasons else "Season archive unavailable",
        "model_artifact_present": (ARTIFACTS_DIR / "f1_models.joblib").exists(),
        "metrics_available": metrics.get("available", False),
        "prediction_engine": _model_readiness()["overall_status"],
        "methodology_status": _model_readiness()["overall_status"],
        "model_readiness": _model_readiness(),
    }


@app.get("/api/weather/forecast")
def race_weather_forecast(
    season: int = Query(..., ge=2018, le=2100),
    round_number: int = Query(..., alias="round", ge=1, le=100),
) -> dict[str, Any]:
    """Return the latest available forecast for the scheduled race start."""
    event = _event_or_404(season, round_number)
    try:
        forecast = OpenMeteoClient().live_event_forecast(event)
    except Exception as exc:  # weather is optional; a provider outage must not take down the API
        return {"event": event, "available": False, "reason": f"Weather provider unavailable: {type(exc).__name__}"}
    if not forecast:
        return {"event": event, "available": False, "reason": "No forecast is available for the scheduled race start or it is outside the forecast horizon."}
    return {"event": event, "available": True, "forecast": _json_safe(forecast)}


@app.get("/api/data/openf1/status")
def openf1_data_status() -> dict[str, Any]:
    """Show local OpenF1 sync coverage without making an external API request."""
    return openf1_status(PROCESSED_DIR)


def _event_lap_source_label(season: int, round_number: int, as_of_session: str = "FP3") -> str:
    laps = load_tables().get("session_laps", pd.DataFrame())
    if laps.empty or not {"season", "round", "session"}.issubset(laps.columns):
        return "Session lap data"
    eligible = {"FP1": {"FP1"}, "FP2": {"FP1", "FP2"}, "FP3": {"FP1", "FP2", "FP3"}}.get(str(as_of_session).upper(), {"FP1", "FP2", "FP3"})
    mask = _number(laps["season"]).eq(season) & _number(laps["round"]).eq(round_number)
    mask &= laps["session"].astype(str).str.upper().isin(eligible)
    event_laps = laps[mask]
    if event_laps.empty:
        return "Session lap data"
    if "data_source" not in event_laps:
        return "FastF1 lap-level data (legacy source metadata)"
    sources = set(event_laps["data_source"].fillna("fastf1").astype(str).str.lower())
    labels = []
    if any(source.startswith("tif1") for source in sources):
        labels.append("TIF1 (FastF1-derived archive)")
    elif "fastf1" in sources:
        labels.append("FastF1")
    if "openf1" in sources:
        labels.append("OpenF1")
    return " + ".join(labels) + " lap-level data"


def _lap_source_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"available": False, "lap_rows": 0, "sessions": []}
    rows = frame.copy()
    if "session" not in rows:
        return {"available": False, "lap_rows": int(len(rows)), "sessions": []}
    items = []
    for session_code, group in rows.groupby(rows["session"].astype(str).str.upper()):
        items.append({
            "session": str(session_code),
            "lap_rows": int(len(group)),
            "event_count": int(group[[c for c in ("season", "round") if c in group]].dropna().drop_duplicates().shape[0]) if {"season", "round"}.issubset(group.columns) else 0,
        })
    return {"available": True, "lap_rows": int(len(rows)), "sessions": sorted(items, key=lambda row: row["session"])}


@app.get("/api/data/readiness")
def data_readiness() -> dict[str, Any]:
    """Report data coverage relevant to the new pace-aware ML pipeline."""
    t = load_tables()
    table_rows = {name: int(len(frame)) for name, frame in t.items()}
    race_calendar = t["races"]
    seasons = sorted(_number(race_calendar.get("season", pd.Series(dtype=float))).dropna().astype(int).unique().tolist()) if not race_calendar.empty else []

    session_laps = t["session_laps"]
    if not session_laps.empty and "data_source" in session_laps.columns:
        fastf1_laps = session_laps[_fastf1_derived_mask(session_laps)]
        openf1_laps = session_laps[session_laps["data_source"].astype(str).str.lower().eq("openf1")]
    else:
        fastf1_laps = session_laps
        openf1_laps = pd.DataFrame()
    weather = t.get("weather", pd.DataFrame())
    session_summary: list[dict[str, Any]] = []
    if not session_laps.empty and {"season", "round", "session"}.issubset(session_laps.columns):
        lap_frame = session_laps.copy()
        lap_frame["season"] = _number(lap_frame["season"])
        lap_frame["round"] = _number(lap_frame["round"])
        lap_frame["session"] = lap_frame["session"].astype(str).str.upper()
        group_cols = ["session"]
        for session_code, group in lap_frame.groupby("session"):
            event_count = int(group[["season", "round"]].dropna().drop_duplicates().shape[0])
            driver_event_count = int(group[[c for c in ("season", "round", "driver_code") if c in group]].dropna().drop_duplicates().shape[0]) if "driver_code" in group else 0
            session_summary.append({
                "session": str(session_code),
                "lap_rows": int(len(group)),
                "event_count": event_count,
                "driver_event_count": driver_event_count,
                "seasons": sorted(_number(group["season"]).dropna().astype(int).unique().tolist()),
            })
    session_summary.sort(key=lambda item: item["session"])

    metrics_payload = _metrics()
    metrics = metrics_payload.get("metrics", {}) if metrics_payload.get("available") else {}
    tasks = metrics.get("tasks", {}) if isinstance(metrics, dict) else {}
    coverage_features = {
        "pole": ["constructor_quali_gap_recent3_pct", "constructor_circuit_quali_gap_prev_avg_pct", "fp_best_gap_pct"],
        "winner": ["constructor_race_longrun_gap_recent3_pct", "constructor_quali_gap_recent3_pct", "race_forecast_precipitation_mm", "race_forecast_is_wet"],
        "dnf": ["constructor_race_valid_laps_prev_avg"],
    }
    trained_coverage: dict[str, dict[str, float]] = {}
    for task, feature_names in coverage_features.items():
        task_data = tasks.get(task, {}) if isinstance(tasks, dict) else {}
        feature_coverage = task_data.get("feature_coverage_fraction", {}) if isinstance(task_data, dict) else {}
        trained_coverage[task] = {
            feature: float(feature_coverage.get(feature, 0.0))
            for feature in feature_names
        }
    for stage in ("FP1", "FP2", "FP3"):
        task_data = tasks.get(f"pole_{stage.lower()}", tasks.get("pole", {}) if stage == "FP3" else {}) if isinstance(tasks, dict) else {}
        feature_coverage = task_data.get("feature_coverage_fraction", {}) if isinstance(task_data, dict) else {}
        trained_coverage[f"pole_{stage.lower()}"] = {
            feature: float(feature_coverage.get(feature, 0.0))
            for feature in ("fp_best_gap_pct", "constructor_quali_gap_recent3_pct")
        }

    lap_sessions = {entry["session"] for entry in session_summary}
    recommendations = []
    required_practice_sessions = {"FP1", "FP2", "FP3"}
    missing_practice_sessions = sorted(required_practice_sessions - lap_sessions)
    if t["race_results"].empty or t["qualifying"].empty:
        recommendations.append("Collect race results and qualifying classifications before training the prediction models.")
    for session_code in missing_practice_sessions:
        recommendations.append(f"Collect historical session laps (FastF1 preferred; OpenF1 is a fallback) for {session_code}; pole models have separate FP1/FP2/FP3 cutoffs.")
    if "Q" not in lap_sessions:
        recommendations.append("Collect historical FastF1 qualifying laps if you want detailed qualifying diagnostics.")
    if "R" not in lap_sessions:
        recommendations.append("Collect historical FastF1 race laps to train the long-run/race-pace feature family.")
    complete_weekend_keys: set[tuple[int, int]] = set()
    complete_by_season: dict[int, set[tuple[int, int]]] = {}
    if not session_laps.empty and {"season", "round", "session"}.issubset(session_laps.columns):
        for (year, round_no), group in lap_frame.groupby(["season", "round"]):
            sessions_for_event = set(group["session"].dropna().astype(str))
            if required_practice_sessions.union({"R"}).issubset(sessions_for_event):
                key = (int(year), int(round_no))
                complete_weekend_keys.add(key)
                complete_by_season.setdefault(int(year), set()).add(key)
    complete_weekend_count = len(complete_weekend_keys)
    complete_weekend_seasons = sorted(complete_by_season)
    if complete_weekend_count < 20 or len(complete_weekend_seasons) < 3:
        recommendations.append(
            "For chronological pace-aware training, collect at least 20 complete GP weekends (FP1, FP2, FP3 and R) across at least three seasons; "
            f"current complete-weekend coverage is {complete_weekend_count} across {len(complete_weekend_seasons)} season(s)."
        )
    if not trained_coverage.get("pole_fp3", {}).get("fp_best_gap_pct", 0.0):
        recommendations.append("The installed pole model has no measured FastF1 practice-feature coverage; current FP inputs cannot influence its learned predictions until it is retrained with historical FP lap data.")

    return {
        "data_ready_for_basic_history": bool(not t["races"].empty and not t["race_results"].empty),
        "data_ready_for_pace_aware_training": bool(
            not t["race_results"].empty and not t["qualifying"].empty
            and complete_weekend_count >= 20 and len(complete_weekend_seasons) >= 3
        ),
        "complete_pace_weekends": complete_weekend_count,
        "complete_pace_weekend_seasons": complete_weekend_seasons,
        "minimum_pace_training_coverage": {"complete_weekends": 20, "seasons": 3},
        "available_lap_sessions": sorted(lap_sessions),
        "available_fastf1_sessions": sorted({row["session"] for row in _lap_source_summary(fastf1_laps)["sessions"]}),
        "available_openf1_sessions": sorted({row["session"] for row in _lap_source_summary(openf1_laps)["sessions"]}),
        "missing_practice_sessions": missing_practice_sessions,
        "season_coverage": seasons,
        "table_rows": table_rows,
        "fastf1": _lap_source_summary(fastf1_laps),
        "tif1": _lap_source_summary(session_laps[_tif1_source_mask(session_laps)] if not session_laps.empty else session_laps),
        "openf1": {
            **openf1_status(PROCESSED_DIR),
            "lap_source_summary": _lap_source_summary(openf1_laps),
            "lap_rows_in_canonical_table": int(len(openf1_laps)),
            "weather_observation_coverage_fraction": float(weather[["season", "round"]].drop_duplicates().shape[0] / max(race_calendar[["season", "round"]].drop_duplicates().shape[0], 1))
            if not weather.empty and {"season", "round"}.issubset(weather.columns) and {"season", "round"}.issubset(race_calendar.columns) else 0.0,
        },
        "model_artifact_present": (ARTIFACTS_DIR / "f1_models.joblib").exists(),
        "metrics_available": bool(metrics_payload.get("available")),
        "trained_feature_coverage": trained_coverage,
        "recommendations": recommendations,
        "interpretation": "Data-ready flags indicate collection coverage, not model accuracy. Pace-aware readiness requires historical practice and race lap coverage; a single current-weekend extract is not enough to validate generalisation.",
    }


@app.get("/api/events")
def events(
    season: int | None = Query(default=None, ge=2018, le=2100),
    upcoming_only: bool = Query(default=True),
) -> dict[str, Any]:
    items = _event_catalog()
    if season is not None:
        items = [event for event in items if event["season"] == season]
    if upcoming_only:
        items = [event for event in items if not event["has_race_result"] and _is_future_event(event)]
        items.sort(key=lambda x: x["start_utc"] or f"{x.get('date', '9999')}T23:59:59Z")
    else:
        items.sort(key=lambda x: (x["season"], x["round"]), reverse=True)
    return {"events": items, "count": len(items), "upcoming_only": upcoming_only}


@app.get("/api/circuits")
def circuit_catalog(
    first_season: int = Query(default=2018, ge=1950, le=2100),
    last_season: int | None = Query(default=None, ge=1950, le=2100),
) -> dict[str, Any]:
    """Return each distinct circuit used by the loaded F1 calendar."""
    races = load_tables()["races"].copy()
    if races.empty or "circuit_id" not in races:
        return {"circuits": [], "count": 0, "first_season": first_season, "last_season": last_season}
    races["season"] = _number(races.get("season", pd.Series(dtype=float)))
    races["round"] = _number(races.get("round", pd.Series(dtype=float)))
    races = races.dropna(subset=["season", "circuit_id"])
    races = races[races["season"] >= first_season]
    effective_last = int(last_season) if last_season is not None else (int(races["season"].max()) if not races.empty else first_season)
    races = races[races["season"] <= effective_last]
    rows: list[dict[str, Any]] = []
    for circuit_id, group in races.groupby(races["circuit_id"].astype(str), sort=True):
        ordered = group.sort_values(["season", "round"] if "round" in group else ["season"])
        latest = ordered.iloc[-1]
        race_names = list(dict.fromkeys(ordered.get("race_name", pd.Series(dtype=object)).dropna().astype(str).tolist()))
        rows.append({
            "circuit_id": circuit_id, "circuit_name": latest.get("circuit_name"),
            "locality": latest.get("locality"), "country": latest.get("country"),
            "first_used_season": int(pd.to_numeric(ordered["season"], errors="coerce").min()),
            "last_used_season": int(pd.to_numeric(ordered["season"], errors="coerce").max()),
            "event_count": int(len(ordered)), "grand_prix_names": race_names,
        })
    rows.sort(key=lambda row: (str(row.get("country") or ""), str(row.get("circuit_name") or "")))
    return {"circuits": _json_safe(rows), "count": len(rows), "first_season": first_season,
            "last_season": effective_last, "season_range": f"{first_season}–{effective_last}"}


@app.get("/api/events/next")
def next_event() -> dict[str, Any]:
    # Pass explicit None because `events()` uses FastAPI Query defaults.
    items = events(season=None, upcoming_only=True)["events"]
    return {"event": items[0] if items else None}


@app.get("/api/events/{season}/{round_number}")
def event_detail(season: int, round_number: int) -> dict[str, Any]:
    event = _event_or_404(season, round_number)
    t = load_tables()
    def subset(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame[(_number(frame["season"]) == season) & (_number(frame["round"]) == round_number)]
    practice_payload = _practice_rows_for_event(season, round_number, t)
    return {
        "event": event,
        "practice": practice_payload["rows"],
        "practice_sessions": practice_payload["sessions"],
        "qualifying": _records(subset(t["qualifying"]).sort_values("qualifying_position") if not subset(t["qualifying"]).empty and "qualifying_position" in subset(t["qualifying"]).columns else subset(t["qualifying"])),
        "race_results": _records(subset(t["race_results"]).sort_values("position_order") if not subset(t["race_results"]).empty and "position_order" in subset(t["race_results"]).columns else subset(t["race_results"])),
    }


@app.get("/api/standings")
def standings(
    season: int = Query(..., ge=2018, le=2100),
    round_number: int | None = Query(default=None, alias="round", ge=1, le=100),
) -> dict[str, Any]:
    t = load_tables()
    drivers = local_driver_standings(t["race_results"], t["sprints"], season, round_number)
    constructors = local_constructor_standings(t["race_results"], t["sprints"], season, round_number)
    return {"season": season, "through_round": round_number, "source": "local race/sprint results; official tie-break rules may differ", "drivers": _records(drivers), "constructors": _records(constructors)}


def _normalise_practice_session(value: Any) -> str | None:
    text = "" if value is None or pd.isna(value) else "".join(ch for ch in str(value).upper() if ch.isalnum())
    aliases = {
        "FP1": "FP1", "P1": "FP1", "PRACTICE1": "FP1", "FREEPRACTICE1": "FP1",
        "FP2": "FP2", "P2": "FP2", "PRACTICE2": "FP2", "FREEPRACTICE2": "FP2",
        "FP3": "FP3", "P3": "FP3", "PRACTICE3": "FP3", "FREEPRACTICE3": "FP3",
    }
    return aliases.get(text)


def _format_lap_seconds(value: Any) -> str | None:
    try:
        seconds = float(value)
        if not np.isfinite(seconds) or seconds <= 0:
            return None
        minutes = int(seconds // 60)
        return f"{minutes}:{seconds - minutes * 60:06.3f}"
    except (TypeError, ValueError):
        return None


def _practice_rows_for_event(
    season: int, round_number: int, tables: dict[str, pd.DataFrame] | None = None
) -> dict[str, Any]:
    """Return official classifications where available and lap-derived fallbacks otherwise."""
    t = tables or load_tables()
    base = t.get("practice", pd.DataFrame()).copy()
    official = pd.DataFrame()
    if not base.empty:
        mask = _number(base.get("season", pd.Series(index=base.index, dtype=float))).eq(season)
        mask &= _number(base.get("round", pd.Series(index=base.index, dtype=float))).eq(round_number)
        official = base[mask].copy()
    if not official.empty:
        if "session" not in official:
            official["session"] = None
        official["session"] = official["session"].map(_normalise_practice_session)
        official = official[official["session"].isin(["FP1", "FP2", "FP3"])].copy()
        if "time_seconds" not in official:
            official["time_seconds"] = np.nan
        official["time_seconds"] = pd.to_numeric(official["time_seconds"], errors="coerce").astype("float64")
        if "time" not in official:
            official["time"] = None
        missing_seconds = official["time_seconds"].isna()
        official.loc[missing_seconds, "time_seconds"] = official.loc[missing_seconds, "time"].map(parse_lap_time_seconds).astype("float64")
        if "source" not in official:
            official["source"] = "Jolpica-F1 classification"
        official.loc[official["source"].isna(), "source"] = "Jolpica-F1 classification"
    official_sessions = set(official.get("session", pd.Series(dtype=object)).dropna().astype(str))
    fallback_rows: list[dict[str, Any]] = []
    laps = t.get("session_laps", pd.DataFrame()).copy()
    if not laps.empty:
        mask = _number(laps.get("season", pd.Series(index=laps.index, dtype=float))).eq(season)
        mask &= _number(laps.get("round", pd.Series(index=laps.index, dtype=float))).eq(round_number)
        laps = laps[mask].copy()
        if "session" in laps:
            laps["session"] = laps["session"].map(_normalise_practice_session)
            laps = laps[laps["session"].isin(["FP1", "FP2", "FP3"])]
        else:
            laps = laps.iloc[0:0]
        for session_code in ("FP1", "FP2", "FP3"):
            # Build a lap-derived supplement for every session, then prefer the
            # official classification per driver during the final deduplication.
            session_laps = laps[laps["session"].eq(session_code)].copy()
            if session_laps.empty:
                continue
            time_col = "lap_time_seconds" if "lap_time_seconds" in session_laps else "lap_duration"
            session_laps["_time_seconds"] = pd.to_numeric(session_laps.get(time_col), errors="coerce")
            if "is_accurate" in session_laps:
                accurate = session_laps["is_accurate"].astype(str).str.lower().isin({"true", "1", "yes"})
                # OpenF1 only marks broad plausibility; FastF1 flags are more strict.
                session_laps = session_laps[accurate | session_laps["is_accurate"].isna()]
            session_laps = session_laps.dropna(subset=["_time_seconds"])
            session_laps = session_laps[session_laps["_time_seconds"].between(40, 200)]
            if session_laps.empty:
                continue
            if "driver_code" not in session_laps and "driver_number" not in session_laps:
                continue
            codes = session_laps.get("driver_code", pd.Series(pd.NA, index=session_laps.index)).astype("string").str.upper()
            codes = codes.replace({"NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA, "": pd.NA})
            numbers = session_laps.get("driver_number", pd.Series(pd.NA, index=session_laps.index)).astype("string")
            session_laps["_driver_key"] = codes.fillna("NUMBER:" + numbers.fillna("unknown"))
            grouped = session_laps.groupby("_driver_key", dropna=True)
            best = grouped.agg(
                best_time_seconds=("_time_seconds", "min"),
                laps_completed=("_time_seconds", "count"),
            ).reset_index().sort_values("best_time_seconds")
            best["position"] = np.arange(1, len(best) + 1)
            roster = t.get("race_results", pd.DataFrame()).copy()
            if not roster.empty:
                roster = roster[(_number(roster.get("season")) == season) & (_number(roster.get("round")) == round_number)]
            code_to_row = {}
            if not roster.empty and "driver_code" in roster:
                for _, rr in roster.dropna(subset=["driver_code"]).drop_duplicates("driver_code").iterrows():
                    code_to_row[str(rr["driver_code"]).upper()] = rr
            official_by_code = {}
            if not official.empty and "driver_code" in official:
                for _, official_row in official.dropna(subset=["driver_code"]).drop_duplicates("driver_code").iterrows():
                    official_by_code[str(official_row["driver_code"]).upper()] = official_row
            constructor_by_code = {}
            driver_name_by_code = {}
            if "constructor_name" in session_laps:
                constructor_by_code = session_laps.groupby("_driver_key")["constructor_name"].agg(lambda x: x.dropna().iloc[-1] if x.dropna().size else None).to_dict()
            if "driver_name" in session_laps:
                driver_name_by_code = session_laps.groupby("_driver_key")["driver_name"].agg(lambda x: x.dropna().iloc[-1] if x.dropna().size else None).to_dict()
            driver_table = t.get("openf1_drivers", pd.DataFrame()).copy()
            if not driver_table.empty and "driver_number" in driver_table:
                if "name_acronym" in driver_table:
                    driver_name_by_code.update({str(row.get("name_acronym")).upper(): row.get("full_name") or row.get("name_acronym") for _, row in driver_table.drop_duplicates("name_acronym").iterrows() if pd.notna(row.get("name_acronym"))})
                if "full_name" in driver_table:
                    driver_name_by_code.update({f"NUMBER:{row.get('driver_number')}".upper(): row.get("full_name") for _, row in driver_table.drop_duplicates("driver_number").iterrows() if pd.notna(row.get("driver_number"))})
            source_counts = session_laps.get("data_source", pd.Series("lap archive", index=session_laps.index)).fillna("lap archive").astype(str).str.lower().value_counts()
            lap_source = "FastF1" if source_counts.get("fastf1", 0) >= source_counts.get("openf1", 0) else "OpenF1"
            for _, item in best.iterrows():
                driver_key = str(item["_driver_key"]).upper()
                is_number_key = driver_key.startswith("NUMBER:")
                code = None if is_number_key else driver_key
                race_row = code_to_row.get(code) if code else None
                official_row = official_by_code.get(code) if code else None
                fallback_rows.append({
                    "season": season, "round": round_number, "session": session_code,
                    "position": int(item["position"]),
                    "driver_id": (race_row.get("driver_id") if race_row is not None else None) or (official_row.get("driver_id") if official_row is not None else None),
                    "driver_code": code,
                    "driver_name": (race_row.get("driver_name") if race_row is not None else None) or (official_row.get("driver_name") if official_row is not None else None) or driver_name_by_code.get(driver_key) or (f"#{driver_key.split(':', 1)[1]}" if is_number_key else code),
                    "constructor_id": (race_row.get("constructor_id") if race_row is not None else None) or (official_row.get("constructor_id") if official_row is not None else None),
                    "constructor_name": (race_row.get("constructor_name") if race_row is not None else None) or (official_row.get("constructor_name") if official_row is not None else None) or constructor_by_code.get(driver_key),
                    "time_seconds": float(item["best_time_seconds"]),
                    "time": _format_lap_seconds(item["best_time_seconds"]),
                    "laps": int(item["laps_completed"]),
                    "source": f"{lap_source} lap-derived classification",
                })
    combined = pd.concat([official, pd.DataFrame(fallback_rows)], ignore_index=True, sort=False) if not official.empty or fallback_rows else pd.DataFrame()
    if not combined.empty:
        combined["_driver_key"] = combined.get("driver_id", pd.Series(index=combined.index, dtype=object)).astype("string").str.strip()
        combined["_driver_key"] = combined["_driver_key"].fillna(combined.get("driver_name", pd.Series(index=combined.index, dtype=object)).astype("string").str.strip().str.casefold())
        combined["_driver_key"] = combined["_driver_key"].fillna(combined.get("driver_code", pd.Series(index=combined.index, dtype=object)).astype("string").str.upper())
        # Official Jolpica classifications take precedence; FastF1/OpenF1
        # fastest-lap reconstructions supplement missing driver/session rows.
        source_text = combined.get("source", pd.Series(index=combined.index, dtype=object)).astype(str)
        combined["_official_priority"] = source_text.str.contains("Jolpica", case=False, regex=False).astype(int) * 2
        combined["_official_priority"] += source_text.str.contains("FastF1", case=False, regex=False).astype(int)
        combined = combined.sort_values("_official_priority").drop_duplicates(["session", "_driver_key"], keep="last")
        combined["session_order"] = combined["session"].map({"FP1": 1, "FP2": 2, "FP3": 3})
        combined["position"] = pd.to_numeric(combined.get("position"), errors="coerce")
        combined["time_seconds"] = pd.to_numeric(combined.get("time_seconds"), errors="coerce")
        combined = combined.sort_values(["session_order", "position", "driver_name"], na_position="last")
        combined = combined.drop(columns=["session_order", "_driver_key", "_official_priority"], errors="ignore")
    session_summary = []
    for session_code in ("FP1", "FP2", "FP3"):
        rows = combined[combined["session"].eq(session_code)] if not combined.empty else pd.DataFrame()
        sources = rows.get("source", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
        session_summary.append({"session": session_code, "rows": int(len(rows)), "available": not rows.empty, "sources": sources})
    return {
        "rows": _records(combined),
        "sessions": session_summary,
        "available": not combined.empty,
        "detail": None if not combined.empty else "No official practice classifications or usable practice laps are present for this Grand Prix.",
        "row_sources": sorted(set(combined.get("source", pd.Series(dtype=object)).dropna().astype(str).tolist())) if not combined.empty else [],
    }


@app.get("/api/practice")
def practice(
    season: int = Query(..., ge=2018, le=2100),
    round_number: int = Query(..., alias="round", ge=1, le=100),
) -> dict[str, Any]:
    event = _event_or_404(season, round_number)
    payload = _practice_rows_for_event(season, round_number)
    return {"event": event, **payload}


@app.get("/api/practice/pace")
def practice_pace(
    season: int = Query(..., ge=2018, le=2100),
    round_number: int = Query(..., alias="round", ge=1, le=100),
    as_of_session: Literal["FP1", "FP2", "FP3"] = Query(default="FP3"),
) -> dict[str, Any]:
    """Return lap-derived one-lap and long-run practice pace separately."""
    event = _event_or_404(season, round_number)
    frame = _event_frame(season, round_number, "pole", as_of_session)
    feature_columns = [
        "fp_best_gap_pct", "fp_latest_best_gap_pct", "fp_best_soft_gap_pct",
        "fp_latest_longrun_gap_pct", "fp_mean_longrun_gap_pct", "fp_valid_laps",
        "fp_longrun_laps", "fp_longrun_stints", "fp_sessions_completed",
        "fp1_best_gap_pct", "fp2_best_gap_pct", "fp3_best_gap_pct",
        "fp1_soft_gap_pct", "fp2_soft_gap_pct", "fp3_soft_gap_pct",
        "fp1_longrun_gap_pct", "fp2_longrun_gap_pct", "fp3_longrun_gap_pct",
    ]
    present_columns = [col for col in feature_columns if col in frame.columns]
    available = bool(present_columns and frame[present_columns].notna().any().any())
    constructor_frame = summarise_constructor_pace(frame, frame[[c for c in ("driver_code", "constructor_id", "constructor_name") if c in frame.columns]])
    return {
        "event": event,
        "as_of_session": as_of_session,
        "available": available,
        "data_source": _event_lap_source_label(season, round_number, as_of_session) if available else None,
        "interpretation": (
            "Session-normalized one-lap gaps and robust long-run stint medians from available FastF1/OpenF1 laps. These are pace indicators, "
            "not pure car-speed measurements; fuel load, engine mode, traffic and tyre programme remain confounders."
        ),
        "drivers": _records(frame[[c for c in ["driver_name", "driver_code", "constructor_name", *present_columns] if c in frame.columns]]),
        "constructors": _records(constructor_frame),
    }


@app.get("/api/intelligence/constructor-pace")
def constructor_pace_intelligence(
    season: int = Query(..., ge=2018, le=2100),
    round_number: int = Query(..., alias="round", ge=1, le=100),
    as_of_session: Literal["FP1", "FP2", "FP3"] = Query(default="FP3"),
) -> dict[str, Any]:
    """Expose six evidence-backed constructor intelligence pillars and coverage."""
    event = _event_or_404(season, round_number)
    frame = _event_frame(season, round_number, "pole", as_of_session)
    metric_groups = {
        "qualifying_pace": {
            "historical_qualifying_gap_pct": "constructor_quali_gap_recent3_pct",
            "season_qualifying_gap_pct": "constructor_quali_gap_season_prev_pct",
            "same_circuit_qualifying_gap_pct": "constructor_circuit_quali_gap_prev_avg_pct",
            "weekend_best_one_lap_gap_pct": "constructor_practice_best_gap_pct",
            "latest_session_one_lap_gap_pct": "constructor_practice_latest_best_gap_pct",
            "weekend_soft_tyre_gap_pct": "constructor_practice_soft_gap_pct",
        },
        "race_pace": {
            "historical_race_longrun_gap_pct": "constructor_race_longrun_gap_recent3_pct",
            "recent_race_fastest_lap_gap_pct": "constructor_fastest_lap_gap_recent3_pct",
            "current_weekend_longrun_gap_pct": "constructor_practice_longrun_gap_pct",
        },
        "long_run_pace": {
            "current_longrun_laps": "constructor_practice_longrun_laps",
            "current_longrun_stints": "constructor_practice_longrun_stints",
            "historical_race_valid_laps_per_event": "constructor_race_valid_laps_prev_avg",
            "historical_stint_time_slope_sec_per_lap": "constructor_race_stint_slope_prev_avg_sec_per_lap",
        },
        "tyre_behaviour": {
            "overall_tyre_age_slope_sec_per_lap": "constructor_practice_tyre_deg_sec_per_lap",
            "soft_tyre_age_slope_sec_per_lap": "constructor_practice_soft_deg_sec_per_lap",
            "medium_tyre_age_slope_sec_per_lap": "constructor_practice_medium_deg_sec_per_lap",
            "hard_tyre_age_slope_sec_per_lap": "constructor_practice_hard_deg_sec_per_lap",
            "observed_compounds_with_slope": "constructor_practice_compound_count",
            "recent_mean_pit_duration_sec": "constructor_pit_stop_mean_duration_recent5_sec",
            "recent_median_pit_duration_sec": "constructor_pit_stop_median_duration_recent5_sec",
            "recent_stops_per_covered_race": "constructor_pit_stop_count_recent5",
        },
        "reliability": {
            "recent_five_gp_dnf_rate": "constructor_dnf_rate_recent5",
            "recent_five_gp_mechanical_dnf_rate": "constructor_mechanical_dnf_rate_recent5",
            "driver_recent_five_gp_dnf_rate": "driver_dnf_rate_recent5",
            "same_circuit_dnf_rate": "constructor_circuit_dnf_rate_prev",
        },
        "circuit_suitability": {
            "same_circuit_average_finish": "constructor_circuit_avg_finish_prev",
            "same_circuit_average_positions_gained": "constructor_circuit_positions_gained_prev",
            "same_circuit_grid_win_conversion": "circuit_grid_win_rate_prev",
            "same_circuit_grid_podium_conversion": "circuit_grid_podium_rate_prev",
            "historical_safety_car_event_rate": "circuit_sc_event_rate_prev",
            "historical_virtual_safety_car_event_rate": "circuit_vsc_event_rate_prev",
        },
    }
    rows: list[dict[str, Any]] = []
    for constructor_id, group in frame.groupby("constructor_id", dropna=False):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "constructor_id": constructor_id,
            "constructor_name": first.get("constructor_name"),
            "drivers": group.get("driver_name", pd.Series(dtype=object)).dropna().tolist(),
            "pillars": {},
        }
        all_values: dict[str, Any] = {}
        for pillar, metrics in metric_groups.items():
            pillar_values: dict[str, Any] = {}
            for label, source in metrics.items():
                value = pd.to_numeric(group[source], errors="coerce").mean() if source in group else np.nan
                pillar_values[label] = None if pd.isna(value) else float(value)
                all_values[label] = pillar_values[label]
            row["pillars"][pillar] = {
                "available": any(value is not None for value in pillar_values.values()),
                "metrics": pillar_values,
            }
        row.update(all_values)
        rows.append(row)
    result = pd.DataFrame(rows)
    pillar_coverage = {}
    for pillar in metric_groups:
        pillar_coverage[pillar] = sum(
            bool(row["pillars"][pillar]["available"]) for row in rows
        )
    all_metric_columns = [column for group in metric_groups.values() for column in group]
    available_metric_columns = [
        column for column in all_metric_columns
        if column in result.columns and pd.to_numeric(result[column], errors="coerce").notna().any()
    ]
    return {
        "event": event,
        "as_of_session": as_of_session,
        "available": any(pillar_coverage.values()),
        "constructor_count": len(rows),
        "pillar_coverage": pillar_coverage,
        "pillars": list(metric_groups.keys()),
        "available_metric_columns": available_metric_columns,
        "note": "Pace gap (%) compares a constructor with the fastest observed constructor; lower is faster. Long-run pace uses representative longer stints. Tyre-age slope is the observed lap-time change per tyre-age lap, not a pure physical degradation rate. Pit duration is an average stop duration. DNF rate is the share of recent starts not classified as finishes. Same-circuit finish and positions gained summarise prior visits. SC/VSC rates are historical event frequencies, not predictions of deployment.",
        "constructors": _records(result),
    }


@app.get("/api/intelligence/driver-performance")
def driver_performance_intelligence(
    season: int = Query(..., ge=2018, le=2100),
    round_number: int = Query(..., alias="round", ge=1, le=100),
    as_of_session: Literal["FP1", "FP2", "FP3"] = Query(default="FP3"),
) -> dict[str, Any]:
    """Expose driver pace indicators and teammate-relative qualifying history."""
    event = _event_or_404(season, round_number)
    frame = _event_frame(season, round_number, "pole", as_of_session)
    fields = [
        "driver_quali_gap_prev_avg_pct", "driver_quali_gap_recent3_pct",
        "driver_teammate_quali_delta_prev_pct", "driver_race_longrun_gap_prev_avg_pct",
        "driver_race_longrun_gap_recent3_pct", "driver_race_stint_slope_prev_avg_sec_per_lap",
        "fp_best_gap_pct", "fp_latest_best_gap_pct", "fp_best_soft_gap_pct",
        "fp_latest_longrun_gap_pct", "fp_valid_laps", "fp_longrun_laps",
    ]
    selected = [c for c in ["driver_name", "driver_code", "constructor_name", *fields] if c in frame.columns]
    rows = frame[selected].copy()
    available = bool([c for c in fields if c in rows and rows[c].notna().any()])
    return {"event": event, "as_of_session": as_of_session, "available": available, "drivers": _records(rows),
            "note": "Teammate qualifying delta is relative to the constructor's mean qualifying gap; negative means faster than the teammate baseline."}


@app.get("/api/history/{circuit_id}")
def circuit_history(
    circuit_id: str,
    season: int = Query(2018, ge=1950, le=2100),
    last_season: int | None = Query(default=None, ge=1950, le=2100),
) -> dict[str, Any]:
    """Return every Grand Prix held at a circuit, including alternate GP titles."""
    t = load_tables()
    if t["races"].empty or t["race_results"].empty:
        return {"circuit_id": circuit_id, "rows": [], "available": False, "season_range": None}
    schedule = t["races"].copy()
    schedule["season"] = _number(schedule.get("season", pd.Series(dtype=float)))
    effective_last = int(last_season) if last_season is not None else (int(schedule["season"].max()) if schedule["season"].notna().any() else season)
    schedule = schedule[(schedule["season"] >= season) & (schedule["season"] <= effective_last)]
    history = build_gp_history(schedule, t["race_results"], t["qualifying"], str(circuit_id), first_season=season, practice_results=t["practice"])
    circuit_rows = schedule[schedule["circuit_id"].astype(str) == str(circuit_id)]
    circuit_name = str(circuit_rows.iloc[-1].get("circuit_name") or circuit_id) if not circuit_rows.empty else circuit_id
    return {"circuit_id": circuit_id, "circuit_name": circuit_name, "race_name": circuit_name,
            "first_season": season, "last_season": effective_last,
            "season_range": f"{season}–{effective_last}", "rows": _records(history), "available": not history.empty}


@app.get("/api/model/metrics")
def model_metrics() -> dict[str, Any]:
    return _metrics()


@app.post("/api/predictions/pole")
def pole_prediction(request: EventPredictionRequest) -> dict[str, Any]:
    event = _event_or_404(request.season, request.round)
    if event["has_race_result"]:
        raise HTTPException(status_code=409, detail="This Grand Prix has already been completed. Historical events belong in backtesting, not live prediction.")
    if event["has_qualifying_result"]:
        raise HTTPException(status_code=409, detail="Qualifying results are already available. Pole prediction is closed for this event.")
    if not event["start_utc"] or datetime.fromisoformat(event["start_utc"].replace("Z", "+00:00")) <= datetime.now(timezone.utc):
        raise HTTPException(status_code=409, detail="The race start has passed or the event timing is unavailable; live pole prediction is disabled.")
    bundle = _load_model_bundle()
    frame = _event_frame(request.season, request.round, "pole", request.as_of_session)
    if frame.empty:
        raise HTTPException(status_code=422, detail="No driver candidates could be assembled for this event.")
    classification_practice_available = bool(
        frame.get("fp_session_count", pd.Series(index=frame.index, dtype=float)).notna().any()
    )
    lap_level_practice_available = bool(
        frame.get("fp_valid_laps", pd.Series(index=frame.index, dtype=float)).fillna(0).gt(0).any()
    )
    practice_payload = _practice_rows_for_event(request.season, request.round)
    available_practice_sessions = {
        item["session"] for item in practice_payload.get("sessions", []) if item.get("available")
    }
    required_sessions = {
        "FP1": {"FP1"},
        "FP2": {"FP1", "FP2"},
        "FP3": {"FP1", "FP2", "FP3"},
    }
    if request.as_of_session != "PRE_FP":
        missing_sessions = sorted(required_sessions[request.as_of_session] - available_practice_sessions)
        if missing_sessions:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"A {request.as_of_session} pole forecast requires practice data for "
                    f"{', '.join(sorted(required_sessions[request.as_of_session]))}. "
                    f"Missing: {', '.join(missing_sessions)}. Choose an earlier completed FP cutoff or collect the missing session data."
                ),
            )
        if not classification_practice_available and not lap_level_practice_available:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"A {request.as_of_session} pole forecast is unavailable because no eligible practice "
                    "classification or lap-level data have been ingested for this Grand Prix."
                ),
            )
    try:
        prediction = predict_pole(bundle, frame, request.as_of_session)
    except ModelBundleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    classification_practice = bool(frame.get("fp_session_count", pd.Series(dtype=float)).notna().any())
    lap_level_data = bool(frame.get("fp_valid_laps", pd.Series(dtype=float)).fillna(0).gt(0).any())
    raw_laps = load_tables().get("session_laps", pd.DataFrame())
    if not raw_laps.empty:
        lap_mask = pd.to_numeric(raw_laps.get("season"), errors="coerce").eq(request.season) & pd.to_numeric(raw_laps.get("round"), errors="coerce").eq(request.round)
        eligible_sessions = {"PRE_FP": [], "FP1": ["FP1"], "FP2": ["FP1", "FP2"], "FP3": ["FP1", "FP2", "FP3"]}[request.as_of_session]
        lap_mask &= raw_laps.get("session", pd.Series(index=raw_laps.index, dtype=object)).astype(str).str.upper().isin(eligible_sessions)
        event_laps = raw_laps[lap_mask]
        source_col = event_laps.get("data_source", pd.Series("fastf1", index=event_laps.index)).fillna("fastf1").astype(str).str.lower()
        is_tif1 = source_col.str.startswith("tif1")
        is_direct_fastf1 = source_col.eq("fastf1")
        lap_source_counts = {
            "fastf1": int((is_direct_fastf1 | is_tif1).sum()),
            "tif1": int(is_tif1.sum()),
            "openf1": int(source_col.eq("openf1").sum()),
        }
    else:
        lap_source_counts = {"fastf1": 0, "tif1": 0, "openf1": 0}
    pole_task_key = "pole_pre_fp" if request.as_of_session == "PRE_FP" else f"pole_{request.as_of_session.lower()}"
    metadata_tasks = bundle.get("metadata", {}).get("tasks", {})
    if pole_task_key not in metadata_tasks and request.as_of_session == "FP3":
        pole_task_key = "pole"
    constructor_pace_coverage = _feature_coverage(bundle, pole_task_key, "constructor_quali_gap_recent3_pct")
    weekend_lap_coverage = _feature_coverage(bundle, pole_task_key, "fp_best_gap_pct")
    model_uses_pace = constructor_pace_coverage > 0.05
    model_uses_lap_level_data = weekend_lap_coverage > 0.05
    warnings = []
    if request.as_of_session != "PRE_FP" and not model_uses_pace:
        warnings.append("The loaded model artifact predates the new constructor-pace features. The API calculates them, but this model does not use them; retrain with python -m src.models.train_models.")
    if request.as_of_session != "PRE_FP" and lap_level_data and not model_uses_lap_level_data:
        warnings.append("Lap-level data are present for this weekend, but the loaded model was not trained with sufficient historical lap coverage to learn from them.")
    if request.as_of_session == "PRE_FP":
        warnings.append("Pre-free-practice forecast: current-weekend practice and observed session weather are deliberately excluded. The model uses prior driver form, constructor season strength, previous Grand Prix performance, teammate-relative pace and circuit history; historical pole records receive the smallest weight.")
    elif not lap_level_data:
        warnings.append("No TIF1/FastF1/OpenF1 lap-level data are available by the selected session cutoff. Jolpica practice classifications are a weaker substitute for detailed qualifying simulations and long-run pace.")
    if lap_source_counts["openf1"] and not lap_source_counts["fastf1"]:
        warnings.append("This event uses OpenF1 lap timing as a fallback. It does not include the same detailed FastF1 lap-quality and track-status flags, so pace filtering is less strict.")
    if request.as_of_session != "PRE_FP":
        warnings.append("Practice lap pace is not fully adjusted for fuel load, engine mode, traffic, tyre programme or track evolution.")
    return {
        "event": event,
        "prediction_type": "pre_free_practice_historical_pole_prediction" if request.as_of_session == "PRE_FP" else "priority_weighted_weekend_aware_pole_prediction",
        "pole_model_key": pole_task_key,
        "is_legacy_baseline": False,
        "prediction_basis": ("Dedicated pre-free-practice model + historical-only expert priorities; score shares are not calibrated probabilities" if request.as_of_session == "PRE_FP" else "50% trained model field ranking + 50% priority-weighted evidence score; score shares are not calibrated probabilities"),
        "priority_weights": ({"constructor_season_strength": 0.35, "driver_recent_form": 0.25, "previous_grand_prix_form": 0.20, "teammate_relative": 0.10, "circuit_suitability": 0.07, "historical_pole_record": 0.03} if request.as_of_session == "PRE_FP" else {"weekend_pace": 0.28, "constructor_strength": 0.23, "driver_form": 0.16, "teammate_relative": 0.11, "weather_context": 0.09, "circuit_suitability": 0.08, "historical_pole_record": 0.05}),
        "practice_data_present": classification_practice or lap_level_data,
        "available_practice_sessions": sorted(available_practice_sessions),
        "practice_session_coverage": practice_payload.get("sessions", []),
        "practice_as_of_session": request.as_of_session if request.as_of_session != "PRE_FP" else None,
        "forecast_stage": "before_free_practice" if request.as_of_session == "PRE_FP" else "after_free_practice",
        "candidate_roster_source": frame["candidate_roster_source"].iloc[0] if "candidate_roster_source" in frame and not frame.empty else "unknown",
        "lap_level_features_present": lap_level_data,
        "lap_rows_by_source": lap_source_counts,
        "fastf1_lap_features_present": lap_source_counts["fastf1"] > 0,
        "openf1_lap_features_present": lap_source_counts["openf1"] > 0,
        "openf1_weather_features_present": bool(frame.get("weather_air_temp_c", pd.Series(dtype=float)).notna().any()),
        "model_uses_constructor_pace_features": model_uses_pace,
        "model_uses_lap_level_features": model_uses_lap_level_data,
        "model_uses_session_lap_features": model_uses_lap_level_data,
        "warnings": warnings,
        "weather_feature_source": "Observed OpenF1 weather through the selected practice cutoff; pole forecasts do not use future race-weather forecasts.",
        "probability_note": "Relative score shares are not calibrated probabilities. Weather is supplied to the trained model as context; the transparent hierarchy score ranks measurable pace/form evidence.",
        "predictions": _records(prediction),
    }


@app.post("/api/predictions/race")
def race_prediction(request: EventPredictionRequest) -> dict[str, Any]:
    event = _event_or_404(request.season, request.round)
    if event["has_race_result"]:
        raise HTTPException(status_code=409, detail="This Grand Prix has already been completed. Historical events belong in backtesting, not live prediction.")
    if not event["start_utc"] or datetime.fromisoformat(event["start_utc"].replace("Z", "+00:00")) <= datetime.now(timezone.utc):
        raise HTTPException(status_code=409, detail="The race start has passed or the event timing is unavailable; live race prediction is disabled.")

    is_pre_quali = request.race_stage == "before_qualifying"
    if is_pre_quali and event["has_qualifying_result"]:
        raise HTTPException(status_code=409, detail="Before-qualifying race forecasts are closed because qualifying results are already available. Select After Qualifying instead.")
    if not is_pre_quali and not event["has_qualifying_result"]:
        raise HTTPException(status_code=409, detail="After-qualifying race prediction is only available after qualifying results. Select Before Qualifying to use the Pole Predictor's projected grid.")

    bundle = _load_model_bundle()
    pole_grid: pd.DataFrame | None = None
    if is_pre_quali:
        # Only FP1/FP2/FP3 evidence is eligible before qualifying. The predicted
        # qualifying order from the Pole Predictor becomes the projected grid.
        pole_frame = _event_frame(request.season, request.round, "pole", request.as_of_session)
        classification_practice = bool(pole_frame.get("fp_session_count", pd.Series(index=pole_frame.index, dtype=float)).notna().any())
        lap_practice = bool(pole_frame.get("fp_valid_laps", pd.Series(index=pole_frame.index, dtype=float)).fillna(0).gt(0).any())
        if request.as_of_session != "PRE_FP":
            practice_payload = _practice_rows_for_event(request.season, request.round)
            available_sessions = {item["session"] for item in practice_payload.get("sessions", []) if item.get("available")}
            required_sessions = {"FP1": {"FP1"}, "FP2": {"FP1", "FP2"}, "FP3": {"FP1", "FP2", "FP3"}}[request.as_of_session]
            missing_sessions = sorted(required_sessions - available_sessions)
            if missing_sessions:
                raise HTTPException(status_code=409, detail=(
                    f"A pre-qualifying race forecast at {request.as_of_session} requires "
                    f"{', '.join(sorted(required_sessions))}; missing {', '.join(missing_sessions)}. "
                    "Choose an earlier completed practice cutoff or collect the missing sessions."
                ))
            if not classification_practice and not lap_practice:
                raise HTTPException(status_code=409, detail=(
                    f"A pre-qualifying race forecast requires practice data through {request.as_of_session}. "
                    "Run the Pole Predictor after the selected practice session has been ingested, or choose the pre-free-practice stage."
                ))
        try:
            pole_grid = predict_pole(bundle, pole_frame, request.as_of_session)
        except ModelBundleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if pole_grid.empty:
            raise HTTPException(status_code=422, detail="The Pole Predictor did not return a projected grid for this event.")
        pole_grid = pole_grid.copy().reset_index(drop=True)
        pole_grid["predicted_grid_position"] = np.arange(1, len(pole_grid) + 1)
        race_frame = _event_frame(
            request.season, request.round, "race", request.as_of_session,
            weather_as_of_session=request.as_of_session,
        )

        def join_key(row: pd.Series) -> str:
            return "|".join(str(row.get(col) or "").strip().casefold() for col in ("driver_name", "constructor_name"))

        grid_map = {join_key(row): int(row["predicted_grid_position"]) for _, row in pole_grid.iterrows()}
        mapped_grid = race_frame.apply(lambda row: grid_map.get(join_key(row)), axis=1)
        if mapped_grid.isna().any():
            unmatched = race_frame.loc[mapped_grid.isna(), "driver_name"].astype(str).tolist()
            raise HTTPException(status_code=422, detail="The Pole Predictor grid did not cover every race entrant: " + ", ".join(unmatched[:8]))
        race_frame["grid"] = mapped_grid.astype(float)
        # Explicitly keep qualifying-time fields empty in this mode.
        race_frame["qualifying_position"] = np.nan
        race_frame["best_qualifying_time_seconds"] = np.nan
        race_frame["starting_grid_source"] = "pole_predictor_predicted_grid"
        grid_source = "pole_predictor_predicted_grid"
    else:
        race_frame = _event_frame(request.season, request.round, "race")
        grid_source = "official_starting_grid_or_qualifying_position"

    frame = race_frame
    if frame.empty:
        raise HTTPException(status_code=422, detail="No driver candidates could be assembled for this event.")
    prediction = predict_race(bundle, frame)
    race_pace_coverage = _feature_coverage(bundle, "winner", "constructor_race_longrun_gap_recent3_pct")
    current_longrun_coverage = _feature_coverage(bundle, "winner", "constructor_practice_longrun_gap_pct")
    reliability_coverage = _feature_coverage(bundle, "winner", "constructor_dnf_rate_recent5")
    pace_model = race_pace_coverage > 0.05 and current_longrun_coverage > 0.05
    warnings = []
    if not pace_model:
        warnings.append("The loaded model was not trained with adequate historical FastF1/OpenF1 race-lap or current-weekend long-run coverage. Race-pace components may be missing or weakly learned; collect FP and R-session laps before retraining.")
    if reliability_coverage <= 0.05:
        warnings.append("Historical constructor reliability features have low training coverage. DNF probabilities remain more uncertain until status history is checked and the model is retrained.")
    forecast_feature_columns = [
        "race_forecast_air_temp_c", "race_forecast_humidity_pct", "race_forecast_precipitation_mm",
        "race_forecast_wind_speed_kmh", "race_forecast_wind_direction_deg", "race_forecast_is_wet",
    ]
    forecast_present = any(frame.get(column, pd.Series(dtype=float)).notna().any() for column in forecast_feature_columns)
    forecast_provider_available = bool(frame.get("race_forecast_source", pd.Series(dtype=object)).notna().any())
    forecast_model_eligible = bool(frame.get("race_forecast_model_eligible", pd.Series(dtype=bool)).fillna(False).any())
    if not forecast_provider_available:
        warnings.append("No Open-Meteo forecast is currently available for the scheduled race start; the model will use its other signals.")
    elif not forecast_model_eligible:
        warnings.append("A race-start weather forecast is available, but its lead time is outside the 12-36 hour window represented by the historical one-day-ahead training forecasts. It is withheld from the ML features.")
    grid_sources = frame.get("starting_grid_source", pd.Series(grid_source, index=frame.index)).fillna(grid_source)
    grid_source_summary = {str(k): int(v) for k, v in grid_sources.value_counts().items()}
    if not is_pre_quali and grid_sources.eq("qualifying_position_proxy").any():
        warnings.append("The official starting grid was not available for every driver; qualifying position is used as the grid proxy for unmatched drivers. Grid penalties may therefore be missing.")
    if not frame.get("weather_air_temp_c", pd.Series(dtype=float)).notna().any():
        warnings.append("No OpenF1 observed weather is available through the selected prediction cutoff; the model will rely on historical/other features for session weather context.")
    neutralisation_history_present = bool(
        frame.get("circuit_sc_event_rate_prev", pd.Series(dtype=float)).notna().any()
        or frame.get("circuit_vsc_event_rate_prev", pd.Series(dtype=float)).notna().any()
    )
    if not neutralisation_history_present:
        warnings.append("No prior OpenF1 race-control coverage is available for this event or a usable global prior. Safety Car/VSC scenarios cannot yet influence this forecast.")
    warnings.append("Weather forecasts can change before lights out. Long-run and tyre-age slopes remain proxies for fuel, traffic, track evolution and run plans; Safety Car/VSC history changes scenario uncertainty but does not predict deployment timing.")
    pole_grid_records = []
    if is_pre_quali and pole_grid is not None:
        pole_grid_records = _records(pole_grid[[c for c in ["predicted_grid_position", "driver_name", "constructor_name", "pole_score_share", "pole_probability"] if c in pole_grid]])
    return {
        "event": event,
        "race_stage": request.race_stage,
        "prediction_type": "pre_qualifying_pole_grid_linked_race_prediction" if is_pre_quali else "hybrid_race_pace_aware_prediction",
        "is_legacy_baseline": False,
        "is_experimental_partial_coverage": not pace_model,
        "model_uses_historical_race_pace": pace_model,
        "model_uses_current_practice_longruns": current_longrun_coverage > 0.05,
        "model_uses_reliability_history": reliability_coverage > 0.05,
        "race_priority_weights": {"constructor_performance": 0.20, "weekend_race_pace": 0.18, "grid_circuit_advantage": 0.18, "driver_form": 0.12, "tyre_strategy": 0.12, "reliability": 0.07, "circuit_characteristics": 0.05, "weather_context": 0.08},
        "candidate_roster_source": frame["candidate_roster_source"].iloc[0] if "candidate_roster_source" in frame and not frame.empty else "unknown",
        "starting_grid_source": grid_source,
        "starting_grid_source_coverage": grid_source_summary,
        "pole_predictor_grid": pole_grid_records,
        "practice_cutoff": (request.as_of_session if request.as_of_session != "PRE_FP" else "before_free_practice") if is_pre_quali else None,
        "warnings": warnings,
        "weather_feature_source": "OpenF1 observed weather through FP3 for pre-qualifying forecasts or through qualifying for post-qualifying forecasts, plus an Open-Meteo race-start forecast when eligible.",
        "circuit_conversion_features_present": bool(frame.get("circuit_pole_to_win_rate_prev", pd.Series(dtype=float)).notna().any()),
        "neutralisation_history_present": neutralisation_history_present,
        "pit_stop_history_present": bool(frame.get("constructor_pit_stop_mean_duration_recent5_sec", pd.Series(dtype=float)).notna().any()),
        "race_weather_forecast_present": forecast_present,
        "race_weather_forecast_provider_available": forecast_provider_available,
        "race_weather_forecast_model_eligible": forecast_model_eligible,
        "race_weather_forecast": _records(frame[[c for c in ["race_forecast_air_temp_c", "race_forecast_humidity_pct", "race_forecast_precip_probability_pct", "race_forecast_precipitation_mm", "race_forecast_wind_speed_kmh", "race_forecast_wind_direction_deg", "race_forecast_weather_code", "race_forecast_source", "race_forecast_lead_hours", "race_forecast_valid_time", "race_forecast_model_eligible"] if c in frame]].head(1)),
        "probability_note": "Win and podium probabilities are field-coherent Monte Carlo estimates combining model rankings, expert-priority scores and DNF risk. They are not externally calibrated; assess calibration on unseen seasons before treating them as betting-grade probabilities.",
        "predictions": _records(prediction),
    }

@app.post("/api/dev/reload-data")
def reload_local_data() -> dict[str, str]:
    """Clear in-process CSV cache after local data ingestion has been rerun."""
    load_tables.cache_clear()
    return {"status": "reloaded", "message": "Local CSV tables will be read again on the next API request."}


if STATIC_DIR.exists() and (STATIC_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
if STATIC_DIR.exists():
    # check_dir=False lets the API be started before the frontend build; the
    # model route becomes live as soon as Vite writes public/models into app/static.
    app.mount("/models", StaticFiles(directory=STATIC_DIR / "models", check_dir=False), name="models")


@app.get("/", include_in_schema=False)
def frontend() -> Any:
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "The Grand Prix Predictor API is running. Start the Vite frontend at http://127.0.0.1:5173 or run npm run build in frontend/.", "docs": "/docs"}
