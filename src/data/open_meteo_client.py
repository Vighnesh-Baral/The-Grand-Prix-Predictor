"""Historical one-day-ahead and live race-start weather forecasts via Open-Meteo.

The historical Previous Runs API is used for model training so the feature is a
forecast that was available before the race, not realised race weather. The
Forecast API provides the corresponding forward-looking value at inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import CACHE_DIR, PROCESSED_DIR

LOGGER = logging.getLogger("f1_open_meteo")
HISTORICAL_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HISTORICAL_VARIABLES = {
    "temperature_2m_previous_day1": "race_forecast_air_temp_c",
    "relative_humidity_2m_previous_day1": "race_forecast_humidity_pct",
    "precipitation_previous_day1": "race_forecast_precipitation_mm",
    "wind_speed_10m_previous_day1": "race_forecast_wind_speed_kmh",
    "wind_direction_10m_previous_day1": "race_forecast_wind_direction_deg",
    "weather_code_previous_day1": "race_forecast_weather_code",
}
LIVE_VARIABLES = {
    "temperature_2m": "race_forecast_air_temp_c",
    "relative_humidity_2m": "race_forecast_humidity_pct",
    "precipitation_probability": "race_forecast_precip_probability_pct",
    "precipitation": "race_forecast_precipitation_mm",
    "wind_speed_10m": "race_forecast_wind_speed_kmh",
    "wind_direction_10m": "race_forecast_wind_direction_deg",
    "weather_code": "race_forecast_weather_code",
}


def _event_start(row: pd.Series | dict[str, Any]) -> pd.Timestamp | None:
    date_value = row.get("date")
    if date_value is None or pd.isna(date_value):
        return None
    time_value = row.get("time")
    text = str(date_value).strip()
    if time_value is not None and not pd.isna(time_value) and str(time_value).strip():
        time_text = str(time_value).strip()
        if "T" not in text and " " not in text:
            text = f"{text}T{time_text}"
    result = pd.to_datetime(text, utc=True, errors="coerce")
    return None if pd.isna(result) else result


def _select_nearest_hour(payload: dict[str, Any], target: pd.Timestamp, variables: dict[str, str], *, source: str, lead_hours: float) -> dict[str, Any] | None:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    parsed = pd.to_datetime(pd.Series(times), utc=True, errors="coerce")
    if parsed.isna().all():
        return None
    deltas = (parsed - target).abs()
    idx = int(deltas.idxmin())
    if deltas.loc[idx] > pd.Timedelta(minutes=45):
        return None
    row: dict[str, Any] = {
        "forecast_valid_time": parsed.loc[idx].isoformat(),
        "forecast_source": source,
        "forecast_lead_hours": float(lead_hours),
    }
    for source_name, target_name in variables.items():
        values = hourly.get(source_name) or []
        value = values[idx] if idx < len(values) else None
        row[target_name] = pd.to_numeric(value, errors="coerce") if value is not None else np.nan
    return row


class OpenMeteoClient:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        cache_dir: Path | str = CACHE_DIR / "open_meteo",
        timeout_seconds: float = 30.0,
        min_interval_seconds: float = 0.5,
        retries: int = 3,
        sleep=time.sleep,
    ) -> None:
        self.session = session or requests.Session()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.min_interval_seconds = float(min_interval_seconds)
        self.retries = max(0, int(retries))
        self._sleep = sleep
        self._last_request = 0.0

    def _get(self, url: str, params: dict[str, Any], *, cache_ttl_seconds: float | None = None) -> dict[str, Any]:
        cache_key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                fresh = cache_ttl_seconds is None or (time.time() - cache_path.stat().st_mtime) <= cache_ttl_seconds
                if fresh:
                    return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        last_error: requests.RequestException | None = None
        for attempt in range(self.retries + 1):
            wait = self.min_interval_seconds - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
                self._last_request = time.monotonic()
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Open-Meteo returned an unexpected payload type")
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise
                self._sleep(min(2 ** attempt, 8))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Open-Meteo request failed without an error detail")

    def historical_event_forecast(self, event: pd.Series | dict[str, Any]) -> dict[str, Any] | None:
        start = _event_start(event)
        if start is None:
            return None
        try:
            latitude, longitude = float(event.get("latitude")), float(event.get("longitude"))
        except (TypeError, ValueError):
            return None
        params = {
            "latitude": latitude, "longitude": longitude,
            "start_date": start.date().isoformat(), "end_date": start.date().isoformat(),
            "hourly": ",".join(HISTORICAL_VARIABLES), "timezone": "UTC", "wind_speed_unit": "kmh",
        }
        try:
            payload = self._get(HISTORICAL_URL, params)
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("Historical weather forecast unavailable for %s: %s", event.get("race_name", "event"), exc)
            return None
        row = _select_nearest_hour(payload, start, HISTORICAL_VARIABLES, source="open_meteo_previous_day1", lead_hours=24.0)
        if row is None:
            return None
        return {"season": int(event["season"]), "round": int(event["round"]), **row}

    def live_event_forecast(self, event: pd.Series | dict[str, Any]) -> dict[str, Any] | None:
        start = _event_start(event)
        now = pd.Timestamp.now(tz="UTC")
        if start is None or start < now - pd.Timedelta(hours=2) or start > now + pd.Timedelta(days=16):
            return None
        try:
            latitude, longitude = float(event.get("latitude")), float(event.get("longitude"))
        except (TypeError, ValueError):
            return None
        params = {
            "latitude": latitude, "longitude": longitude,
            "hourly": ",".join(LIVE_VARIABLES), "forecast_days": 16,
            "timezone": "UTC", "wind_speed_unit": "kmh",
        }
        try:
            payload = self._get(FORECAST_URL, params, cache_ttl_seconds=900)
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("Live weather forecast unavailable for %s: %s", event.get("race_name", "event"), exc)
            return None
        row = _select_nearest_hour(payload, start, LIVE_VARIABLES, source="open_meteo_live_forecast", lead_hours=(start - now).total_seconds() / 3600)
        if row is None:
            return None
        return {"season": int(event["season"]), "round": int(event["round"]), **row}


def build_historical_forecast_table(
    races: pd.DataFrame,
    *,
    client: OpenMeteoClient | None = None,
    start_year: int = 2024,
    end_year: int | None = None,
    output_path: Path | str = PROCESSED_DIR / "race_weather_forecasts.csv",
) -> pd.DataFrame:
    """Fetch archived one-day-ahead forecasts at the scheduled race start."""
    end_year = int(end_year or datetime.now(timezone.utc).year)
    if races is None or races.empty:
        raise ValueError("races.csv is empty; collect the Jolpica calendar first")
    required = {"season", "round", "date", "time", "latitude", "longitude"}
    missing = required.difference(races.columns)
    if missing:
        raise ValueError(f"Race calendar missing weather location/time columns: {', '.join(sorted(missing))}")
    calendar = races.copy()
    calendar["season"] = pd.to_numeric(calendar["season"], errors="coerce")
    calendar["round"] = pd.to_numeric(calendar["round"], errors="coerce")
    calendar = calendar[calendar["season"].between(start_year, end_year)].dropna(subset=["season", "round"])
    client = client or OpenMeteoClient()
    output = Path(output_path)
    existing = pd.read_csv(output) if output.exists() else pd.DataFrame()
    existing_keys = set(zip(existing.get("season", []), existing.get("round", []))) if not existing.empty else set()
    rows: list[dict[str, Any]] = []
    for _, event in calendar.sort_values(["season", "round"]).iterrows():
        key = (int(event["season"]), int(event["round"]))
        if key in existing_keys:
            continue
        row = client.historical_event_forecast(event)
        if row:
            rows.append(row)
            LOGGER.info("Archived race weather forecast: %s R%s", key[0], key[1])
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False) if not existing.empty or rows else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(["season", "round"], keep="last").sort_values(["season", "round"])
        output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output, index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--races", default=str(PROCESSED_DIR / "races.csv"))
    parser.add_argument("--output", default=str(PROCESSED_DIR / "race_weather_forecasts.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    races = pd.read_csv(args.races)
    result = build_historical_forecast_table(races, start_year=args.start_year, end_year=args.end_year, output_path=args.output)
    print(f"Saved {len(result)} historical pre-race weather forecast rows to {args.output}")


if __name__ == "__main__":
    main()
