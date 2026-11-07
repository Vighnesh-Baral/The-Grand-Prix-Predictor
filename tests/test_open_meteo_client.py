from __future__ import annotations

import pandas as pd

from src.data.open_meteo_client import (
    HISTORICAL_URL, FORECAST_URL, HISTORICAL_VARIABLES, LIVE_VARIABLES,
    OpenMeteoClient,
)
from src.features.weather_context import RACE_WEATHER_FORECAST_FEATURES, attach_race_forecast_features


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return FakeResponse(self.payload)


def _payload(variable_names):
    hourly = {
        "time": ["2025-10-26T13:00", "2025-10-26T14:00", "2025-10-26T15:00"],
    }
    for name in variable_names:
        hourly[name] = [10, 20, 30]
    return {"hourly": hourly}


def _event():
    return pd.Series({
        "season": 2025, "round": 21, "race_name": "Test Grand Prix",
        "date": "2025-10-26", "time": "14:00:00Z",
        "latitude": 19.4, "longitude": -99.1,
    })


def test_historical_forecast_uses_one_day_ahead_archive_at_race_start(tmp_path):
    session = FakeSession(_payload(HISTORICAL_VARIABLES))
    client = OpenMeteoClient(session=session, cache_dir=tmp_path, sleep=lambda _: None)

    result = client.historical_event_forecast(_event())

    assert result is not None
    assert result["season"] == 2025
    assert result["round"] == 21
    assert result["forecast_source"] == "open_meteo_previous_day1"
    assert result["forecast_lead_hours"] == 24.0
    assert result["race_forecast_air_temp_c"] == 20
    assert result["race_forecast_precipitation_mm"] == 20
    assert session.calls[0][0] == HISTORICAL_URL
    assert session.calls[0][1]["timezone"] == "UTC"
    assert "temperature_2m_previous_day1" in session.calls[0][1]["hourly"]


def test_live_forecast_uses_scheduled_start_and_16_day_window(tmp_path):
    future = _event()
    future["date"] = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=3)).date().isoformat()
    future["time"] = "14:00:00Z"
    start = pd.to_datetime(future["date"] + "T14:00:00Z", utc=True)
    times = [(start + pd.Timedelta(hours=offset)).strftime("%Y-%m-%dT%H:%M") for offset in (-1, 0, 1)]
    payload = {"hourly": {"time": times}}
    for name in LIVE_VARIABLES:
        payload["hourly"][name] = [1, 2, 3]
    session = FakeSession(payload)
    client = OpenMeteoClient(session=session, cache_dir=tmp_path, sleep=lambda _: None)

    result = client.live_event_forecast(future)

    assert result is not None
    assert result["forecast_source"] == "open_meteo_live_forecast"
    assert result["race_forecast_air_temp_c"] == 2
    assert session.calls[0][0] == FORECAST_URL
    assert session.calls[0][1]["forecast_days"] == 16
    assert session.calls[0][1]["wind_speed_unit"] == "kmh"


def test_race_forecast_features_are_attached_by_event_key_without_changing_rows():
    candidates = pd.DataFrame([
        {"season": 2025, "round": 21, "driver_id": "a"},
        {"season": 2025, "round": 21, "driver_id": "b"},
    ])
    forecasts = pd.DataFrame([{
        "season": 2025, "round": 21,
        "race_forecast_air_temp_c": 24.0,
        "race_forecast_precipitation_mm": 0.2,
        "race_forecast_weather_code": 61,
        "forecast_source": "open_meteo_previous_day1",
        "forecast_lead_hours": 24.0,
    }])

    result = attach_race_forecast_features(candidates, forecasts)

    assert len(result) == 2
    assert result["race_forecast_air_temp_c"].tolist() == [24.0, 24.0]
    assert result["race_forecast_precipitation_mm"].tolist() == [0.2, 0.2]
    assert result["race_forecast_is_wet"].tolist() == [1.0, 1.0]
    assert result["race_forecast_source"].eq("open_meteo_previous_day1").all()
    assert set(RACE_WEATHER_FORECAST_FEATURES).issubset(result.columns)


def test_open_meteo_retries_transient_request_error(tmp_path):
    import requests

    class FlakySession(FakeSession):
        def get(self, url, params=None, timeout=None):
            self.calls.append((url, params, timeout))
            if len(self.calls) == 1:
                raise requests.Timeout("temporary timeout")
            return FakeResponse(self.payload)

    session = FlakySession(_payload(HISTORICAL_VARIABLES))
    sleeps = []
    client = OpenMeteoClient(
        session=session, cache_dir=tmp_path, retries=1,
        sleep=lambda duration: sleeps.append(duration), min_interval_seconds=0,
    )

    result = client.historical_event_forecast(_event())

    assert result is not None
    assert len(session.calls) == 2
    assert sleeps == [1]
