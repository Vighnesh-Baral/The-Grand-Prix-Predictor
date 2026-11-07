import pandas as pd

from src.features.weather_context import build_event_weather_features, attach_race_forecast_features


def test_weather_uses_latest_eligible_session_not_future_session():
    rows = pd.DataFrame([
        {"season": 2026, "round": 4, "session": "FP1", "air_temperature": 20, "track_temperature": 30, "humidity": 50, "wind_speed": 2, "rainfall": 0},
        {"season": 2026, "round": 4, "session": "FP2", "air_temperature": 21, "track_temperature": 32, "humidity": 52, "wind_speed": 3, "rainfall": 1},
        {"season": 2026, "round": 4, "session": "Q", "air_temperature": 15, "track_temperature": 18, "humidity": 80, "wind_speed": 8, "rainfall": 1},
        {"season": 2026, "round": 4, "session": "R", "air_temperature": 10, "track_temperature": 12, "humidity": 90, "wind_speed": 10, "rainfall": 1},
    ])
    fp1 = build_event_weather_features(rows, as_of_session="FP1").iloc[0]
    fp2 = build_event_weather_features(rows, as_of_session="FP2").iloc[0]
    race = build_event_weather_features(rows, as_of_session="Q").iloc[0]
    assert fp1.weather_air_temp_c == 20
    assert fp2.weather_air_temp_c == 21
    assert race.weather_air_temp_c == 15
    assert race.weather_source_session == "Q"


def test_race_forecast_probability_is_context_only_not_a_model_feature():
    from src.features.build_features import RACE_NUMERIC_FEATURES
    from src.features.weather_context import RACE_WEATHER_CONTEXT_FEATURES

    assert "race_forecast_precip_probability_pct" in RACE_WEATHER_CONTEXT_FEATURES
    assert "race_forecast_precip_probability_pct" not in RACE_NUMERIC_FEATURES
    assert "race_forecast_precipitation_mm" in RACE_NUMERIC_FEATURES
    assert "race_forecast_is_wet" in RACE_NUMERIC_FEATURES


def test_live_weather_forecast_outside_training_lead_window_is_withheld_from_model():
    candidates = pd.DataFrame([{"season": 2026, "round": 5, "driver_id": "driver"}])
    forecasts = pd.DataFrame([{
        "season": 2026, "round": 5,
        "race_forecast_air_temp_c": 25.0,
        "race_forecast_precipitation_mm": 0.3,
        "forecast_source": "open_meteo_live_forecast",
        "forecast_lead_hours": 96.0,
    }])

    result = attach_race_forecast_features(candidates, forecasts)

    assert result.loc[0, "race_forecast_source"] == "open_meteo_live_forecast"
    assert result.loc[0, "race_forecast_lead_hours"] == 96.0
    assert result.loc[0, "race_forecast_model_eligible"] == False
    assert pd.isna(result.loc[0, "race_forecast_air_temp_c"])


def test_live_weather_forecast_near_one_day_lead_is_model_eligible():
    candidates = pd.DataFrame([{"season": 2026, "round": 5, "driver_id": "driver"}])
    forecasts = pd.DataFrame([{
        "season": 2026, "round": 5,
        "race_forecast_air_temp_c": 25.0,
        "race_forecast_precipitation_mm": 0.3,
        "forecast_source": "open_meteo_live_forecast",
        "forecast_lead_hours": 24.0,
    }])

    result = attach_race_forecast_features(candidates, forecasts)

    assert result.loc[0, "race_forecast_model_eligible"] == True
    assert result.loc[0, "race_forecast_air_temp_c"] == 25.0
