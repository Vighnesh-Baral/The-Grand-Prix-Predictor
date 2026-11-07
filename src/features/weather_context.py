"""Weather features restricted to information available at a prediction cutoff."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_FEATURES = [
    "weather_air_temp_c",
    "weather_track_temp_c",
    "weather_humidity_pct",
    "weather_wind_speed_mps",
    "weather_rainfall_fraction",
    "weather_track_air_temp_delta_c",
    "weather_observation_count",
]

SESSION_ORDER = {"FP1": 1, "FP2": 2, "FP3": 3, "SQ": 3, "Q": 4}
SESSION_ALIASES = {
    "PRACTICE 1": "FP1", "PRACTICE 2": "FP2", "PRACTICE 3": "FP3",
    "FREE PRACTICE 1": "FP1", "FREE PRACTICE 2": "FP2", "FREE PRACTICE 3": "FP3",
    "QUALIFYING": "Q", "SPRINT QUALIFYING": "SQ",
}


def normalise_session(value: object) -> str:
    text = str(value or "").strip().upper()
    return SESSION_ALIASES.get(text, text)


def build_event_weather_features(
    weather_observations: pd.DataFrame | None,
    *,
    as_of_session: str = "FP3",
) -> pd.DataFrame:
    """Aggregate weather from the latest completed eligible session per GP.

    For pole forecasts, pass FP1/FP2/FP3. For post-qualifying race forecasts,
    pass Q; if Q weather is unavailable the most recent practice session is used.
    Actual race weather is never used as a pre-race feature.
    """
    columns = ["season", "round", *WEATHER_FEATURES, "weather_source_session"]
    if weather_observations is None or weather_observations.empty:
        return pd.DataFrame(columns=columns)
    cutoff = normalise_session(as_of_session)
    if cutoff not in {"FP1", "FP2", "FP3", "Q"}:
        raise ValueError("Weather cutoff must be FP1, FP2, FP3 or Q")
    frame = weather_observations.copy()
    required = {"season", "round", "session"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    for col in ("season", "round"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["session"] = frame["session"].map(normalise_session)
    frame["_order"] = frame["session"].map(SESSION_ORDER)
    frame = frame[frame["_order"].notna() & frame["_order"].le(SESSION_ORDER[cutoff])].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    measure_map = {
        "air_temperature": "weather_air_temp_c",
        "track_temperature": "weather_track_temp_c",
        "humidity": "weather_humidity_pct",
        "wind_speed": "weather_wind_speed_mps",
        "rainfall": "weather_rainfall_fraction",
    }
    for source in measure_map:
        frame[source] = pd.to_numeric(frame.get(source, np.nan), errors="coerce")
    frame = frame.dropna(subset=["season", "round"])
    frame["season"] = frame["season"].astype(int)
    frame["round"] = frame["round"].astype(int)
    max_order = frame.groupby(["season", "round"])["_order"].transform("max")
    frame = frame[frame["_order"].eq(max_order)].copy()
    aggregations = {target: (source, "mean") for source, target in measure_map.items()}
    result = frame.groupby(["season", "round", "session"], as_index=False).agg(**aggregations, weather_observation_count=("session", "size"))
    result["weather_track_air_temp_delta_c"] = result["weather_track_temp_c"] - result["weather_air_temp_c"]
    result = result.rename(columns={"session": "weather_source_session"})
    return result[columns]


def attach_weather_features(
    frame: pd.DataFrame,
    weather_observations: pd.DataFrame | None,
    *,
    as_of_session: str = "FP3",
) -> pd.DataFrame:
    result = frame.copy()
    weather = build_event_weather_features(weather_observations, as_of_session=as_of_session)
    if weather.empty:
        for feature in WEATHER_FEATURES:
            result[feature] = np.nan
        result["weather_source_session"] = None
        return result
    return result.drop(columns=[*WEATHER_FEATURES, "weather_source_session"], errors="ignore").merge(
        weather, on=["season", "round"], how="left", validate="many_to_one"
    )

RACE_WEATHER_FORECAST_FEATURES = [
    "race_forecast_air_temp_c",
    "race_forecast_humidity_pct",
    "race_forecast_precipitation_mm",
    "race_forecast_wind_speed_kmh",
    "race_forecast_wind_direction_deg",
    "race_forecast_is_wet",
]
RACE_WEATHER_CONTEXT_FEATURES = ["race_forecast_precip_probability_pct", "race_forecast_weather_code"]


def attach_race_forecast_features(
    candidates: pd.DataFrame,
    race_weather_forecasts: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach pre-race weather forecast at scheduled race start.

    Historical rows should come from an archived forecast issued one day before
    the race. At inference, the caller may provide the latest live forecast. The
    source/lead metadata are retained for audit but not passed to the estimator.
    """
    result = candidates.copy()
    if race_weather_forecasts is None or race_weather_forecasts.empty:
        for column in RACE_WEATHER_FORECAST_FEATURES:
            result[column] = np.nan
        result["race_forecast_source"] = None
        result["race_forecast_lead_hours"] = np.nan
        result["race_forecast_valid_time"] = None
        result["race_forecast_model_eligible"] = False
        return result
    forecast = race_weather_forecasts.copy()
    for column in ("season", "round"):
        forecast[column] = pd.to_numeric(forecast.get(column), errors="coerce")
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    for column in [*RACE_WEATHER_FORECAST_FEATURES, *RACE_WEATHER_CONTEXT_FEATURES]:
        if column not in forecast:
            forecast[column] = np.nan
    if "race_forecast_is_wet" not in forecast or forecast["race_forecast_is_wet"].isna().all():
        weather_code = pd.to_numeric(forecast.get("race_forecast_weather_code"), errors="coerce")
        precipitation = pd.to_numeric(forecast.get("race_forecast_precipitation_mm"), errors="coerce")
        forecast["race_forecast_is_wet"] = (
            weather_code.between(51, 67) | weather_code.between(80, 82) | weather_code.between(95, 99)
            | precipitation.gt(0.1)
        ).astype(float)
    for column in ("forecast_source", "forecast_lead_hours", "forecast_valid_time"):
        if column not in forecast:
            forecast[column] = None if column != "forecast_lead_hours" else np.nan
    forecast = forecast.dropna(subset=["season", "round"]).drop_duplicates(["season", "round"], keep="last")
    forecast = forecast[[column for column in ["season", "round", *RACE_WEATHER_FORECAST_FEATURES, *RACE_WEATHER_CONTEXT_FEATURES, "forecast_source", "forecast_lead_hours", "forecast_valid_time"] if column in forecast]]
    result = result.drop(columns=[*RACE_WEATHER_FORECAST_FEATURES, *RACE_WEATHER_CONTEXT_FEATURES, "race_forecast_source", "race_forecast_lead_hours", "race_forecast_valid_time", "race_forecast_model_eligible"], errors="ignore").merge(
        forecast.rename(columns={
            "forecast_source": "race_forecast_source",
            "forecast_lead_hours": "race_forecast_lead_hours",
            "forecast_valid_time": "race_forecast_valid_time",
        }),
        on=["season", "round"], how="left", validate="many_to_one",
    )
    source = result["race_forecast_source"].fillna("").astype(str)
    lead = pd.to_numeric(result["race_forecast_lead_hours"], errors="coerce")
    live = source.eq("open_meteo_live_forecast")
    # The training archive is a one-day-ahead forecast. Inference live forecasts
    # outside a 12-36h window have a materially different lead-time distribution,
    # so preserve them for display/audit but withhold them from the ML features.
    live_eligible = lead.between(12.0, 36.0, inclusive="both")
    eligible = (~live) | live_eligible
    result["race_forecast_model_eligible"] = eligible & source.ne("")
    result.loc[live & ~live_eligible, RACE_WEATHER_FORECAST_FEATURES] = np.nan
    return result
