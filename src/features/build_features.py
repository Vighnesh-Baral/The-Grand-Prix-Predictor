"""Leakage-aware feature engineering for pole and race prediction."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.normalize import parse_lap_time_seconds
from src.data.identity import normalize_identity_name
from src.features.constructor_intelligence import (
    CONSTRUCTOR_INTELLIGENCE_FEATURES, build_historical_constructor_intelligence_table,
    build_constructor_intelligence_for_event,
)
from src.features.circuit_context import (
    CIRCUIT_RACE_FEATURES, build_historical_circuit_features, circuit_features_for_candidates,
)
from src.features.weather_context import (
    WEATHER_FEATURES, RACE_WEATHER_FORECAST_FEATURES, attach_weather_features, attach_race_forecast_features,
)
from src.features.weekend_pace import (
    SESSION_FEATURES as WEEKEND_PACE_FEATURES,
    RACE_PACE_FEATURES,
    build_event_weekend_pace_features,
    build_historical_weekend_pace_table,
    build_historical_race_pace_table,
)

CATEGORICAL_FEATURES = ["driver_id", "constructor_id", "circuit_id"]
POLE_NUMERIC_FEATURES = [
    "round", "driver_prev_avg_finish", "driver_recent_avg_finish", "driver_prev_wins",
    "driver_prev_podiums", "driver_prev_poles", "driver_prev_points", "driver_prev_avg_grid",
    "driver_last_gp_finish", "constructor_last_gp_avg_finish",
    "driver_prev_avg_quali", "driver_circuit_prev_avg_finish", "driver_circuit_prev_poles",
    "driver_season_prev_avg_finish", "driver_season_prev_wins", "driver_season_prev_podiums",
    "driver_season_prev_poles", "driver_season_prev_points", "constructor_prev_avg_finish",
    "constructor_prev_wins", "constructor_prev_points", "constructor_prev_podiums",
    "constructor_prev_poles", "constructor_season_prev_avg_finish", "constructor_season_prev_wins",
    "constructor_season_prev_podiums", "constructor_season_prev_poles", "constructor_season_prev_points",
    "fp_best_position", "fp_mean_position", "fp_best_time_seconds", "fp_best_time_gap_pct",
    "fp_mean_time_gap_pct", "fp_session_count", "fp_total_laps",
    "driver_quali_gap_prev_avg_pct", "driver_quali_gap_recent3_pct",
    "driver_teammate_quali_delta_prev_pct", "constructor_quali_gap_prev_avg_pct",
    "constructor_quali_gap_recent3_pct", "constructor_quali_gap_season_prev_pct",
    "constructor_circuit_quali_gap_prev_avg_pct",
    *WEEKEND_PACE_FEATURES, *WEATHER_FEATURES,
    "constructor_practice_best_gap_pct", "constructor_practice_latest_best_gap_pct",
    "constructor_practice_soft_gap_pct",
 ]
# Inputs available before the first free-practice session. Keep weekend practice,
# current-session weather and lap-derived features out of this model entirely.
POLE_PRE_FP_NUMERIC_FEATURES = [
    "round", "driver_prev_avg_finish", "driver_recent_avg_finish", "driver_last_gp_finish",
    "driver_prev_wins", "driver_prev_podiums", "driver_prev_poles", "driver_prev_points",
    "driver_prev_avg_grid", "driver_prev_avg_quali", "driver_circuit_prev_avg_finish",
    "driver_circuit_prev_poles", "driver_season_prev_avg_finish", "driver_season_prev_wins",
    "driver_season_prev_podiums", "driver_season_prev_poles", "driver_season_prev_points",
    "constructor_prev_avg_finish", "constructor_last_gp_avg_finish", "constructor_prev_wins",
    "constructor_prev_points", "constructor_prev_podiums", "constructor_prev_poles",
    "constructor_season_prev_avg_finish", "constructor_season_prev_wins",
    "constructor_season_prev_podiums", "constructor_season_prev_poles",
    "constructor_season_prev_points", "driver_quali_gap_prev_avg_pct",
    "driver_quali_gap_recent3_pct", "driver_teammate_quali_delta_prev_pct",
    "constructor_quali_gap_prev_avg_pct", "constructor_quali_gap_recent3_pct",
    "constructor_quali_gap_season_prev_pct", "constructor_circuit_quali_gap_prev_avg_pct",
]
RACE_NUMERIC_FEATURES = POLE_NUMERIC_FEATURES + [
    *CIRCUIT_RACE_FEATURES, *RACE_WEATHER_FORECAST_FEATURES,
    "grid", "qualifying_position", "best_qualifying_time_seconds",
    "driver_fastest_lap_gap_prev_avg_pct", "constructor_fastest_lap_gap_prev_avg_pct",
    "constructor_fastest_lap_gap_recent3_pct", *RACE_PACE_FEATURES,
    *[feature for feature in CONSTRUCTOR_INTELLIGENCE_FEATURES if feature not in POLE_NUMERIC_FEATURES],
]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result


def _normalise_name(value: object) -> str:
    """Use the shared identity normalizer, including null/placeholder guards."""
    return normalize_identity_name(value)


def _practice_session_order(value: object) -> int | None:
    """Map Jolpica/FastF1 practice labels to weekend order; unknown=FP3-only."""
    if pd.isna(value):
        return 3
    text = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    aliases = {
        "FP1": 1, "PRACTICE1": 1, "FREEPRACTICE1": 1,
        "FP2": 2, "PRACTICE2": 2, "FREEPRACTICE2": 2,
        "FP3": 3, "PRACTICE3": 3, "FREEPRACTICE3": 3,
        # Some historical alpha extracts contain a single combined FP row.
        # It is only safe to use after the full practice window (FP3).
        "FP": 3, "PRACTICE": 3, "UNKNOWN": 3, "": 3,
    }
    return aliases.get(text)


def _filter_practice_cutoff(practice: pd.DataFrame | None, as_of_session: str) -> pd.DataFrame:
    if practice is None or practice.empty:
        return pd.DataFrame() if practice is None else practice.copy()
    cutoff = {"FP1": 1, "FP2": 2, "FP3": 3}.get(str(as_of_session).upper())
    if cutoff is None:
        raise ValueError("practice_as_of_session must be FP1, FP2 or FP3")
    if "session" not in practice.columns:
        return practice.iloc[0:0].copy()
    frame = practice.copy()
    frame["_session_order"] = frame["session"].map(_practice_session_order)
    frame = frame[frame["_session_order"].notna() & frame["_session_order"].le(cutoff)].copy()
    return frame.drop(columns=["_session_order"])


def _add_practice_aggregates(
    races: pd.DataFrame, practice: pd.DataFrame | None, as_of_session: str = "FP3"
) -> pd.DataFrame:
    """Attach session-summary practice features to each race entrant.

    Time gaps are expressed relative to the fastest driver in each session so
    they are more comparable across circuits than raw lap time alone.
    """
    practice = _filter_practice_cutoff(practice, as_of_session)
    defaults = [
        "fp_best_position", "fp_mean_position", "fp_best_time_seconds",
        "fp_best_time_gap_pct", "fp_mean_time_gap_pct", "fp_session_count", "fp_total_laps",
    ]
    frame = races.copy()
    if practice is None or practice.empty:
        for column in defaults:
            frame[column] = np.nan
        return frame

    p = _ensure_columns(
        practice,
        ["season", "round", "driver_id", "driver_name", "position", "time_seconds", "laps", "session"],
    )
    # Resolve provider-specific practice IDs against the matching race-entry
    # record for that exact GP. The old global name->ID dictionary could apply a
    # driver's latest-season ID to every earlier season. Ambiguous event/name
    # matches are left unmatched rather than guessed.
    for column in ("season", "round", "position", "time_seconds", "laps"):
        p[column] = pd.to_numeric(p[column], errors="coerce")
    p["session"] = p["session"].fillna("Unknown").astype(str)
    p["_model_driver_id"] = pd.NA

    race_identity = races[[c for c in ("season", "round", "driver_id", "driver_name", "driver_code") if c in races.columns]].copy()
    race_identity["season"] = pd.to_numeric(race_identity["season"], errors="coerce")
    race_identity["round"] = pd.to_numeric(race_identity["round"], errors="coerce")
    race_identity["_name_key"] = race_identity["driver_name"].map(_normalise_name)
    if "driver_code" in race_identity:
        race_codes = race_identity["driver_code"].map(_normalise_name)
    else:
        race_codes = pd.Series("", index=race_identity.index, dtype=object)
    valid_race_names = race_identity["_name_key"].ne("") & race_identity["_name_key"].ne(race_codes)
    race_identity = race_identity[valid_race_names].dropna(subset=["season", "round", "driver_id"])
    identity_keys = ["season", "round", "_name_key"]
    if not race_identity.empty:
        id_counts = race_identity.groupby(identity_keys)["driver_id"].nunique()
        safe_keys = set(id_counts[id_counts.eq(1)].index.tolist())
        safe_race = race_identity.drop_duplicates(identity_keys + ["driver_id"])
        name_to_id = {
            (row["season"], row["round"], row["_name_key"]): row["driver_id"]
            for row in safe_race.to_dict("records")
            if (row["season"], row["round"], row["_name_key"]) in safe_keys
        }
        practice_names = p["driver_name"].map(_normalise_name)
        if "driver_code" in p:
            practice_codes = p["driver_code"].map(_normalise_name)
        else:
            practice_codes = pd.Series("", index=p.index, dtype=object)
        valid_practice_names = practice_names.ne("") & practice_names.ne(practice_codes)
        p["_model_driver_id"] = [
            name_to_id.get((season, rnd, name)) if valid else pd.NA
            for season, rnd, name, valid in zip(p["season"], p["round"], practice_names, valid_practice_names)
        ]

    # A source ID is usable only if it is already present in the race roster for
    # the same event. Opaque IDs that exist only in another provider are not.
    valid_race_ids = set(
        zip(
            pd.to_numeric(races["season"], errors="coerce"),
            pd.to_numeric(races["round"], errors="coerce"),
            races["driver_id"],
        )
    ) if {"season", "round", "driver_id"}.issubset(races.columns) else set()
    direct_match = [
        source_id if pd.isna(mapped) and pd.notna(source_id) and (season, rnd, source_id) in valid_race_ids else mapped
        for season, rnd, source_id, mapped in zip(p["season"], p["round"], p["driver_id"], p["_model_driver_id"])
    ]
    p["_model_driver_id"] = direct_match
    p = p.dropna(subset=["season", "round", "_model_driver_id"])
    if p.empty:
        for column in defaults:
            frame[column] = np.nan
        return frame

    session_keys = ["season", "round", "session"]
    p["session_best_time_seconds"] = p.groupby(session_keys)["time_seconds"].transform("min")
    p["time_gap_pct"] = np.where(
        p["session_best_time_seconds"].gt(0) & p["time_seconds"].notna(),
        (p["time_seconds"] / p["session_best_time_seconds"] - 1.0) * 100.0,
        np.nan,
    )
    aggregated = p.groupby(["season", "round", "_model_driver_id"], as_index=False).agg(
        fp_best_position=("position", "min"),
        fp_mean_position=("position", "mean"),
        fp_best_time_seconds=("time_seconds", "min"),
        fp_best_time_gap_pct=("time_gap_pct", "min"),
        fp_mean_time_gap_pct=("time_gap_pct", "mean"),
        fp_session_count=("session", "nunique"),
        fp_total_laps=("laps", "sum"),
    ).rename(columns={"_model_driver_id": "driver_id"})
    return frame.merge(aggregated, on=["season", "round", "driver_id"], how="left", validate="many_to_one")


def _shifted_expanding(grouped: pd.core.groupby.SeriesGroupBy, operation: str) -> pd.Series:
    if operation == "mean":
        return grouped.transform(lambda values: values.shift(1).expanding(min_periods=1).mean())
    if operation == "sum":
        return grouped.transform(lambda values: values.shift(1).fillna(0).cumsum())
    if operation == "rolling5":
        return grouped.transform(lambda values: values.shift(1).rolling(5, min_periods=1).mean())
    raise ValueError(f"Unknown operation: {operation}")


def _lagged_pace_history(
    races: pd.DataFrame, qualifying_results: pd.DataFrame | None, *, include_current_event: bool = False
) -> pd.DataFrame:
    """Create car/driver pace features using only earlier events.

    Qualifying gaps are a direct one-lap pace signal. Race fastest-lap gaps are
    retained only as a weaker one-lap race-session proxy, not labelled as full
    race pace. True race pace is added from FastF1 long-run data when available.
    """
    keys = ["season", "round"]
    base = races[[c for c in [*keys, "date", "circuit_id", "driver_id", "constructor_id"] if c in races]].drop_duplicates(keys + ["driver_id"]).copy()
    if base.empty:
        return base
    base["date"] = pd.to_datetime(base.get("date"), errors="coerce")
    base = base.sort_values(["date", "season", "round"])

    q = qualifying_results.copy() if qualifying_results is not None else pd.DataFrame()
    if not q.empty:
        q = _ensure_columns(q, ["season", "round", "driver_id", "constructor_id", "circuit_id", "q1", "q2", "q3"])
        q["season"] = pd.to_numeric(q["season"], errors="coerce")
        q["round"] = pd.to_numeric(q["round"], errors="coerce")
        if "circuit_id" in races:
            circuit_map = races[[*keys, "circuit_id"]].drop_duplicates(keys).rename(columns={"circuit_id": "_race_circuit_id"})
            q = q.merge(circuit_map, on=keys, how="left", validate="many_to_one")
            q["circuit_id"] = q["circuit_id"].where(q["circuit_id"].notna(), q["_race_circuit_id"])
            q = q.drop(columns=["_race_circuit_id"])
        q["best_q_s"] = q[["q1", "q2", "q3"]].apply(lambda col: col.map(parse_lap_time_seconds)).min(axis=1, skipna=True)
        q = q.dropna(subset=["season", "round", "driver_id", "best_q_s"])
        if not q.empty:
            q["event_best_q_s"] = q.groupby(keys)["best_q_s"].transform("min")
            q["driver_quali_gap_pct"] = np.where(
                q["event_best_q_s"].gt(0), (q["best_q_s"] / q["event_best_q_s"] - 1.0) * 100.0, np.nan
            )
            q_event = q.groupby([*keys, "driver_id", "constructor_id", "circuit_id"], as_index=False, dropna=False).agg(
                driver_quali_gap_pct=("driver_quali_gap_pct", "min")
            )
            q_event["constructor_event_quali_gap_pct"] = q_event.groupby([*keys, "constructor_id"])["driver_quali_gap_pct"].transform("mean")
            q_event["driver_teammate_quali_delta_pct"] = q_event["driver_quali_gap_pct"] - q_event["constructor_event_quali_gap_pct"]
            q_event = q_event.sort_values(["season", "round"])
            q_event["_driver_q_values"] = q_event["driver_quali_gap_pct"] if include_current_event else q_event.groupby("driver_id", sort=False)["driver_quali_gap_pct"].shift(1)
            q_event["driver_quali_gap_prev_avg_pct"] = q_event.groupby("driver_id", sort=False)["_driver_q_values"].transform(lambda x: x.expanding(min_periods=1).mean())
            q_event["driver_quali_gap_recent3_pct"] = q_event.groupby("driver_id", sort=False)["_driver_q_values"].transform(lambda x: x.rolling(3, min_periods=1).mean())
            q_event["_driver_teammate_values"] = q_event["driver_teammate_quali_delta_pct"] if include_current_event else q_event.groupby("driver_id", sort=False)["driver_teammate_quali_delta_pct"].shift(1)
            q_event["driver_teammate_quali_delta_prev_pct"] = q_event.groupby("driver_id", sort=False)["_driver_teammate_values"].transform(lambda x: x.expanding(min_periods=1).mean())
            team_events = q_event.drop_duplicates([*keys, "constructor_id"]).sort_values(["season", "round"]).copy()
            team_events["_team_q_values"] = team_events["constructor_event_quali_gap_pct"] if include_current_event else team_events.groupby("constructor_id", sort=False)["constructor_event_quali_gap_pct"].shift(1)
            team_events["constructor_quali_gap_prev_avg_pct"] = team_events.groupby("constructor_id", sort=False)["_team_q_values"].transform(lambda x: x.expanding(min_periods=1).mean())
            team_events["constructor_quali_gap_recent3_pct"] = team_events.groupby("constructor_id", sort=False)["_team_q_values"].transform(lambda x: x.rolling(3, min_periods=1).mean())
            def _season_pace(values: pd.Series) -> pd.Series:
                source = values if include_current_event else values.shift(1)
                return source.expanding(min_periods=1).mean()
            team_events["constructor_quali_gap_season_prev_pct"] = team_events.groupby(["season", "constructor_id"], sort=False)["constructor_event_quali_gap_pct"].transform(_season_pace)
            team_cols = [*keys, "constructor_id", "constructor_quali_gap_prev_avg_pct", "constructor_quali_gap_recent3_pct", "constructor_quali_gap_season_prev_pct"]
            q_event = q_event.merge(team_events[team_cols], on=[*keys, "constructor_id"], how="left")

            # Circuit-specific car pace is a supporting signal, not a replacement
            # for current-season/global pace. It is lagged within constructor × circuit.
            circuit_team_events = q_event.drop_duplicates([*keys, "constructor_id", "circuit_id"]).sort_values(["season", "round"]).copy()
            if include_current_event:
                circuit_team_events["_circuit_q_values"] = circuit_team_events["constructor_event_quali_gap_pct"]
            else:
                circuit_team_events["_circuit_q_values"] = circuit_team_events.groupby(["constructor_id", "circuit_id"], sort=False)["constructor_event_quali_gap_pct"].shift(1)
            circuit_team_events["constructor_circuit_quali_gap_prev_avg_pct"] = circuit_team_events.groupby(
                ["constructor_id", "circuit_id"], sort=False
            )["_circuit_q_values"].transform(lambda x: x.expanding(min_periods=1).mean())
            circuit_cols = [*keys, "constructor_id", "circuit_id", "constructor_circuit_quali_gap_prev_avg_pct"]
            q_event = q_event.merge(circuit_team_events[circuit_cols], on=[*keys, "constructor_id", "circuit_id"], how="left")
            base = base.merge(q_event[[*keys, "driver_id", "driver_quali_gap_prev_avg_pct", "driver_quali_gap_recent3_pct", "driver_teammate_quali_delta_prev_pct", "constructor_quali_gap_prev_avg_pct", "constructor_quali_gap_recent3_pct", "constructor_quali_gap_season_prev_pct", "constructor_circuit_quali_gap_prev_avg_pct"]], on=[*keys, "driver_id"], how="left")

    # Fastest-lap times from race results are a one-lap proxy only. They do not
    # substitute for clean long-run lap sequences and degradation measurements.
    race = _ensure_columns(races, ["fastest_lap_time", "driver_id", "constructor_id", *keys]).copy()
    race["fastest_lap_s"] = race["fastest_lap_time"].map(parse_lap_time_seconds)
    race = race.dropna(subset=[*keys, "driver_id", "fastest_lap_s"])
    if not race.empty:
        race["event_best_fl_s"] = race.groupby(keys)["fastest_lap_s"].transform("min")
        race["driver_fastest_lap_gap_pct"] = np.where(race["event_best_fl_s"].gt(0), (race["fastest_lap_s"] / race["event_best_fl_s"] - 1.0) * 100.0, np.nan)
        driver_events = race.groupby([*keys, "driver_id", "constructor_id"], as_index=False).agg(driver_fastest_lap_gap_pct=("driver_fastest_lap_gap_pct", "min"))
        driver_events["constructor_event_fastest_lap_gap_pct"] = driver_events.groupby([*keys, "constructor_id"])["driver_fastest_lap_gap_pct"].transform("mean")
        driver_events = driver_events.sort_values(["season", "round"])
        driver_events["_driver_fl_values"] = driver_events["driver_fastest_lap_gap_pct"] if include_current_event else driver_events.groupby("driver_id", sort=False)["driver_fastest_lap_gap_pct"].shift(1)
        driver_events["driver_fastest_lap_gap_prev_avg_pct"] = driver_events.groupby("driver_id", sort=False)["_driver_fl_values"].transform(lambda x: x.expanding(min_periods=1).mean())
        team_events = driver_events.drop_duplicates([*keys, "constructor_id"]).copy()
        team_events["_team_fl_values"] = team_events["constructor_event_fastest_lap_gap_pct"] if include_current_event else team_events.groupby("constructor_id", sort=False)["constructor_event_fastest_lap_gap_pct"].shift(1)
        team_events["constructor_fastest_lap_gap_prev_avg_pct"] = team_events.groupby("constructor_id", sort=False)["_team_fl_values"].transform(lambda x: x.expanding(min_periods=1).mean())
        team_events["constructor_fastest_lap_gap_recent3_pct"] = team_events.groupby("constructor_id", sort=False)["_team_fl_values"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        driver_events = driver_events.merge(team_events[[*keys, "constructor_id", "constructor_fastest_lap_gap_prev_avg_pct", "constructor_fastest_lap_gap_recent3_pct"]], on=[*keys, "constructor_id"], how="left")
        base = base.merge(driver_events[[*keys, "driver_id", "driver_fastest_lap_gap_prev_avg_pct", "constructor_fastest_lap_gap_prev_avg_pct", "constructor_fastest_lap_gap_recent3_pct"]], on=[*keys, "driver_id"], how="left")

    return base


def build_race_feature_table(
    race_results: pd.DataFrame,
    qualifying_results: pd.DataFrame | None = None,
    practice_results: pd.DataFrame | None = None,
    sprint_results: pd.DataFrame | None = None,
    session_laps: pd.DataFrame | None = None,
    practice_as_of_session: str = "FP3",
    weather_observations: pd.DataFrame | None = None,
    weather_as_of_session: str | None = None,
    race_weather_forecasts: pd.DataFrame | None = None,
    race_control: pd.DataFrame | None = None,
    pit_stops: pd.DataFrame | None = None,
    openf1_drivers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one row per race entrant, with all historical features lagged.

    The pole task only sees practice and historical form. The race task also sees
    the current event's qualifying/grid features because that prediction is made
    after qualifying and before lights out.
    """
    if race_results is None or race_results.empty:
        raise ValueError("race_results is empty; download data before feature engineering")

    races = _ensure_columns(race_results, [
        "season", "round", "date", "race_name", "circuit_id", "driver_id", "driver_code",
        "driver_name", "constructor_id", "constructor_name", "grid", "position", "position_order",
        "points", "status",
    ])
    for col in ("season", "round", "grid", "position_order", "points"):
        races[col] = pd.to_numeric(races[col], errors="coerce")
    races["date"] = pd.to_datetime(races["date"], errors="coerce")
    races = races.dropna(subset=["season", "round", "driver_id", "position_order"]).copy()
    races["season"] = races["season"].astype(int)
    races["round"] = races["round"].astype(int)

    quali = qualifying_results.copy() if qualifying_results is not None else pd.DataFrame()
    if not quali.empty:
        quali = _ensure_columns(quali, ["season", "round", "driver_id", "qualifying_position", "q1", "q2", "q3"])
        quali["season"] = pd.to_numeric(quali["season"], errors="coerce")
        quali["round"] = pd.to_numeric(quali["round"], errors="coerce")
        quali["qualifying_position"] = pd.to_numeric(quali["qualifying_position"], errors="coerce")
        for q_col in ("q1", "q2", "q3"):
            quali[f"{q_col}_seconds"] = quali[q_col].map(parse_lap_time_seconds)
        q_times = quali[["q1_seconds", "q2_seconds", "q3_seconds"]].min(axis=1, skipna=True)
        quali["best_qualifying_time_seconds"] = q_times
        quali = quali[["season", "round", "driver_id", "qualifying_position", "best_qualifying_time_seconds"]]
        frame = races.merge(quali, on=["season", "round", "driver_id"], how="left")
    else:
        frame = races.copy()
        frame["qualifying_position"] = np.nan
        frame["best_qualifying_time_seconds"] = np.nan

    frame = _add_practice_aggregates(frame, practice_results, practice_as_of_session)

    # These are deliberately computed from earlier completed events only. Circuit
    # conversion rates give the race model evidence for track-specific grid effects
    # (e.g. pole conversion versus P2/P3 starts) without hard-coded track bonuses.
    circuit_context = build_historical_circuit_features(races)
    if not circuit_context.empty:
        frame = frame.merge(circuit_context, on=["season", "round", "driver_id"], how="left", validate="one_to_one")
    else:
        for feature in CIRCUIT_RACE_FEATURES:
            frame[feature] = np.nan

    frame = attach_weather_features(
        frame, weather_observations,
        as_of_session=weather_as_of_session or practice_as_of_session,
    )
    frame = attach_race_forecast_features(frame, race_weather_forecasts)
    lagged_pace = _lagged_pace_history(races, qualifying_results)
    pace_cols = [c for c in lagged_pace.columns if c not in {"date", "circuit_id", "driver_id", "constructor_id"}]
    if pace_cols:
        frame = frame.merge(lagged_pace[["season", "round", "driver_id", *[c for c in pace_cols if c not in {"season", "round"}]]], on=["season", "round", "driver_id"], how="left")
    historical_weekend = build_historical_weekend_pace_table(session_laps, as_of_session=practice_as_of_session)
    if not historical_weekend.empty and "driver_code" in frame.columns:
        frame = frame.merge(historical_weekend, on=["season", "round", "driver_code"], how="left", validate="many_to_one")
    historical_race_pace = build_historical_race_pace_table(session_laps, races)
    if not historical_race_pace.empty and "driver_code" in frame.columns:
        frame = frame.merge(
            historical_race_pace[["season", "round", "driver_code", *RACE_PACE_FEATURES]],
            on=["season", "round", "driver_code"], how="left", validate="many_to_one",
        )
    constructor_intelligence = build_historical_constructor_intelligence_table(
        races, session_laps=session_laps, race_control=race_control, as_of_session=practice_as_of_session,
        pit_stops=pit_stops, openf1_drivers=openf1_drivers,
    )
    if not constructor_intelligence.empty:
        frame = frame.merge(
            constructor_intelligence[["season", "round", "driver_id", "constructor_id", *CONSTRUCTOR_INTELLIGENCE_FEATURES]],
            on=["season", "round", "driver_id", "constructor_id"], how="left", validate="one_to_one",
        )
    else:
        for feature in CONSTRUCTOR_INTELLIGENCE_FEATURES:
            frame[feature] = np.nan
    frame["sprint_points"] = 0.0
    if sprint_results is not None and not sprint_results.empty:
        sprint = _ensure_columns(sprint_results, ["season", "round", "driver_id", "points"]).copy()
        sprint["season"] = pd.to_numeric(sprint["season"], errors="coerce")
        sprint["round"] = pd.to_numeric(sprint["round"], errors="coerce")
        sprint["points"] = pd.to_numeric(sprint["points"], errors="coerce").fillna(0)
        sprint_points = sprint.groupby(["season", "round", "driver_id"], as_index=False)["points"].sum()
        sprint_points = sprint_points.rename(columns={"points": "sprint_points"})
        frame = frame.drop(columns=["sprint_points"], errors="ignore").merge(
            sprint_points, on=["season", "round", "driver_id"], how="left"
        )
        frame["sprint_points"] = frame["sprint_points"].fillna(0)
    frame = frame.sort_values(["date", "season", "round", "driver_id"], na_position="last").reset_index(drop=True)

    frame["is_winner"] = (frame["position_order"] == 1).astype(int)
    frame["is_podium"] = frame["position_order"].between(1, 3).astype(int)
    status = frame["status"].fillna("").astype(str).str.strip()
    finished = status.str.fullmatch(r"Finished|\+\s*\d+\s*Lap(?:s)?", case=False)
    administrative_non_starters = status.str.contains(
        r"disqualified|\bDSQ\b|\bDQ\b|did not start|\bDNS\b|withdrawn|did not qualify|excluded|not started|did not take part",
        case=False, regex=True, na=False,
    )
    unknown_status = status.eq("")
    # DNF risk is intended to model on-track retirements. Administrative outcomes,
    # non-starters and missing status values are unknown/not applicable.
    frame["is_dnf"] = np.where(
        finished, 0.0, np.where(administrative_non_starters | unknown_status, np.nan, 1.0)
    )
    # Missing qualifying records are unknown labels, not negative pole outcomes.
    qualifying_position = pd.to_numeric(frame["qualifying_position"], errors="coerce")
    frame["is_pole"] = np.where(qualifying_position.notna(), qualifying_position.eq(1).astype(float), np.nan)

    # Driver-level previous career form; shift(1) ensures the current race is excluded.
    frame["_points_total_event"] = frame["points"].fillna(0) + frame["sprint_points"].fillna(0)
    driver_group = frame.groupby("driver_id", sort=False)
    frame["driver_last_gp_finish"] = driver_group["position_order"].shift(1)
    frame["driver_prev_avg_finish"] = _shifted_expanding(driver_group["position_order"], "mean")
    frame["driver_recent_avg_finish"] = _shifted_expanding(driver_group["position_order"], "rolling5")
    frame["driver_prev_wins"] = _shifted_expanding(driver_group["is_winner"], "sum")
    frame["driver_prev_podiums"] = _shifted_expanding(driver_group["is_podium"], "sum")
    frame["driver_prev_poles"] = _shifted_expanding(driver_group["is_pole"], "sum")
    frame["driver_prev_points"] = _shifted_expanding(driver_group["_points_total_event"], "sum")
    frame["driver_prev_avg_grid"] = _shifted_expanding(driver_group["grid"], "mean")
    frame["driver_prev_avg_quali"] = _shifted_expanding(driver_group["qualifying_position"], "mean")

    # Circuit-specific historical performance, excluding the current GP.
    circuit_group = frame.groupby(["driver_id", "circuit_id"], sort=False)
    frame["driver_circuit_prev_avg_finish"] = _shifted_expanding(circuit_group["position_order"], "mean")
    frame["driver_circuit_prev_poles"] = _shifted_expanding(circuit_group["is_pole"], "sum")

    # Constructor values are aggregated by event first. If we lagged raw driver rows,
    # the second car could see the first car's result from the same Grand Prix.
    team_event = frame.groupby(["season", "round", "constructor_id"], as_index=False, dropna=False).agg(
        _team_event_avg_finish=("position_order", "mean"),
        _team_event_points=("_points_total_event", "sum"),
        _team_event_podiums=("is_podium", "sum"),
        _team_event_wins=("is_winner", "sum"),
        _team_event_poles=("is_pole", "sum"),
    )
    event_dates = frame[["season", "round", "date"]].drop_duplicates()
    team_event = team_event.merge(event_dates, on=["season", "round"], how="left")
    team_event = team_event.sort_values(["date", "season", "round"], na_position="last")
    team_group = team_event.groupby("constructor_id", sort=False)
    team_event["constructor_last_gp_avg_finish"] = team_group["_team_event_avg_finish"].shift(1)
    team_event["constructor_prev_avg_finish"] = _shifted_expanding(team_group["_team_event_avg_finish"], "mean")
    team_event["constructor_prev_points"] = _shifted_expanding(team_group["_team_event_points"], "sum")
    team_event["constructor_prev_podiums"] = _shifted_expanding(team_group["_team_event_podiums"], "sum")
    team_event["constructor_prev_wins"] = _shifted_expanding(team_group["_team_event_wins"], "sum")
    team_event["constructor_prev_poles"] = _shifted_expanding(team_group["_team_event_poles"], "sum")

    # Same-season form is reset at the start of each championship year.
    season_driver = frame.groupby(["season", "driver_id"], sort=False)
    frame["driver_season_prev_avg_finish"] = _shifted_expanding(season_driver["position_order"], "mean")
    frame["driver_season_prev_wins"] = _shifted_expanding(season_driver["is_winner"], "sum")
    frame["driver_season_prev_podiums"] = _shifted_expanding(season_driver["is_podium"], "sum")
    frame["driver_season_prev_poles"] = _shifted_expanding(season_driver["is_pole"], "sum")
    frame["driver_season_prev_points"] = _shifted_expanding(season_driver["_points_total_event"], "sum")
    team_event["constructor_season_prev_avg_finish"] = team_event.groupby(
        ["season", "constructor_id"], sort=False
    )["_team_event_avg_finish"].transform(lambda values: values.shift(1).expanding(min_periods=1).mean())
    team_event["constructor_season_prev_wins"] = team_event.groupby(
        ["season", "constructor_id"], sort=False
    )["_team_event_wins"].transform(lambda values: values.shift(1).fillna(0).cumsum())
    team_event["constructor_season_prev_poles"] = team_event.groupby(
        ["season", "constructor_id"], sort=False
    )["_team_event_poles"].transform(lambda values: values.shift(1).fillna(0).cumsum())
    team_event["constructor_season_prev_points"] = team_event.groupby(
        ["season", "constructor_id"], sort=False
    )["_team_event_points"].transform(lambda values: values.shift(1).fillna(0).cumsum())
    team_event["constructor_season_prev_podiums"] = team_event.groupby(
        ["season", "constructor_id"], sort=False
    )["_team_event_podiums"].transform(lambda values: values.shift(1).fillna(0).cumsum())
    merge_cols = [
        "season", "round", "constructor_id", "constructor_prev_avg_finish",
        "constructor_prev_points", "constructor_prev_podiums", "constructor_prev_wins", "constructor_prev_poles",
        "constructor_season_prev_avg_finish", "constructor_season_prev_wins",
        "constructor_season_prev_podiums", "constructor_season_prev_poles", "constructor_season_prev_points",
    ]
    frame = frame.merge(team_event[merge_cols], on=["season", "round", "constructor_id"], how="left")

    frame = frame.drop(columns=["_points_total_event"], errors="ignore")
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].fillna("unknown").astype(str)
    return frame


def make_model_frame(
    features: pd.DataFrame, task: str, numeric_features: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return leak-safe model features and a target for a supported task."""
    if task == "pole":
        target = "is_pole"
        columns = CATEGORICAL_FEATURES + (numeric_features or POLE_NUMERIC_FEATURES)
        # An event without a qualifying classification has no valid pole label.
        features = features.loc[pd.to_numeric(features.get("qualifying_position"), errors="coerce").notna()].copy()
    elif task == "position":
        target = "position_order"
        columns = CATEGORICAL_FEATURES + RACE_NUMERIC_FEATURES
        # Finishing-position regression is conditional on being classified.
        features = features.loc[features["is_dnf"] == 0].copy()
    elif task in {"winner", "podium", "dnf"}:
        target = {"winner": "is_winner", "podium": "is_podium", "dnf": "is_dnf"}[task]
        columns = CATEGORICAL_FEATURES + RACE_NUMERIC_FEATURES
    else:
        raise ValueError(f"Unsupported task: {task}")
    clean = _ensure_columns(features, columns + [target]).dropna(subset=[target]).copy()
    X = clean[columns].copy()
    y = pd.to_numeric(clean[target], errors="coerce")
    valid = y.notna()
    return X.loc[valid], y.loc[valid].astype(int if task != "position" else float)


def build_event_prediction_frame(
    season: int,
    round_number: int,
    race_results: pd.DataFrame,
    qualifying_results: pd.DataFrame | None = None,
    practice_results: pd.DataFrame | None = None,
    race_schedule: pd.DataFrame | None = None,
    sprint_results: pd.DataFrame | None = None,
    stage: str = "race",
    session_laps: pd.DataFrame | None = None,
    as_of_session: str = "FP3",
    driver_roster: pd.DataFrame | None = None,
    weather_observations: pd.DataFrame | None = None,
    weather_as_of_session: str | None = None,
    race_weather_forecasts: pd.DataFrame | None = None,
    starting_grid: pd.DataFrame | None = None,
    race_control: pd.DataFrame | None = None,
    pit_stops: pd.DataFrame | None = None,
    openf1_drivers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build driver rows for a selected event without using that event's race outcome.

    ``driver_roster`` is an optional curated event entry list. It is especially
    useful before the first race of a season, when the previous season's final
    event roster may contain outdated driver/constructor pairings.
    """
    if stage not in {"pole", "race"}:
        raise ValueError("stage must be 'pole' or 'race'")
    races = _ensure_columns(race_results, [
        "season", "round", "date", "race_name", "circuit_id", "driver_id", "driver_code",
        "driver_name", "constructor_id", "constructor_name", "grid", "position_order", "points", "status",
    ]).copy()
    races["season"] = pd.to_numeric(races["season"], errors="coerce")
    races["round"] = pd.to_numeric(races["round"], errors="coerce")
    target_rows = races[(races["season"] == season) & (races["round"] == round_number)]

    schedule_row: dict = {}
    if race_schedule is not None and not race_schedule.empty:
        sch = race_schedule.copy()
        sch["season"] = pd.to_numeric(sch["season"], errors="coerce")
        sch["round"] = pd.to_numeric(sch["round"], errors="coerce")
        selected = sch[(sch["season"] == season) & (sch["round"] == round_number)]
        if not selected.empty:
            schedule_row = selected.iloc[0].to_dict()
    circuit_id = schedule_row.get("circuit_id")
    if not circuit_id and not target_rows.empty:
        circuit_id = target_rows.iloc[0].get("circuit_id")

    # Strictly earlier events only: the selected event's result never enters history.
    history = races[(races["season"] < season) | ((races["season"] == season) & (races["round"] < round_number))].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.sort_values(["date", "season", "round"])
    quali = qualifying_results.copy() if qualifying_results is not None else pd.DataFrame()
    if not quali.empty:
        quali["season"] = pd.to_numeric(quali.get("season"), errors="coerce")
        quali["round"] = pd.to_numeric(quali.get("round"), errors="coerce")
    event_quali = quali[(quali["season"] == season) & (quali["round"] == round_number)].copy() if not quali.empty else pd.DataFrame()

    practice = practice_results.copy() if practice_results is not None else pd.DataFrame()
    if not practice.empty:
        practice["season"] = pd.to_numeric(practice.get("season"), errors="coerce")
        practice["round"] = pd.to_numeric(practice.get("round"), errors="coerce")
    event_practice = practice[(practice["season"] == season) & (practice["round"] == round_number)].copy() if not practice.empty else pd.DataFrame()
    event_practice = _filter_practice_cutoff(event_practice, as_of_session)

    # Resolve the entrant roster from the selected event where possible. When it
    # is not yet published, use the *latest completed GP's field*, not one latest
    # row for every driver across the entire historical dataset (which retains
    # retired drivers and creates invalid future prediction candidates).
    candidate_columns = ["driver_id", "driver_code", "driver_name", "constructor_id", "constructor_name"]

    def _dedupe_candidate_rows(source: pd.DataFrame) -> pd.DataFrame:
        """Keep one row per entrant even when an upstream source omits driver IDs.

        Missing IDs are replaced only inside this prediction frame with a stable,
        name-derived key. The original source tables and provider IDs are untouched.
        """
        if source is None or source.empty:
            return pd.DataFrame(columns=candidate_columns)
        selected = source[[column for column in candidate_columns if column in source.columns]].copy()
        for column in candidate_columns:
            if column not in selected.columns:
                selected[column] = pd.NA
        selected["driver_id"] = selected["driver_id"].astype("object")
        name_keys = selected["driver_name"].map(_normalise_name)
        code_keys = selected["driver_code"].map(_normalise_name)
        missing_id = selected["driver_id"].isna() | selected["driver_id"].astype(str).str.strip().isin({"", "nan", "None", "<NA>"})
        fallback_keys = name_keys.where(name_keys.ne(""), "code:" + code_keys)
        usable_fallback = fallback_keys.ne("") & fallback_keys.ne("code:")
        selected.loc[missing_id & usable_fallback, "driver_id"] = "identity:" + fallback_keys[missing_id & usable_fallback]
        selected = selected[~(missing_id & ~usable_fallback)].copy()
        selected["_candidate_dedupe_key"] = selected["driver_id"].astype("string")
        return selected.drop_duplicates("_candidate_dedupe_key").drop(columns="_candidate_dedupe_key").reset_index(drop=True)

    candidates = _dedupe_candidate_rows(target_rows)
    roster_source = "selected_event_race_entries" if not candidates.empty else None
    if candidates.empty and stage == "race" and not event_quali.empty:
        candidates = _dedupe_candidate_rows(event_quali)
        roster_source = "selected_event_qualifying_entries"
    if candidates.empty and not event_practice.empty:
        candidates = _dedupe_candidate_rows(event_practice)
        roster_source = "selected_event_practice_entries"
    if candidates.empty and driver_roster is not None and not driver_roster.empty:
        roster = _ensure_columns(driver_roster, [
            "season", "round", "driver_id", "driver_code", "driver_name",
            "constructor_id", "constructor_name",
        ]).copy()
        roster["season"] = pd.to_numeric(roster["season"], errors="coerce")
        roster["round"] = pd.to_numeric(roster["round"], errors="coerce")
        roster = roster[(roster["season"] == season) & (roster["round"] == round_number)]
        roster = roster.dropna(subset=["driver_id", "constructor_id"])
        if not roster.empty:
            candidates = _dedupe_candidate_rows(roster)
            roster_source = "curated_event_roster"
    if candidates.empty and not history.empty:
        ordered_history = history.sort_values(["date", "season", "round"], na_position="last")
        latest_event = ordered_history[["season", "round"]].dropna().drop_duplicates().tail(1)
        if not latest_event.empty:
            latest_season = int(latest_event.iloc[0]["season"])
            latest_round = int(latest_event.iloc[0]["round"])
            roster = history[(pd.to_numeric(history["season"], errors="coerce") == latest_season)
                             & (pd.to_numeric(history["round"], errors="coerce") == latest_round)]
            candidates = _dedupe_candidate_rows(roster)
            roster_source = f"latest_completed_gp_{latest_season}_round_{latest_round}"
    if candidates.empty:
        raise ValueError("No current driver roster is available. Refresh race results or collect selected-event practice/qualifying entries.")

    for col in ("driver_code", "driver_name", "constructor_id", "constructor_name"):
        if col not in candidates:
            candidates[col] = None
    candidates = candidates.copy()
    candidates["candidate_roster_source"] = roster_source or "unknown"
    candidates["season"] = season
    candidates["round"] = round_number
    candidates["circuit_id"] = circuit_id or "unknown"
    candidates["race_name"] = schedule_row.get("race_name") or (target_rows.iloc[0].get("race_name") if not target_rows.empty else f"Round {round_number}")

    # Match event practice records by driver ID; use normalised names as a fallback.
    # Practice classifications are session-relative, so calculate gaps against the
    # fastest driver in each session before aggregating across FP1/FP2/FP3.
    practice_feature_columns = [
        "fp_best_position", "fp_mean_position", "fp_best_time_seconds",
        "fp_best_time_gap_pct", "fp_mean_time_gap_pct", "fp_session_count", "fp_total_laps",
    ]
    if not event_practice.empty:
        event_practice = event_practice.copy()
        if "driver_id" not in event_practice:
            event_practice["driver_id"] = None
        event_practice["_model_driver_id"] = pd.NA
        candidate_names = candidates.copy()
        candidate_names["_name_key"] = candidate_names["driver_name"].map(_normalise_name)
        if "driver_code" in candidate_names:
            candidate_codes = candidate_names["driver_code"].map(_normalise_name)
        else:
            candidate_codes = pd.Series("", index=candidate_names.index, dtype=object)
        candidate_names = candidate_names[
            candidate_names["_name_key"].ne("") & candidate_names["_name_key"].ne(candidate_codes)
        ]
        counts = candidate_names.groupby("_name_key")["driver_id"].nunique()
        safe_names = set(counts[counts.eq(1)].index)
        candidate_names = candidate_names.drop_duplicates(["_name_key", "driver_id"])
        name_map = {
            row["_name_key"]: row["driver_id"]
            for row in candidate_names.to_dict("records")
            if row["_name_key"] in safe_names
        }
        if "driver_name" in event_practice:
            practice_names = event_practice["driver_name"].map(_normalise_name)
            if "driver_code" in event_practice:
                practice_codes = event_practice["driver_code"].map(_normalise_name)
            else:
                practice_codes = pd.Series("", index=event_practice.index, dtype=object)
            valid_names = practice_names.ne("") & practice_names.ne(practice_codes)
            event_practice.loc[valid_names, "_model_driver_id"] = practice_names[valid_names].map(name_map)
        candidate_id_set = set(candidates["driver_id"].dropna())
        direct_ids = event_practice["driver_id"].where(event_practice["driver_id"].isin(candidate_id_set))
        event_practice["_model_driver_id"] = event_practice["_model_driver_id"].fillna(direct_ids)
        if "time_seconds" not in event_practice.columns and "time" in event_practice.columns:
            event_practice["time_seconds"] = event_practice["time"].map(parse_lap_time_seconds)
        if "session" not in event_practice.columns:
            event_practice["session"] = "Unknown"
        for col in ("position", "time_seconds", "laps"):
            if col not in event_practice:
                event_practice[col] = np.nan
            event_practice[col] = pd.to_numeric(event_practice[col], errors="coerce")
        event_practice["session"] = event_practice["session"].fillna("Unknown").astype(str)
        event_practice = event_practice.dropna(subset=["_model_driver_id"])
        if not event_practice.empty:
            session_best = event_practice.groupby("session")["time_seconds"].transform("min")
            event_practice["time_gap_pct"] = np.where(
                session_best.gt(0) & event_practice["time_seconds"].notna(),
                (event_practice["time_seconds"] / session_best - 1.0) * 100.0,
                np.nan,
            )
            fp_agg = event_practice.groupby("_model_driver_id", as_index=False).agg(
                fp_best_position=("position", "min"), fp_mean_position=("position", "mean"),
                fp_best_time_seconds=("time_seconds", "min"),
                fp_best_time_gap_pct=("time_gap_pct", "min"),
                fp_mean_time_gap_pct=("time_gap_pct", "mean"),
                fp_session_count=("session", "nunique"), fp_total_laps=("laps", "sum"),
            ).rename(columns={"_model_driver_id": "driver_id"})
            candidates = candidates.merge(fp_agg, on="driver_id", how="left", validate="one_to_one")
    for col in practice_feature_columns:
        if col not in candidates:
            candidates[col] = np.nan

    # Higher-resolution FastF1 lap features are kept separate from Jolpica FP
    # classification aggregates. They are restricted to sessions available by
    # the requested prediction cutoff (FP1/FP2/FP3).
    weekend_features = build_event_weekend_pace_features(session_laps, season, round_number, as_of_session)
    if not weekend_features.empty and "driver_code" in candidates.columns:
        candidates = candidates.merge(weekend_features, on="driver_code", how="left", validate="one_to_one")
    else:
        for col in WEEKEND_PACE_FEATURES:
            candidates[col] = np.nan

    # Current-event qualifying is allowed only for post-qualifying race predictions.
    if stage == "race" and not event_quali.empty:
        qkeep = [c for c in ["driver_id", "qualifying_position", "q1", "q2", "q3"] if c in event_quali.columns]
        candidates = candidates.merge(event_quali[qkeep].drop_duplicates("driver_id"), on="driver_id", how="left")
    else:
        candidates["qualifying_position"] = np.nan
        candidates["q1"] = None
        candidates["q2"] = None
        candidates["q3"] = None
    if "qualifying_position" not in candidates:
        candidates["qualifying_position"] = np.nan

    # Historical rows use the actual grid from race results. For a future event,
    # prefer OpenF1's published starting grid (which can include penalties); use
    # qualifying position only as an explicitly labelled fallback.
    if stage == "race" and not target_rows.empty:
        grid_map = target_rows.drop_duplicates("driver_id").set_index("driver_id")["grid"]
        candidates["grid"] = candidates["driver_id"].map(grid_map)
        candidates["starting_grid_source"] = "jolpica_actual_grid"
    elif stage == "race":
        candidates["grid"] = np.nan
        candidates["starting_grid_source"] = "qualifying_position_proxy"
        if starting_grid is not None and not starting_grid.empty:
            grid = starting_grid.copy()
            grid["season"] = pd.to_numeric(grid.get("season"), errors="coerce")
            grid["round"] = pd.to_numeric(grid.get("round"), errors="coerce")
            grid = grid[(grid["season"] == season) & (grid["round"] == round_number)]
            if not grid.empty:
                grid["grid"] = pd.to_numeric(grid.get("grid"), errors="coerce")
                if "driver_id" in grid:
                    by_id = grid.dropna(subset=["driver_id", "grid"]).drop_duplicates("driver_id").set_index("driver_id")["grid"]
                    mapped = candidates["driver_id"].map(by_id)
                else:
                    mapped = pd.Series(np.nan, index=candidates.index)
                if "driver_code" in grid and "driver_code" in candidates:
                    by_code = grid.dropna(subset=["driver_code", "grid"]).copy()
                    by_code["driver_code"] = by_code["driver_code"].astype(str).str.upper()
                    by_code = by_code.drop_duplicates("driver_code").set_index("driver_code")["grid"]
                    code_mapped = candidates["driver_code"].astype(str).str.upper().map(by_code)
                    mapped = mapped.fillna(code_mapped)
                valid = mapped.notna()
                candidates.loc[valid, "grid"] = mapped[valid]
                candidates.loc[valid, "starting_grid_source"] = "openf1_starting_grid"
        candidates["grid"] = pd.to_numeric(candidates["grid"], errors="coerce").fillna(
            pd.to_numeric(candidates["qualifying_position"], errors="coerce")
        )
    else:
        candidates["grid"] = np.nan
        candidates["starting_grid_source"] = "not_applicable"
    for q_col in ("q1", "q2", "q3"):
        if q_col not in candidates:
            candidates[q_col] = None
        candidates[f"{q_col}_seconds"] = candidates[q_col].map(parse_lap_time_seconds)
    candidates["best_qualifying_time_seconds"] = candidates[["q1_seconds", "q2_seconds", "q3_seconds"]].min(axis=1, skipna=True)

    history["is_winner"] = pd.to_numeric(history["position_order"], errors="coerce").eq(1).astype(int)
    history["is_podium"] = pd.to_numeric(history["position_order"], errors="coerce").between(1, 3).astype(int)
    history["points"] = pd.to_numeric(history["points"], errors="coerce").fillna(0)
    if sprint_results is not None and not sprint_results.empty:
        sprint_history = sprint_results.copy()
        sprint_history["season"] = pd.to_numeric(sprint_history["season"], errors="coerce")
        sprint_history["round"] = pd.to_numeric(sprint_history["round"], errors="coerce")
        sprint_history = sprint_history[
            (sprint_history["season"] < season) |
            ((sprint_history["season"] == season) & (sprint_history["round"] < round_number))
        ]
        sprint_history["points"] = pd.to_numeric(sprint_history["points"], errors="coerce").fillna(0)
        sprint_points = sprint_history.groupby(["season", "round", "driver_id"], as_index=False)["points"].sum()
        history = history.merge(sprint_points, on=["season", "round", "driver_id"], how="left", suffixes=("", "_sprint"))
        history["points"] = history["points"] + history["points_sprint"].fillna(0)
    history["grid"] = pd.to_numeric(history["grid"], errors="coerce")
    history["position_order"] = pd.to_numeric(history["position_order"], errors="coerce")

    def driver_stats(driver_id: str, circuit: str, season_history: pd.DataFrame) -> dict:
        dh = history[history["driver_id"].astype(str) == str(driver_id)]
        ch = dh[dh["circuit_id"].astype(str) == str(circuit)]
        return {
            "driver_prev_avg_finish": dh["position_order"].mean(),
            "driver_recent_avg_finish": dh["position_order"].tail(5).mean(),
            "driver_last_gp_finish": pd.to_numeric(dh.sort_values(["date", "season", "round"])["position_order"], errors="coerce").iloc[-1] if not dh.empty else np.nan,
            "driver_prev_wins": dh["is_winner"].sum(),
            "driver_prev_podiums": dh["is_podium"].sum(),
            "driver_prev_poles": np.nan,
            "driver_prev_points": dh["points"].sum(),
            "driver_prev_avg_grid": dh["grid"].mean(),
            "driver_prev_avg_quali": np.nan,
            "driver_circuit_prev_avg_finish": ch["position_order"].mean(),
            "driver_circuit_prev_poles": 0,
            "driver_season_prev_avg_finish": season_history["position_order"].mean(),
            "driver_season_prev_wins": season_history["is_winner"].sum(),
            "driver_season_prev_podiums": season_history["is_podium"].sum(),
            "driver_season_prev_poles": np.nan,
            "driver_season_prev_points": season_history["points"].sum(),
        }

    quali_history = quali[(quali["season"] < season) | ((quali["season"] == season) & (quali["round"] < round_number))] if not quali.empty else pd.DataFrame()
    lagged_history_pace = _lagged_pace_history(history, quali_history, include_current_event=True)
    session_laps_history = session_laps.copy() if session_laps is not None else pd.DataFrame()
    if not session_laps_history.empty:
        session_laps_history["season"] = pd.to_numeric(session_laps_history.get("season"), errors="coerce")
        session_laps_history["round"] = pd.to_numeric(session_laps_history.get("round"), errors="coerce")
        session_laps_history = session_laps_history[
            (session_laps_history["season"] < season) |
            ((session_laps_history["season"] == season) & (session_laps_history["round"] < round_number))
        ]
    historical_race_pace_state = build_historical_race_pace_table(
        session_laps_history, history, include_current_event=True
    )
    for col in RACE_PACE_FEATURES:
        candidates[col] = np.nan
    if not historical_race_pace_state.empty:
        for index, candidate in candidates.iterrows():
            driver_state = historical_race_pace_state[
                historical_race_pace_state["driver_id"].astype(str).eq(str(candidate["driver_id"]))
            ].sort_values(["season", "round"])
            team_state = historical_race_pace_state[
                historical_race_pace_state["constructor_id"].astype(str).eq(str(candidate["constructor_id"]))
            ].sort_values(["season", "round"])
            driver_cols = [c for c in RACE_PACE_FEATURES if c.startswith("driver_")]
            team_cols = [c for c in RACE_PACE_FEATURES if c.startswith("constructor_")]
            if not driver_state.empty:
                for col in driver_cols:
                    candidates.loc[index, col] = driver_state.iloc[-1][col]
            if not team_state.empty:
                for col in team_cols:
                    candidates.loc[index, col] = team_state.iloc[-1][col]
    pace_defaults = [
        "driver_quali_gap_prev_avg_pct", "driver_quali_gap_recent3_pct",
        "driver_teammate_quali_delta_prev_pct", "constructor_quali_gap_prev_avg_pct",
        "constructor_quali_gap_recent3_pct", "constructor_quali_gap_season_prev_pct",
        "constructor_circuit_quali_gap_prev_avg_pct", "driver_fastest_lap_gap_prev_avg_pct",
        "constructor_fastest_lap_gap_prev_avg_pct",
        "constructor_fastest_lap_gap_recent3_pct",
    ]
    for col in pace_defaults:
        candidates[col] = np.nan
    driver_pace_columns = [
        "driver_quali_gap_prev_avg_pct", "driver_quali_gap_recent3_pct",
        "driver_teammate_quali_delta_prev_pct", "driver_fastest_lap_gap_prev_avg_pct",
    ]
    team_pace_columns = [
        "constructor_quali_gap_prev_avg_pct", "constructor_quali_gap_recent3_pct",
        "constructor_quali_gap_season_prev_pct", "constructor_circuit_quali_gap_prev_avg_pct",
        "constructor_fastest_lap_gap_prev_avg_pct", "constructor_fastest_lap_gap_recent3_pct",
    ]
    if not lagged_history_pace.empty:
        for index, candidate in candidates.iterrows():
            driver_pace = lagged_history_pace[lagged_history_pace["driver_id"].astype(str).eq(str(candidate["driver_id"]))].sort_values(["season", "round"])
            team_pace = lagged_history_pace[lagged_history_pace["constructor_id"].astype(str).eq(str(candidate["constructor_id"]))].sort_values(["season", "round"])
            circuit_team_pace = team_pace.iloc[0:0]
            if "circuit_id" in team_pace and pd.notna(candidate.get("circuit_id")):
                circuit_team_pace = team_pace[team_pace["circuit_id"].astype(str).eq(str(candidate["circuit_id"]))]
            if not driver_pace.empty:
                for col in driver_pace_columns:
                    if col in driver_pace.columns:
                        candidates.loc[index, col] = driver_pace.iloc[-1][col]
            if not team_pace.empty:
                for col in team_pace_columns:
                    if col in team_pace.columns:
                        if col == "constructor_circuit_quali_gap_prev_avg_pct":
                            # Never substitute a different circuit's pace when the
                            # target circuit has no prior sample.
                            if not circuit_team_pace.empty:
                                candidates.loc[index, col] = circuit_team_pace.iloc[-1][col]
                        else:
                            candidates.loc[index, col] = team_pace.iloc[-1][col]
    for index, candidate in candidates.iterrows():
        driver_id = candidate["driver_id"]
        d_hist = history[history["driver_id"].astype(str) == str(driver_id)]
        season_hist = d_hist[d_hist["season"] == season]
        stats = driver_stats(driver_id, str(circuit_id or "unknown"), season_hist)
        if not quali_history.empty:
            dq = quali_history[quali_history["driver_id"].astype(str) == str(driver_id)]
            dq_position = pd.to_numeric(dq.get("qualifying_position"), errors="coerce")
            stats["driver_prev_avg_quali"] = dq_position.mean()
            stats["driver_prev_poles"] = dq_position.eq(1).sum()
            current_season_quali = dq[dq["season"] == season] if "season" in dq else pd.DataFrame()
            stats["driver_season_prev_poles"] = pd.to_numeric(current_season_quali.get("qualifying_position"), errors="coerce").eq(1).sum()
            if not dq.empty and "circuit_id" in dq:
                cq = dq[dq["circuit_id"].astype(str) == str(circuit_id)]
                stats["driver_circuit_prev_poles"] = pd.to_numeric(cq.get("qualifying_position"), errors="coerce").eq(1).sum()
        for name, value in stats.items():
            candidates.loc[index, name] = value

    # Rebuild constructor history at event level. Aggregating before summing/averaging
    # keeps this definition consistent with the training feature table.
    history["is_pole"] = 0
    if not quali_history.empty:
        poles = quali_history[pd.to_numeric(quali_history["qualifying_position"], errors="coerce").eq(1)][
            ["season", "round", "driver_id"]
        ].drop_duplicates()
        pole_keys = set(zip(poles["season"], poles["round"], poles["driver_id"]))
        history["is_pole"] = [int((r.season, r.round, r.driver_id) in pole_keys) for r in history.itertuples()]

    team_event_history = history.groupby(
        ["season", "round", "constructor_id"], as_index=False, dropna=False
    ).agg(
        team_avg_finish=("position_order", "mean"),
        team_points=("points", "sum"),
        team_podiums=("is_podium", "sum"),
        team_wins=("is_winner", "sum"),
        team_poles=("is_pole", "sum"),
    )
    for col in (
        "constructor_prev_avg_finish", "constructor_last_gp_avg_finish", "constructor_prev_points", "constructor_prev_podiums",
        "constructor_prev_wins", "constructor_prev_poles", "constructor_season_prev_avg_finish",
        "constructor_season_prev_wins", "constructor_season_prev_podiums",
        "constructor_season_prev_poles", "constructor_season_prev_points",
    ):
        candidates[col] = np.nan

    for index, candidate in candidates.iterrows():
        team_history = team_event_history[
            team_event_history["constructor_id"].astype(str) == str(candidate["constructor_id"])
        ]
        team_season = team_history[team_history["season"] == season]
        candidates.loc[index, "constructor_prev_avg_finish"] = team_history["team_avg_finish"].mean()
        latest_team_event = team_history.sort_values(["season", "round"]).tail(1)
        candidates.loc[index, "constructor_last_gp_avg_finish"] = latest_team_event["team_avg_finish"].iloc[0] if not latest_team_event.empty else np.nan
        candidates.loc[index, "constructor_prev_points"] = team_history["team_points"].sum()
        candidates.loc[index, "constructor_prev_podiums"] = team_history["team_podiums"].sum()
        candidates.loc[index, "constructor_prev_wins"] = team_history["team_wins"].sum()
        candidates.loc[index, "constructor_prev_poles"] = team_history["team_poles"].sum()
        candidates.loc[index, "constructor_season_prev_avg_finish"] = team_season["team_avg_finish"].mean()
        candidates.loc[index, "constructor_season_prev_wins"] = team_season["team_wins"].sum()
        candidates.loc[index, "constructor_season_prev_podiums"] = team_season["team_podiums"].sum()
        candidates.loc[index, "constructor_season_prev_poles"] = team_season["team_poles"].sum()
        candidates.loc[index, "constructor_season_prev_points"] = team_season["team_points"].sum()

    # Current-event circuit conversion features are available only to the race
    # model after qualifying/grid publication. Historical rows are strictly prior.
    if stage == "race":
        circuit_context = circuit_features_for_candidates(history, candidates, str(circuit_id) if circuit_id else None)
        candidates = candidates.drop(columns=CIRCUIT_RACE_FEATURES, errors="ignore").merge(
            circuit_context, on="driver_id", how="left", validate="one_to_one"
        )
    else:
        for feature in CIRCUIT_RACE_FEATURES:
            candidates[feature] = np.nan

    constructor_intelligence = build_constructor_intelligence_for_event(
        races, candidates, session_laps, race_control, season, round_number, circuit_id, as_of_session,
        pit_stops, openf1_drivers,
    )
    if not constructor_intelligence.empty:
        candidates = candidates.drop(columns=CONSTRUCTOR_INTELLIGENCE_FEATURES, errors="ignore").merge(
            constructor_intelligence[["driver_id", "constructor_id", *CONSTRUCTOR_INTELLIGENCE_FEATURES]],
            on=["driver_id", "constructor_id"], how="left", validate="one_to_one",
        )
    else:
        for feature in CONSTRUCTOR_INTELLIGENCE_FEATURES:
            candidates[feature] = np.nan

    candidates["season"] = season
    candidates["round"] = round_number
    candidates = attach_weather_features(
        candidates, weather_observations,
        as_of_session=weather_as_of_session or ("Q" if stage == "race" else as_of_session),
    )
    if stage == "race":
        candidates = attach_race_forecast_features(candidates, race_weather_forecasts)
    else:
        candidates = attach_race_forecast_features(candidates, None)
    for col in CATEGORICAL_FEATURES:
        candidates[col] = candidates[col].fillna("unknown").astype(str)
    for col in POLE_NUMERIC_FEATURES + RACE_PACE_FEATURES + CIRCUIT_RACE_FEATURES + CONSTRUCTOR_INTELLIGENCE_FEATURES + WEATHER_FEATURES + RACE_WEATHER_FORECAST_FEATURES + ["grid", "qualifying_position", "best_qualifying_time_seconds"]:
        if col not in candidates:
            candidates[col] = np.nan
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    return candidates.sort_values("driver_name").reset_index(drop=True)
