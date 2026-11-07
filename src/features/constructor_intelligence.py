"""Leakage-aware constructor intelligence: pace, tyres, reliability and circuit fit.

Practice-derived fields describe the selected weekend. Historical fields are built
from completed events strictly before the target Grand Prix. Missing source data
remain NaN and are surfaced as coverage, never silently converted to zero.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.features.weekend_pace import build_event_weekend_pace_features

CONSTRUCTOR_INTELLIGENCE_FEATURES = [
    "constructor_practice_best_gap_pct",
    "constructor_practice_latest_best_gap_pct",
    "constructor_practice_soft_gap_pct",
    "constructor_practice_longrun_gap_pct",
    "constructor_practice_longrun_laps",
    "constructor_practice_longrun_stints",
    "constructor_practice_compound_count",
    "constructor_practice_tyre_deg_sec_per_lap",
    "constructor_practice_soft_deg_sec_per_lap",
    "constructor_practice_medium_deg_sec_per_lap",
    "constructor_practice_hard_deg_sec_per_lap",
    "constructor_pit_stop_mean_duration_recent5_sec",
    "constructor_pit_stop_median_duration_recent5_sec",
    "constructor_pit_stop_count_recent5",
    "constructor_dnf_rate_recent5",
    "constructor_mechanical_dnf_rate_recent5",
    "driver_dnf_rate_recent5",
    "constructor_circuit_avg_finish_prev",
    "constructor_circuit_positions_gained_prev",
    "constructor_circuit_dnf_rate_prev",
    "circuit_sc_event_rate_prev",
    "circuit_vsc_event_rate_prev",
]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _dnf_classification(status: Any) -> tuple[float, float]:
    """Return (DNF, mechanical DNF); administrative/missing outcomes are unknown."""
    text = "" if status is None or pd.isna(status) else str(status).strip().lower()
    if not text:
        return np.nan, np.nan
    if text == "finished" or re.fullmatch(r"\+\s*\d+\s*lap(?:s)?", text):
        return 0.0, 0.0
    if re.search(r"disqualified|\bdsq\b|\bdq\b|did not start|\bdns\b|withdrawn|did not qualify|excluded|not started|did not take part", text):
        return np.nan, np.nan
    mechanical = bool(re.search(
        r"engine|gearbox|hydraulic|electrical|transmission|brake|fuel|oil|water|clutch|suspension|exhaust|power unit|turbo|ers|battery|alternator|radiator|pressure|mechanical",
        text,
    ))
    return 1.0, float(mechanical)


def _clean_results(results: pd.DataFrame | None) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame(columns=["season", "round", "date", "circuit_id", "driver_id", "driver_code", "constructor_id", "grid", "position_order", "status"])
    frame = results.copy()
    for col in ("season", "round", "grid", "position_order"):
        frame[col] = _num(frame, col)
    for col in ("circuit_id", "driver_id", "driver_code", "constructor_id", "constructor_name", "status"):
        if col not in frame:
            frame[col] = None
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    frame[["is_dnf", "is_mechanical_dnf"]] = frame["status"].map(_dnf_classification).apply(pd.Series)
    frame["positions_gained"] = _num(frame, "grid") - _num(frame, "position_order")
    return frame.dropna(subset=["season", "round", "driver_id", "constructor_id"]).sort_values(
        ["date", "season", "round"], na_position="last"
    )


def _recent_team_reliability(history: pd.DataFrame, constructor_id: Any) -> tuple[float, float]:
    team = history[history["constructor_id"].astype(str).eq(str(constructor_id))].copy()
    if team.empty:
        return np.nan, np.nan
    events = team.groupby(["season", "round"], as_index=False).agg(
        dnf_rate=("is_dnf", "mean"), mechanical_rate=("is_mechanical_dnf", "mean"),
    ).sort_values(["season", "round"])
    recent = events.tail(5)
    return float(recent["dnf_rate"].mean()), float(recent["mechanical_rate"].mean())


def _recent_driver_reliability(history: pd.DataFrame, driver_id: Any) -> float:
    driver = history[history["driver_id"].astype(str).eq(str(driver_id))].copy()
    if driver.empty:
        return np.nan
    return float(driver.sort_values(["season", "round"])["is_dnf"].tail(5).mean())


def _circuit_team_history(history: pd.DataFrame, constructor_id: Any, circuit_id: Any) -> tuple[float, float, float]:
    team = history[
        history["constructor_id"].astype(str).eq(str(constructor_id))
        & history["circuit_id"].astype(str).eq(str(circuit_id))
    ].copy()
    if team.empty:
        return np.nan, np.nan, np.nan
    events = team.groupby(["season", "round"], as_index=False).agg(
        avg_finish=("position_order", "mean"), positions_gained=("positions_gained", "mean"),
        dnf_rate=("is_dnf", "mean"),
    )
    return (
        float(events["avg_finish"].mean()),
        float(events["positions_gained"].mean()),
        float(events["dnf_rate"].mean()),
    )


def _circuit_neutralisation_history(
    history: pd.DataFrame, race_control: pd.DataFrame | None, circuit_id: Any
) -> tuple[float, float]:
    if race_control is None or race_control.empty or history.empty:
        return np.nan, np.nan
    control = race_control.copy()
    session = control.get("session_code", control.get("session", pd.Series("", index=control.index))).astype(str).str.upper()
    control = control[session.isin(["R", "RACE"])].copy()
    if control.empty or "message" not in control:
        return np.nan, np.nan
    for col in ("season", "round"):
        control[col] = pd.to_numeric(control.get(col), errors="coerce")
    control = control.dropna(subset=["season", "round"])
    if control.empty:
        return np.nan, np.nan
    event_messages = control.groupby(["season", "round"])["message"].apply(
        lambda values: " | ".join(values.dropna().astype(str).str.upper())
    ).reset_index(name="messages")
    event_messages["has_vsc"] = event_messages["messages"].str.contains(r"VIRTUAL SAFETY CAR|\bVSC\b", regex=True).astype(float)
    event_messages["has_sc"] = event_messages["messages"].str.contains(r"\bSAFETY CAR\b", regex=True).astype(float)
    event_messages["has_sc"] = np.where(event_messages["has_vsc"].eq(1), 0.0, event_messages["has_sc"])
    event_meta = history[["season", "round", "circuit_id"]].drop_duplicates(["season", "round"])
    known = event_messages.merge(event_meta, on=["season", "round"], how="inner")
    if known.empty:
        return np.nan, np.nan
    # Only events with race-control records enter the denominator. Missing coverage
    # is unknown, not a no-Safety-Car event. Sparse circuit samples are shrunk
    # toward the prior-event global rate (five-event prior weight).
    global_sc = float(known["has_sc"].mean())
    global_vsc = float(known["has_vsc"].mean())
    local = known[known["circuit_id"].astype(str).eq(str(circuit_id))]
    if local.empty:
        return global_sc, global_vsc
    prior_weight = 5.0
    sc_rate = (float(local["has_sc"].sum()) + global_sc * prior_weight) / (len(local) + prior_weight)
    vsc_rate = (float(local["has_vsc"].sum()) + global_vsc * prior_weight) / (len(local) + prior_weight)
    return sc_rate, vsc_rate


def _valid_practice_laps(laps: pd.DataFrame) -> pd.DataFrame:
    frame = laps.copy()
    frame["lap_time_seconds"] = _num(frame, "lap_time_seconds")
    frame = frame[frame["lap_time_seconds"].between(40, 200)].copy()
    if "is_accurate" in frame and frame["is_accurate"].notna().any():
        accurate = frame["is_accurate"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame = frame[accurate]
    for flag in ("pit_in", "pit_out", "deleted", "fastf1_generated"):
        if flag in frame and frame[flag].notna().any():
            truth = frame[flag].astype(str).str.lower().isin(["true", "1", "yes"])
            frame = frame[~truth]
    if "track_status" in frame and frame["track_status"].notna().any():
        frame = frame[frame["track_status"].astype(str).str.strip().eq("1")]
    return frame


def _current_constructor_practice(
    session_laps: pd.DataFrame | None, season: int, round_number: int, candidates: pd.DataFrame, as_of_session: str = "FP3"
) -> pd.DataFrame:
    """Aggregate FP1-FP3 one-lap, long-run and compound degradation evidence by team."""
    empty = pd.DataFrame(columns=["constructor_id", *CONSTRUCTOR_INTELLIGENCE_FEATURES[:11]])
    if session_laps is None or session_laps.empty or candidates.empty:
        return empty
    laps = session_laps.copy()
    laps["season"] = _num(laps, "season")
    laps["round"] = _num(laps, "round")
    laps["session"] = laps.get("session", pd.Series("", index=laps.index)).astype(str).str.upper()
    session_order = {"FP1": 1, "FP2": 2, "FP3": 3}
    cutoff = session_order.get(str(as_of_session).upper())
    if cutoff is None:
        raise ValueError("as_of_session must be FP1, FP2 or FP3")
    allowed_sessions = [name for name, order in session_order.items() if order <= cutoff]
    laps = laps[
        laps["season"].eq(season) & laps["round"].eq(round_number)
        & laps["session"].isin(allowed_sessions)
    ].copy()
    if laps.empty or "driver_code" not in laps:
        return empty

    roster = candidates[[c for c in ("driver_id", "driver_code", "constructor_id") if c in candidates]].dropna(subset=["constructor_id"]).copy()
    roster["driver_code"] = roster["driver_code"].astype(str).str.upper().str.strip()
    roster = roster.drop_duplicates("driver_code")
    if roster.empty:
        return empty
    weekend = build_event_weekend_pace_features(laps, season, round_number, as_of_session)
    if not weekend.empty:
        weekend = weekend.merge(roster, on="driver_code", how="left", validate="many_to_one")
        weekend = weekend.dropna(subset=["constructor_id"])
    else:
        weekend = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    teams = roster["constructor_id"].dropna().unique().tolist()
    if not weekend.empty:
        pace_by_team = weekend.groupby("constructor_id").agg(
            constructor_practice_best_gap_pct=("fp_best_gap_pct", "mean"),
            constructor_practice_latest_best_gap_pct=("fp_latest_best_gap_pct", "mean"),
            constructor_practice_soft_gap_pct=("fp_best_soft_gap_pct", "mean"),
            constructor_practice_longrun_gap_pct=("fp_mean_longrun_gap_pct", "mean"),
            constructor_practice_longrun_laps=("fp_longrun_laps", "sum"),
            constructor_practice_longrun_stints=("fp_longrun_stints", "sum"),
        )
    else:
        pace_by_team = pd.DataFrame()

    valid = _valid_practice_laps(laps)
    valid["driver_code"] = valid["driver_code"].astype(str).str.upper().str.strip()
    valid = valid.merge(roster, on="driver_code", how="left", validate="many_to_one").dropna(subset=["constructor_id"])
    for col in ("stint", "tyre_life"):
        valid[col] = _num(valid, col)
    valid["compound"] = valid.get("compound", pd.Series("UNKNOWN", index=valid.index)).fillna("UNKNOWN").astype(str).str.upper()

    slope_rows: list[dict[str, Any]] = []
    if not valid.empty and valid["stint"].notna().any() and valid["tyre_life"].notna().any():
        for (team, driver, session, stint, compound), group in valid.dropna(subset=["stint", "tyre_life"]).groupby(
            ["constructor_id", "driver_code", "session", "stint", "compound"]
        ):
            run = group.sort_values("tyre_life").drop_duplicates("tyre_life")
            if len(run) < 5 or run["tyre_life"].nunique() < 4:
                continue
            run = run.iloc[1:]  # remove one warm-up lap
            slope = float(np.polyfit(run["tyre_life"].to_numpy(dtype=float), run["lap_time_seconds"].to_numpy(dtype=float), 1)[0])
            if np.isfinite(slope):
                slope_rows.append({"constructor_id": team, "compound": compound, "slope": slope, "laps": len(run)})
    slopes = pd.DataFrame(slope_rows)
    team_slopes = {}
    if not slopes.empty:
        grouped = slopes.groupby(["constructor_id", "compound"])["slope"].median()
        for team in teams:
            team_slopes[team] = {
                str(compound).lower(): float(value)
                for (constructor, compound), value in grouped.items() if str(constructor) == str(team)
            }

    for team in teams:
        row = {"constructor_id": team}
        if team in pace_by_team.index:
            row.update(pace_by_team.loc[team].to_dict())
        else:
            row.update({c: np.nan for c in [
                "constructor_practice_best_gap_pct", "constructor_practice_latest_best_gap_pct",
                "constructor_practice_soft_gap_pct",
                "constructor_practice_longrun_gap_pct", "constructor_practice_longrun_laps",
                "constructor_practice_longrun_stints",
            ]})
        compound_slopes = team_slopes.get(team, {})
        row["constructor_practice_compound_count"] = len(compound_slopes) if compound_slopes else np.nan
        for compound in ("soft", "medium", "hard"):
            row[f"constructor_practice_{compound}_deg_sec_per_lap"] = compound_slopes.get(compound, np.nan)
        row["constructor_practice_tyre_deg_sec_per_lap"] = (
            float(np.nanmedian(list(compound_slopes.values()))) if compound_slopes else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)



def _recent_constructor_pit_metrics(
    history: pd.DataFrame, pit_stops: pd.DataFrame | None, openf1_drivers: pd.DataFrame | None, constructor_id: Any
) -> tuple[float, float, float]:
    """Summarise recent race pit-stop durations for a constructor, when linkable."""
    if history.empty or pit_stops is None or pit_stops.empty or openf1_drivers is None or openf1_drivers.empty:
        return np.nan, np.nan, np.nan
    pit = pit_stops.copy()
    drivers = openf1_drivers.copy()
    if not {"session_key", "driver_number"}.issubset(pit.columns) or not {"session_key", "driver_number", "name_acronym"}.issubset(drivers.columns):
        return np.nan, np.nan, np.nan
    session = pit.get("session_code", pit.get("session", pd.Series("R", index=pit.index))).astype(str).str.upper()
    pit = pit[session.isin(["R", "RACE"])].copy()
    if pit.empty:
        return np.nan, np.nan, np.nan
    pit["pit_duration"] = pd.to_numeric(pit.get("pit_duration", pit.get("duration", pd.Series(np.nan, index=pit.index))), errors="coerce")
    pit = pit[pit["pit_duration"].between(1.0, 120.0)].copy()
    if pit.empty:
        return np.nan, np.nan, np.nan
    pit["driver_number"] = pd.to_numeric(pit["driver_number"], errors="coerce")
    drivers["driver_number"] = pd.to_numeric(drivers["driver_number"], errors="coerce")
    lookup = drivers[["session_key", "driver_number", "name_acronym"]].dropna().drop_duplicates(["session_key", "driver_number"])
    pit = pit.merge(lookup, on=["session_key", "driver_number"], how="inner")
    pit["driver_code"] = pit["name_acronym"].astype(str).str.upper().str.strip()
    if not {"season", "round", "driver_code", "constructor_id"}.issubset(history.columns):
        return np.nan, np.nan, np.nan
    entries = history[["season", "round", "driver_code", "constructor_id"]].dropna().copy()
    entries["season"] = pd.to_numeric(entries["season"], errors="coerce")
    entries["round"] = pd.to_numeric(entries["round"], errors="coerce")
    for col in ("season", "round"):
        pit[col] = pd.to_numeric(pit.get(col), errors="coerce")
    linked = pit.merge(entries, on=["season", "round", "driver_code"], how="inner")
    team = linked[linked["constructor_id"].astype(str).eq(str(constructor_id))]
    if team.empty:
        return np.nan, np.nan, np.nan
    events = team.groupby(["season", "round"], as_index=False).agg(
        mean_duration=("pit_duration", "mean"), median_duration=("pit_duration", "median"),
        stop_count=("pit_duration", "count"),
    ).sort_values(["season", "round"]).tail(5)
    return (
        float(events["mean_duration"].mean()),
        float(events["median_duration"].median()),
        float(events["stop_count"].mean()),
    )

def _features_for_event(
    history: pd.DataFrame,
    event_candidates: pd.DataFrame,
    session_laps: pd.DataFrame | None,
    race_control: pd.DataFrame | None,
    season: int,
    round_number: int,
    circuit_id: Any,
    as_of_session: str = "FP3",
    pit_stops: pd.DataFrame | None = None,
    openf1_drivers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current = _current_constructor_practice(session_laps, season, round_number, event_candidates, as_of_session)
    current_map = current.set_index("constructor_id").to_dict(orient="index") if not current.empty else {}
    for candidate in event_candidates.drop_duplicates("driver_id").itertuples(index=False):
        driver_id = getattr(candidate, "driver_id", None)
        constructor_id = getattr(candidate, "constructor_id", None)
        constructor_dnf, mechanical_dnf = _recent_team_reliability(history, constructor_id)
        pit_mean, pit_median, pit_count = _recent_constructor_pit_metrics(history, pit_stops, openf1_drivers, constructor_id)
        circuit_finish, circuit_gain, circuit_dnf = _circuit_team_history(history, constructor_id, circuit_id)
        sc_rate, vsc_rate = _circuit_neutralisation_history(history, race_control, circuit_id)
        row = {
            "season": season, "round": round_number, "driver_id": driver_id,
            "constructor_id": constructor_id,
            "constructor_dnf_rate_recent5": constructor_dnf,
            "constructor_mechanical_dnf_rate_recent5": mechanical_dnf,
            "constructor_pit_stop_mean_duration_recent5_sec": pit_mean,
            "constructor_pit_stop_median_duration_recent5_sec": pit_median,
            "constructor_pit_stop_count_recent5": pit_count,
            "driver_dnf_rate_recent5": _recent_driver_reliability(history, driver_id),
            "constructor_circuit_avg_finish_prev": circuit_finish,
            "constructor_circuit_positions_gained_prev": circuit_gain,
            "constructor_circuit_dnf_rate_prev": circuit_dnf,
            "circuit_sc_event_rate_prev": sc_rate,
            "circuit_vsc_event_rate_prev": vsc_rate,
        }
        row.update(current_map.get(constructor_id, {}))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id", *CONSTRUCTOR_INTELLIGENCE_FEATURES])
    result = pd.DataFrame(rows)
    for feature in CONSTRUCTOR_INTELLIGENCE_FEATURES:
        if feature not in result:
            result[feature] = np.nan
    return result[["season", "round", "driver_id", "constructor_id", *CONSTRUCTOR_INTELLIGENCE_FEATURES]]


def build_historical_constructor_intelligence_table(
    race_results: pd.DataFrame,
    session_laps: pd.DataFrame | None = None,
    race_control: pd.DataFrame | None = None,
    as_of_session: str = "FP3",
    pit_stops: pd.DataFrame | None = None,
    openf1_drivers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build pre-race constructor feature rows for all historical event entries."""
    races = _clean_results(race_results)
    if races.empty:
        return pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id", *CONSTRUCTOR_INTELLIGENCE_FEATURES])
    output: list[pd.DataFrame] = []
    for (season, rnd), event in races.groupby(["season", "round"], sort=True):
        history = races[(races["season"] < season) | ((races["season"] == season) & (races["round"] < rnd))]
        circuit = event["circuit_id"].dropna().iloc[0] if event["circuit_id"].notna().any() else None
        features = _features_for_event(
            history, event, session_laps, race_control, int(season), int(rnd), circuit,
            as_of_session, pit_stops, openf1_drivers,
        )
        output.append(features)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def build_constructor_intelligence_for_event(
    race_results: pd.DataFrame,
    candidates: pd.DataFrame,
    session_laps: pd.DataFrame | None,
    race_control: pd.DataFrame | None,
    season: int,
    round_number: int,
    circuit_id: Any,
    as_of_session: str = "FP3",
    pit_stops: pd.DataFrame | None = None,
    openf1_drivers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build inference features; only prior race results and current FP laps are used."""
    races = _clean_results(race_results)
    history = races[(races["season"] < season) | ((races["season"] == season) & (races["round"] < round_number))]
    return _features_for_event(
        history, candidates, session_laps, race_control, season, round_number,
        circuit_id, as_of_session, pit_stops, openf1_drivers,
    )
