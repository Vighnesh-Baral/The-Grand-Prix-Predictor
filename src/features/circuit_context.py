"""Leakage-safe circuit/grid conversion features for race forecasting.

Features are estimated only from completed races strictly before the target GP.
Bayesian smoothing prevents one or two historical observations from dominating.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CIRCUIT_RACE_FEATURES = [
    "circuit_grid_win_rate_prev",
    "circuit_grid_podium_rate_prev",
    "circuit_pole_to_win_rate_prev",
    "circuit_p2_p3_to_win_rate_prev",
    "circuit_front3_podium_rate_prev",
    "circuit_grid_positions_gained_prev",
    "circuit_history_event_count_prev",
    "circuit_grid_sample_size_prev",
]


def _clean_results(results: pd.DataFrame | None) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame(columns=["season", "round", "date", "circuit_id", "driver_id", "grid", "position_order"])
    frame = results.copy()
    for column in ("season", "round", "grid", "position_order"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("circuit_id", "driver_id"):
        if column not in frame:
            frame[column] = None
    if "date" not in frame:
        frame["date"] = pd.NaT
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["season", "round", "circuit_id", "driver_id", "grid", "position_order"]).copy()
    frame = frame[(frame["grid"] > 0) & (frame["position_order"] > 0)]
    frame["is_win"] = frame["position_order"].eq(1).astype(int)
    frame["is_podium"] = frame["position_order"].le(3).astype(int)
    frame["positions_gained"] = frame["grid"] - frame["position_order"]
    return frame.sort_values(["date", "season", "round"], na_position="last")


def _rate(successes: int, attempts: int, prior_rate: float, prior_strength: float = 15.0) -> float:
    return float((successes + prior_rate * prior_strength) / (attempts + prior_strength))


def _stats_for_grid(history: pd.DataFrame, grid: float, global_history: pd.DataFrame) -> dict[str, float]:
    """Compute smoothed circuit/grid signals from strictly prior records."""
    local_grid = history[history["grid"].eq(grid)] if not history.empty else history
    global_grid = global_history[global_history["grid"].eq(grid)] if not global_history.empty else global_history
    global_win_prior = float(global_grid["is_win"].mean()) if not global_grid.empty else 1.0 / 20.0
    global_podium_prior = float(global_grid["is_podium"].mean()) if not global_grid.empty else 3.0 / 20.0
    grid_win = _rate(int(local_grid["is_win"].sum()), len(local_grid), global_win_prior)
    grid_podium = _rate(int(local_grid["is_podium"].sum()), len(local_grid), global_podium_prior)

    pole = history[history["grid"].eq(1)] if not history.empty else history
    front23 = history[history["grid"].isin([2, 3])] if not history.empty else history
    front3 = history[history["grid"].isin([1, 2, 3])] if not history.empty else history
    global_pole = global_history[global_history["grid"].eq(1)] if not global_history.empty else global_history
    global_front23 = global_history[global_history["grid"].isin([2, 3])] if not global_history.empty else global_history
    global_front3 = global_history[global_history["grid"].isin([1, 2, 3])] if not global_history.empty else global_history
    pole_prior = float(global_pole["is_win"].mean()) if not global_pole.empty else 1.0 / 20.0
    p23_prior = float(global_front23["is_win"].mean()) if not global_front23.empty else 1.0 / 20.0
    front3_podium_prior = float(global_front3["is_podium"].mean()) if not global_front3.empty else 3.0 / 20.0
    gain_prior = float(global_history["positions_gained"].mean()) if not global_history.empty else 0.0

    return {
        "circuit_grid_win_rate_prev": grid_win,
        "circuit_grid_podium_rate_prev": grid_podium,
        "circuit_pole_to_win_rate_prev": _rate(int(pole["is_win"].sum()), len(pole), pole_prior),
        "circuit_p2_p3_to_win_rate_prev": _rate(int(front23["is_win"].sum()), len(front23), p23_prior),
        "circuit_front3_podium_rate_prev": _rate(int(front3["is_podium"].sum()), len(front3), front3_podium_prior),
        "circuit_grid_positions_gained_prev": _rate(
            0, 0, gain_prior, 1.0
        ) if history.empty else float((history["positions_gained"].sum() + gain_prior * 15.0) / (len(history) + 15.0)),
        "circuit_history_event_count_prev": float(history[["season", "round"]].drop_duplicates().shape[0]),
        "circuit_grid_sample_size_prev": float(len(local_grid)),
    }


def build_historical_circuit_features(race_results: pd.DataFrame | None) -> pd.DataFrame:
    """Return one row per event entrant with pre-event circuit conversion features."""
    frame = _clean_results(race_results)
    if frame.empty:
        return pd.DataFrame(columns=["season", "round", "driver_id", *CIRCUIT_RACE_FEATURES])
    output: list[dict[str, Any]] = []
    all_prior = frame.iloc[0:0].copy()
    for (season, rnd), event in frame.groupby(["season", "round"], sort=False):
        circuit = event["circuit_id"].iloc[0]
        circuit_prior = all_prior[all_prior["circuit_id"].astype(str).eq(str(circuit))]
        all_stats = all_prior
        for row in event.itertuples(index=False):
            stats = _stats_for_grid(circuit_prior, float(row.grid), all_stats)
            output.append({
                "season": int(season), "round": int(rnd), "driver_id": row.driver_id,
                **stats,
            })
        all_prior = pd.concat([all_prior, event], ignore_index=True)
    return pd.DataFrame(output)


def circuit_features_for_candidates(
    prior_results: pd.DataFrame | None,
    candidates: pd.DataFrame,
    circuit_id: str | None,
) -> pd.DataFrame:
    """Build circuit conversion features for future entrants after grid is known."""
    history = _clean_results(prior_results)
    rows: list[dict[str, Any]] = []
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=["driver_id", *CIRCUIT_RACE_FEATURES])
    local = history[history["circuit_id"].astype(str).eq(str(circuit_id))] if circuit_id is not None else history.iloc[0:0]
    for candidate in candidates.itertuples(index=False):
        grid = getattr(candidate, "grid", np.nan)
        try:
            grid_value = float(grid)
        except (TypeError, ValueError):
            grid_value = np.nan
        stats = _stats_for_grid(local, grid_value, history) if np.isfinite(grid_value) and grid_value > 0 else {name: np.nan for name in CIRCUIT_RACE_FEATURES}
        rows.append({"driver_id": getattr(candidate, "driver_id", None), **stats})
    return pd.DataFrame(rows)
