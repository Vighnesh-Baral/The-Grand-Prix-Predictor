"""Inference helpers for the trained F1 prediction bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.features.build_features import (
    CATEGORICAL_FEATURES, POLE_NUMERIC_FEATURES, POLE_PRE_FP_NUMERIC_FEATURES,
    RACE_NUMERIC_FEATURES,
)


class ModelBundleError(RuntimeError):
    pass


def load_model_bundle(path: Path | str = ARTIFACTS_DIR / "f1_models.joblib") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ModelBundleError(f"Trained models not found at {path}. Run: python -m src.models.train_models")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "models" not in bundle:
        raise ModelBundleError(f"The model file at {path} is not a recognised F1 Race Intelligence bundle")
    required = {"pole", "position", "winner", "podium", "dnf"}
    missing = required.difference(bundle["models"])
    if missing:
        raise ModelBundleError(f"The model bundle is missing tasks: {', '.join(sorted(missing))}")
    return bundle


def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    probabilities = model.predict_proba(X)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def _relative_share(values: np.ndarray) -> np.ndarray:
    """Normalize non-negative model scores within a race field, not a calibration."""
    scores = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(scores.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(len(scores), 1.0 / len(scores)) if len(scores) else scores
    return scores / total


def _percentile_strength(frame: pd.DataFrame, columns: list[tuple[str, bool]]) -> pd.Series:
    """Field-relative feature strength; lower_is_better controls rank direction."""
    values: list[pd.Series] = []
    for column, lower_is_better in columns:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() < 2:
            continue
        rank = numeric.rank(pct=True, ascending=not lower_is_better)
        values.append(rank.where(numeric.notna()))
    if not values:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(values, axis=1).mean(axis=1, skipna=True)


def _priority_pole_score(
    event_frame: pd.DataFrame, *, before_free_practice: bool = False
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Transparent priority-weighted evidence score for pole prediction.

    Before FP, only strictly historical driver/constructor/circuit information is
    considered. After FP begins, current-session pace is added progressively.
    These are design weights, not calibrated probabilities.
    """
    if before_free_practice:
        groups = {
            "constructor_season_strength": (0.35, [
                ("constructor_quali_gap_season_prev_pct", True),
                ("constructor_quali_gap_recent3_pct", True),
                ("constructor_season_prev_avg_finish", True),
                ("constructor_last_gp_avg_finish", True),
            ]),
            "driver_recent_form": (0.25, [
                ("driver_quali_gap_recent3_pct", True),
                ("driver_recent_avg_finish", True),
                ("driver_season_prev_avg_finish", True),
            ]),
            "previous_grand_prix_form": (0.20, [
                ("driver_last_gp_finish", True),
                ("constructor_last_gp_avg_finish", True),
            ]),
            "teammate_relative": (0.10, [
                ("driver_teammate_quali_delta_prev_pct", True),
            ]),
            "circuit_suitability": (0.07, [
                ("driver_circuit_prev_avg_finish", True),
                ("constructor_circuit_quali_gap_prev_avg_pct", True),
            ]),
            "historical_pole_record": (0.03, [
                ("driver_circuit_prev_poles", False),
                ("driver_prev_poles", False),
            ]),
        }
    else:
        groups = {
        "weekend_pace": (0.28, [
            ("fp_best_gap_pct", True), ("fp_latest_best_gap_pct", True),
            ("fp_best_soft_gap_pct", True), ("fp_best_time_gap_pct", True),
            ("fp_mean_time_gap_pct", True), ("fp1_best_gap_pct", True),
            ("fp2_best_gap_pct", True), ("fp3_best_gap_pct", True),
            ("fp1_soft_gap_pct", True), ("fp2_soft_gap_pct", True),
            ("fp3_soft_gap_pct", True),
        ]),
        "constructor_strength": (0.23, [
            ("constructor_quali_gap_recent3_pct", True),
            ("constructor_quali_gap_season_prev_pct", True),
            ("constructor_circuit_quali_gap_prev_avg_pct", True),
            ("constructor_practice_latest_best_gap_pct", True),
            ("constructor_quali_gap_prev_avg_pct", True),
            ("constructor_season_prev_avg_finish", True),
            ("constructor_prev_avg_finish", True),
        ]),
        "driver_form": (0.16, [
            ("driver_quali_gap_recent3_pct", True),
            ("driver_recent_avg_finish", True),
            ("driver_season_prev_avg_finish", True),
            ("driver_prev_avg_finish", True),
            ("driver_prev_wins", False), ("driver_prev_podiums", False),
        ]),
        "teammate_relative": (0.11, [("driver_teammate_quali_delta_prev_pct", True)]),
        "circuit_suitability": (0.08, [
            ("driver_circuit_prev_avg_finish", True),
        ]),
        "historical_pole_record": (0.05, [
            ("driver_circuit_prev_poles", False), ("driver_prev_poles", False),
        ]),
    }
    component_scores: dict[str, pd.Series] = {}
    weighted_total = pd.Series(0.0, index=event_frame.index)
    available_weight = pd.Series(0.0, index=event_frame.index)
    for name, (weight, columns) in groups.items():
        score = _percentile_strength(event_frame, columns)
        component_scores[name] = score
        valid = score.notna()
        weighted_total.loc[valid] += score.loc[valid] * weight
        available_weight.loc[valid] += weight
    weather_columns = [
        "weather_air_temp_c", "weather_track_temp_c", "weather_humidity_pct",
        "weather_wind_speed_mps", "weather_rainfall_fraction",
    ]
    has_weather = any(
        column in event_frame and pd.to_numeric(event_frame[column], errors="coerce").notna().any()
        for column in weather_columns
    )
    weather_score = pd.Series(0.5, index=event_frame.index, dtype=float) if has_weather else pd.Series(np.nan, index=event_frame.index, dtype=float)
    component_scores["weather_context"] = weather_score
    if has_weather:
        weighted_total += weather_score * 0.09
        available_weight += 0.09
    priority_score = weighted_total.div(available_weight.replace(0, np.nan))
    return priority_score.fillna(0.5), component_scores


def predict_pole(
    bundle: dict[str, Any], event_frame: pd.DataFrame, as_of_session: str = "FP3"
) -> pd.DataFrame:
    if event_frame.empty:
        return pd.DataFrame(columns=["driver_name", "constructor_name", "pole_probability", "pole_score_share"])
    cutoff = str(as_of_session).upper()
    if cutoff == "PRE_FP":
        model_key = "pole_pre_fp"
        numeric_features = POLE_PRE_FP_NUMERIC_FEATURES
    elif cutoff in {"FP1", "FP2", "FP3"}:
        model_key = f"pole_{cutoff.lower()}"
        numeric_features = POLE_NUMERIC_FEATURES
    else:
        raise ValueError("as_of_session must be PRE_FP, FP1, FP2 or FP3")
    model = bundle["models"].get(model_key)
    if model is None and cutoff == "FP3":
        model = bundle["models"].get("pole")
    if model is None:
        raise ModelBundleError(
            f"This model bundle has no {cutoff}-specific pole model. Retrain with "
            "python -m src.models.train_models. A model trained with later-session features cannot be reused at an earlier cutoff."
        )
    columns = CATEGORICAL_FEATURES + numeric_features
    X = event_frame.reindex(columns=columns)
    result = event_frame[["driver_name", "constructor_name"]].copy()
    ml_probability = _positive_probability(model, X)
    priority_score, components = _priority_pole_score(event_frame, before_free_practice=cutoff == "PRE_FP")
    ml_strength = pd.Series(ml_probability, index=event_frame.index).rank(pct=True)
    result["ml_model_strength"] = ml_strength.to_numpy()
    result["priority_score"] = priority_score.to_numpy()
    for name, score in components.items():
        result[f"{name}_score"] = score.to_numpy()
    # Equal blend of the trained classifier's field ranking and the explicit
    # priority hierarchy. The displayed share remains relative, not calibrated.
    result["pole_probability"] = (0.50 * ml_strength + 0.50 * priority_score).to_numpy()
    result["pole_score_share"] = _relative_share(result["pole_probability"].to_numpy())
    result["prediction_basis"] = "pre_free_practice_historical_form_ensemble" if cutoff == "PRE_FP" else "priority_weighted_weekend_aware_ml_ensemble"
    result["forecast_cutoff"] = "before_free_practice" if cutoff == "PRE_FP" else cutoff
    return result.sort_values(["pole_score_share", "pole_probability"], ascending=False).reset_index(drop=True)


RACE_PRIORITY_WEIGHTS = {
    "constructor_performance": 0.20,
    "weekend_race_pace": 0.18,
    "grid_circuit_advantage": 0.18,
    "driver_form": 0.12,
    "tyre_strategy": 0.12,
    "reliability": 0.07,
    "circuit_characteristics": 0.05,
    "weather_context": 0.08,
}


def _priority_race_score(event_frame: pd.DataFrame) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Expert-weighted field score. Weights are design priors, not measured accuracy."""
    groups: dict[str, tuple[float, list[tuple[str, bool]]]] = {
        "constructor_performance": (0.20, [
            ("constructor_quali_gap_recent3_pct", True),
            ("constructor_quali_gap_season_prev_pct", True),
            ("constructor_fastest_lap_gap_recent3_pct", True),
            ("constructor_race_longrun_gap_recent3_pct", True),
            ("constructor_practice_latest_best_gap_pct", True),
        ]),
        "weekend_race_pace": (0.18, [
            ("constructor_practice_longrun_gap_pct", True),
            ("fp_latest_longrun_gap_pct", True),
            ("fp_mean_longrun_gap_pct", True),
            ("constructor_practice_longrun_laps", False),
            ("constructor_practice_longrun_stints", False),
        ]),
        "grid_circuit_advantage": (0.18, [
            ("grid", True), ("circuit_grid_win_rate_prev", False),
            ("circuit_grid_podium_rate_prev", False),
            ("circuit_pole_to_win_rate_prev", False),
            ("circuit_p2_p3_to_win_rate_prev", False),
            ("circuit_grid_positions_gained_prev", False),
        ]),
        "driver_form": (0.12, [
            ("driver_recent_avg_finish", True), ("driver_prev_avg_finish", True),
            ("driver_prev_wins", False), ("driver_prev_podiums", False),
            ("driver_fastest_lap_gap_prev_avg_pct", True),
        ]),
        "tyre_strategy": (0.12, [
            ("constructor_practice_tyre_deg_sec_per_lap", True),
            ("constructor_practice_soft_deg_sec_per_lap", True),
            ("constructor_practice_medium_deg_sec_per_lap", True),
            ("constructor_practice_hard_deg_sec_per_lap", True),
            ("constructor_race_stint_slope_prev_avg_sec_per_lap", True),
            ("constructor_pit_stop_mean_duration_recent5_sec", True),
            ("constructor_pit_stop_median_duration_recent5_sec", True),
        ]),
        "reliability": (0.07, [
            ("constructor_dnf_rate_recent5", True),
            ("constructor_mechanical_dnf_rate_recent5", True),
            ("driver_dnf_rate_recent5", True),
        ]),
        "circuit_characteristics": (0.05, [
            ("constructor_circuit_avg_finish_prev", True),
            ("constructor_circuit_positions_gained_prev", False),
            ("constructor_circuit_dnf_rate_prev", True),
        ]),
    }
    components: dict[str, pd.Series] = {}
    weighted = pd.Series(0.0, index=event_frame.index)
    available_weight = pd.Series(0.0, index=event_frame.index)
    for name, (weight, columns) in groups.items():
        score = _percentile_strength(event_frame, columns)
        components[name] = score
        valid = score.notna()
        weighted.loc[valid] += score.loc[valid] * weight
        available_weight.loc[valid] += weight

    # Weather is an event-level context shared by all entrants. It informs the
    # trained ML model and is exposed as coverage, but must not fake a driver edge.
    weather_columns = [
        "race_forecast_air_temp_c", "race_forecast_precipitation_mm",
        "race_forecast_wind_speed_kmh", "race_forecast_is_wet",
        "weather_air_temp_c", "weather_track_temp_c", "weather_rainfall_fraction",
    ]
    has_weather = any(
        column in event_frame and pd.to_numeric(event_frame[column], errors="coerce").notna().any()
        for column in weather_columns
    )
    weather_score = pd.Series(0.5, index=event_frame.index, dtype=float) if has_weather else pd.Series(np.nan, index=event_frame.index, dtype=float)
    components["weather_context"] = weather_score
    if has_weather:
        weighted += weather_score * RACE_PRIORITY_WEIGHTS["weather_context"]
        available_weight += RACE_PRIORITY_WEIGHTS["weather_context"]
    score = weighted.div(available_weight.replace(0, np.nan)).fillna(0.5)
    return score, components


def _position_residual_scale(bundle: dict[str, Any], field_size: int) -> tuple[float, str]:
    metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
    task = metadata.get("tasks", {}).get("position", {}) if isinstance(metadata, dict) else {}
    leaderboard = task.get("leaderboard", {}) if isinstance(task, dict) else {}
    selected = task.get("selected_model") if isinstance(task, dict) else None
    validation = leaderboard.get(selected, {}).get("validation", {}) if selected in leaderboard else {}
    value = validation.get("validation_residual_std_positions") if isinstance(validation, dict) else None
    try:
        scale = float(value)
        if np.isfinite(scale) and scale > 0:
            return float(np.clip(scale, 1.0, max(2.0, field_size / 2))), "held_out_validation_residuals"
    except (TypeError, ValueError):
        pass
    return max(1.75, field_size * 0.12), "conservative_fallback_no_validation_residual_metadata"


def _simulate_race_outcomes(
    expected_position: np.ndarray,
    dnf_probability: np.ndarray,
    *,
    residual_scale: float,
    season: int,
    round_number: int,
    neutralisation_probability: float = 0.0,
    weather_risk: float = 0.0,
    simulations: int = 4000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Field-coherent Monte Carlo including Safety-Car/weather volatility.

    A neutralisation compresses pace gaps and increases outcome variance. This
    represents uncertainty from a Safety Car/VSC scenario; it does not claim to
    predict whether one will occur or when it will deploy.
    """
    n = len(expected_position)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    rng = np.random.default_rng(int(season) * 1009 + int(round_number) * 9176 + 42)
    base_position = np.asarray(expected_position, dtype=float)
    neutralisation_probability = float(np.clip(neutralisation_probability, 0.0, 0.95))
    weather_risk = float(np.clip(weather_risk, 0.0, 1.0))
    neutralised = rng.random((simulations, 1)) < neutralisation_probability
    field_centre = float(np.nanmean(base_position))
    # A Safety Car/VSC compresses the field's expected pace gaps. Wet conditions
    # and neutralisations both increase uncertainty around the expected order.
    compressed_position = field_centre + (base_position - field_centre) * 0.72
    scenario_mean = np.where(neutralised, compressed_position[None, :], base_position[None, :])
    scenario_scale = max(float(residual_scale), 0.5) * (1.0 + 0.35 * neutralised + 0.25 * weather_risk)
    latent = rng.normal(loc=scenario_mean, scale=scenario_scale, size=(simulations, n))
    retired = rng.random((simulations, n)) < np.clip(np.asarray(dnf_probability, dtype=float), 0.0, 0.95)
    # DNF drivers are ordered behind finishers in the projected classification.
    # Their relative DNF order is deliberately treated as uncertain.
    latent += retired * (n + rng.uniform(0.1, n * 0.5, size=(simulations, n)))
    order = np.argsort(latent, axis=1)
    ranks = np.empty_like(order)
    row_idx = np.arange(simulations)[:, None]
    ranks[row_idx, order] = np.arange(1, n + 1)[None, :]
    win_prob = (ranks == 1).mean(axis=0)
    podium_prob = (ranks <= min(3, n)).mean(axis=0)
    expected_finish = ranks.mean(axis=0)
    return win_prob, podium_prob, expected_finish


def predict_race(bundle: dict[str, Any], event_frame: pd.DataFrame) -> pd.DataFrame:
    if event_frame.empty:
        return pd.DataFrame(columns=[
            "driver_name", "constructor_name", "expected_classified_position", "expected_finish_position",
            "win_probability", "win_score_share", "podium_probability", "dnf_probability", "predicted_order",
        ])
    models = bundle["models"]
    columns = CATEGORICAL_FEATURES + RACE_NUMERIC_FEATURES
    X = event_frame.reindex(columns=columns)
    result = event_frame[["driver_name", "constructor_name"]].copy()
    if "grid" in event_frame:
        result["starting_grid_position"] = pd.to_numeric(event_frame["grid"], errors="coerce").to_numpy()
    ml_position = np.asarray(models["position"].predict(X), dtype=float)
    ml_win_probability = _positive_probability(models["winner"], X)
    dnf_probability = np.clip(_positive_probability(models["dnf"], X), 0.0, 0.95)

    priority_score, components = _priority_race_score(event_frame)
    ml_position_strength = pd.Series(-ml_position, index=event_frame.index).rank(pct=True).to_numpy()
    ml_win_strength = pd.Series(ml_win_probability, index=event_frame.index).rank(pct=True).to_numpy()
    ml_strength = 0.65 * ml_position_strength + 0.35 * ml_win_strength
    combined_strength = 0.50 * ml_strength + 0.50 * priority_score.to_numpy()
    expert_rank = priority_score.rank(method="average", ascending=False).to_numpy(dtype=float)
    n = len(event_frame)
    rank_position_from_strength = 1.0 + (n - 1) * (1.0 - combined_strength)
    expected_classified = 0.50 * ml_position + 0.50 * (0.65 * expert_rank + 0.35 * rank_position_from_strength)

    first = event_frame.iloc[0]
    season = int(pd.to_numeric(first.get("season", 0), errors="coerce") or 0)
    round_number = int(pd.to_numeric(first.get("round", 0), errors="coerce") or 0)
    residual_scale, residual_source = _position_residual_scale(bundle, n)
    sc_values = pd.to_numeric(event_frame.get("circuit_sc_event_rate_prev", pd.Series(np.nan, index=event_frame.index)), errors="coerce")
    vsc_values = pd.to_numeric(event_frame.get("circuit_vsc_event_rate_prev", pd.Series(np.nan, index=event_frame.index)), errors="coerce")
    sc_rate = float(sc_values.median()) if sc_values.notna().any() else 0.0
    vsc_rate = float(vsc_values.median()) if vsc_values.notna().any() else 0.0
    neutralisation_probability = 1.0 - (1.0 - np.clip(sc_rate, 0.0, 1.0)) * (1.0 - np.clip(vsc_rate, 0.0, 1.0))
    wet_values = pd.to_numeric(event_frame.get("race_forecast_is_wet", pd.Series(np.nan, index=event_frame.index)), errors="coerce")
    precip_values = pd.to_numeric(event_frame.get("race_forecast_precipitation_mm", pd.Series(np.nan, index=event_frame.index)), errors="coerce")
    if wet_values.notna().any():
        weather_risk = float(wet_values.max())
    elif precip_values.notna().any():
        weather_risk = float(precip_values.max() > 0.5)
    else:
        weather_risk = 0.0
    win_prob, podium_prob, expected_finish = _simulate_race_outcomes(
        expected_classified, dnf_probability, residual_scale=residual_scale,
        season=season, round_number=round_number,
        neutralisation_probability=neutralisation_probability, weather_risk=weather_risk,
    )
    result["model_expected_classified_position"] = ml_position
    result["priority_score"] = priority_score.to_numpy()
    result["priority_rank"] = expert_rank.astype(int)
    result["ml_model_strength"] = ml_strength
    result["combined_race_strength"] = combined_strength
    result["expected_classified_position"] = expected_classified
    result["expected_finish_position"] = expected_finish
    result["win_probability"] = win_prob
    result["win_score_share"] = win_prob  # event-coherent Monte Carlo probability estimate
    result["podium_probability"] = podium_prob
    result["dnf_probability"] = dnf_probability
    result["predicted_order"] = pd.Series(expected_finish, index=event_frame.index).rank(method="first").astype(int)
    result["prediction_method"] = "hybrid_ml_expert_priority_monte_carlo"
    result["position_uncertainty_scale"] = residual_scale
    result["position_uncertainty_source"] = residual_source
    result["historical_sc_event_rate"] = sc_rate if sc_values.notna().any() else np.nan
    result["historical_vsc_event_rate"] = vsc_rate if vsc_values.notna().any() else np.nan
    result["neutralisation_scenario_probability"] = neutralisation_probability if (sc_values.notna().any() or vsc_values.notna().any()) else np.nan
    result["weather_risk_scenario"] = weather_risk if (wet_values.notna().any() or precip_values.notna().any()) else np.nan
    for name, score in components.items():
        result[f"priority_{name}_score"] = score.to_numpy()
    return result.sort_values(["predicted_order", "win_probability"], ascending=[True, False]).reset_index(drop=True)

