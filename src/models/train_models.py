"""Train leakage-aware pole and race prediction models.

The model-selection test is chronological. If a partial current season is also
present in the downloaded data, it is retained for future-event feature
engineering but excluded from fitting the deployed model.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import ARTIFACTS_DIR, PROCESSED_DIR
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    POLE_NUMERIC_FEATURES,
    POLE_PRE_FP_NUMERIC_FEATURES,
    RACE_NUMERIC_FEATURES,
    build_race_feature_table,
    make_model_frame,
)

LOGGER = logging.getLogger("f1_model_training")


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values and non-finite floats into strict JSON values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def _candidate_estimators(
    task: str,
    positive_count: int | None = None,
    negative_count: int | None = None,
) -> dict[str, Any]:
    """Return CPU-friendly candidates; XGBoost is optional but preferred for comparison."""
    if task == "position":
        candidates: dict[str, Any] = {
            "random_forest": RandomForestRegressor(
                n_estimators=250, min_samples_leaf=2, max_features=0.9,
                random_state=42, n_jobs=-1,
            ),
        }
        try:
            from xgboost import XGBRegressor

            candidates["xgboost"] = XGBRegressor(
                n_estimators=250, max_depth=4, learning_rate=0.035,
                subsample=0.85, colsample_bytree=0.85,
                objective="reg:squarederror", random_state=42,
                n_jobs=-1, tree_method="hist",
            )
        except ImportError:
            LOGGER.warning("XGBoost is not installed; Random Forest will be used for position regression")
        return candidates

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, max_features=0.9,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        positive = max(int(positive_count or 0), 1)
        negative = max(int(negative_count or 0), 1)
        candidates["xgboost"] = XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1, tree_method="hist",
            scale_pos_weight=negative / positive,
        )
    except ImportError:
        LOGGER.warning("XGBoost is not installed; Random Forest will be used for %s", task)
    return candidates


def _positive_probability(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def _score(
    task: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    if task == "position":
        return {
            "mae": float(mean_absolute_error(y_true, predictions)),
            "r2": float(r2_score(y_true, predictions)) if len(y_true) > 1 else float("nan"),
            "within_one_position": float(np.mean(np.abs(np.asarray(y_true) - predictions) <= 1.0)),
        }

    scores = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if probabilities is not None:
        positives = int(pd.Series(y_true).sum())
        scores["average_precision"] = float(average_precision_score(y_true, probabilities)) if positives else 0.0
        if len(np.unique(y_true)) == 2:
            scores["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return scores


def _event_metrics(task: str, aligned: pd.DataFrame, y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    """Evaluate the selection users actually care about, grouped by Grand Prix."""
    if task not in {"pole", "winner", "podium"}:
        return {}
    rows = aligned[["season", "round"]].copy()
    rows["actual"] = np.asarray(y_true, dtype=int)
    rows["score"] = np.asarray(scores, dtype=float)
    event_hits: list[float] = []
    for _, group in rows.groupby(["season", "round"], dropna=False):
        if not group["actual"].eq(1).any():
            continue
        if task in {"pole", "winner"}:
            picked_index = group["score"].idxmax()
            event_hits.append(float(group.loc[picked_index, "actual"] == 1))
        else:
            top_three = group.nlargest(min(3, len(group)), "score")
            event_hits.append(float(top_three["actual"].sum() / 3.0))
    if not event_hits:
        return {}
    return {
        "event_top_pick_accuracy" if task in {"pole", "winner"} else "event_top3_recall":
            float(np.mean(event_hits)),
        "events_evaluated": float(len(event_hits)),
    }


def _baseline_event_metrics(task: str, aligned: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    """Measure simple, transparent event-level baselines on the same test rows.

    Pole baseline: fastest recorded practice classification time (or best practice
    position when timing is absent). Race baseline: current grid order. These are
    deliberately simple comparators, not competing ML models.
    """
    if task not in {"pole", "winner", "podium"}:
        return {}
    rows = aligned[["season", "round"]].copy()
    rows["actual"] = np.asarray(y_true, dtype=int)
    if task == "pole":
        time_values = pd.to_numeric(aligned.get("fp_best_time_seconds"), errors="coerce")
        position_values = pd.to_numeric(aligned.get("fp_best_position"), errors="coerce")
        rows["baseline_rank"] = time_values if time_values.notna().any() else position_values
        ascending = True
    else:
        rows["baseline_rank"] = pd.to_numeric(aligned.get("grid"), errors="coerce")
        ascending = True
    event_scores: list[float] = []
    for _, group in rows.groupby(["season", "round"], dropna=False):
        if not group["actual"].eq(1).any():
            continue
        eligible = group.dropna(subset=["baseline_rank"])
        if eligible.empty:
            continue
        if task in {"pole", "winner"}:
            picked = eligible.sort_values("baseline_rank", ascending=ascending).iloc[0]
            event_scores.append(float(picked["actual"] == 1))
        else:
            top = eligible.sort_values("baseline_rank", ascending=ascending).head(3)
            event_scores.append(float(top["actual"].sum() / 3.0))
    if not event_scores:
        return {"events_evaluated": 0.0}
    key = "event_top3_recall" if task == "podium" else "event_top_pick_accuracy"
    return {key: float(np.mean(event_scores)), "events_evaluated": float(len(event_scores))}


def _selection_score(task: str, metrics: dict[str, float]) -> float:
    if task == "position":
        return metrics.get("mae", float("inf"))
    event_key = "event_top3_recall" if task == "podium" else "event_top_pick_accuracy"
    return metrics.get(event_key, metrics.get("average_precision", metrics.get("f1", 0.0)))


def _temporal_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seasons = sorted(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int).unique())
    if len(seasons) < 3:
        raise ValueError(f"At least three seasons are needed for a train/validation/test split; found {seasons}")

    # Preserve the intended fixed split when those seasons exist. This ensures a
    # partial 2026 season does not silently become the final test set.
    if {2023, 2024, 2025}.issubset(seasons):
        train_end, validation_season, test_season = 2023, 2024, 2025
    else:
        validation_season, test_season = seasons[-2], seasons[-1]
        train_end = seasons[-3]
    train = frame[frame["season"] <= train_end]
    validation = frame[frame["season"] == validation_season]
    test = frame[frame["season"] == test_season]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition. Check the downloaded season coverage.")
    return train, validation, test, {
        "train_through": int(train_end),
        "validation_season": int(validation_season),
        "test_season": int(test_season),
        "available_seasons": seasons,
    }


def _fit_select(
    task: str,
    frame: pd.DataFrame,
    numeric_features: list[str],
    split_meta: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    X, y = make_model_frame(frame, task, numeric_features)
    aligned = frame.loc[X.index]
    seasons = pd.to_numeric(aligned["season"], errors="coerce")
    train_mask = seasons <= split_meta["train_through"]
    val_mask = seasons == split_meta["validation_season"]
    test_mask = seasons == split_meta["test_season"]
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    if X_train.empty or X_val.empty or X_test.empty:
        raise ValueError(f"Task {task} has insufficient samples in one of the temporal splits")
    if task != "position" and y_train.nunique() < 2:
        raise ValueError(f"Task {task} has only one class in training data")

    candidate_models = _candidate_estimators(
        task,
        positive_count=int((y_train == 1).sum()) if task != "position" else None,
        negative_count=int((y_train == 0).sum()) if task != "position" else None,
    )
    leaderboard: dict[str, Any] = {}
    best_name: str | None = None
    best_value = float("inf") if task == "position" else -float("inf")

    for name, estimator in candidate_models.items():
        try:
            pipeline = Pipeline([
                ("preprocessor", _preprocessor(numeric_features)),
                ("model", estimator),
            ])
            pipeline.fit(X_train, y_train)
            val_pred = pipeline.predict(X_val)
            val_prob = _positive_probability(pipeline, X_val) if task != "position" else None
            val_metrics = _score(task, y_val, val_pred, val_prob)
            if task == "position":
                val_metrics["validation_residual_std_positions"] = float(np.std(np.asarray(y_val, dtype=float) - np.asarray(val_pred, dtype=float)))
            if val_prob is not None:
                val_metrics.update(_event_metrics(task, aligned.loc[val_mask], y_val, val_prob))
            leaderboard[name] = {"validation": val_metrics}
            selection = _selection_score(task, val_metrics)
            if (task == "position" and selection < best_value) or (task != "position" and selection > best_value):
                best_name, best_value = name, selection
        except Exception as exc:
            LOGGER.exception("Candidate %s failed for task %s", name, task)
            leaderboard[name] = {"error": f"{type(exc).__name__}: {exc}"}

    if best_name is None:
        reasons = "; ".join(f"{name}: {data.get('error', 'no score')}" for name, data in leaderboard.items())
        raise RuntimeError(f"No model candidate succeeded for task {task}. {reasons}")

    # Evaluate once on the untouched test season after selecting on validation.
    selected_estimator = _candidate_estimators(
        task,
        positive_count=int((y_train == 1).sum()) if task != "position" else None,
        negative_count=int((y_train == 0).sum()) if task != "position" else None,
    )[best_name]
    refit = Pipeline([("preprocessor", _preprocessor(numeric_features)), ("model", selected_estimator)])
    refit.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_pred = refit.predict(X_test)
    test_prob = _positive_probability(refit, X_test) if task != "position" else None
    test_metrics = _score(task, y_test, test_pred, test_prob)
    if test_prob is not None:
        test_metrics.update(_event_metrics(task, aligned.loc[test_mask], y_test, test_prob))
    leaderboard[best_name]["test"] = test_metrics
    baseline_metrics = _baseline_event_metrics(task, aligned.loc[test_mask], y_test)

    # The 2026 season can be partial. Keep it available for event features but do
    # not train the deployed model on future rounds that would leak into earlier
    # 2026 predictions. Train the deployable bundle through the historical test year.
    deployment_mask = seasons <= split_meta["test_season"]
    X_deploy, y_deploy = X.loc[deployment_mask], y.loc[deployment_mask]
    final_estimator = _candidate_estimators(
        task,
        positive_count=int((y_deploy == 1).sum()) if task != "position" else None,
        negative_count=int((y_deploy == 0).sum()) if task != "position" else None,
    )[best_name]
    final_pipeline = Pipeline([("preprocessor", _preprocessor(numeric_features)), ("model", final_estimator)])
    final_pipeline.fit(X_deploy, y_deploy)
    metadata = {
        "selected_model": best_name,
        "feature_coverage_fraction": {
            column: float(X[column].notna().mean()) for column in X.columns
        },
        "samples": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
            "deployment": int(len(X_deploy)),
            "all_available": int(len(X)),
        },
        "leaderboard": leaderboard,
        "test_baseline": baseline_metrics,
        "features": CATEGORICAL_FEATURES + numeric_features,
        "deployment_training_through": int(split_meta["test_season"]),
    }
    return final_pipeline, metadata


def train_all_models(
    race_results_path: Path | str = PROCESSED_DIR / "race_results.csv",
    qualifying_path: Path | str = PROCESSED_DIR / "qualifying_results.csv",
    practice_path: Path | str = PROCESSED_DIR / "practice_results.csv",
    output_dir: Path | str = ARTIFACTS_DIR,
    sprint_path: Path | str = PROCESSED_DIR / "sprint_results.csv",
    session_laps_path: Path | str = PROCESSED_DIR / "session_laps.csv",
    weather_path: Path | str = PROCESSED_DIR / "weather_observations.csv",
    race_weather_forecasts_path: Path | str = PROCESSED_DIR / "race_weather_forecasts.csv",
    race_control_path: Path | str = PROCESSED_DIR / "openf1_race_control.csv",
    pit_stops_path: Path | str = PROCESSED_DIR / "openf1_pit.csv",
    openf1_drivers_path: Path | str = PROCESSED_DIR / "openf1_drivers.csv",
) -> dict[str, Any]:
    # Convert already-downloaded OpenF1 raw exports into canonical session/weather
    # tables before reading the default project paths. This is local-only; no API
    # requests are made during training.
    if Path(session_laps_path) == Path(PROCESSED_DIR / "session_laps.csv"):
        from src.data.openf1_sync import OpenF1Sync
        OpenF1Sync().rebuild_local_tables()

    race_path = Path(race_results_path)
    if not race_path.exists():
        raise FileNotFoundError(
            f"Race results not found at {race_path}. Download historical data first with "
            "python -m src.data.build_dataset --start-year 2018 --end-year 2026"
        )
    race_results = pd.read_csv(race_path)
    if race_results.empty:
        raise ValueError("Race-results CSV is empty. Re-run the data ingestion step.")
    qualifying = pd.read_csv(qualifying_path) if Path(qualifying_path).exists() else pd.DataFrame()
    practice = pd.read_csv(practice_path) if Path(practice_path).exists() else pd.DataFrame()
    sprints = pd.read_csv(sprint_path) if Path(sprint_path).exists() else pd.DataFrame()
    session_laps = pd.read_csv(session_laps_path) if Path(session_laps_path).exists() else pd.DataFrame()
    weather = pd.read_csv(weather_path) if Path(weather_path).exists() else pd.DataFrame()
    race_weather_forecasts = pd.read_csv(race_weather_forecasts_path) if Path(race_weather_forecasts_path).exists() else pd.DataFrame()
    race_control = pd.read_csv(race_control_path) if Path(race_control_path).exists() else pd.DataFrame()
    pit_stops = pd.read_csv(pit_stops_path) if Path(pit_stops_path).exists() else pd.DataFrame()
    openf1_drivers = pd.read_csv(openf1_drivers_path) if Path(openf1_drivers_path).exists() else pd.DataFrame()
    if not practice.empty and "time_seconds" not in practice.columns and "time" in practice.columns:
        from src.data.normalize import normalize_practice
        practice = normalize_practice(practice.to_dict("records"))

    # The race model is a post-qualifying forecast: qualifying/grid and the latest
    # observed Q weather are allowed. The pole models are trained separately at
    # FP1/FP2/FP3 and each receives only weather/practice data available by cutoff.
    race_features = build_race_feature_table(
        race_results, qualifying, practice, sprints, session_laps,
        practice_as_of_session="FP3", weather_observations=weather, weather_as_of_session="Q",
        race_weather_forecasts=race_weather_forecasts, race_control=race_control,
        pit_stops=pit_stops, openf1_drivers=openf1_drivers,
    )
    _, _, _, split_meta = _temporal_splits(race_features)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {
        "temporal_split": split_meta,
        "tasks": {},
        "pole_priority_hierarchy": {
            "weekend_pace": 0.28,
            "constructor_strength": 0.23,
            "driver_form": 0.16,
            "teammate_relative": 0.11,
            "weather_context": 0.09,
            "circuit_suitability": 0.08,
            "historical_pole_record": 0.05,
            "note": "Starting design weights for the inference ensemble, not fitted coefficients or calibrated probabilities. Weather remains a supervised-model context feature because raw weather values are not intrinsically good or bad for a driver."
        },
        "race_priority_hierarchy": {
            "constructor_performance": 0.20,
            "weekend_race_pace": 0.18,
            "grid_circuit_advantage": 0.18,
            "driver_form": 0.12,
            "tyre_strategy": 0.12,
            "reliability": 0.07,
            "circuit_characteristics": 0.05,
            "weather_context": 0.08,
            "note": "Hybrid expert-priority and ML ensemble. These are expert starting weights; the ML component is selected on a chronological validation season. Weather is shared event context and influences the trained model rather than creating a fabricated driver-specific edge."
        },
        "modeling_note": (
            "Validation selects the model; the test season is untouched for reporting. "
            "The deployed model is refit through the test season. Pole models are trained "
            "separately at FP1, FP2 and FP3 cutoffs. Race forecasts are post-qualifying "
            "forecasts and use current grid/qualifying plus Q-or-earlier weather, never race weather. "
            "Circuit/grid conversion features use only earlier races at the same circuit. Constructor intelligence includes current practice pace, compound-specific observed tyre-age slopes, lagged reliability and circuit results, and available OpenF1 Safety Car/VSC history. These pace/degradation fields remain proxies when fuel, traffic and run plans are unavailable."
        ),
    }

    pole_data_rows: dict[str, int] = {}
    # A dedicated historical-only model is trained with practice, session-lap and
    # observed-weather inputs removed. This is the only model used before FP1.
    pre_fp_features = build_race_feature_table(
        race_results, qualifying, pd.DataFrame(), sprints, pd.DataFrame(),
        practice_as_of_session="FP1", weather_observations=pd.DataFrame(),
        weather_as_of_session="FP1", race_weather_forecasts=pd.DataFrame(),
    )
    LOGGER.info("Training pole model for pre-free-practice cutoff")
    pre_fp_pipeline, pre_fp_meta = _fit_select(
        "pole", pre_fp_features, POLE_PRE_FP_NUMERIC_FEATURES, split_meta
    )
    models["pole_pre_fp"] = pre_fp_pipeline
    metrics["tasks"]["pole_pre_fp"] = pre_fp_meta
    pole_data_rows["PRE_FP"] = int(len(pre_fp_features))

    for cutoff in ("FP1", "FP2", "FP3"):
        pole_features = build_race_feature_table(
            race_results, qualifying, practice, sprints, session_laps,
            practice_as_of_session=cutoff, weather_observations=weather, weather_as_of_session=cutoff,
            race_weather_forecasts=race_weather_forecasts,
        )
        LOGGER.info("Training pole model for %s cutoff", cutoff)
        pipeline, task_meta = _fit_select("pole", pole_features, POLE_NUMERIC_FEATURES, split_meta)
        model_key = f"pole_{cutoff.lower()}"
        models[model_key] = pipeline
        metrics["tasks"][model_key] = task_meta
        pole_data_rows[cutoff] = int(len(pole_features))
        if cutoff == "FP3":
            models["pole"] = pipeline
            metrics["tasks"]["pole"] = task_meta

    for task, numeric_features in {
        "position": RACE_NUMERIC_FEATURES,
        "winner": RACE_NUMERIC_FEATURES,
        "podium": RACE_NUMERIC_FEATURES,
        "dnf": RACE_NUMERIC_FEATURES,
    }.items():
        LOGGER.info("Training %s model", task)
        pipeline, task_meta = _fit_select(task, race_features, numeric_features, split_meta)
        models[task] = pipeline
        metrics["tasks"][task] = task_meta

    available_seasons = sorted(pd.to_numeric(race_features["season"], errors="coerce").dropna().astype(int).unique().tolist())
    practice_coverage = float(race_features["fp_session_count"].notna().mean()) if "fp_session_count" in race_features else 0.0
    if "data_source" in session_laps.columns:
        source_series = session_laps["data_source"].fillna("fastf1").astype(str).str.lower()
        tif1_mask = source_series.str.startswith("tif1")
        fastf1_mask = source_series.eq("fastf1") | tif1_mask
        fastf1_laps = session_laps[fastf1_mask]
        tif1_laps = session_laps[tif1_mask]
        openf1_laps = session_laps[source_series.eq("openf1")]
        unknown_laps = session_laps[~(fastf1_mask | source_series.eq("openf1"))]
    else:
        # Earlier project archives predate the source column; their lap table was
        # populated by the FastF1 collector, so preserve backwards compatibility.
        fastf1_laps, tif1_laps, openf1_laps, unknown_laps = session_laps, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def _lap_driver_event_coverage(source_frame: pd.DataFrame) -> float:
        if source_frame.empty or not {"season", "round", "driver_code"}.issubset(source_frame.columns):
            return 0.0
        target_keys = race_features[["season", "round", "driver_code"]].dropna(subset=["driver_code"]).drop_duplicates()
        source_keys = source_frame[["season", "round", "driver_code"]].dropna(subset=["driver_code"]).drop_duplicates()
        return float(target_keys.merge(source_keys, on=["season", "round", "driver_code"], how="inner").shape[0] / max(len(target_keys), 1))
    metrics["data"] = {
        "rows": int(len(race_features)),
        "pole_feature_rows_by_cutoff": pole_data_rows,
        "training_rows": int(race_features["season"].le(split_meta["test_season"]).sum()),
        "seasons": available_seasons,
        "deployment_training_seasons": [year for year in available_seasons if year <= split_meta["test_season"]],
        "inference_only_seasons": [year for year in available_seasons if year > split_meta["test_season"]],
        "practice_rows": int(len(practice)),
        "practice_coverage_fraction": practice_coverage,
        "session_lap_rows_by_source": {
            "fastf1": int(len(fastf1_laps)),
            "tif1": int(len(tif1_laps)),
            "openf1": int(len(openf1_laps)),
            "unknown": int(len(unknown_laps)),
            "combined_model_input": int(len(session_laps)),
        },
        "fastf1_session_event_driver_coverage_fraction": _lap_driver_event_coverage(fastf1_laps),
        "openf1_session_event_driver_coverage_fraction": _lap_driver_event_coverage(openf1_laps),
        "combined_session_event_driver_coverage_fraction": _lap_driver_event_coverage(session_laps),
        "weather_observation_rows": int(len(weather)),
        "race_weather_forecast_rows": int(len(race_weather_forecasts)),
        "race_weather_forecast_feature_coverage_fraction": {
            feature: float(race_features[feature].notna().mean()) if feature in race_features else 0.0
            for feature in ("race_forecast_air_temp_c", "race_forecast_precipitation_mm", "race_forecast_is_wet", "race_forecast_wind_speed_kmh")
        },
        "weather_feature_coverage_fraction": {
            feature: float(race_features[feature].notna().mean()) if feature in race_features else 0.0
            for feature in ("weather_air_temp_c", "weather_track_temp_c", "weather_rainfall_fraction")
        },
        "circuit_conversion_feature_coverage_fraction": {
            feature: float(race_features[feature].notna().mean()) if feature in race_features else 0.0
            for feature in ("circuit_grid_win_rate_prev", "circuit_pole_to_win_rate_prev", "circuit_p2_p3_to_win_rate_prev")
        },
        "pole_training_cutoffs": ["FP1", "FP2", "FP3"],
        "race_weather_cutoff": "Q",
        "note": (
            "No target-event qualifying is used for pole forecasts. Pole practice/weather features are stage-specific. "
            "Race forecasts use post-qualifying grid, strictly lagged circuit/grid conversion rates, observed weather through "
            "qualifying, and archived one-day-ahead race-start weather forecasts when available (never realised race weather). "
            "TIF1 (FastF1-derived)/OpenF1 long-run features remain proxies and require care with fuel, "
            "traffic, track evolution and tyre programme."
        ),
    }
    bundle = {"models": models, "metadata": _json_safe(metrics)}
    joblib.dump(bundle, output / "f1_models.joblib")
    (output / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-results", default=str(PROCESSED_DIR / "race_results.csv"))
    parser.add_argument("--qualifying", default=str(PROCESSED_DIR / "qualifying_results.csv"))
    parser.add_argument("--practice", default=str(PROCESSED_DIR / "practice_results.csv"))
    parser.add_argument("--output-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--sprints", default=str(PROCESSED_DIR / "sprint_results.csv"))
    parser.add_argument("--session-laps", default=str(PROCESSED_DIR / "session_laps.csv"))
    parser.add_argument("--weather", default=str(PROCESSED_DIR / "weather_observations.csv"))
    parser.add_argument("--race-weather-forecasts", default=str(PROCESSED_DIR / "race_weather_forecasts.csv"))
    parser.add_argument("--race-control", default=str(PROCESSED_DIR / "openf1_race_control.csv"))
    parser.add_argument("--pit-stops", default=str(PROCESSED_DIR / "openf1_pit.csv"))
    parser.add_argument("--openf1-drivers", default=str(PROCESSED_DIR / "openf1_drivers.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    metrics = train_all_models(
        args.race_results, args.qualifying, args.practice, args.output_dir,
        args.sprints, args.session_laps, args.weather, args.race_weather_forecasts,
        args.race_control, args.pit_stops, args.openf1_drivers,
    )
    print(json.dumps(_json_safe(metrics), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
