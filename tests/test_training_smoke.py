from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.models import train_models
from src.models.predict import load_model_bundle, predict_pole, predict_race
from src.features.build_features import build_event_prediction_frame


def _synthetic_tables():
    race_rows, quali_rows, practice_rows = [], [], []
    for season in range(2020, 2025):
        for round_no in range(1, 5):
            date = f"{season}-{3 + (round_no - 1) // 2:02d}-{(round_no % 2) * 12 + 3:02d}"
            drivers = [(f"d{i}", f"team_{i // 2}", i) for i in range(8)]
            race_order = sorted(drivers, key=lambda row: (row[2] + (round_no * season) % 5 * 0.1, row[0]))
            quali_order = sorted(drivers, key=lambda row: (row[2] + (round_no * season) % 7 * 0.1, row[0]))
            quali_positions = {row[0]: rank + 1 for rank, row in enumerate(quali_order)}
            for rank, (driver_id, constructor_id, skill) in enumerate(race_order, start=1):
                dnf = (skill + round_no + season) % 11 == 0
                race_rows.append({
                    "season": season, "round": round_no, "date": date,
                    "race_name": f"Synthetic GP {round_no}", "circuit_id": f"circuit_{round_no}",
                    "driver_id": driver_id, "driver_code": driver_id.upper(), "driver_name": driver_id.upper(),
                    "constructor_id": constructor_id, "constructor_name": constructor_id.upper(),
                    "grid": quali_positions[driver_id], "position": rank, "position_order": rank,
                    "points": [25, 18, 15, 12, 10, 8, 6, 4][rank - 1] if not dnf else 0,
                    "status": "Engine" if dnf else "Finished",
                })
                quali_rows.append({
                    "season": season, "round": round_no, "date": date,
                    "race_name": f"Synthetic GP {round_no}", "circuit_id": f"circuit_{round_no}",
                    "driver_id": driver_id, "driver_code": driver_id.upper(), "driver_name": driver_id.upper(),
                    "constructor_id": constructor_id, "constructor_name": constructor_id.upper(),
                    "qualifying_position": quali_positions[driver_id],
                    "q1": "1:20.000", "q2": "1:19.000", "q3": "1:18.000",
                })
                for session, offset in (("FP1", 0.4), ("FP2", 0.2), ("FP3", 0.0)):
                    practice_rows.append({
                        "season": season, "round": round_no, "race_name": f"Synthetic GP {round_no}",
                        "circuit_id": f"circuit_{round_no}", "session": session,
                        "position": quali_positions[driver_id], "driver_id": driver_id,
                        "driver_name": driver_id.upper(), "constructor_id": constructor_id,
                        "constructor_name": constructor_id.upper(),
                        "time_seconds": 80.0 + skill * 0.12 + offset, "laps": 15 + skill,
                    })
    return pd.DataFrame(race_rows), pd.DataFrame(quali_rows), pd.DataFrame(practice_rows)


def test_all_five_models_train_save_load_and_predict(tmp_path, monkeypatch):
    race, quali, practice = _synthetic_tables()
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    processed.mkdir()
    race.to_csv(processed / "race_results.csv", index=False)
    quali.to_csv(processed / "qualifying_results.csv", index=False)
    practice.to_csv(processed / "practice_results.csv", index=False)

    # Keep the automated test fast. The separate development smoke run compares
    # the full Random Forest and XGBoost candidate sets.
    def small_estimators(task, positive_count=None, negative_count=None):
        if task == "position":
            return {"random_forest": RandomForestRegressor(n_estimators=12, min_samples_leaf=1, random_state=42)}
        return {"random_forest": RandomForestClassifier(n_estimators=12, min_samples_leaf=1, random_state=42)}

    monkeypatch.setattr(train_models, "_candidate_estimators", small_estimators)
    metrics = train_models.train_all_models(
        processed / "race_results.csv", processed / "qualifying_results.csv",
        processed / "practice_results.csv", artifacts,
    )
    assert set(metrics["tasks"]) == {
        "pole", "pole_pre_fp", "pole_fp1", "pole_fp2", "pole_fp3", "position", "winner", "podium", "dnf"
    }
    pre_fp_features = metrics["tasks"]["pole_pre_fp"]["features"]
    assert "driver_last_gp_finish" in pre_fp_features
    assert "constructor_last_gp_avg_finish" in pre_fp_features
    assert not any(str(feature).startswith(("fp_", "weather_")) for feature in pre_fp_features)
    assert metrics["temporal_split"]["validation_season"] == 2023
    assert metrics["temporal_split"]["test_season"] == 2024
    assert "test_baseline" in metrics["tasks"]["winner"]
    assert "event_top_pick_accuracy" in metrics["tasks"]["winner"]["test_baseline"]
    assert "test_baseline" in metrics["tasks"]["pole_fp1"]
    assert (artifacts / "f1_models.joblib").exists()
    assert (artifacts / "metrics.json").exists()

    bundle = load_model_bundle(artifacts / "f1_models.joblib")
    event = build_event_prediction_frame(
        2024, 5, race, quali, practice,
        pd.DataFrame([{"season": 2024, "round": 5, "race_name": "Future Synthetic GP", "circuit_id": "circuit_1"}]),
        stage="pole",
    )
    pole_pre_fp = predict_pole(bundle, event, as_of_session="PRE_FP")
    pole = predict_pole(bundle, event, as_of_session="FP1")
    pole_fp3 = predict_pole(bundle, event, as_of_session="FP3")
    race_event = build_event_prediction_frame(
        2024, 5, race, quali, practice,
        pd.DataFrame([{"season": 2024, "round": 5, "race_name": "Future Synthetic GP", "circuit_id": "circuit_1"}]),
        stage="race",
    )
    race_predictions = predict_race(bundle, race_event)
    assert len(pole_pre_fp) == 8
    assert pole_pre_fp["forecast_cutoff"].eq("before_free_practice").all()
    assert pole_pre_fp["prediction_basis"].eq("pre_free_practice_historical_form_ensemble").all()
    assert np.isclose(pole_pre_fp["pole_score_share"].sum(), 1.0)
    assert len(pole) == 8
    assert np.isclose(pole["pole_score_share"].sum(), 1.0)
    assert len(pole_fp3) == 8
    assert np.isclose(pole_fp3["pole_score_share"].sum(), 1.0)
    assert len(race_predictions) == 8
    assert np.isclose(race_predictions["win_score_share"].sum(), 1.0)
    assert np.isclose(race_predictions["podium_probability"].sum(), 3.0)
    assert race_predictions["dnf_probability"].between(0, 1).all()
    assert race_predictions["prediction_method"].eq("hybrid_ml_expert_priority_monte_carlo").all()


def test_event_level_baselines_are_reported_for_pole_and_race_tasks():
    aligned = pd.DataFrame({
        "season": [2025, 2025, 2025, 2025, 2025, 2025],
        "round": [1, 1, 1, 2, 2, 2],
        "fp_best_time_seconds": [90.0, 91.0, 92.0, 90.0, 91.0, 92.0],
        "fp_best_position": [1, 2, 3, 1, 2, 3],
        "grid": [1, 2, 3, 1, 2, 3],
    })
    pole_y = pd.Series([1, 0, 0, 0, 1, 0])
    race_y = pd.Series([0, 1, 0, 1, 0, 0])

    pole_baseline = train_models._baseline_event_metrics("pole", aligned, pole_y)
    winner_baseline = train_models._baseline_event_metrics("winner", aligned, race_y)

    assert pole_baseline["events_evaluated"] == 2
    assert pole_baseline["event_top_pick_accuracy"] == 0.5
    assert winner_baseline["event_top_pick_accuracy"] == 0.5


def test_future_race_frame_prefers_official_starting_grid_over_qualifying_proxy():
    race, quali, practice = _synthetic_tables()
    schedule = pd.DataFrame([{
        "season": 2024, "round": 5, "race_name": "Future Synthetic GP", "circuit_id": "circuit_1",
    }])
    official_grid = pd.DataFrame([
        {"season": 2024, "round": 5, "driver_id": "d0", "driver_code": "D0", "grid": 2},
        {"season": 2024, "round": 5, "driver_id": "d1", "driver_code": "D1", "grid": 1},
    ])

    frame = build_event_prediction_frame(
        2024, 5, race, quali, practice, schedule, stage="race", starting_grid=official_grid,
    )

    mapped = frame.set_index("driver_id")
    assert mapped.loc["d0", "grid"] == 2
    assert mapped.loc["d1", "grid"] == 1
    assert mapped.loc["d0", "starting_grid_source"] == "openf1_starting_grid"
    assert mapped.loc["d1", "starting_grid_source"] == "openf1_starting_grid"


def test_priority_pole_score_rewards_weekend_pace_before_history():
    from src.models.predict import _priority_pole_score

    event = pd.DataFrame({
        "driver_name": ["Driver A", "Driver B", "Driver C"],
        "fp_best_gap_pct": [0.0, 0.2, 0.4],
        "constructor_quali_gap_recent3_pct": [0.0, 0.1, 0.2],
        "driver_quali_gap_recent3_pct": [0.0, 0.1, 0.2],
        "driver_teammate_quali_delta_prev_pct": [0.0, 0.1, 0.2],
        "driver_circuit_prev_poles": [0, 8, 4],
    })
    score, components = _priority_pole_score(event)
    assert score.iloc[0] > score.iloc[1]
    assert score.iloc[1] > score.iloc[2]
    assert components["weekend_pace"].notna().all()
    assert components["historical_pole_record"].notna().all()


def test_priority_pole_score_is_neutral_when_no_relevant_features_exist():
    from src.models.predict import _priority_pole_score

    event = pd.DataFrame({"driver_name": ["A", "B"]})
    score, _ = _priority_pole_score(event)
    assert np.allclose(score.to_numpy(), [0.5, 0.5])


def test_safety_car_scenarios_increase_race_outcome_uncertainty():
    from src.models.predict import _simulate_race_outcomes

    args = {
        "expected_position": np.arange(1, 11, dtype=float),
        "dnf_probability": np.zeros(10),
        "residual_scale": 1.5,
        "season": 2025,
        "round_number": 3,
    }
    no_neutralisation = _simulate_race_outcomes(**args, neutralisation_probability=0.0)
    frequent_neutralisation = _simulate_race_outcomes(**args, neutralisation_probability=0.8)
    assert np.isclose(no_neutralisation[0].sum(), 1.0)
    assert np.isclose(frequent_neutralisation[0].sum(), 1.0)
    assert np.isclose(frequent_neutralisation[1].sum(), 3.0)
    assert frequent_neutralisation[0].max() < no_neutralisation[0].max()
