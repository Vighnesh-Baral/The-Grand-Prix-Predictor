import pandas as pd

from src.features.build_features import build_event_prediction_frame, build_race_feature_table, make_model_frame


def sample_tables():
    race_rows = []
    for season, round_no, date, offset in [
        (2023, 1, "2023-03-05", 0),
        (2023, 2, "2023-03-19", 1),
        (2024, 1, "2024-03-03", 2),
    ]:
        for index, (driver, team) in enumerate([
            ("driver_a", "team_x"), ("driver_b", "team_x"),
            ("driver_c", "team_y"), ("driver_d", "team_y"),
        ]):
            position = ((index + offset) % 4) + 1
            race_rows.append({
                "season": season, "round": round_no, "date": date,
                "race_name": "Test GP", "circuit_id": "test_circuit",
                "driver_id": driver, "driver_code": driver[-1].upper(),
                "driver_name": driver.title(), "constructor_id": team,
                "constructor_name": team.title(), "grid": index + 1,
                "position": position, "position_order": position,
                "points": [25, 18, 15, 12][position - 1],
                "status": "Finished",
            })
    race = pd.DataFrame(race_rows)
    quali = race[["season", "round", "date", "race_name", "circuit_id", "driver_id", "driver_code", "driver_name", "constructor_id", "constructor_name", "grid"]].copy()
    quali["qualifying_position"] = quali["grid"]
    quali["q1"] = "1:20.000"
    quali["q2"] = "1:19.000"
    quali["q3"] = "1:18.000"
    return race, quali


def test_lagged_features_exclude_current_event():
    race, quali = sample_tables()
    features = build_race_feature_table(race, quali)
    round_two = features[(features["season"] == 2023) & (features["round"] == 2)]
    driver_a = round_two[round_two["driver_id"] == "driver_a"].iloc[0]
    assert driver_a["driver_prev_avg_finish"] == 1
    # Team X's prior event average is based on both cars in round 1 (positions 1 and 2).
    team_a = round_two[round_two["constructor_id"] == "team_x"]
    assert team_a["constructor_prev_avg_finish"].nunique() == 1
    assert team_a["constructor_prev_avg_finish"].iloc[0] == 1.5
    assert team_a["constructor_prev_points"].nunique() == 1
    assert team_a["constructor_prev_points"].iloc[0] == 43


def test_pole_features_do_not_include_current_qualifying_fields():
    race, quali = sample_tables()
    features = build_race_feature_table(race, quali)
    X, y = make_model_frame(features, "pole")
    assert "qualifying_position" not in X.columns
    assert "grid" not in X.columns
    assert len(X) == len(features)


def test_event_prediction_excludes_selected_event_race_outcomes():
    race, quali = sample_tables()
    # Deliberately change the selected event result to a very bad finish. It must not
    # change that driver's historical-form feature for the selected event.
    first = build_event_prediction_frame(2023, 2, race, quali, race_schedule=None)
    race2 = race.copy()
    race2.loc[(race2["season"] == 2023) & (race2["round"] == 2) & (race2["driver_id"] == "driver_a"), "position_order"] = 99
    second = build_event_prediction_frame(2023, 2, race2, quali, race_schedule=None)
    a = first[first["driver_id"] == "driver_a"].iloc[0]
    b = second[second["driver_id"] == "driver_a"].iloc[0]
    assert a["driver_prev_avg_finish"] == b["driver_prev_avg_finish"] == 1


def test_event_practice_features_include_session_relative_pace():
    race, quali = sample_tables()
    practice = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "driver_a", "driver_name": "Driver A", "session": "FP1", "position": 1, "time_seconds": 80.0, "laps": 20},
        {"season": 2024, "round": 1, "driver_id": "driver_b", "driver_name": "Driver B", "session": "FP1", "position": 2, "time_seconds": 81.0, "laps": 21},
        {"season": 2024, "round": 1, "driver_id": "driver_a", "driver_name": "Driver A", "session": "FP2", "position": 2, "time_seconds": 80.5, "laps": 22},
        {"season": 2024, "round": 1, "driver_id": "driver_b", "driver_name": "Driver B", "session": "FP2", "position": 1, "time_seconds": 80.2, "laps": 23},
    ])
    event = build_event_prediction_frame(2024, 1, race, quali, practice, stage="pole")
    driver_a = event[event["driver_id"] == "driver_a"].iloc[0]
    assert driver_a["fp_session_count"] == 2
    assert driver_a["fp_total_laps"] == 42
    assert driver_a["fp_best_time_gap_pct"] == 0
    assert driver_a["fp_mean_time_gap_pct"] > 0


def test_sprint_points_are_included_in_lagged_driver_form():
    race, quali = sample_tables()
    sprint = pd.DataFrame([{
        "season": 2023, "round": 1, "driver_id": "driver_a", "points": 8,
    }])
    features = build_race_feature_table(race, quali, sprint_results=sprint)
    driver_a_round_two = features[(features["season"] == 2023) & (features["round"] == 2) & (features["driver_id"] == "driver_a")].iloc[0]
    assert driver_a_round_two["driver_prev_points"] == 33


def test_practice_classification_features_obey_fp_cutoff():
    race, quali = sample_tables()
    practice = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "driver_a", "driver_name": "Driver A", "session": "FP1", "position": 1, "time_seconds": 80.0, "laps": 20},
        {"season": 2024, "round": 1, "driver_id": "driver_b", "driver_name": "Driver B", "session": "FP1", "position": 2, "time_seconds": 81.0, "laps": 20},
        {"season": 2024, "round": 1, "driver_id": "driver_a", "driver_name": "Driver A", "session": "FP2", "position": 2, "time_seconds": 81.0, "laps": 20},
        {"season": 2024, "round": 1, "driver_id": "driver_b", "driver_name": "Driver B", "session": "FP2", "position": 1, "time_seconds": 80.0, "laps": 20},
        {"season": 2024, "round": 1, "driver_id": "driver_a", "driver_name": "Driver A", "session": "FP3", "position": 2, "time_seconds": 82.0, "laps": 20},
        {"season": 2024, "round": 1, "driver_id": "driver_b", "driver_name": "Driver B", "session": "FP3", "position": 1, "time_seconds": 79.5, "laps": 20},
    ])
    fp1 = build_race_feature_table(race, quali, practice, practice_as_of_session="FP1")
    fp3 = build_race_feature_table(race, quali, practice, practice_as_of_session="FP3")
    fp1_rows = fp1[(fp1["season"] == 2024) & (fp1["round"] == 1)].set_index("driver_id")
    fp3_rows = fp3[(fp3["season"] == 2024) & (fp3["round"] == 1)].set_index("driver_id")

    assert fp1_rows.loc["driver_a", "fp_session_count"] == 1
    assert fp1_rows.loc["driver_b", "fp_session_count"] == 1
    assert fp1_rows.loc["driver_a", "fp_best_time_seconds"] == 80.0
    assert fp1_rows.loc["driver_b", "fp_best_time_seconds"] == 81.0
    assert fp3_rows.loc["driver_a", "fp_session_count"] == 3
    assert fp3_rows.loc["driver_a", "fp_best_time_seconds"] == 80.0
    assert fp3_rows.loc["driver_b", "fp_best_time_seconds"] == 79.5


def test_curated_event_roster_overrides_stale_previous_season_roster():
    from src.features.build_features import build_event_prediction_frame

    history = pd.DataFrame([
        {
            "season": 2025, "round": 24, "date": "2025-12-07", "race_name": "Finale",
            "circuit_id": "old_circuit", "driver_id": "retired_driver", "driver_code": "RET",
            "driver_name": "Retired Driver", "constructor_id": "old_team",
            "constructor_name": "Old Team", "grid": 1, "position_order": 1,
            "points": 25, "status": "Finished",
        }
    ])
    schedule = pd.DataFrame([
        {"season": 2026, "round": 1, "race_name": "Season Opener", "date": "2026-03-08", "circuit_id": "new_circuit"}
    ])
    roster = pd.DataFrame([
        {"season": 2026, "round": 1, "driver_id": "new_driver_a", "driver_code": "NVA",
         "driver_name": "New Driver A", "constructor_id": "new_team", "constructor_name": "New Team"},
        {"season": 2026, "round": 1, "driver_id": "new_driver_b", "driver_code": "NVB",
         "driver_name": "New Driver B", "constructor_id": "new_team", "constructor_name": "New Team"},
    ])
    frame = build_event_prediction_frame(
        2026, 1, history, race_schedule=schedule, stage="pole", driver_roster=roster
    )
    assert set(frame["driver_id"]) == {"new_driver_a", "new_driver_b"}
    assert set(frame["constructor_id"]) == {"new_team"}
    assert frame["candidate_roster_source"].iloc[0] == "curated_event_roster"
