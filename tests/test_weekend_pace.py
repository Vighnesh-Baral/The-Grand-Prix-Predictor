import numpy as np
import pandas as pd

from src.features.build_features import build_race_feature_table
from src.features.weekend_pace import build_event_weekend_pace_features


def _lap_rows():
    rows = []
    session_offsets = {"FP1": 0.30, "FP2": 0.15, "FP3": 0.00}
    for session, offset in session_offsets.items():
        for driver, one_lap, longrun in (("VER", 80.0 + offset, 91.0 + offset), ("HAM", 80.8 + offset, 91.8 + offset)):
            for lap in range(1, 9):
                is_short_run = lap <= 2
                rows.append({
                    "season": 2026, "round": 5, "session": session,
                    "driver_code": driver, "lap_number": lap, "stint": 1,
                    "lap_time_seconds": (one_lap if lap == 1 else one_lap + 0.15) if is_short_run else longrun + (lap - 3) * 0.08,
                    "compound": "SOFT" if is_short_run else "MEDIUM",
                    "tyre_life": lap, "is_accurate": True, "deleted": False,
                    "fastf1_generated": False, "pit_in": None, "pit_out": None,
                    "track_status": "1",
                })
    return pd.DataFrame(rows)


def test_pace_features_respect_session_cutoff_and_split_quali_longrun():
    laps = _lap_rows()
    fp1 = build_event_weekend_pace_features(laps, 2026, 5, "FP1").set_index("driver_code")
    fp3 = build_event_weekend_pace_features(laps, 2026, 5, "FP3").set_index("driver_code")

    assert fp1.loc["VER", "fp_sessions_completed"] == 1
    assert pd.isna(fp1.loc["VER", "fp2_best_gap_pct"])
    assert fp3.loc["VER", "fp_sessions_completed"] == 3
    assert fp3.loc["VER", "fp3_best_gap_pct"] == 0
    assert fp3.loc["HAM", "fp3_best_gap_pct"] > 0
    assert fp3.loc["HAM", "fp3_soft_gap_pct"] > 0
    assert fp3.loc["HAM", "fp3_longrun_gap_pct"] > 0
    assert fp3.loc["VER", "fp_longrun_laps"] > 0


def test_constructor_qualifying_pace_features_are_lagged():
    races = []
    qualifying = []
    for round_no in (1, 2):
        for driver, team, time_ms in (("a", "fast_team", 0), ("b", "fast_team", 100), ("c", "slow_team", 400), ("d", "slow_team", 500)):
            races.append({
                "season": 2025, "round": round_no, "date": f"2025-03-{round_no * 7:02d}",
                "race_name": "Test GP", "circuit_id": f"c{round_no}", "driver_id": driver,
                "driver_code": driver.upper(), "driver_name": driver.upper(), "constructor_id": team,
                "constructor_name": team, "grid": 1, "position_order": 1, "points": 25,
                "status": "Finished", "fastest_lap_time": "1:30.000",
            })
            qualifying.append({
                "season": 2025, "round": round_no, "driver_id": driver, "constructor_id": team,
                "qualifying_position": 1, "q1": "1:20.000", "q2": "1:19.000",
                "q3": f"1:18.{time_ms:03d}",
            })
    features = build_race_feature_table(pd.DataFrame(races), pd.DataFrame(qualifying))
    first_round = features[features["round"] == 1]
    second_round = features[features["round"] == 2]

    assert first_round["constructor_quali_gap_prev_avg_pct"].isna().all()
    team_gaps = second_round.groupby("constructor_id")["constructor_quali_gap_prev_avg_pct"].first()
    assert team_gaps["fast_team"] < team_gaps["slow_team"]


def test_dnf_label_excludes_non_starters_and_disqualifications():
    from src.features.build_features import build_race_feature_table

    base = pd.DataFrame([
        {"season": 2025, "round": 1, "date": "2025-03-01", "driver_id": "a", "constructor_id": "x", "position_order": 1, "points": 25, "status": "Finished"},
        {"season": 2025, "round": 1, "date": "2025-03-01", "driver_id": "b", "constructor_id": "x", "position_order": 2, "points": 18, "status": "Engine"},
        {"season": 2025, "round": 1, "date": "2025-03-01", "driver_id": "c", "constructor_id": "y", "position_order": 3, "points": 15, "status": "Disqualified"},
    ])
    # A minimal single-season table is sufficient for validating target construction.
    features = build_race_feature_table(base)
    assert features.loc[features.driver_id.eq("a"), "is_dnf"].iloc[0] == 0
    assert features.loc[features.driver_id.eq("b"), "is_dnf"].iloc[0] == 1
    assert np.isnan(features.loc[features.driver_id.eq("c"), "is_dnf"].iloc[0])
    base.loc[base.driver_id.eq("c"), "status"] = None
    missing_status_features = build_race_feature_table(base)
    assert np.isnan(missing_status_features.loc[missing_status_features.driver_id.eq("c"), "is_dnf"].iloc[0])


def test_historical_race_pace_is_lagged_and_constructor_specific():
    from src.features.weekend_pace import build_historical_race_pace_table

    race_rows = []
    lap_rows = []
    for round_no, pace_shift in ((1, 0.0), (2, 0.2)):
        for driver, code, team, base in (
            ("a", "AAA", "fast_team", 90.0), ("b", "BBB", "fast_team", 90.4),
            ("c", "CCC", "slow_team", 92.0), ("d", "DDD", "slow_team", 92.4),
        ):
            race_rows.append({
                "season": 2025, "round": round_no, "driver_id": driver,
                "driver_code": code, "constructor_id": team,
            })
            for lap_no in range(1, 9):
                lap_rows.append({
                    "season": 2025, "round": round_no, "session": "R",
                    "driver_code": code, "lap_number": lap_no, "stint": 1,
                    "lap_time_seconds": base + pace_shift + 0.06 * lap_no,
                    "tyre_life": lap_no, "is_accurate": True,
                    "track_status": "1", "deleted": False,
                    "pit_in": None, "pit_out": None,
                })

    result = build_historical_race_pace_table(pd.DataFrame(lap_rows), pd.DataFrame(race_rows))
    first_round = result[result["round"] == 1]
    second_round = result[result["round"] == 2]
    assert first_round["constructor_race_longrun_gap_prev_avg_pct"].isna().all()
    team_gap = second_round.groupby("constructor_id")["constructor_race_longrun_gap_prev_avg_pct"].first()
    assert team_gap["fast_team"] < team_gap["slow_team"]
    assert second_round["driver_race_stint_slope_prev_avg_sec_per_lap"].notna().all()


def test_constructor_circuit_qualifying_pace_uses_only_prior_visits_to_same_circuit():
    race_rows = []
    qualifying_rows = []
    events = [
        (1, "circuit_a", {"team_a": (78.0, 78.2), "team_b": (79.0, 79.2)}),
        (2, "circuit_b", {"team_a": (80.0, 80.2), "team_b": (78.0, 78.2)}),
        (3, "circuit_a", {"team_a": (78.4, 78.6), "team_b": (79.4, 79.6)}),
    ]
    for round_no, circuit, teams in events:
        for team, times in teams.items():
            for car_no, q_time in enumerate(times, start=1):
                driver = f"{team}_{car_no}"
                race_rows.append({
                    "season": 2025, "round": round_no, "date": f"2025-03-{round_no * 7:02d}",
                    "race_name": "Test GP", "circuit_id": circuit, "driver_id": driver,
                    "driver_code": driver.upper(), "driver_name": driver, "constructor_id": team,
                    "constructor_name": team, "grid": car_no, "position_order": car_no,
                    "points": 10, "status": "Finished",
                })
                q_text = f"1:{int(q_time):02d}.{int(round(q_time % 1 * 1000)):03d}"
                qualifying_rows.append({
                    "season": 2025, "round": round_no, "circuit_id": circuit,
                    "driver_id": driver, "constructor_id": team,
                    "qualifying_position": car_no, "q1": q_text, "q2": q_text, "q3": q_text,
                })

    features = build_race_feature_table(pd.DataFrame(race_rows), pd.DataFrame(qualifying_rows))
    target = features[features["round"] == 3].groupby("constructor_id")["constructor_circuit_quali_gap_prev_avg_pct"].first()
    assert target["team_a"] < target["team_b"]


def test_future_roster_fallback_uses_latest_completed_gp_not_all_historical_drivers():
    from src.features.build_features import build_event_prediction_frame

    rows = []
    for round_no, drivers in ((1, ("active_a", "active_b", "retired_driver")), (2, ("active_a", "active_b"))):
        for position, driver in enumerate(drivers, start=1):
            rows.append({
                "season": 2026, "round": round_no, "date": f"2026-03-{round_no * 7:02d}",
                "race_name": "Past GP", "circuit_id": "circuit_a", "driver_id": driver,
                "driver_code": driver.upper(), "driver_name": driver, "constructor_id": f"team_{position}",
                "constructor_name": f"Team {position}", "grid": position, "position_order": position,
                "points": 10, "status": "Finished",
            })
    schedule = pd.DataFrame([{
        "season": 2026, "round": 3, "race_name": "Upcoming GP", "date": "2026-03-21",
        "circuit_id": "circuit_b",
    }])
    frame = build_event_prediction_frame(
        2026, 3, pd.DataFrame(rows), race_schedule=schedule, stage="pole"
    )
    assert set(frame["driver_id"]) == {"active_a", "active_b"}
    assert frame["candidate_roster_source"].iloc[0] == "latest_completed_gp_2026_round_2"


def test_upcoming_event_does_not_borrow_constructor_pace_from_a_different_circuit():
    from src.features.build_features import build_event_prediction_frame

    race_rows = []
    quali_rows = []
    for round_no, circuit, pace in ((1, "circuit_a", 80.0), (2, "circuit_a", 80.2)):
        for car_no, driver in enumerate(("driver_a", "driver_b"), start=1):
            race_rows.append({
                "season": 2026, "round": round_no, "date": f"2026-03-{round_no * 7:02d}",
                "race_name": "Past GP", "circuit_id": circuit, "driver_id": driver,
                "driver_code": driver.upper(), "driver_name": driver, "constructor_id": "team_x",
                "constructor_name": "Team X", "grid": car_no, "position_order": car_no,
                "points": 10, "status": "Finished",
            })
            q_time = f"1:{int(pace + (car_no - 1) * 0.2):02d}.000"
            quali_rows.append({
                "season": 2026, "round": round_no, "circuit_id": circuit,
                "driver_id": driver, "constructor_id": "team_x", "qualifying_position": car_no,
                "q1": q_time, "q2": q_time, "q3": q_time,
            })

    schedule = pd.DataFrame([{
        "season": 2026, "round": 3, "race_name": "New Circuit GP", "date": "2026-03-21",
        "circuit_id": "circuit_b",
    }])
    frame = build_event_prediction_frame(
        2026, 3, pd.DataFrame(race_rows), pd.DataFrame(quali_rows),
        race_schedule=schedule, stage="pole",
    )
    assert frame["constructor_quali_gap_recent3_pct"].notna().all()
    assert frame["constructor_circuit_quali_gap_prev_avg_pct"].isna().all()


def test_practice_longrun_gap_uses_a_compound_shared_by_multiple_drivers():
    rows = []
    # Both drivers run a medium stint and a soft stint. Their cross-compound
    # ranking differs, so the feature must select the shared medium benchmark.
    for driver, code, medium_base, soft_base in (
        ("driver_a", "AAA", 90.0, 101.0),
        ("driver_b", "BBB", 91.0, 96.0),
    ):
        for stint, compound, base in ((1, "MEDIUM", medium_base), (2, "SOFT", soft_base)):
            for lap_no in range(1, 8):
                rows.append({
                    "season": 2026, "round": 4, "session": "FP1", "driver_code": code,
                    "lap_number": lap_no, "stint": stint, "lap_time_seconds": base + lap_no * 0.05,
                    "compound": compound, "tyre_life": lap_no, "is_accurate": True,
                    "deleted": False, "fastf1_generated": False, "track_status": "1",
                    "pit_in": None, "pit_out": None,
                })
    result = build_event_weekend_pace_features(pd.DataFrame(rows), 2026, 4, "FP1").set_index("driver_code")
    assert result.loc["AAA", "fp1_longrun_gap_pct"] == 0
    assert result.loc["BBB", "fp1_longrun_gap_pct"] > 0


def test_race_longrun_history_uses_same_compound_before_lagging():
    from src.features.weekend_pace import build_historical_race_pace_table

    results = []
    laps = []
    for round_no in (1, 2):
        for driver, code, team, medium_base, soft_base in (
            ("a", "AAA", "team_a", 90.0, 101.0),
            ("b", "BBB", "team_b", 91.0, 96.0),
        ):
            results.append({
                "season": 2025, "round": round_no, "driver_id": driver,
                "driver_code": code, "constructor_id": team,
            })
            for stint, compound, base in ((1, "MEDIUM", medium_base), (2, "SOFT", soft_base)):
                for lap_no in range(1, 8):
                    laps.append({
                        "season": 2025, "round": round_no, "session": "R", "driver_code": code,
                        "lap_number": lap_no, "stint": stint, "lap_time_seconds": base + lap_no * 0.05,
                        "compound": compound, "tyre_life": lap_no, "is_accurate": True,
                        "track_status": "1", "deleted": False, "fastf1_generated": False,
                        "pit_in": None, "pit_out": None,
                    })
    features = build_historical_race_pace_table(pd.DataFrame(laps), pd.DataFrame(results))
    next_round = features[features["round"] == 2].set_index("driver_id")
    assert next_round.loc["a", "driver_race_longrun_gap_prev_avg_pct"] == 0
    assert next_round.loc["b", "driver_race_longrun_gap_prev_avg_pct"] > 0
