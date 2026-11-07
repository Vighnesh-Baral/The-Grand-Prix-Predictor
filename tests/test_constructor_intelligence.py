import numpy as np
import pandas as pd

from src.features.constructor_intelligence import (
    _current_constructor_practice,
    build_historical_constructor_intelligence_table,
)


def _historical_races():
    rows = []
    for rnd, results in [
        (1, [("d1", "A", "team_a", "Engine", 2, 1), ("d2", "B", "team_b", "Finished", 1, 2)]),
        (2, [("d1", "A", "team_a", "Finished", 1, 1), ("d2", "B", "team_b", "Finished", 2, 2)]),
    ]:
        for driver, code, team, status, position, grid in results:
            rows.append({
                "season": 2025, "round": rnd, "date": f"2025-03-{rnd + 1:02d}",
                "circuit_id": "same_circuit", "driver_id": driver, "driver_code": code,
                "constructor_id": team, "constructor_name": team.upper(),
                "grid": grid, "position_order": position, "status": status,
            })
    return pd.DataFrame(rows)


def test_constructor_reliability_and_safety_car_features_are_lagged():
    races = _historical_races()
    control = pd.DataFrame([
        {"season": 2025, "round": 1, "session_code": "R", "message": "SAFETY CAR DEPLOYED"},
        {"season": 2025, "round": 2, "session_code": "R", "message": "VIRTUAL SAFETY CAR DEPLOYED"},
    ])
    table = build_historical_constructor_intelligence_table(races, race_control=control)
    event_one = table[table["round"].eq(1)]
    event_two = table[table["round"].eq(2)].set_index("driver_id")

    # A race cannot use its own retirement or its own neutralisation messages.
    assert event_one["constructor_dnf_rate_recent5"].isna().all()
    assert event_one["circuit_sc_event_rate_prev"].isna().all()

    assert event_two.loc["d1", "constructor_dnf_rate_recent5"] == 1.0
    assert event_two.loc["d1", "constructor_mechanical_dnf_rate_recent5"] == 1.0
    assert event_two.loc["d1", "driver_dnf_rate_recent5"] == 1.0
    assert event_two.loc["d1", "circuit_sc_event_rate_prev"] == 1.0
    assert event_two.loc["d1", "circuit_vsc_event_rate_prev"] == 0.0


def test_constructor_practice_cutoff_excludes_later_fp_sessions():
    candidates = pd.DataFrame([
        {"driver_id": "d1", "driver_code": "AAA", "constructor_id": "team_a"},
        {"driver_id": "d2", "driver_code": "BBB", "constructor_id": "team_b"},
    ])
    laps = []
    for session, times in [("FP1", {"AAA": 80.0, "BBB": 80.3}), ("FP2", {"AAA": 81.0, "BBB": 79.0})]:
        for code, base in times.items():
            for lap in range(1, 7):
                laps.append({
                    "season": 2026, "round": 3, "session": session,
                    "driver_code": code, "lap_number": lap,
                    "lap_time_seconds": base + 0.04 * (lap - 1),
                    "stint": 1, "compound": "MEDIUM", "tyre_life": lap,
                    "is_accurate": True, "track_status": "1",
                })
    frame = pd.DataFrame(laps)

    fp1 = _current_constructor_practice(frame, 2026, 3, candidates, "FP1").set_index("constructor_id")
    fp2 = _current_constructor_practice(frame, 2026, 3, candidates, "FP2").set_index("constructor_id")

    assert fp1.loc["team_a", "constructor_practice_latest_best_gap_pct"] < fp1.loc["team_b", "constructor_practice_latest_best_gap_pct"]
    assert fp2.loc["team_b", "constructor_practice_latest_best_gap_pct"] < fp2.loc["team_a", "constructor_practice_latest_best_gap_pct"]
    assert fp1.loc["team_a", "constructor_practice_medium_deg_sec_per_lap"] > 0


def test_recent_pit_stop_metrics_are_mapped_to_constructor_without_future_data():
    races = _historical_races()
    drivers = pd.DataFrame([
        {"session_key": "S1", "driver_number": 1, "name_acronym": "A"},
        {"session_key": "S1", "driver_number": 2, "name_acronym": "B"},
    ])
    pits = pd.DataFrame([
        {"season": 2025, "round": 1, "session": "R", "session_key": "S1", "driver_number": 1, "pit_duration": 22.0},
        {"season": 2025, "round": 1, "session": "R", "session_key": "S1", "driver_number": 2, "pit_duration": 24.0},
    ])
    table = build_historical_constructor_intelligence_table(
        races, pit_stops=pits, openf1_drivers=drivers,
    )
    event_one = table[table["round"].eq(1)]
    event_two = table[table["round"].eq(2)].set_index("driver_id")

    assert event_one["constructor_pit_stop_mean_duration_recent5_sec"].isna().all()
    assert event_two.loc["d1", "constructor_pit_stop_mean_duration_recent5_sec"] == 22.0
    assert event_two.loc["d2", "constructor_pit_stop_mean_duration_recent5_sec"] == 24.0


def test_neutralisation_rate_is_a_global_fallback_when_no_local_circuit_history_exists():
    races = _historical_races().copy()
    races.loc[races["round"].eq(2), "circuit_id"] = "new_circuit"
    control = pd.DataFrame([
        {"season": 2025, "round": 1, "session_code": "R", "message": "SAFETY CAR DEPLOYED"},
    ])
    table = build_historical_constructor_intelligence_table(races, race_control=control)
    event_two = table[table["round"].eq(2)]
    # The current circuit has no prior visit, but a global prior exists from the
    # earlier event; the absence of a local sample is not interpreted as zero.
    assert event_two["circuit_sc_event_rate_prev"].notna().all()
    assert event_two["circuit_sc_event_rate_prev"].eq(1.0).all()
