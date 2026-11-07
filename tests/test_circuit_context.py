import pandas as pd

from src.features.circuit_context import build_historical_circuit_features, circuit_features_for_candidates


def _races():
    rows = []
    # Historical Monaco: pole converts strongly; P2/P3 convert less often.
    for season, wins in [(2022, {1: 1, 2: 0, 3: 0}), (2023, {1: 1, 2: 0, 3: 0}), (2024, {1: 1, 2: 0, 3: 0})]:
        for grid in range(1, 5):
            rows.append({
                "season": season, "round": 1, "date": f"{season}-05-28",
                "circuit_id": "monaco", "driver_id": f"d{grid}",
                "grid": grid, "position_order": 1 if wins.get(grid, 0) else grid + (1 if grid == 1 else 0),
            })
    return pd.DataFrame(rows)


def test_circuit_conversion_features_use_only_prior_events():
    races = _races()
    features = build_historical_circuit_features(races)
    target = features[(features.season == 2024) & (features.driver_id == "d1")].iloc[0]
    p2 = features[(features.season == 2024) & (features.driver_id == "d2")].iloc[0]
    assert target.circuit_history_event_count_prev == 2
    assert target.circuit_pole_to_win_rate_prev > p2.circuit_p2_p3_to_win_rate_prev
    # Altering the target event's result cannot change features for that event.
    changed = races.copy()
    changed.loc[(changed.season == 2024) & (changed.driver_id == "d1"), "position_order"] = 19
    changed_features = build_historical_circuit_features(changed)
    changed_target = changed_features[(changed_features.season == 2024) & (changed_features.driver_id == "d1")].iloc[0]
    assert target.circuit_pole_to_win_rate_prev == changed_target.circuit_pole_to_win_rate_prev


def test_future_candidates_use_grid_specific_circuit_history():
    prior = _races()[lambda x: x.season < 2024]
    candidates = pd.DataFrame([{"driver_id": "new_a", "grid": 1}, {"driver_id": "new_b", "grid": 2}])
    features = circuit_features_for_candidates(prior, candidates, "monaco")
    pole = features[features.driver_id == "new_a"].iloc[0]
    p2 = features[features.driver_id == "new_b"].iloc[0]
    assert pole.circuit_grid_win_rate_prev > p2.circuit_grid_win_rate_prev
