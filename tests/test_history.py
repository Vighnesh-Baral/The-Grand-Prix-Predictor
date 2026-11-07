import pandas as pd
from src.historical.gp_history import build_gp_history


def test_gp_history_links_winner_pole_and_podium():
    races = pd.DataFrame([{"season": 2024, "round": 1, "race_name": "Test GP", "circuit_id": "test", "circuit_name": "Test Circuit", "date": "2024-03-03"}])
    results = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_name": "Winner", "constructor_name": "Team A", "position_order": 1},
        {"season": 2024, "round": 1, "driver_name": "Second", "constructor_name": "Team B", "position_order": 2},
        {"season": 2024, "round": 1, "driver_name": "Third", "constructor_name": "Team C", "position_order": 3},
    ])
    qualifying = pd.DataFrame([{"season": 2024, "round": 1, "driver_name": "Pole", "qualifying_position": 1}])
    history = build_gp_history(races, results, qualifying, "test")
    assert history.iloc[0]["winner"] == "Winner"
    assert history.iloc[0]["pole_sitter"] == "Pole"
    assert history.iloc[0]["podium"] == "Winner · Second · Third"


def test_gp_history_uses_exact_gp_name_when_requested():
    races = pd.DataFrame([
        {"season": 2021, "round": 1, "race_name": "Austrian Grand Prix", "circuit_id": "red_bull_ring", "circuit_name": "Red Bull Ring", "date": "2021-07-04"},
        {"season": 2021, "round": 2, "race_name": "Styrian Grand Prix", "circuit_id": "red_bull_ring", "circuit_name": "Red Bull Ring", "date": "2021-06-27"},
    ])
    results = pd.DataFrame([
        {"season": 2021, "round": 1, "driver_name": "Austrian Winner", "constructor_name": "Team A", "position_order": 1},
        {"season": 2021, "round": 2, "driver_name": "Styrian Winner", "constructor_name": "Team B", "position_order": 1},
    ])
    history = build_gp_history(races, results, pd.DataFrame(), "red_bull_ring", race_name="Austrian Grand Prix")
    assert history["grand_prix"].tolist() == ["Austrian Grand Prix"]
    assert history["winner"].tolist() == ["Austrian Winner"]


def test_local_standings_include_sprint_points_and_limit_round_snapshot():
    from src.historical.standings import local_constructor_standings, local_driver_standings

    races = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "a", "driver_name": "Driver A", "constructor_id": "team_a", "constructor_name": "Team A", "points": 25, "position_order": 1},
        {"season": 2024, "round": 1, "driver_id": "b", "driver_name": "Driver B", "constructor_id": "team_b", "constructor_name": "Team B", "points": 18, "position_order": 2},
        {"season": 2024, "round": 2, "driver_id": "a", "driver_name": "Driver A", "constructor_id": "team_a", "constructor_name": "Team A", "points": 18, "position_order": 2},
    ])
    sprints = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "a", "constructor_id": "team_a", "points": 8},
        {"season": 2024, "round": 2, "driver_id": "a", "constructor_id": "team_a", "points": 7},
    ])
    drivers = local_driver_standings(races, sprints, 2024, round_number=1)
    constructors = local_constructor_standings(races, sprints, 2024, round_number=1)
    assert drivers.iloc[0]["driver"] == "Driver A"
    assert drivers.iloc[0]["points"] == 33
    assert constructors.iloc[0]["constructor"] == "Team A"
    assert constructors.iloc[0]["points"] == 33
    assert constructors["points"].sum() == 51


def test_gp_history_includes_available_practice_session_fastest_drivers():
    races = pd.DataFrame([{
        "season": 2024, "round": 1, "race_name": "Test GP", "circuit_id": "test",
        "circuit_name": "Test Circuit", "date": "2024-03-03",
    }])
    results = pd.DataFrame([{
        "season": 2024, "round": 1, "driver_name": "Winner", "constructor_name": "Team A", "position_order": 1,
    }])
    practice = pd.DataFrame([
        {"season": 2024, "round": 1, "session": "FP1", "driver_name": "Fast FP1", "time_seconds": 80.0, "position": 1},
        {"season": 2024, "round": 1, "session": "FP1", "driver_name": "Slow FP1", "time_seconds": 81.0, "position": 2},
        {"season": 2024, "round": 1, "session": "FP2", "driver_name": "Fast FP2", "time_seconds": 79.5, "position": 1},
    ])
    history = build_gp_history(races, results, pd.DataFrame(), "test", race_name="Test GP", practice_results=practice)
    assert history.iloc[0]["fp1_fastest"] == "Fast FP1"
    assert history.iloc[0]["fp2_fastest"] == "Fast FP2"
    assert history.iloc[0]["fp3_fastest"] == "Not available"
