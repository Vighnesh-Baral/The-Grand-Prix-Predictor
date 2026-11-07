from src.data.normalize import normalize_qualifying, normalize_race_results, normalize_schedule, parse_lap_time_seconds


def test_parse_lap_time_seconds():
    assert parse_lap_time_seconds("1:23.500") == 83.5
    assert parse_lap_time_seconds("59.500") == 59.5
    assert parse_lap_time_seconds(None) is None


def test_normalize_race_result():
    record = {
        "position": "1", "positionOrder": "1", "grid": "2", "points": "25", "laps": "58", "status": "Finished",
        "Driver": {"driverId": "driver_a", "code": "DRA", "givenName": "Alex", "familyName": "Driver"},
        "Constructor": {"constructorId": "team_a", "name": "Team A"},
        "race": {"season": "2024", "round": "1", "raceName": "Test Grand Prix", "date": "2024-03-03", "Circuit": {"circuitId": "test"}},
    }
    row = normalize_race_results([record]).iloc[0]
    assert row["driver_name"] == "Alex Driver"
    assert row["constructor_name"] == "Team A"
    assert row["position_order"] == 1


def test_normalize_schedule():
    row = normalize_schedule([{"season": "2024", "round": "1", "raceName": "Test GP", "date": "2024-03-03", "Circuit": {"circuitId": "test", "circuitName": "Test Circuit", "Location": {"country": "Testland"}}}]).iloc[0]
    assert row["season"] == 2024
    assert row["circuit_id"] == "test"
