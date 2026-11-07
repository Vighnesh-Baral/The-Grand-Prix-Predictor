from pathlib import Path

from src.data.jolpica_client import JolpicaClient


def test_expand_race_session_rows():
    payload = [{"raceName": "Example GP", "Results": [{"position": "1"}, {"position": "2"}]}]
    rows = JolpicaClient._expand_race_session_rows(payload, "Results")
    assert len(rows) == 2
    assert rows[0]["position"] == "1"
    assert rows[0]["race"]["raceName"] == "Example GP"


def test_fetch_nested_collection_paginates_by_driver_rows(tmp_path: Path):
    client = JolpicaClient(cache_dir=tmp_path, min_request_interval=0)
    calls = []

    def fake_get_json(base, path, params=None, refresh=False):
        calls.append(params["offset"])
        if params["offset"] == 0:
            races = [
                {"raceName": "GP A", "Results": [{"position": "1"}]},
                {"raceName": "GP B", "Results": [{"position": "1"}]},
            ]
        else:
            races = [
                {"raceName": "GP C", "Results": [{"position": "1"}]},
                {"raceName": "GP D", "Results": [{"position": "1"}]},
            ]
        return {"MRData": {"total": "4", "RaceTable": {"Races": races}}}

    client._get_json = fake_get_json
    records = client._fetch_ergast_collection("2024/results", "Races", page_size=2, nested_key="Results")
    assert len(records) == 4
    assert calls == [0, 2]


def test_standings_normalizer_shapes():
    from src.historical.standings import driver_standings_frame, constructor_standings_frame

    driver = driver_standings_frame([{
        "position": "1", "points": "100", "wins": "3",
        "Driver": {"driverId": "driver_a", "givenName": "Alex", "familyName": "Driver"},
        "Constructors": [{"name": "Team A"}],
    }])
    constructor = constructor_standings_frame([{
        "position": "1", "points": "150", "wins": "4",
        "Constructor": {"constructorId": "team_a", "name": "Team A"},
    }])
    assert driver.iloc[0]["driver"] == "Alex Driver"
    assert constructor.iloc[0]["constructor"] == "Team A"


def test_practice_client_uses_combined_fp_fallback_when_split_sessions_absent(tmp_path: Path):
    client = JolpicaClient(cache_dir=tmp_path, min_request_interval=0)
    requested = []

    def fake_get_json(base, path, params=None, refresh=False):
        requested.append(path)
        if path == "schedules/2024/":
            return {"data": {"events": [{
                "name": "Example GP", "round": {"id": "round_test", "number": 1},
                "circuit": {"id": "test_circuit"},
            }]}}
        if path.endswith("/FP/"):
            return {"data": {"results": [{
                "position": 1,
                "driver": {"id": "driver_a", "name": "Alex Driver", "code": "ALX"},
                "team": {"id": "team_a", "name": "Team A"},
                "time": {"value": "1:20.000"}, "laps": 20,
            }]}}
        return {"data": {"results": []}}

    client._get_json = fake_get_json
    rows = client.get_practice_results(2024)
    assert len(rows) == 1
    assert rows[0]["session"] == "FP"
    assert rows[0]["driver_name"] == "Alex Driver"
    assert "results/round_test/FP/" in requested
