from datetime import datetime, timezone, timedelta

import pandas as pd

from src.data.openf1_sync import OpenF1Sync, _map_sessions_to_calendar


class FakeOpenF1Client:
    def __init__(self, sessions, endpoints):
        self._sessions = sessions
        self._endpoints = endpoints
        self.calls = []

    def sessions(self, year, refresh=False):
        return [row for row in self._sessions if int(row["year"]) == int(year)]

    def session_endpoint(self, endpoint, session_key):
        self.calls.append((endpoint, session_key))
        return self._endpoints.get(endpoint, [])


def test_session_mapping_uses_circuit_and_weekend_date():
    sessions = pd.DataFrame([{
        "year": 2025, "circuit_short_name": "Monaco", "date_start": "2025-05-23T10:00:00Z",
    }])
    calendar = pd.DataFrame([{
        "season": 2025, "round": 8, "date": "2025-05-25", "circuit_id": "monaco", "race_name": "Monaco Grand Prix",
    }])
    mapped = _map_sessions_to_calendar(sessions, calendar)
    assert mapped.loc[0, "round"] == 8
    assert mapped.loc[0, "circuit_id"] == "monaco"


def test_sync_waits_for_free_window_and_persists_normalized_weather_and_laps(tmp_path):
    now = datetime(2025, 5, 23, 14, 40, tzinfo=timezone.utc)
    session_row = {
        "year": 2025, "session_key": 9001, "meeting_key": 8001,
        "circuit_short_name": "Monaco", "session_name": "Qualifying",
        "date_start": "2025-05-23T13:00:00Z", "date_end": "2025-05-23T14:00:00Z",
        "is_cancelled": False,
    }
    calendar = pd.DataFrame([{
        "season": 2025, "round": 8, "date": "2025-05-25", "circuit_id": "monaco", "race_name": "Monaco Grand Prix",
    }])
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    calendar.to_csv(processed / "races.csv", index=False)
    endpoints = {
        "drivers": [{"session_key": 9001, "driver_number": 16, "name_acronym": "LEC", "full_name": "Charles Leclerc", "team_name": "Ferrari"}],
        "laps": [{"session_key": 9001, "driver_number": 16, "lap_number": 1, "lap_duration": 89.2, "is_pit_out_lap": False}],
        "stints": [{"session_key": 9001, "driver_number": 16, "stint_number": 1, "lap_start": 1, "lap_end": 10, "compound": "SOFT", "tyre_age_at_start": 2}],
        "weather": [{"session_key": 9001, "date": "2025-05-23T13:30:00Z", "air_temperature": 24, "track_temperature": 40, "humidity": 60, "wind_speed": 2, "rainfall": 0}],
        "race_control": [], "session_result": [],
    }
    client = FakeOpenF1Client([session_row], endpoints)
    syncer = OpenF1Sync(
        client, processed_dir=processed, raw_dir=raw, state_path=tmp_path / "state.json",
        now_fn=lambda: now,
    )
    # Only 40 minutes since session end: it is eligible after the 35-minute guard.
    result = syncer.sync(2025, 2025)
    assert result["sessions_eligible"] == 1
    weather = pd.read_csv(processed / "weather_observations.csv")
    assert weather.loc[0, "round"] == 8
    assert weather.loc[0, "air_temperature"] == 24
    laps = pd.read_csv(processed / "session_laps.csv")
    assert laps.loc[0, "driver_code"] == "LEC"
    assert laps.loc[0, "compound"] == "SOFT"
    assert laps.loc[0, "tyre_life"] == 2
    assert result["endpoints_fetched"] == len(client.calls)


def test_openf1_austin_session_maps_to_cota_jolpica_circuit():
    from src.data.openf1_sync import _map_sessions_to_calendar

    sessions = pd.DataFrame([{
        "year": 2024, "date_start": "2024-10-18T17:00:00+00:00",
        "circuit_short_name": "Austin",
    }])
    calendar = pd.DataFrame([{
        "season": 2024, "round": 19, "date": "2024-10-20",
        "race_name": "United States Grand Prix", "circuit_id": "circuit_of_the_americas",
    }])

    result = _map_sessions_to_calendar(sessions, calendar)

    assert result.loc[0, "round"] == 19
    assert result.loc[0, "circuit_id"] == "circuit_of_the_americas"


def test_sync_fetches_published_race_starting_grid_before_race(tmp_path):
    now = datetime(2025, 5, 24, 14, 0, tzinfo=timezone.utc)
    sessions = [
        {
            "year": 2025, "session_key": 9001, "meeting_key": 8001,
            "circuit_short_name": "Monaco", "session_name": "Qualifying",
            "date_start": "2025-05-24T10:00:00Z", "date_end": "2025-05-24T12:00:00Z",
            "is_cancelled": False,
        },
        {
            "year": 2025, "session_key": 9002, "meeting_key": 8001,
            "circuit_short_name": "Monaco", "session_name": "Race",
            "date_start": "2025-05-25T13:00:00Z", "date_end": "2025-05-25T15:00:00Z",
            "is_cancelled": False,
        },
    ]
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    pd.DataFrame([{
        "season": 2025, "round": 8, "date": "2025-05-25",
        "circuit_id": "monaco", "race_name": "Monaco Grand Prix",
    }]).to_csv(processed / "races.csv", index=False)
    pd.DataFrame([{
        "season": 2025, "round": 8, "driver_code": "LEC", "driver_id": "leclerc",
    }]).to_csv(processed / "qualifying_results.csv", index=False)
    client = FakeOpenF1Client(sessions, {
        "drivers": [{
            "session_key": 9001, "meeting_key": 8001, "driver_number": 16,
            "name_acronym": "LEC", "full_name": "Charles Leclerc", "team_name": "Ferrari",
        }],
        "starting_grid": [{
            "session_key": 9002, "meeting_key": 8001, "driver_number": 16, "position": 3,
        }],
    })
    syncer = OpenF1Sync(
        client, processed_dir=processed, raw_dir=raw, state_path=tmp_path / "state.json",
        now_fn=lambda: now,
    )

    result = syncer.sync(2025, 2025)

    assert ("starting_grid", 9002) in client.calls
    assert result["rows_written"]["starting_grid"] == 1
    grid = pd.read_csv(processed / "starting_grid.csv")
    assert grid.loc[0, "driver_id"] == "leclerc"
    assert grid.loc[0, "grid"] == 3
    assert grid.loc[0, "grid_source"] == "openf1_starting_grid"


def test_rebuild_local_tables_uses_raw_openf1_exports_when_processed_tables_missing(tmp_path):
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    pd.DataFrame([{
        "season": 2025, "round": 8, "session": "FP1", "session_key": 9001,
        "driver_number": 16, "lap_number": 1, "lap_duration": 89.2,
        "is_pit_out_lap": False,
    }]).to_csv(raw / "openf1_laps.csv", index=False)
    pd.DataFrame([{
        "session_key": 9001, "driver_number": 16, "name_acronym": "LEC",
        "full_name": "Charles Leclerc", "team_name": "Ferrari",
    }]).to_csv(raw / "openf1_drivers.csv", index=False)
    pd.DataFrame([{
        "session_key": 9001, "driver_number": 16, "stint_number": 1,
        "lap_start": 1, "lap_end": 10, "compound": "SOFT", "tyre_age_at_start": 2,
    }]).to_csv(raw / "openf1_stints.csv", index=False)
    pd.DataFrame([{
        "season": 2025, "round": 8, "session": "FP1", "session_key": 9001,
        "date": "2025-05-23T13:30:00Z", "air_temperature": 24,
        "track_temperature": 40, "humidity": 60, "wind_speed": 2, "rainfall": 0,
    }]).to_csv(raw / "openf1_weather.csv", index=False)

    syncer = OpenF1Sync(
        FakeOpenF1Client([], {}), processed_dir=processed, raw_dir=raw,
        state_path=tmp_path / "state.json",
    )
    syncer.rebuild_local_tables()

    canonical = pd.read_csv(processed / "session_laps.csv")
    assert canonical.loc[0, "driver_code"] == "LEC"
    assert canonical.loc[0, "driver_name"] == "Charles Leclerc"
    assert canonical.loc[0, "constructor_name"] == "Ferrari"
    assert canonical.loc[0, "lap_time_seconds"] == 89.2
    assert canonical.loc[0, "compound"] == "SOFT"
    assert canonical.loc[0, "tyre_life"] == 2
    assert canonical.loc[0, "data_source"] == "openf1"
    weather = pd.read_csv(processed / "weather_observations.csv")
    assert len(weather) == 1


def test_rebuild_local_tables_merges_new_raw_rows_with_processed_openf1_exports(tmp_path):
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    processed.mkdir()
    raw.mkdir()
    pd.DataFrame([
        {"session_key": 1, "driver_number": 44, "lap_number": 1, "lap_duration": 90.1, "season": 2024, "round": 1, "session": "FP1"},
    ]).to_csv(processed / "openf1_laps.csv", index=False)
    pd.DataFrame([
        {"session_key": 1, "driver_number": 44, "lap_number": 1, "lap_duration": 90.1, "season": 2024, "round": 1, "session": "FP1"},
        {"session_key": 1, "driver_number": 63, "lap_number": 1, "lap_duration": 90.8, "season": 2024, "round": 1, "session": "FP1"},
    ]).to_csv(raw / "openf1_laps.csv", index=False)

    syncer = OpenF1Sync(client=FakeOpenF1Client([], {}), processed_dir=processed, raw_dir=raw, state_path=tmp_path / "state.json")
    merged = syncer._read_endpoint("laps")
    assert len(merged) == 2
    syncer.rebuild_local_tables()
    persisted = pd.read_csv(processed / "openf1_laps.csv")
    assert len(persisted) == 2
    canonical = pd.read_csv(processed / "session_laps.csv")
    assert set(canonical["driver_number"].astype(int)) == {44, 63}
