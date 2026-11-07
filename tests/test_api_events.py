from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import app.main as api


def test_upcoming_event_api_excludes_completed_and_past_events(monkeypatch):
    future_date = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=8)).date().isoformat()
    tables = {
        "races": pd.DataFrame([
            {"season": 2026, "round": 1, "race_name": "Past GP", "date": "2026-09-01", "time": "13:00:00Z", "circuit_id": "past"},
            {"season": 2026, "round": 2, "race_name": "Completed GP", "date": future_date, "time": "11:00:00Z", "circuit_id": "done"},
            {"season": 2026, "round": 3, "race_name": "Upcoming GP", "date": future_date, "time": "15:00:00Z", "circuit_id": "next"},
        ]),
        "race_results": pd.DataFrame([
            {"season": 2026, "round": 1, "driver_id": "old"},
            {"season": 2026, "round": 2, "driver_id": "finished"},
        ]),
        "qualifying": pd.DataFrame(columns=["season", "round", "driver_id"]),
        "practice": pd.DataFrame(),
        "sprints": pd.DataFrame(),
    }
    monkeypatch.setattr(api, "load_tables", lambda: tables)
    client = TestClient(api.app)

    result = client.get("/api/events?upcoming_only=true")
    assert result.status_code == 200
    payload = result.json()
    assert [event["race_name"] for event in payload["events"]] == ["Upcoming GP"]

    next_result = client.get("/api/events/next")
    assert next_result.status_code == 200
    assert next_result.json()["event"]["race_name"] == "Upcoming GP"


def test_pace_intelligence_endpoints_expose_components_without_composite_score(monkeypatch):
    future_date = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=20)).date().isoformat()
    races = pd.DataFrame([
        {"season": 2026, "round": 1, "race_name": "Previous GP", "date": "2026-03-01", "time": "13:00:00Z", "circuit_id": "old"},
        {"season": 2027, "round": 1, "race_name": "Next GP", "date": future_date, "time": "13:00:00Z", "circuit_id": "next"},
    ])
    race_results = pd.DataFrame([
        {"season": 2026, "round": 1, "date": "2026-03-01", "race_name": "Previous GP", "circuit_id": "old", "driver_id": "a", "driver_code": "AAA", "driver_name": "Driver A", "constructor_id": "team_a", "constructor_name": "Team A", "grid": 1, "position_order": 1, "points": 25, "status": "Finished", "fastest_lap_time": "1:30.000"},
        {"season": 2026, "round": 1, "date": "2026-03-01", "race_name": "Previous GP", "circuit_id": "old", "driver_id": "b", "driver_code": "BBB", "driver_name": "Driver B", "constructor_id": "team_b", "constructor_name": "Team B", "grid": 2, "position_order": 2, "points": 18, "status": "Finished", "fastest_lap_time": "1:31.000"},
    ])
    qualifying = pd.DataFrame([
        {"season": 2026, "round": 1, "driver_id": "a", "driver_code": "AAA", "driver_name": "Driver A", "constructor_id": "team_a", "constructor_name": "Team A", "qualifying_position": 1, "q1": "1:20.000", "q2": "1:19.000", "q3": "1:18.000"},
        {"season": 2026, "round": 1, "driver_id": "b", "driver_code": "BBB", "driver_name": "Driver B", "constructor_id": "team_b", "constructor_name": "Team B", "qualifying_position": 2, "q1": "1:20.500", "q2": "1:19.500", "q3": "1:18.500"},
    ])
    session_laps = []
    for driver, code, team, lap_base in (("a", "AAA", "Team A", 80.0), ("b", "BBB", "Team B", 81.0)):
        for lap_no in range(1, 9):
            session_laps.append({
                "season": 2027, "round": 1, "session": "FP1", "driver_code": code,
                "constructor_name": team, "lap_number": lap_no, "stint": 1,
                "lap_time_seconds": lap_base + (0.1 * lap_no), "compound": "SOFT" if lap_no <= 2 else "MEDIUM",
                "tyre_life": lap_no, "is_accurate": True, "deleted": False,
                "fastf1_generated": False, "track_status": "1", "pit_in": None, "pit_out": None,
            })
    tables = {
        "races": races, "race_results": race_results, "qualifying": qualifying,
        "practice": pd.DataFrame(), "sprints": pd.DataFrame(), "session_laps": pd.DataFrame(session_laps),
    }
    monkeypatch.setattr(api, "load_tables", lambda: tables)
    client = TestClient(api.app)

    practice_result = client.get("/api/practice/pace?season=2027&round=1&as_of_session=FP1")
    assert practice_result.status_code == 200
    assert practice_result.json()["available"] is True
    assert len(practice_result.json()["drivers"]) == 2
    assert len(practice_result.json()["constructors"]) == 2

    constructor_result = client.get("/api/intelligence/constructor-pace?season=2027&round=1&as_of_session=FP1")
    assert constructor_result.status_code == 200
    body = constructor_result.json()
    assert len(body["constructors"]) == 2
    assert set(body["pillars"]) == {
        "qualifying_pace", "race_pace", "long_run_pace", "tyre_behaviour",
        "reliability", "circuit_suitability",
    }
    assert set(body["pillar_coverage"]) == set(body["pillars"])
    assert "composite_score" not in body["constructors"][0]

    driver_result = client.get("/api/intelligence/driver-performance?season=2027&round=1&as_of_session=FP1")
    assert driver_result.status_code == 200
    assert len(driver_result.json()["drivers"]) == 2


def test_data_readiness_endpoint_reports_session_coverage_and_training_gap(monkeypatch):
    tables = {
        "races": pd.DataFrame([{"season": 2025, "round": 1}]),
        "race_results": pd.DataFrame([{"season": 2025, "round": 1}]),
        "qualifying": pd.DataFrame([{"season": 2025, "round": 1}]),
        "practice": pd.DataFrame(),
        "sprints": pd.DataFrame(),
        "session_laps": pd.DataFrame([
            {"season": 2025, "round": 1, "session": "FP1", "driver_code": "AAA"},
            {"season": 2025, "round": 1, "session": "R", "driver_code": "AAA"},
        ]),
    }
    monkeypatch.setattr(api, "load_tables", lambda: tables)
    monkeypatch.setattr(api, "_metrics", lambda: {
        "available": True,
        "metrics": {"tasks": {"pole": {"feature_coverage_fraction": {"fp_best_gap_pct": 0.0}}}},
    })
    client = TestClient(api.app)

    result = client.get("/api/data/readiness")
    assert result.status_code == 200
    body = result.json()
    assert body["data_ready_for_basic_history"] is True
    assert body["data_ready_for_pace_aware_training"] is False
    assert body["fastf1"]["lap_rows"] == 2
    assert {row["session"] for row in body["fastf1"]["sessions"]} == {"FP1", "R"}
    assert body["missing_practice_sessions"] == ["FP2", "FP3"]
    assert any("FP2" in item for item in body["recommendations"])


def test_live_prediction_endpoints_block_historical_and_wrong_stage_requests(monkeypatch):
    future_date = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=20)).date().isoformat()
    tables = {
        "races": pd.DataFrame([
            {"season": 2027, "round": 1, "race_name": "Upcoming GP", "date": future_date, "time": "13:00:00Z", "circuit_id": "next"},
            {"season": 2026, "round": 1, "race_name": "Completed GP", "date": "2026-03-01", "time": "13:00:00Z", "circuit_id": "past"},
        ]),
        "race_results": pd.DataFrame([{"season": 2026, "round": 1, "driver_id": "old"}]),
        "qualifying": pd.DataFrame([{"season": 2027, "round": 1, "driver_id": "current"}]),
        "practice": pd.DataFrame(),
        "sprints": pd.DataFrame(),
        "session_laps": pd.DataFrame(),
        "driver_roster": pd.DataFrame(),
    }
    monkeypatch.setattr(api, "load_tables", lambda: tables)
    client = TestClient(api.app)

    # Pole prediction is no longer valid once qualifying classifications exist.
    pole_after_quali = client.post("/api/predictions/pole", json={"season": 2027, "round": 1, "as_of_session": "FP3"})
    assert pole_after_quali.status_code == 409
    assert "Qualifying results are already available" in pole_after_quali.json()["detail"]

    # Race prediction cannot run before qualifying has been collected.
    tables["qualifying"] = pd.DataFrame(columns=["season", "round", "driver_id"])
    race_before_quali = client.post("/api/predictions/race", json={"season": 2027, "round": 1})
    assert race_before_quali.status_code == 409
    assert "only available after qualifying" in race_before_quali.json()["detail"]

    # A completed historic event is never exposed as a live prediction target.
    past_pole = client.post("/api/predictions/pole", json={"season": 2026, "round": 1, "as_of_session": "FP3"})
    assert past_pole.status_code == 409
    assert "already been completed" in past_pole.json()["detail"]


def test_date_only_future_event_is_listed_but_live_prediction_is_blocked(monkeypatch):
    future_date = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=30)).date().isoformat()
    tables = {
        "races": pd.DataFrame([
            {"season": 2027, "round": 1, "race_name": "Time Pending GP", "date": future_date, "circuit_id": "pending"}
        ]),
        "race_results": pd.DataFrame(columns=["season", "round", "driver_id"]),
        "qualifying": pd.DataFrame(columns=["season", "round", "driver_id"]),
        "practice": pd.DataFrame(), "sprints": pd.DataFrame(),
        "session_laps": pd.DataFrame(), "driver_roster": pd.DataFrame(),
    }
    monkeypatch.setattr(api, "load_tables", lambda: tables)
    client = TestClient(api.app)

    listing = client.get("/api/events?upcoming_only=true")
    assert listing.status_code == 200
    event = listing.json()["events"][0]
    assert event["status"] == "scheduled_time_unknown"
    assert event["pole_prediction_available"] is False

    prediction = client.post("/api/predictions/pole", json={"season": 2027, "round": 1, "as_of_session": "FP1"})
    assert prediction.status_code == 409
    assert "event timing is unavailable" in prediction.json()["detail"]


def test_weather_forecast_endpoint_returns_latest_provider_forecast(monkeypatch):
    future_start = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=4)
    races = pd.DataFrame([{
        "season": 2027, "round": 1, "race_name": "Forecast GP",
        "date": future_start.date().isoformat(), "time": future_start.strftime("%H:%M:%SZ"),
        "circuit_id": "forecast", "latitude": 25.0, "longitude": 25.0,
    }])
    tables = {
        "races": races,
        "race_results": pd.DataFrame(columns=["season", "round", "driver_id"]),
        "qualifying": pd.DataFrame(columns=["season", "round", "driver_id"]),
        "practice": pd.DataFrame(), "sprints": pd.DataFrame(),
        "session_laps": pd.DataFrame(), "driver_roster": pd.DataFrame(),
        "weather": pd.DataFrame(), "race_weather_forecasts": pd.DataFrame(),
    }

    class FakeWeatherClient:
        def live_event_forecast(self, event):
            return {
                "season": 2027, "round": 1,
                "race_forecast_air_temp_c": 24.0,
                "forecast_source": "open_meteo_live_forecast",
            }

    monkeypatch.setattr(api, "load_tables", lambda: tables)
    monkeypatch.setattr(api, "OpenMeteoClient", FakeWeatherClient)
    client = TestClient(api.app)

    response = client.get("/api/weather/forecast?season=2027&round=1")

    assert response.status_code == 200
    assert response.json()["available"] is True
    assert response.json()["forecast"]["race_forecast_air_temp_c"] == 24.0


def test_circuit_catalog_covers_all_distinct_tracks_since_2018(monkeypatch):
    tables = {
        "races": pd.DataFrame([
            {"season": 2018, "round": 1, "race_name": "Australian Grand Prix", "circuit_id": "albert_park", "circuit_name": "Albert Park", "country": "Australia"},
            {"season": 2021, "round": 1, "race_name": "Bahrain Grand Prix", "circuit_id": "bahrain", "circuit_name": "Bahrain International Circuit", "country": "Bahrain"},
            {"season": 2021, "round": 2, "race_name": "Sakhir Grand Prix", "circuit_id": "bahrain", "circuit_name": "Bahrain International Circuit", "country": "Bahrain"},
            {"season": 2026, "round": 1, "race_name": "Australian Grand Prix", "circuit_id": "albert_park", "circuit_name": "Albert Park", "country": "Australia"},
        ]),
        "race_results": pd.DataFrame(columns=["season", "round", "driver_id"]),
        "qualifying": pd.DataFrame(), "practice": pd.DataFrame(), "sprints": pd.DataFrame(),
        "session_laps": pd.DataFrame(),
    }
    monkeypatch.setattr(api, "load_tables", lambda: tables)
    client = TestClient(api.app)

    response = client.get("/api/circuits?first_season=2018&last_season=2026")
    assert response.status_code == 200
    body = response.json()
    assert body["season_range"] == "2018–2026"
    assert body["count"] == 2
    bahrain = next(row for row in body["circuits"] if row["circuit_id"] == "bahrain")
    assert bahrain["event_count"] == 2
    assert bahrain["grand_prix_names"] == ["Bahrain Grand Prix", "Sakhir Grand Prix"]


def test_circuit_history_includes_alternate_grand_prix_titles(monkeypatch):
    races = pd.DataFrame([
        {"season": 2021, "round": 1, "race_name": "Bahrain Grand Prix", "circuit_id": "bahrain", "circuit_name": "Bahrain International Circuit", "date": "2021-03-28"},
        {"season": 2021, "round": 2, "race_name": "Sakhir Grand Prix", "circuit_id": "bahrain", "circuit_name": "Bahrain International Circuit", "date": "2021-12-06"},
    ])
    results = pd.DataFrame([
        {"season": 2021, "round": 1, "driver_name": "Winner A", "constructor_name": "Team A", "position_order": 1},
        {"season": 2021, "round": 2, "driver_name": "Winner B", "constructor_name": "Team B", "position_order": 1},
    ])
    tables = {"races": races, "race_results": results, "qualifying": pd.DataFrame(), "practice": pd.DataFrame()}
    monkeypatch.setattr(api, "load_tables", lambda: tables)
    response = TestClient(api.app).get("/api/history/bahrain?season=2018&last_season=2026")
    assert response.status_code == 200
    assert [row["grand_prix"] for row in response.json()["rows"]] == ["Bahrain Grand Prix", "Sakhir Grand Prix"]
    assert [row["winner"] for row in response.json()["rows"]] == ["Winner A", "Winner B"]


def test_pre_qualifying_race_forecast_uses_pole_predictor_grid_and_fp3_cutoff(monkeypatch):
    future_start = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)
    races = pd.DataFrame([{
        "season": 2027, "round": 1, "race_name": "Upcoming GP", "date": future_start.date().isoformat(),
        "time": future_start.strftime("%H:%M:%SZ"), "circuit_id": "next",
    }])
    tables = {
        "races": races,
        "race_results": pd.DataFrame([{"season": 2026, "round": 1, "driver_id": "historic"}]),
        "qualifying": pd.DataFrame(columns=["season", "round", "driver_id"]),
    }
    pole_frame = pd.DataFrame([
        {"driver_name": "Driver One", "constructor_name": "Team A", "fp_session_count": 3, "fp_valid_laps": 30},
        {"driver_name": "Driver Two", "constructor_name": "Team B", "fp_session_count": 3, "fp_valid_laps": 30},
    ])
    race_frame = pd.DataFrame([
        {"driver_name": "Driver Two", "constructor_name": "Team B", "grid": np.nan, "qualifying_position": np.nan,
         "best_qualifying_time_seconds": np.nan, "starting_grid_source": "qualifying_position_proxy",
         "candidate_roster_source": "selected_event_practice_entries", "weather_air_temp_c": np.nan},
        {"driver_name": "Driver One", "constructor_name": "Team A", "grid": np.nan, "qualifying_position": np.nan,
         "best_qualifying_time_seconds": np.nan, "starting_grid_source": "qualifying_position_proxy",
         "candidate_roster_source": "selected_event_practice_entries", "weather_air_temp_c": np.nan},
    ])
    frame_calls = []
    def fake_event_frame(season, round_number, stage, as_of_session="FP3", weather_as_of_session=None):
        frame_calls.append((stage, as_of_session, weather_as_of_session))
        return pole_frame.copy() if stage == "pole" else race_frame.copy()
    pole_output = pd.DataFrame([
        {"driver_name": "Driver One", "constructor_name": "Team A", "pole_score_share": 0.6, "pole_probability": 0.6},
        {"driver_name": "Driver Two", "constructor_name": "Team B", "pole_score_share": 0.4, "pole_probability": 0.4},
    ])
    captured = {}
    def fake_race_prediction(bundle, frame):
        captured["frame"] = frame.copy()
        return pd.DataFrame([
            {"driver_name": row.driver_name, "constructor_name": row.constructor_name,
             "starting_grid_position": row.grid, "predicted_order": i + 1,
             "win_probability": 0.5, "podium_probability": 0.8, "dnf_probability": 0.05}
            for i, row in enumerate(frame.itertuples())
        ])
    monkeypatch.setattr(api, "load_tables", lambda: tables)
    monkeypatch.setattr(api, "_practice_rows_for_event", lambda season, round_number: {
        "sessions": [
            {"session": "FP1", "available": True},
            {"session": "FP2", "available": True},
            {"session": "FP3", "available": True},
        ],
        "rows": [], "available": True,
    })
    monkeypatch.setattr(api, "_load_model_bundle", lambda: {"models": {}})
    monkeypatch.setattr(api, "_event_frame", fake_event_frame)
    monkeypatch.setattr(api, "predict_pole", lambda bundle, frame, cutoff: pole_output.copy())
    monkeypatch.setattr(api, "predict_race", fake_race_prediction)
    monkeypatch.setattr(api, "_feature_coverage", lambda bundle, task, feature: 0.0)
    client = TestClient(api.app)

    response = client.post("/api/predictions/race", json={
        "season": 2027, "round": 1, "race_stage": "before_qualifying", "as_of_session": "FP3",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["starting_grid_source"] == "pole_predictor_predicted_grid"
    assert body["race_stage"] == "before_qualifying"
    assert body["pole_predictor_grid"][0]["driver_name"] == "Driver One"
    assert frame_calls == [("pole", "FP3", None), ("race", "FP3", "FP3")]
    captured_frame = captured["frame"].set_index("driver_name")
    assert captured_frame.loc["Driver One", "grid"] == 1
    assert captured_frame.loc["Driver Two", "grid"] == 2
    assert captured_frame["qualifying_position"].isna().all()
    assert captured_frame["best_qualifying_time_seconds"].isna().all()
    assert set(captured_frame["starting_grid_source"]) == {"pole_predictor_predicted_grid"}


def test_events_archive_lists_historical_and_future_while_live_selector_filters_history(monkeypatch):
    future_date = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=20)).date().isoformat()
    tables = {
        "races": pd.DataFrame([
            {"season": 2018, "round": 1, "race_name": "Historical GP", "date": "2018-03-25", "time": "05:00:00Z", "circuit_id": "old"},
            {"season": 2027, "round": 1, "race_name": "Future GP", "date": future_date, "time": "13:00:00Z", "circuit_id": "next"},
        ]),
        "race_results": pd.DataFrame([{"season": 2018, "round": 1, "driver_id": "old"}]),
        "qualifying": pd.DataFrame(columns=["season", "round", "driver_id"]),
    }
    monkeypatch.setattr(api, "load_tables", lambda: tables)
    client = TestClient(api.app)
    archive = client.get("/api/events?upcoming_only=false").json()["events"]
    live = client.get("/api/events?upcoming_only=true").json()["events"]
    assert {event["season"] for event in archive} == {2018, 2027}
    assert [event["race_name"] for event in live] == ["Future GP"]


def test_practice_archive_reconstructs_fp_classifications_from_session_laps():
    from app.main import _practice_rows_for_event

    tables = {
        "practice": pd.DataFrame([{
            "season": 2025, "round": 8, "session": "Practice 1", "position": 1,
            "driver_id": "leclerc", "driver_code": "LEC", "driver_name": "Charles Leclerc",
            "constructor_name": "Ferrari", "time_seconds": 89.2, "laps": 20,
            "source": "Jolpica-F1 classification",
        }]),
        "session_laps": pd.DataFrame([
            {"season": 2025, "round": 8, "session": "FP1", "driver_code": "LEC", "driver_name": "Charles Leclerc", "constructor_name": "Ferrari", "lap_time_seconds": 89.2, "data_source": "openf1"},
            {"season": 2025, "round": 8, "session": "FP1", "driver_code": "HAM", "driver_name": "Lewis Hamilton", "constructor_name": "Ferrari", "lap_time_seconds": 89.8, "data_source": "openf1"},
            {"season": 2025, "round": 8, "session": "FP2", "driver_code": "LEC", "driver_name": "Charles Leclerc", "constructor_name": "Ferrari", "lap_time_seconds": 88.9, "data_source": "openf1"},
            {"season": 2025, "round": 8, "session": "FP2", "driver_code": "HAM", "driver_name": "Lewis Hamilton", "constructor_name": "Ferrari", "lap_time_seconds": 89.5, "data_source": "openf1"},
        ]),
        "race_results": pd.DataFrame(),
        "openf1_drivers": pd.DataFrame(),
    }

    payload = _practice_rows_for_event(2025, 8, tables)

    assert len(payload["rows"]) == 4  # Official FP1 row is preferred; missing FP1 and FP2 drivers are filled from laps.
    assert [row["session"] for row in payload["rows"]] == ["FP1", "FP1", "FP2", "FP2"]
    assert len([row for row in payload["rows"] if row["session"] == "FP1" and row["driver_code"] == "LEC"]) == 1
    assert payload["sessions"] == [
        {"session": "FP1", "rows": 2, "available": True, "sources": ["Jolpica-F1 classification", "OpenF1 lap-derived classification"]},
        {"session": "FP2", "rows": 2, "available": True, "sources": ["OpenF1 lap-derived classification"]},
        {"session": "FP3", "rows": 0, "available": False, "sources": []},
    ]


def test_pre_free_practice_race_frame_excludes_session_weather_and_normalizes_cutoff(monkeypatch):
    tables = {
        "races": pd.DataFrame([{
            "season": 2027, "round": 1, "race_name": "Future GP",
            "date": "2027-03-01", "circuit_id": "future",
        }]),
        "race_results": pd.DataFrame([{"season": 2026, "round": 1, "driver_id": "prior"}]),
        "qualifying": pd.DataFrame(), "practice": pd.DataFrame([{"season": 2027, "round": 1, "session": "FP1"}]),
        "sprints": pd.DataFrame(),
        "session_laps": pd.DataFrame([{"season": 2027, "round": 1, "session": "FP1"}]),
        "weather": pd.DataFrame([{"season": 2027, "round": 1, "session": "FP1", "air_temperature": 25}]),
        "race_weather_forecasts": pd.DataFrame(), "starting_grid": pd.DataFrame(),
        "race_control": pd.DataFrame(), "pit_stops": pd.DataFrame(), "openf1_drivers": pd.DataFrame(),
        "driver_roster": pd.DataFrame(),
    }
    captured = {}
    monkeypatch.setattr(api, "load_tables", lambda: tables)

    class NoForecast:
        def live_event_forecast(self, event):
            return None
    monkeypatch.setattr(api, "OpenMeteoClient", NoForecast)

    def fake_builder(*args, **kwargs):
        captured["_args"] = args
        captured.update(kwargs)
        return pd.DataFrame([{"driver_name": "Driver A"}])
    monkeypatch.setattr(api, "build_event_prediction_frame", fake_builder)

    result = api._event_frame(2027, 1, "race", "PRE_FP", weather_as_of_session="PRE_FP")

    assert not result.empty
    # practice_source is the fifth positional argument to build_event_prediction_frame.
    assert captured["_args"][4].empty
    assert captured["session_laps"].empty
    assert captured["weather_observations"].empty
    assert captured["weather_as_of_session"] == "FP1"


def test_practice_reconstruction_keeps_openf1_drivers_without_acronyms():
    from app.main import _practice_rows_for_event

    tables = {
        "practice": pd.DataFrame(),
        "session_laps": pd.DataFrame([
            {"season": 2025, "round": 8, "session": "FP1", "driver_code": pd.NA,
             "driver_number": 44, "driver_name": "Lewis Hamilton", "constructor_name": "Ferrari",
             "lap_time_seconds": 89.2, "data_source": "openf1"},
            {"season": 2025, "round": 8, "session": "FP1", "driver_code": pd.NA,
             "driver_number": 63, "driver_name": "George Russell", "constructor_name": "Mercedes",
             "lap_time_seconds": 89.5, "data_source": "openf1"},
        ]),
        "race_results": pd.DataFrame(), "openf1_drivers": pd.DataFrame(),
    }
    payload = _practice_rows_for_event(2025, 8, tables)
    assert len(payload["rows"]) == 2
    assert {row["driver_name"] for row in payload["rows"]} == {"Lewis Hamilton", "George Russell"}
