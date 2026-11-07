from types import SimpleNamespace
import sys

import pandas as pd

from src.data.tif1_client import load_session, normalize_session_code, session_driver_metadata


def test_session_codes_map_to_tif1_session_names():
    assert normalize_session_code("FP1") == "Practice 1"
    assert normalize_session_code("fp2") == "Practice 2"
    assert normalize_session_code("FP3") == "Practice 3"
    assert normalize_session_code("Q") == "Qualifying"


def test_tif1_loader_uses_public_session_api_and_materializes_laps(monkeypatch):
    calls = []
    fake_session = SimpleNamespace(laps=pd.DataFrame([{"Driver": "NOR"}]))

    def fake_get_session(year, event, session_name):
        calls.append((year, event, session_name))
        return fake_session

    monkeypatch.setitem(sys.modules, "tif1", SimpleNamespace(get_session=fake_get_session))
    loaded = load_session(2025, "Australian Grand Prix", "FP1")

    assert loaded is fake_session
    assert calls == [(2025, "Australian Grand Prix", "Practice 1")]


def test_driver_metadata_normalizes_tif1_driver_fields():
    session = SimpleNamespace(
        drivers_df=pd.DataFrame([{
            "Abbreviation": "NOR", "FirstName": "Lando", "LastName": "Norris",
            "TeamName": "McLaren",
        }]),
        results=pd.DataFrame(),
    )
    metadata = session_driver_metadata(session)
    row = metadata.iloc[0]
    assert row["FullName"] == "Lando Norris"
    assert row["Team"] == "McLaren"


def test_historical_session_helper_does_not_call_fastf1(monkeypatch):
    from src.data import fastf1_client
    calls = []
    sentinel = object()
    monkeypatch.setattr(fastf1_client, "load_tif1_session", lambda year, event, code: calls.append((year, event, code)) or sentinel)

    result = fastf1_client.load_session(2024, 5, "FP2")
    assert result is sentinel
    assert calls == [(2024, 5, "FP2")]


def test_api_counts_tif1_as_fastf1_derived_but_reports_it_separately():
    import app.main as api
    frame = pd.DataFrame({"data_source": ["tif1_public_archive", "fastf1", "openf1"]})
    assert api._fastf1_derived_mask(frame).tolist() == [True, True, False]
    assert api._tif1_source_mask(frame).tolist() == [True, False, False]
