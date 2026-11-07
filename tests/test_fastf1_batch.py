import pandas as pd
import pytest

from src.data.collect_fastf1_batch import event_list


def test_event_list_filters_range_deduplicates_and_sorts(tmp_path):
    calendar = tmp_path / "races.csv"
    pd.DataFrame([
        {"season": 2025, "round": 2, "race_name": "Race B"},
        {"season": 2024, "round": 1, "race_name": "Race A"},
        {"season": 2025, "round": 1, "race_name": "Race C"},
        {"season": 2025, "round": 2, "race_name": "Duplicate"},
        {"season": 2026, "round": 1, "race_name": "Outside range"},
        {"season": 2025, "round": None, "race_name": "No round"},
    ]).to_csv(calendar, index=False)

    assert event_list(calendar, 2024, 2025) == [
        (2024, 1, "Race A"),
        (2025, 1, "Race C"),
        (2025, 2, "Race B"),
    ]


def test_event_list_reports_missing_calendar(tmp_path):
    with pytest.raises(FileNotFoundError, match="Race calendar not found"):
        event_list(tmp_path / "missing.csv", 2024, 2025)


def test_event_list_requires_season_and_round_columns(tmp_path):
    calendar = tmp_path / "bad.csv"
    pd.DataFrame([{"year": 2025, "round": 1}]).to_csv(calendar, index=False)
    with pytest.raises(ValueError, match="round, season"):
        event_list(calendar, 2024, 2025)
