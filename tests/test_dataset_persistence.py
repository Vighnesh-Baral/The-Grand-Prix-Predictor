import pandas as pd

from src.data.build_dataset import _merge_existing_table


def test_merge_existing_table_preserves_old_seasons_and_replaces_refreshed_rows(tmp_path):
    path = tmp_path / "race_results.csv"
    pd.DataFrame([
        {"season": 2023, "round": 1, "driver_id": "a", "points": 25},
        {"season": 2024, "round": 1, "driver_id": "a", "points": 18},
    ]).to_csv(path, index=False)
    new_rows = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "a", "points": 25},
        {"season": 2025, "round": 1, "driver_id": "a", "points": 25},
    ])
    merged = _merge_existing_table("race_results", new_rows, tmp_path)
    assert set(merged["season"]) == {2023, 2024, 2025}
    assert merged.loc[merged["season"] == 2024, "points"].iloc[0] == 25
