import numpy as np
import pandas as pd

from src.data.collect_fastf1_practice import _merge_practice, build_classification_rows


def test_fastf1_classification_filters_string_false_lap_flags():
    laps = pd.DataFrame([
        {"Driver": "LEC", "LapTime": pd.Timedelta(seconds=89.2), "IsAccurate": "True", "Deleted": "False", "Team": "Ferrari"},
        {"Driver": "LEC", "LapTime": pd.Timedelta(seconds=88.0), "IsAccurate": "False", "Deleted": "False", "Team": "Ferrari"},
        {"Driver": "HAM", "LapTime": pd.Timedelta(seconds=90.0), "IsAccurate": "True", "Deleted": "True", "Team": "Ferrari"},
        {"Driver": "RUS", "LapTime": pd.Timedelta(seconds=90.3), "IsAccurate": "True", "Deleted": "False", "Team": "Mercedes"},
    ])
    result = build_classification_rows(2025, 8, "FP1", laps)
    assert result["driver_code"].tolist() == ["LEC", "RUS"]
    assert result.iloc[0]["time_seconds"] == 89.2


def test_practice_merge_prefers_fastf1_but_does_not_collapse_missing_codes():
    existing = pd.DataFrame([
        {"season": 2025, "round": 8, "session": "FP1", "driver_code": np.nan, "driver_id": np.nan, "driver_name": "Charles Leclerc", "time_seconds": 90.0, "source": "Jolpica-F1"},
        {"season": 2025, "round": 8, "session": "FP1", "driver_code": np.nan, "driver_id": np.nan, "driver_name": "Lewis Hamilton", "time_seconds": 91.0, "source": "Jolpica-F1"},
    ])
    incoming = pd.DataFrame([
        {"season": 2025, "round": 8, "session": "FP1", "driver_code": "LEC", "driver_name": "Charles Leclerc", "time_seconds": 89.5, "source": "FastF1"},
    ])
    result = _merge_practice(existing, incoming)
    assert len(result) == 2
    assert result.loc[result["driver_name"].eq("Charles Leclerc"), "time_seconds"].iloc[0] == 89.5
    assert "Lewis Hamilton" in result["driver_name"].tolist()


def test_jolpica_official_practice_classification_outranks_fastf1_lap_reconstruction():
    from app.main import _practice_rows_for_event

    tables = {
        "practice": pd.DataFrame([{
            "season": 2025, "round": 8, "session": "FP1", "position": 1,
            "driver_id": "leclerc", "driver_code": "LEC", "driver_name": "Charles Leclerc",
            "constructor_name": "Ferrari", "time_seconds": 89.2, "laps": 20,
            "source": "Jolpica-F1 classification",
        }]),
        "session_laps": pd.DataFrame([{
            "season": 2025, "round": 8, "session": "FP1", "driver_code": "LEC",
            "driver_name": "Charles Leclerc", "constructor_name": "Ferrari",
            "lap_time_seconds": 88.9, "data_source": "fastf1",
        }]),
        "race_results": pd.DataFrame(), "openf1_drivers": pd.DataFrame(),
    }
    payload = _practice_rows_for_event(2025, 8, tables)
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["source"] == "Jolpica-F1 classification"
    assert payload["rows"][0]["time_seconds"] == 89.2


def test_practice_rows_accept_tif1_numeric_lap_time_seconds():
    laps = pd.DataFrame([
        {"Driver": "NOR", "LapTimeSeconds": 88.75, "Team": "McLaren", "IsAccurate": True},
        {"Driver": "PIA", "LapTimeSeconds": 89.10, "Team": "McLaren", "IsAccurate": True},
    ])
    result = build_classification_rows(2025, 1, "FP1", laps)
    assert result["driver_code"].tolist() == ["NOR", "PIA"]
    assert result["time_seconds"].tolist() == [88.75, 89.10]
    assert result["source"].str.contains("TIF1").all()


def test_tif1_practice_rows_get_internal_keys_without_fabricating_upstream_ids():
    laps = pd.DataFrame([
        {"Driver": "KUB", "LapTimeSeconds": 90.1, "Team": "Williams", "IsAccurate": True},
        {"Driver": "NOR", "LapTimeSeconds": 90.3, "Team": "McLaren", "IsAccurate": True},
    ])
    metadata = pd.DataFrame([
        {"Abbreviation": "KUB", "FullName": "Robert Kubica", "TeamName": "Williams"},
        {"Abbreviation": "NOR", "FullName": "Lando Norris", "TeamName": "McLaren"},
    ])
    result = build_classification_rows(2018, 1, "FP1", laps, metadata)
    assert result["driver_key"].tolist() == ["name:robertkubica", "name:landonorris"]
    assert result["driver_id"].isna().all()
    assert result["driver_identity_status"].eq("name_key_only").all()
    assert result["constructor_key"].tolist() == ["season:2018:team:williams", "season:2018:team:mclaren"]


def test_canonical_identity_does_not_trust_code_only_or_merge_constructor_seasons():
    from src.data.identity import add_canonical_identities

    frame = pd.DataFrame([
        {"season": 2025, "driver_code": "SHW", "driver_name": "SHW", "constructor_name": "AlphaTauri"},
        {"season": 2026, "driver_code": "LAW", "driver_name": "Liam Lawson", "constructor_name": "Racing Bulls"},
        {"season": 2025, "driver_code": "LAW", "driver_name": "Liam Lawson", "constructor_name": "AlphaTauri"},
    ])
    result = add_canonical_identities(frame)
    assert pd.isna(result.loc[0, "driver_key"])
    assert result.loc[0, "driver_identity_status"] == "unresolved"
    assert result.loc[1, "driver_key"] == result.loc[2, "driver_key"] == "name:liamlawson"
    assert result.loc[1, "constructor_key"] == "season:2026:team:racingbulls"
    assert result.loc[2, "constructor_key"] == "season:2025:team:racingbulls"


def test_tif1_code_name_conflict_does_not_attach_wrong_jolpica_driver_id():
    laps = pd.DataFrame([
        {"Driver": "BRO", "LapTimeSeconds": 90.1, "Team": "Ferrari", "IsAccurate": True},
    ])
    metadata = pd.DataFrame([
        {"Abbreviation": "BRO", "FullName": "Luke Browning", "TeamName": "Williams"},
    ])
    entries = pd.DataFrame([
        {"driver_id": "leclerc", "driver_code": "LEC", "driver_name": "Charles Leclerc", "constructor_id": "ferrari", "constructor_name": "Ferrari"},
    ])
    result = build_classification_rows(2025, 1, "FP1", laps, metadata, entries)
    assert result.loc[0, "driver_name"] == "Luke Browning"
    assert pd.isna(result.loc[0, "driver_id"])
    assert pd.isna(result.loc[0, "constructor_id"])


def test_practice_merge_matches_name_when_only_one_source_has_upstream_id():
    existing = pd.DataFrame([{
        "season": 2025, "round": 8, "session": "FP1", "driver_id": "norris",
        "driver_code": "NOR", "driver_name": "Lando Norris", "time_seconds": 90.0,
        "source": "Jolpica-F1",
    }])
    incoming = pd.DataFrame([{
        "season": 2025, "round": 8, "session": "FP1", "driver_id": pd.NA,
        "driver_code": "NOR", "driver_name": "Lando Norris", "time_seconds": 89.5,
        "source": "TIF1 public archive (FastF1-derived lap classification)",
    }])
    result = _merge_practice(existing, incoming)
    assert len(result) == 1
    assert result.iloc[0]["driver_id"] == "norris"
    assert result.iloc[0]["time_seconds"] == 89.5


def test_backfill_practice_identity_maps_only_unique_same_event_full_name():
    from src.data.backfill_practice_identity import enrich_practice_identities

    practice = pd.DataFrame([
        {"season": 2025, "round": 1, "driver_name": "Lando Norris", "driver_code": "NOR", "constructor_name": "McLaren"},
        {"season": 2025, "round": 1, "driver_name": "Reserve Driver", "driver_code": "RES", "constructor_name": "McLaren"},
    ])
    races = pd.DataFrame([
        {"season": 2025, "round": 1, "driver_name": "Lando Norris", "driver_id": "norris", "constructor_id": "mclaren"},
        {"season": 2025, "round": 1, "driver_name": "Oscar Piastri", "driver_id": "piastri", "constructor_id": "mclaren"},
    ])
    result = enrich_practice_identities(practice, races)
    assert len(result) == 2
    assert result.loc[0, "driver_id"] == "norris"
    assert result.loc[0, "constructor_id"] == "mclaren"
    assert pd.isna(result.loc[1, "driver_id"])
    assert result.loc[1, "driver_identity_status"] == "name_key_only"


def test_placeholder_driver_metadata_is_not_treated_as_a_real_name():
    from src.data.identity import add_canonical_identities, normalize_identity_name

    assert normalize_identity_name("None None") == ""
    frame = pd.DataFrame([{
        "season": 2026, "driver_code": "HER", "driver_name": "None None",
        "constructor_name": None,
    }])
    result = add_canonical_identities(frame)
    assert pd.isna(result.loc[0, "driver_key"])
    assert result.loc[0, "driver_identity_status"] == "unresolved"
    assert pd.isna(result.loc[0, "constructor_key"])


def test_season_code_history_resolves_only_unambiguous_missing_metadata():
    from src.data.identity import add_canonical_identities, resolve_season_code_metadata

    frame = pd.DataFrame([
        {"season": 2026, "driver_code": "HER", "driver_name": "None None", "constructor_name": None},
        {"season": 2026, "driver_code": "HER", "driver_name": "HER", "constructor_name": None},
        {"season": 2026, "driver_code": "HER", "driver_name": "Colton Herta", "constructor_name": "Cadillac"},
        {"season": 2026, "driver_code": "HER", "driver_name": "Colton Herta", "constructor_name": "Cadillac"},
        {"season": 2026, "driver_code": "BEA", "driver_name": "Oliver Bearman", "constructor_name": "Haas"},
        {"season": 2026, "driver_code": "BEA", "driver_name": "Oliver Bearman", "constructor_name": "Ferrari"},
    ])
    resolved = resolve_season_code_metadata(frame)
    result = add_canonical_identities(resolved)
    assert result.loc[0, "driver_name"] == "Colton Herta"
    assert result.loc[0, "driver_key"] == "name:coltonherta"
    assert result.loc[0, "constructor_name"] == "Cadillac"
    assert result.loc[1, "driver_name"] == "Colton Herta"
    assert result.loc[1, "driver_key"] == "name:coltonherta"
    # Same driver code/name across multiple teams: don't guess the missing team.
    assert result.loc[4, "driver_key"] == result.loc[5, "driver_key"] == "name:oliverbearman"
    assert result.loc[4, "constructor_key"] != result.loc[5, "constructor_key"]


def test_canonical_driver_key_ignores_provider_specific_ids_and_normalizes_accents():
    from src.data.identity import add_canonical_identities

    frame = pd.DataFrame([
        {"season": 2025, "round": 1, "driver_id": "driver_VkTWWTC5", "driver_code": "VER", "driver_name": "Max Verstappen", "constructor_id": "opaque_red_bull", "constructor_name": "Red Bull Racing"},
        {"season": 2025, "round": 1, "driver_id": "verstappen", "driver_code": "VER", "driver_name": "Max Verstappen", "constructor_id": "red_bull", "constructor_name": "Red Bull"},
        {"season": 2025, "round": 2, "driver_id": "hulkenberg", "driver_code": "HUL", "driver_name": "Nico Hülkenberg", "constructor_id": "haas", "constructor_name": "Haas"},
        {"season": 2025, "round": 2, "driver_id": "driver_hulkenberg", "driver_code": "HUL", "driver_name": "Nico Hulkenberg", "constructor_id": "haas_2", "constructor_name": "Haas"},
    ])
    result = add_canonical_identities(frame)

    assert result.loc[0, "driver_key"] == result.loc[1, "driver_key"] == "name:maxverstappen"
    assert result.loc[2, "driver_key"] == result.loc[3, "driver_key"] == "name:nicohulkenberg"
    assert result.loc[0, "driver_id"] == "driver_VkTWWTC5"  # provider ID is not overwritten
    assert result.loc[1, "driver_id"] == "verstappen"
    assert result.loc[0, "driver_identity_status"] == "name_key_with_source_id"
    assert result.loc[0, "constructor_key"] == result.loc[1, "constructor_key"]


def test_identity_key_does_not_fabricate_a_cross_source_key_from_id_only():
    from src.data.identity import add_canonical_identities

    result = add_canonical_identities(pd.DataFrame([{
        "season": 2025, "driver_id": "driver_VkTWWTC5", "driver_code": "VER",
        "driver_name": "VER", "constructor_id": "opaque_team",
    }]))
    assert pd.isna(result.loc[0, "driver_key"])
    assert result.loc[0, "driver_identity_status"] == "unresolved_id_only"
    assert pd.isna(result.loc[0, "constructor_key"])


def test_backfill_preserves_opaque_provider_id_while_adding_cross_source_key():
    from src.data.backfill_practice_identity import enrich_practice_identities

    practice = pd.DataFrame([
        {"season": 2025, "round": 1, "session": "FP1", "driver_id": "driver_VkTWWTC5", "driver_name": "Max Verstappen", "driver_code": "VER", "constructor_id": "opaque_red_bull", "constructor_name": "Red Bull Racing"},
        {"season": 2025, "round": 1, "session": "FP1", "driver_id": None, "driver_name": "Luke Browning", "driver_code": "BRO", "constructor_id": None, "constructor_name": "Williams", "source": "Jolpica alpha"},
    ])
    races = pd.DataFrame([
        {"season": 2025, "round": 1, "driver_id": "verstappen", "driver_name": "Max Verstappen", "constructor_id": "red_bull", "constructor_name": "Red Bull Racing"},
    ])
    result = enrich_practice_identities(practice, races)

    assert len(result) == 2
    assert result.loc[0, "driver_id"] == "driver_VkTWWTC5"
    assert result.loc[0, "driver_key"] == "name:maxverstappen"
    assert pd.isna(result.loc[1, "driver_id"])
    assert result.loc[1, "driver_key"] == "name:lukebrowning"
    assert result.loc[1, "driver_identity_status"] == "name_key_only"


def test_practice_feature_join_uses_same_event_name_mapping_not_global_driver_id():
    from src.features.build_features import _add_practice_aggregates

    races = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "driver_old", "driver_name": "Max Verstappen", "driver_code": "VER"},
        {"season": 2025, "round": 1, "driver_id": "driver_new", "driver_name": "Max Verstappen", "driver_code": "VER"},
    ])
    practice = pd.DataFrame([
        {"season": 2024, "round": 1, "session": "FP1", "driver_id": "opaque_2024", "driver_name": "Max Verstappen", "driver_code": "VER", "position": 1, "time_seconds": 90.0, "laps": 20},
        {"season": 2025, "round": 1, "session": "FP1", "driver_id": "opaque_2025", "driver_name": "Max Verstappen", "driver_code": "VER", "position": 2, "time_seconds": 91.0, "laps": 21},
        {"season": 2025, "round": 1, "session": "FP1", "driver_id": "reserve_id", "driver_name": "Luke Browning", "driver_code": "BRO", "position": 1, "time_seconds": 89.0, "laps": 10},
    ])

    result = _add_practice_aggregates(races, practice, "FP1").set_index(["season", "driver_id"])
    assert result.loc[(2024, "driver_old"), "fp_best_time_seconds"] == 90.0
    assert result.loc[(2025, "driver_new"), "fp_best_time_seconds"] == 91.0
    assert pd.isna(result.loc[(2024, "driver_old"), "fp_total_laps"]) is False


def test_practice_table_merge_does_not_collapse_distinct_rows_with_missing_external_ids(tmp_path):
    from src.data.build_dataset import _merge_existing_table

    practice = pd.DataFrame([
        {"season": 2025, "round": 1, "session": "FP1", "driver_id": None, "driver_name": "Max Verstappen", "driver_code": "VER", "time_seconds": 90.0},
        {"season": 2025, "round": 1, "session": "FP1", "driver_id": None, "driver_name": "Lewis Hamilton", "driver_code": "HAM", "time_seconds": 90.5},
    ])
    result = _merge_existing_table("practice_results", practice, tmp_path)
    assert len(result) == 2
    assert set(result["driver_key"]) == {"name:maxverstappen", "name:lewishamilton"}


def test_ergast_null_token_is_not_a_driver_or_constructor_identity():
    from src.data.identity import add_canonical_identities, normalize_identity_name

    assert normalize_identity_name(r"\N") == ""
    result = add_canonical_identities(pd.DataFrame([{
        "season": 2025, "driver_id": r"\N", "driver_code": r"\N", "driver_name": r"\N",
        "constructor_id": r"\N", "constructor_name": r"\N",
    }]))
    assert pd.isna(result.loc[0, "driver_key"])
    assert pd.isna(result.loc[0, "constructor_key"])
    assert result.loc[0, "driver_identity_status"] == "unresolved"
