from pathlib import Path

import pandas as pd

from src.data.backfill_practice_identity import migrate_processed_tables
from src.data.validate_identities import audit_identity_tables


def _write_fixture_tables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"season": 2025, "round": 1, "session": "FP1", "driver_id": "driver_VkTWWTC5", "driver_name": "Max Verstappen", "driver_code": "VER", "constructor_id": "opaque_red_bull", "constructor_name": "Red Bull Racing", "source": "Jolpica alpha"},
        {"season": 2025, "round": 1, "session": "FP2", "driver_id": "driver_VkTWWTC5", "driver_name": "Max Verstappen", "driver_code": "VER", "constructor_id": "opaque_red_bull", "constructor_name": "Red Bull Racing", "source": "Jolpica alpha"},
        {"season": 2025, "round": 1, "session": "FP1", "driver_id": "driver_HUL", "driver_name": "Nico Hulkenberg", "driver_code": "HUL", "constructor_id": "opaque_kick", "constructor_name": "Kick Sauber", "source": "Jolpica alpha"},
        {"season": 2025, "round": 1, "session": "FP1", "driver_id": None, "driver_name": "Luke Browning", "driver_code": "BRO", "constructor_id": None, "constructor_name": "Williams", "source": "Jolpica alpha"},
    ]).to_csv(root / "practice_results.csv", index=False)
    pd.DataFrame([
        {"season": 2025, "round": 1, "driver_id": "verstappen", "driver_name": "Max Verstappen", "driver_code": "VER", "constructor_id": "red_bull", "constructor_name": "Red Bull Racing"},
        {"season": 2025, "round": 1, "driver_id": "hulkenberg", "driver_name": "Nico Hülkenberg", "driver_code": "HUL", "constructor_id": "sauber", "constructor_name": "Kick Sauber"},
    ]).to_csv(root / "race_results.csv", index=False)
    pd.DataFrame([
        {"season": 2025, "round": 1, "driver_id": "verstappen", "driver_name": "Max Verstappen", "driver_code": "VER", "constructor_id": "red_bull", "constructor_name": "Red Bull Racing"},
        {"season": 2025, "round": 1, "driver_id": "hulkenberg", "driver_name": "Nico Hülkenberg", "driver_code": "HUL", "constructor_id": "sauber", "constructor_name": "Kick Sauber"},
    ]).to_csv(root / "qualifying_results.csv", index=False)


def test_migration_preserves_source_ids_and_creates_backup(tmp_path):
    _write_fixture_tables(tmp_path)
    before = pd.read_csv(tmp_path / "practice_results.csv")

    preview = migrate_processed_tables(tmp_path, dry_run=True)
    assert preview["practice_results"]["rows"] == 4
    assert "driver_key" not in pd.read_csv(tmp_path / "practice_results.csv").columns

    report = migrate_processed_tables(tmp_path, dry_run=False)
    migrated = pd.read_csv(tmp_path / "practice_results.csv")
    assert report["race_results"]["rows"] == 2
    assert len(migrated) == len(before)
    assert migrated.loc[0, "driver_id"] == "driver_VkTWWTC5"
    assert migrated.loc[0, "driver_key"] == "name:maxverstappen"
    assert (tmp_path / "practice_results.pre-v0.9.3.bak.csv").exists()
    assert (tmp_path / "race_results.pre-v0.9.3.bak.csv").exists()


def test_audit_matches_opaque_ids_by_canonical_name_and_keeps_fp_only_driver_unmatched(tmp_path):
    _write_fixture_tables(tmp_path)
    migrate_processed_tables(tmp_path)

    summary, reports = audit_identity_tables(tmp_path)
    race = summary["practice_to_race"]
    qualifying = summary["practice_to_qualifying"]

    # Three unique practice driver-event identities: Max, Nico, and FP-only Luke.
    assert race["practice_unique_driver_events"] == 3
    assert race["matched_driver_events"] == 2
    assert race["unmatched_driver_events"] == 1
    assert qualifying["matched_driver_events"] == 2

    unmatched = reports["unmatched_practice_vs_race.csv"]
    assert set(unmatched["driver_name"]) == {"Luke Browning"}
    assert unmatched["match_status"].eq("not_found_in_target_table").all()
    assert summary["practice_source_id_differs_from_race_source_id"] == 2

    variants = reports["normalized_name_variants.csv"]
    assert "name:nicohulkenberg" in set(variants["driver_key"])
