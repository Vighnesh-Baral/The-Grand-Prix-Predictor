"""Migrate processed result tables to provider-independent identity keys.

Run a safe preview:
    python -m src.data.backfill_practice_identity --dry-run

Apply the migration (original external IDs are preserved):
    python -m src.data.backfill_practice_identity

The command only touches processed CSVs and stores one pre-v0.9.3 backup per
file before the first write.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.data.identity import (
    add_canonical_identities,
    normalize_identity_name,
    resolve_season_code_metadata,
)


def _present(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().casefold() not in {"", "nan", "none", "null", "<na>", "n/a", "na", r"\n"}


def enrich_practice_identities(practice: pd.DataFrame, races: pd.DataFrame | None) -> pd.DataFrame:
    """Add canonical keys and fill only missing IDs from unique same-event matches.

    Provider IDs that are already present (including opaque IDs) are retained as
    supplied. A same-event, normalized full-name match can populate a missing
    Ergast/Jolpica ID, but never overwrites a non-empty source ID.
    """
    result = practice.copy()
    for col in ("season", "round", "driver_id", "constructor_id"):
        if col not in result:
            result[col] = pd.NA
    for col in ("season", "round"):
        result[col] = pd.to_numeric(result[col], errors="coerce")

    # Recover placeholders only from an unambiguous season+code history.
    result = resolve_season_code_metadata(result)
    if races is None or races.empty:
        return add_canonical_identities(result)

    required = {"season", "round", "driver_name", "driver_id"}
    if not required.issubset(races.columns):
        return add_canonical_identities(result)

    race = races.copy()
    for col in ("season", "round"):
        race[col] = pd.to_numeric(race[col], errors="coerce")
    race["_name_key"] = race["driver_name"].map(normalize_identity_name)
    if "driver_code" in race:
        race_code = race["driver_code"].map(normalize_identity_name)
    else:
        race_code = pd.Series("", index=race.index, dtype=object)
    valid_race_name = race["_name_key"].ne("") & race["_name_key"].ne(race_code)
    race = race[valid_race_name].copy()
    race = race[race["driver_id"].map(_present)].copy()

    if race.empty:
        return add_canonical_identities(result)

    group_keys = ["season", "round", "_name_key"]
    id_counts = race.groupby(group_keys)["driver_id"].nunique(dropna=True)
    id_lookup: dict[tuple[object, object, str], object] = {}
    for key, group in race.groupby(group_keys, dropna=False):
        ids = group.loc[group["driver_id"].map(_present), "driver_id"].astype(str).unique()
        if len(ids) == 1 and id_counts.get(key, 0) == 1:
            id_lookup[key] = ids[0]

    constructor_lookup: dict[tuple[object, object, str], object] = {}
    if "constructor_id" in race.columns:
        for key, group in race.groupby(group_keys, dropna=False):
            ids = group.loc[group["constructor_id"].map(_present), "constructor_id"].astype(str).unique()
            if len(ids) == 1:
                constructor_lookup[key] = ids[0]

    practice_names = result["driver_name"].map(normalize_identity_name)
    if "driver_code" in result:
        practice_codes = result["driver_code"].map(normalize_identity_name)
    else:
        practice_codes = pd.Series("", index=result.index, dtype=object)
    valid_practice_name = practice_names.ne("") & practice_names.ne(practice_codes)

    for idx in result.index[valid_practice_name]:
        key = (result.at[idx, "season"], result.at[idx, "round"], practice_names.at[idx])
        # Driver ID and constructor ID mappings are checked independently. An
        # ambiguous constructor mapping must not block a safe driver mapping.
        if not _present(result.at[idx, "driver_id"]) and key in id_lookup:
            result.at[idx, "driver_id"] = id_lookup[key]
        if (
            "constructor_id" in result.columns
            and not _present(result.at[idx, "constructor_id"])
            and key in constructor_lookup
        ):
            result.at[idx, "constructor_id"] = constructor_lookup[key]

    return add_canonical_identities(result)


def _atomic_csv_write(frame: pd.DataFrame, path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_name(f"{path.stem}.pre-v0.9.3.bak{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def migrate_processed_tables(processed_dir: Path, *, dry_run: bool = False) -> dict[str, dict[str, object]]:
    """Add identity fields to practice, race, qualifying and sprint result CSVs."""
    practice_path = processed_dir / "practice_results.csv"
    if not practice_path.exists():
        raise FileNotFoundError(f"Practice data not found: {practice_path}")

    race_path = processed_dir / "race_results.csv"
    races = pd.read_csv(race_path) if race_path.exists() else pd.DataFrame()
    report: dict[str, dict[str, object]] = {}

    practice = pd.read_csv(practice_path)
    enriched_practice = enrich_practice_identities(practice, races)
    _atomic_csv_write(enriched_practice, practice_path, dry_run=dry_run)
    report["practice_results"] = {
        "rows": len(enriched_practice),
        "driver_status": enriched_practice["driver_identity_status"].value_counts(dropna=False).to_dict(),
        "constructor_status": enriched_practice["constructor_identity_status"].value_counts(dropna=False).to_dict(),
    }

    for table_name in ("race_results", "qualifying_results", "sprint_results"):
        path = processed_dir / f"{table_name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        enriched = add_canonical_identities(frame)
        _atomic_csv_write(enriched, path, dry_run=dry_run)
        report[table_name] = {
            "rows": len(enriched),
            "driver_status": enriched["driver_identity_status"].value_counts(dropna=False).to_dict(),
            "constructor_status": enriched["constructor_identity_status"].value_counts(dropna=False).to_dict(),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview row/status counts without changing CSVs.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR, help="Processed data directory (defaults to project data/processed).")
    args = parser.parse_args()

    report = migrate_processed_tables(args.processed_dir, dry_run=args.dry_run)
    print("Identity migration preview" if args.dry_run else "Identity migration complete")
    for table, details in report.items():
        print(f"\n{table}: {details['rows']:,} rows")
        print(f"  driver identity: {details['driver_status']}")
        print(f"  constructor identity: {details['constructor_status']}")
    if args.dry_run:
        print("\nNo files changed. Re-run without --dry-run to apply; a pre-v0.9.3 backup is created before each first write.")
    else:
        print("\nRaw provider IDs were preserved. Only processed CSV identity metadata was updated.")


if __name__ == "__main__":
    main()
