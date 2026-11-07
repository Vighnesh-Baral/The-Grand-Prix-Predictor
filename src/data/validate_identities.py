"""Audit cross-source driver identity joins without forcing reserve-driver matches.

Run after migration:
    python -m src.data.validate_identities

The audit compares event-level canonical full-name keys. Unmatched practice
entries are reported for review; they are not automatically classified as bad
records because FP-only/reserve drivers may not appear in race classifications.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.data.identity import add_canonical_identities

EVENT_IDENTITY = ["season", "round", "driver_key"]


def _read_identity_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return add_canonical_identities(frame)


def _unique_event_identity(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not set(EVENT_IDENTITY).issubset(frame.columns):
        return pd.DataFrame(columns=EVENT_IDENTITY)
    result = frame.copy()
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["round"] = pd.to_numeric(result["round"], errors="coerce")
    return result.dropna(subset=EVENT_IDENTITY).drop_duplicates(EVENT_IDENTITY)[EVENT_IDENTITY]


def _source_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    source = frame.get("source", pd.Series("unspecified", index=frame.index)).fillna("unspecified").astype(str)
    return {str(key): int(value) for key, value in source.value_counts(dropna=False).items()}


def _practice_comparison(practice: pd.DataFrame, target: pd.DataFrame, target_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    p_unique = _unique_event_identity(practice)
    t_unique = _unique_event_identity(target)
    target_keys = set(map(tuple, t_unique[EVENT_IDENTITY].itertuples(index=False, name=None)))
    if p_unique.empty:
        unmatched = pd.DataFrame(columns=[*EVENT_IDENTITY, "match_status"])
        return {
            "target": target_name,
            "practice_unique_driver_events": 0,
            "matched_driver_events": 0,
            "unmatched_driver_events": 0,
            "match_rate": None,
        }, unmatched

    p_keys = list(map(tuple, p_unique[EVENT_IDENTITY].itertuples(index=False, name=None)))
    matched_mask = [key in target_keys for key in p_keys]
    p_unique["match_status"] = ["matched" if matched else "not_found_in_target_table" for matched in matched_mask]
    unmatched_keys = p_unique.loc[~pd.Series(matched_mask, index=p_unique.index), EVENT_IDENTITY]
    unmatched = practice.merge(unmatched_keys, on=EVENT_IDENTITY, how="inner") if not unmatched_keys.empty else pd.DataFrame(columns=practice.columns)
    if not unmatched.empty:
        unmatched["match_status"] = "not_found_in_target_table"

    matched_count = int(sum(matched_mask))
    total = len(p_unique)
    summary = {
        "target": target_name,
        "practice_unique_driver_events": int(total),
        "matched_driver_events": matched_count,
        "unmatched_driver_events": int(total - matched_count),
        "match_rate": round(matched_count / total, 6) if total else None,
    }
    return summary, unmatched


def audit_identity_tables(processed_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    practice = _read_identity_table(processed_dir / "practice_results.csv")
    race = _read_identity_table(processed_dir / "race_results.csv")
    qualifying = _read_identity_table(processed_dir / "qualifying_results.csv")
    sprint = _read_identity_table(processed_dir / "sprint_results.csv")

    tables = {
        "practice_results": practice,
        "race_results": race,
        "qualifying_results": qualifying,
        "sprint_results": sprint,
    }
    table_summary: dict[str, Any] = {}
    for name, frame in tables.items():
        statuses = frame.get("driver_identity_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
        table_summary[name] = {
            "rows": int(len(frame)),
            "rows_with_canonical_driver_key": int(frame.get("driver_key", pd.Series(dtype=object)).notna().sum()),
            "rows_without_canonical_driver_key": int(frame.get("driver_key", pd.Series(dtype=object)).isna().sum()),
            "driver_identity_status": {str(key): int(value) for key, value in statuses.items()},
        }

    if practice.empty:
        raise ValueError(f"No practice rows found in {processed_dir / 'practice_results.csv'}")

    race_summary, unmatched_race = _practice_comparison(practice, race, "race_results")
    quali_summary, unmatched_quali = _practice_comparison(practice, qualifying, "qualifying_results")

    # Count actual source-ID differences only after canonical event/name matching.
    id_mismatch_count = 0
    id_pairs = pd.DataFrame()
    join_cols = [*EVENT_IDENTITY]
    if not race.empty and {"driver_id", *join_cols}.issubset(practice.columns) and {"driver_id", *join_cols}.issubset(race.columns):
        practice_ids = practice.dropna(subset=join_cols).groupby(join_cols, as_index=False)["driver_id"].agg(
            practice_source_ids=lambda values: " ; ".join(sorted({str(v) for v in values if pd.notna(v)}))
        )
        race_ids = race.dropna(subset=join_cols).groupby(join_cols, as_index=False)["driver_id"].agg(
            race_source_ids=lambda values: " ; ".join(sorted({str(v) for v in values if pd.notna(v)}))
        )
        id_pairs = practice_ids.merge(race_ids, on=join_cols, how="inner")
        id_mismatch = id_pairs[
            id_pairs["practice_source_ids"].ne(id_pairs["race_source_ids"])
            & id_pairs["practice_source_ids"].ne("")
            & id_pairs["race_source_ids"].ne("")
        ].copy()
        id_mismatch_count = int(len(id_mismatch))
        id_pairs = id_mismatch

    # Different display spellings that collapse to the same accent/punctuation-
    # normalized key are visible for review, but not treated as a conflict.
    name_variant_frames = []
    for table_name, frame in tables.items():
        if frame.empty or not {"driver_key", "driver_name"}.issubset(frame.columns):
            continue
        variants = frame.dropna(subset=["driver_key", "driver_name"])[["driver_key", "driver_name"]].drop_duplicates()
        variants["table"] = table_name
        name_variant_frames.append(variants)
    name_variants = pd.concat(name_variant_frames, ignore_index=True) if name_variant_frames else pd.DataFrame(columns=["driver_key", "driver_name", "table"])
    if not name_variants.empty:
        variant_counts = name_variants.groupby("driver_key")["driver_name"].nunique()
        variant_keys = set(variant_counts[variant_counts.gt(1)].index)
        name_variants = name_variants[name_variants["driver_key"].isin(variant_keys)].sort_values(["driver_key", "table", "driver_name"])

    # Per-season match diagnostics make a provider-specific ID migration obvious.
    per_season: list[dict[str, Any]] = []
    p_unique = _unique_event_identity(practice)
    r_keys = set(map(tuple, _unique_event_identity(race)[EVENT_IDENTITY].itertuples(index=False, name=None)))
    q_keys = set(map(tuple, _unique_event_identity(qualifying)[EVENT_IDENTITY].itertuples(index=False, name=None)))
    for season, group in p_unique.groupby("season"):
        keys = list(map(tuple, group[EVENT_IDENTITY].itertuples(index=False, name=None)))
        per_season.append({
            "season": int(season),
            "practice_driver_events": len(keys),
            "matched_race": int(sum(key in r_keys for key in keys)),
            "unmatched_race": int(sum(key not in r_keys for key in keys)),
            "matched_qualifying": int(sum(key in q_keys for key in keys)),
            "unmatched_qualifying": int(sum(key not in q_keys for key in keys)),
        })

    summary: dict[str, Any] = {
        "tables": table_summary,
        "practice_to_race": race_summary,
        "practice_to_qualifying": quali_summary,
        "practice_source_id_differs_from_race_source_id": id_mismatch_count,
        "unmatched_practice_source_rows": _source_counts(unmatched_race),
        "per_season": per_season,
        "interpretation": (
            "Unmatched practice drivers may be reserve or FP-only entrants and are not automatically errors. "
            "Canonical keys use normalized full names; source driver IDs remain unchanged and are not used "
            "as cross-provider identity keys. Name variants are surfaced for review; fuzzy matching is not applied."
        ),
    }
    return summary, {
        "unmatched_practice_vs_race.csv": unmatched_race,
        "unmatched_practice_vs_qualifying.csv": unmatched_quali,
        "provider_id_mismatches.csv": id_pairs,
        "normalized_name_variants.csv": name_variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "identity_audit")
    args = parser.parse_args()

    summary, reports = audit_identity_tables(args.processed_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "identity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for filename, frame in reports.items():
        frame.to_csv(args.output_dir / filename, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetailed reports: {args.output_dir}")
    print("Note: unmatched practice entries are review items, not automatic data errors.")


if __name__ == "__main__":
    main()
