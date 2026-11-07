"""Download and persist historical F1 data from 2018 onward."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_LAST_SEASON, FIRST_SEASON, PROCESSED_DIR, RAW_DIR
from src.data.jolpica_client import JolpicaClient
from src.data.normalize import normalize_practice, normalize_qualifying, normalize_race_results, normalize_schedule, normalize_sprints
from src.data.identity import add_canonical_identities

LOGGER = logging.getLogger("f1_data_ingest")


def _merge_existing_table(key: str, new_frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Merge new rows into a local table without deleting seasons that failed to refresh."""
    path = data_dir / f"{key}.csv"
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            LOGGER.warning("Could not read existing %s table %s: %s", key, path, exc)
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if existing.empty and new_frame.empty:
        return existing
    combined = pd.concat([existing, new_frame], ignore_index=True, sort=False)
    if key == "practice_results":
        # Practice sources do not share one reliable external-ID namespace.
        # Deduplicate on the canonical full-name key, not driver_id: a provider
        # may assign opaque IDs, and some legitimate TIF1 rows have no ID at all.
        combined = add_canonical_identities(combined)
        event_keys = ["season", "round", "session", "driver_key"]
        if set(event_keys).issubset(combined.columns):
            identified = combined[combined["driver_key"].notna()].drop_duplicates(event_keys, keep="last")
            unresolved = combined[combined["driver_key"].isna()].drop_duplicates()
            combined = pd.concat([identified, unresolved], ignore_index=True, sort=False)
        else:
            combined = combined.drop_duplicates()
    else:
        key_columns = {
            "races": ["season", "round"],
            "race_results": ["season", "round", "driver_id"],
            "qualifying_results": ["season", "round", "driver_id"],
            "sprint_results": ["season", "round", "driver_id"],
        }.get(key, [])
        if key_columns and set(key_columns).issubset(combined.columns):
            combined = combined.drop_duplicates(subset=key_columns, keep="last")
        else:
            combined = combined.drop_duplicates()
    return combined.reset_index(drop=True)


def _write_table_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def build_dataset(start_year: int, end_year: int, include_practice: bool = False, refresh: bool = False) -> dict[str, pd.DataFrame]:
    if start_year < FIRST_SEASON:
        raise ValueError(f"FastF1-era project data starts in {FIRST_SEASON}; choose a start year >= {FIRST_SEASON}.")
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    client = JolpicaClient()
    collected: dict[str, list[pd.DataFrame]] = {
        "races": [], "race_results": [], "qualifying_results": [], "sprint_results": [], "practice_results": []
    }
    failed_seasons: list[int] = []
    for season in range(start_year, end_year + 1):
        LOGGER.info("Downloading %s season", season)
        try:
            schedule = client.get_schedule(season, refresh=refresh)
            race_results = client.get_race_results(season, refresh=refresh)
            qualifying = client.get_qualifying_results(season, refresh=refresh)
            sprints = client.get_sprint_results(season, refresh=refresh)
        except Exception as exc:
            failed_seasons.append(season)
            LOGGER.exception("Could not collect stable historical records for %s: %s", season, exc)
            continue

        for key, frame in (
            ("races", normalize_schedule(schedule)),
            ("race_results", normalize_race_results(race_results)),
            ("qualifying_results", normalize_qualifying(qualifying)),
            ("sprint_results", normalize_sprints(sprints)),
        ):
            if not frame.empty:
                collected[key].append(frame)

        if include_practice:
            try:
                practice_rows = client.get_practice_results(season, refresh=refresh)
                practice = normalize_practice(practice_rows)
                if not practice.empty:
                    collected["practice_results"].append(practice)
                LOGGER.info("Collected %s practice classification rows for %s", len(practice), season)
            except Exception as exc:
                LOGGER.warning("Practice data skipped for %s: %s", season, exc)

    result: dict[str, pd.DataFrame] = {}
    for key, frames in collected.items():
        new_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not new_frame.empty:
            new_frame = new_frame.drop_duplicates().reset_index(drop=True)
        # Merge rather than overwrite: a transient failure in one requested season
        # must not erase previously downloaded seasons or a prior practice extract.
        merged = _merge_existing_table(key, new_frame, PROCESSED_DIR)
        if key in {"race_results", "qualifying_results", "sprint_results", "practice_results"} and not merged.empty:
            # Keep provider IDs as received; project-owned keys are a separate
            # identity layer used for auditable cross-source comparisons.
            merged = add_canonical_identities(merged)
        result[key] = merged
        if not merged.empty:
            _write_table_atomic(merged, RAW_DIR / f"{key}.csv")
            _write_table_atomic(merged, PROCESSED_DIR / f"{key}.csv")
            LOGGER.info("Persisted %s rows to %s", len(merged), PROCESSED_DIR / f"{key}.csv")

    if failed_seasons:
        LOGGER.warning("Retained existing data where collection failed. Seasons with API errors: %s", failed_seasons)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=DEFAULT_LAST_SEASON)
    parser.add_argument("--include-practice", action="store_true", help="Also query Jolpica alpha practice classifications (FP1/FP2/FP3 or combined FP where published). Coverage is incomplete and the endpoint is alpha.")
    parser.add_argument("--refresh", action="store_true", help="Ignore local API response cache and re-download data.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    tables = build_dataset(args.start_year, args.end_year, args.include_practice, args.refresh)
    print("\nCollection summary")
    for name, frame in tables.items():
        print(f"{name:22s} {len(frame):>7,} rows")
    if tables["race_results"].empty:
        raise SystemExit("No race-result data were downloaded. Check internet access and the Jolpica API status.")


if __name__ == "__main__":
    main()
