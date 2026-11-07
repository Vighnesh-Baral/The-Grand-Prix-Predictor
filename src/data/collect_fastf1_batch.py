"""Collect completed TIF1 public-archive sessions without F1TV credentials.

Example:
  python -m src.data.collect_fastf1_batch --start-year 2022 --end-year 2026 --sessions FP1 FP2 FP3 R
  python -m src.data.collect_fastf1_batch --start-year 2024 --end-year 2024 --sessions FP1 FP2 FP3 R --max-events 3
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.data.fastf1_client import collect_fastf1_sessions

LOGGER = logging.getLogger("f1_fastf1_batch")


def event_list(calendar_path: Path | str, start_year: int, end_year: int) -> list[tuple[int, int, str]]:
    path = Path(calendar_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Race calendar not found at {path}. Run `python -m src.data.build_dataset` first."
        )
    calendar = pd.read_csv(path)
    required = {"season", "round"}
    if not required.issubset(calendar.columns):
        raise ValueError(f"Race calendar must contain columns: {', '.join(sorted(required))}")
    calendar["season"] = pd.to_numeric(calendar["season"], errors="coerce")
    calendar["round"] = pd.to_numeric(calendar["round"], errors="coerce")
    selected = calendar[
        calendar["season"].between(start_year, end_year)
        & calendar["round"].notna()
    ].copy()
    selected = selected.drop_duplicates(["season", "round"]).sort_values(["season", "round"])
    name_col = "race_name" if "race_name" in selected.columns else None
    return [
        (int(row.season), int(row.round), str(getattr(row, name_col)) if name_col else "")
        for row in selected.itertuples(index=False)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--sessions", nargs="+", default=["FP1", "FP2", "FP3", "R"],
                        help="Session codes to collect; e.g. FP1 FP2 FP3 Q R")
    parser.add_argument("--calendar", default=str(PROCESSED_DIR / "races.csv"))
    parser.add_argument("--output", default=str(PROCESSED_DIR / "session_laps.csv"))
    parser.add_argument("--max-events", type=int, default=None,
                        help="Optional limit for a small test run")
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year must be greater than or equal to --start-year")
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be at least 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    events = event_list(args.calendar, args.start_year, args.end_year)
    if args.max_events is not None:
        events = events[:args.max_events]
    if not events:
        raise SystemExit("No calendar events found in the selected season range.")

    successes = 0
    failures: list[str] = []
    for index, (season, round_number, name) in enumerate(events, start=1):
        label = f"{season} Round {round_number}" + (f" — {name}" if name else "")
        LOGGER.info("[%s/%s] Collecting %s: %s", index, len(events), label, ", ".join(args.sessions))
        try:
            frame = collect_fastf1_sessions(
                season, round_number, args.sessions, destination=Path(args.output)
            )
            LOGGER.info("Stored/updated %s total lap rows after %s", f"{len(frame):,}", label)
            successes += 1
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            LOGGER.warning("Could not collect %s: %s", label, exc)

    print(f"Finished no-login TIF1 batch: {successes}/{len(events)} events processed.")
    print(f"Lap archive: {args.output}")
    if failures:
        print("Events with collection errors:")
        for failure in failures:
            print(f" - {failure}")


if __name__ == "__main__":
    main()
