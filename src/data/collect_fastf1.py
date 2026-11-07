"""Collect completed-session lap data from TIF1 public archives (no F1 login).

Examples:
  python -m src.data.collect_fastf1 --season 2026 --round 22
  python -m src.data.collect_fastf1 --season 2025 --round 1 --sessions FP1 FP2 FP3 Q R
"""
from __future__ import annotations

import argparse
import logging

from src.data.fastf1_client import collect_fastf1_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--sessions", nargs="+", default=["FP1", "FP2", "FP3"],
                        help="Session codes: FP1 FP2 FP3 Q SQ S R (TIF1 public archive; no login)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    kwargs = {"destination": args.output} if args.output else {}
    frame = collect_fastf1_sessions(args.season, args.round, args.sessions, **kwargs)
    print(f"Stored {len(frame):,} lap rows for {args.season} Round {args.round}.")
    print("The model needs historical weekends collected as well before the new session features can be trained and evaluated.")


if __name__ == "__main__":
    main()
