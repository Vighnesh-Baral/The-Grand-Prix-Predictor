"""Read and print the latest chronological evaluation report."""
from __future__ import annotations
import json
from pathlib import Path
from src.config import ARTIFACTS_DIR


def main() -> None:
    report_path = ARTIFACTS_DIR / "metrics.json"
    if not report_path.exists():
        raise SystemExit("No evaluation report found. Train models first: python -m src.models.train_models")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
