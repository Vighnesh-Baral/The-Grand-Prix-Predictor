"""Central paths and project settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIRST_SEASON = 2018
DEFAULT_LAST_SEASON = 2026  # Current season; its rows are partial until the season is complete.
JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
JOLPICA_ALPHA_BASE_URL = "https://api.jolpi.ca/f1/alpha"

for _path in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, ARTIFACTS_DIR, FIGURES_DIR):
    _path.mkdir(parents=True, exist_ok=True)
