# Client Handoff Checklist — F1 Race Intelligence v0.9.3

## Included in this release

- FastAPI prediction and intelligence backend source.
- Existing React/Vite application source and package manifest.
- Historical ingestion, feature engineering, model training/inference and API modules.
- Provider-independent cross-source identity keys and migration/audit tooling.
- Automated backend regression tests and model card.

## Required before a client accepts model predictions

1. Copy the client's `data/raw/` and `data/processed/` CSV tables into the project. This source archive intentionally does not contain private/local datasets or model artifacts.
2. Create the Python 3.12 ML/API environment and install `requirements.txt` (Pandas 2.x). If historical TIF1 collection is required, create the separate `.venv-tif1` and install `requirements-tif1.txt` there (Pandas 3.x); never install TIF1 into the ML/API environment.
3. Run `python -m src.data.backfill_practice_identity --dry-run`, review the status counts, then apply the migration with `python -m src.data.backfill_practice_identity`.
4. Run `python -m src.data.validate_identities` and review the per-season summary plus unmatched/source-ID reports. Do not force-match reserve or FP-only drivers.
5. Train fresh artifacts with `python -m src.models.train_models` after returning to the main `.venv`.
6. Review `artifacts/metrics.json`: chronological train/validation/test seasons, event-level metrics, baseline comparison and `feature_coverage_fraction` for every task. Reject any task whose features are absent or whose test coverage is insufficient for its intended use.
7. Run `python -m pytest -q` in the client's Python 3.12 environment.
8. In `frontend/`, run `npm install` and `npm run build`; resolve any dependency/build issues before deployment.
9. Start the API and UI, verify `/api/health`, `/api/data/readiness`, upcoming event selection, practice cutoffs, and both before-/after-qualifying race prediction modes using a completed test event and a future event.

## Release verification performed in this build environment

- `python -m compileall -q src app tests`: passed.
- `python -m pytest -q`: 94 passed.
- FastAPI import/version smoke test: passed; API reports `0.9.3`.
- Automated tests ran with Python 3.13.5, not the documented Python 3.12 client interpreter.
- React/Vite production build: not verified. npm registry access timed out and packages were not present in the offline cache.
- Real client data migration, audit and model training: not run because the client CSV data and trained model artifact were not included in the source package.

## Acceptance boundary

This is a tested **source-code release**, not a claim that the client's trained F1 forecasts have passed real-world validation. Do not describe its predictions as client-validated until the real-data migration, audit, retraining and chronological test review above are complete.
