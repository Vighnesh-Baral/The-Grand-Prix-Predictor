# The Grand Prix Predictor — PitchIQ Architecture + ML Backend v0.9.3

The Grand Prix Predictor is an independent Formula 1 analytics platform. This revision updates the FastAPI backend, ML feature engineering and the Constructor Intelligence page. The existing React application structure is retained; the Constructor Intelligence view now reports actual feature coverage instead of planned placeholders.

## Architecture

- **Frontend:** React 18 + Vite 6 + Tailwind CSS 4 (unchanged)
- **Backend:** FastAPI
- **Historical results/calendar:** Jolpica F1 API
- **Session timing/laps:** TIF1 public historical archive (FastF1-derived, no-login) + OpenF1 (historical fallback from 2023)
- **Weather:** OpenF1 observed session weather + Open-Meteo archived/live race-start forecasts
- **ML:** scikit-learn + XGBoost, selected using chronological validation
- **Persistence:** local CSV data tables and joblib model artifacts

## What's new in v0.9.3 — client release / identity integrity

- Canonical driver keys now derive from conservatively normalized full names, not upstream provider IDs. This prevents provider-specific IDs (including opaque `driver_*` IDs in practice data) from splitting one driver into multiple identities.
- Original `driver_id` and `constructor_id` values are retained exactly as supplied. The project does not replace, synthesize or relabel an external ID to force a join.
- Diacritic and punctuation normalization makes spellings such as `Nico Hülkenberg` and `Nico Hulkenberg` share a canonical key. No fuzzy matching or driver-code-only matching is used.
- Constructor keys remain season-scoped and are built from normalized constructor names. A team name in one season is not merged into another season's constructor entity.
- The processed-table migration now adds canonical keys to practice, race, qualifying and sprint results. It creates a pre-v0.9.3 backup before its first write and supports a dry run.
- TIF1 0.9.0 is isolated in `requirements-tif1.txt` and a separate `.venv-tif1` because it requires Pandas 3.x, while the ML/API environment is kept on Pandas 2.x. This avoids an unsatisfiable dependency set and protects the modelling/runtime environment.
- Practice-to-race feature joins now resolve names inside the exact season and round. This removes the previous global name-to-ID lookup that could assign a driver's ID from another season. Ambiguous mappings are excluded rather than guessed.
- Practice table deduplication now uses event + session + canonical driver key. Distinct drivers with missing external IDs are retained rather than collapsed into one null-ID row.
- Added a cross-source audit that reports practice-to-race and practice-to-qualifying matches, source-ID differences, name spelling variants and unmatched practice entries. FP-only/reserve drivers are reported for review, not treated automatically as errors.
- Added regression coverage for opaque-ID alignment, accented names, provider-ID preservation, event-scoped joins, FP-only drivers, and missing-ID row preservation.
- Full automated suite: 96 tests passed in the build environment. Real local-data migration and the real-data audit must still be run on the client's dataset before any data-quality sign-off.

## What's new in v0.9.2

- Permanent TIF1 practice identity resolution: null-like and code-only labels are not treated as full driver names.
- Missing identities are filled only from unambiguous same-season driver-code history; constructor history is resolved separately and conservatively.
- No upstream IDs are fabricated.

## What's new in v0.9.1

- Added conservative canonical identity keys to practice records. Project-owned `driver_key` and season-scoped `constructor_key` support identity continuity without inventing or overwriting Jolpica IDs.
- Driver mapping uses verified full-name agreement when matching TIF1 session metadata to Jolpica race entries; conflicting code/name pairs are left unmapped. Reserve and FP-only drivers remain valid practice rows with nullable upstream IDs.
- Added explicit identity-status fields so unresolved/code-only identities are visible for audit rather than silently guessed. Constructor aliases are normalized only within a season; historic team entities are not merged across seasons.
- Added an offline migration command, `python -m src.data.backfill_practice_identity`, to enrich the existing practice CSV by exact full-name matches within the same season and GP. It retains all practice rows and never fabricates upstream IDs.

## What's new in v0.9

- Historical practice/lap backfill now uses TIF1 public CDN archives by default. The collector never invokes FastF1 3.9+ account authentication and will not prompt for an F1 email or password. TIF1 serves completed sessions only; it is not a live-timing source.
- Existing `collect_fastf1_practice` and `collect_fastf1` command names are retained for compatibility, but historical data reads are routed through TIF1.
- Added TIF1 0.9.0 to requirements. Its archive documents lap, sector, tyre and telemetry coverage for 2018-current with public CDN failover. TIF1 is currently marked beta by its publisher, so the project stores provenance and should continue cross-checking important classifications against Jolpica/OpenF1 where possible.

## What's new in v0.8

- Practice Intelligence has a historical Grand Prix selector and reports separate FP1, FP2 and FP3 results. It prefers official Jolpica session classifications and supplements missing entries with TIF1 (FastF1-derived)/OpenF1 fastest-valid-lap reconstructions.
- OpenF1 raw and processed endpoint CSVs are merged by endpoint identity before canonical weather/lap/grid tables are rebuilt. This prevents a stale processed export from hiding additional raw records and keeps OpenF1 drivers without an abbreviation instead of collapsing them.
- Added a checkpointed no-login TIF1 practice backfill command for 2018–2026. It saves after each session, skips sessions with at least 15 usable driver times, logs unavailable sessions and can resume after interruption. Sprint-format weekends do not necessarily contain FP1/FP2/FP3.
- Pole Predictor now offers Before Free Practice (PRE_FP) using historical driver/constructor/previous-GP/circuit data, and After Free Practice with progressive FP1, FP1+FP2, and FP1+FP2+FP3 cutoffs. Dedicated pre-FP and stage-specific models are trained separately to enforce information cutoffs.
- Constructor Intelligence hides metric columns with no usable numeric observations and adds plain-language labels, definitions and interpretation guidance.
- Pre-FP race prediction normalises its empty practice-weather cutoff safely and continues to use only historical performance plus the Pole Predictor's projected grid.

## What's new in v0.7

- Prediction event selectors are restricted to upcoming Grands Prix; historical events are not offered for live prediction.
- Constructor Intelligence has a separate Grand Prix archive selector covering loaded weekends from 2018 onward.
- Circuit Intelligence now lists distinct circuits from the loaded 2018–2026 calendar and shows all available Grand Prix titles/results at each circuit.
- Race Predictor offers Before Qualifying and After Qualifying modes. Before Qualifying can use the historical-only PRE_FP forecast or a selected FP1/FP2/FP3 Pole Predictor cutoff, injects that ranked output as the projected grid, and excludes actual qualifying position/time features. After Qualifying requires the qualifying result and uses the official starting grid when available.

## What's new in v0.6

1. **Race prediction:** expert-priority + ML ensemble. The initial race feature hierarchy is constructor performance (20%), current weekend race pace (18%), grid/circuit advantage (18%), driver form (12%), tyre/strategy (12%), weather context (8%), reliability (7%) and circuit characteristics (5%). These are design priors, not measured coefficients.
2. **Field-coherent race outcomes:** 4,000-run Monte Carlo estimates produce one winner and up to three podium finishers per simulation. Winner probabilities sum to 100%; podium probabilities sum to three expected places. The simulator uses validation residual spread and driver DNF risk. These estimates are not externally calibrated.
3. **Circuit-aware grid influence:** actual grid position is combined with lagged circuit/grid conversion history. Sparse circuit samples are smoothed toward overall grid-conversion rates. No hard-coded Monaco/Mexico bonus is used; physical first-corner and slipstream simulation is not yet implemented.
4. **Safety Car/VSC context:** prior OpenF1 race-control messages are converted to circuit event frequencies, shrunk toward a global historical prior when local samples are sparse. Those rates affect race simulation uncertainty/field compression; they do not predict deployment lap or detailed restart strategy.
5. **Weather context:** the post-qualifying race model uses the pre-race forecast only when its lead time matches the archived one-day-ahead training window. Wet forecasts increase scenario uncertainty; actual race weather is never used for pre-race predictions.
6. **Constructor Intelligence:** all six sections now have feature calculations and coverage reporting: qualifying pace, race pace, long-run pace, tyre behaviour, reliability and circuit suitability. The UI shows `DATA AVAILABLE` or `AWAITING DATA` based on the selected event's actual local data.
7. **Tyres and strategy proxies:** FP compound-specific lap-time slope versus tyre age is estimated from sufficiently long stints, and recent OpenF1 pit-stop duration is linked to constructors through event driver metadata. These are proxies, not a fully fuel-corrected degradation or optimal pit-strategy simulator.
8. **Reliability:** recent-five-GP constructor and driver DNF rates, mechanical DNF rates and same-circuit retirement history are built using prior events only. Administrative outcomes are treated as unknown rather than mechanical failures.
9. **Constructor practice:** current-session constructor one-lap and long-run pace are normalised within session; latest-session pace is stored separately so an FP2 reversal is not hidden by an earlier FP1 best. FP1/FP2/FP3 cutoffs are enforced.
10. **Regression tests:** tests now cover lagged reliability, SC/VSC history, pit-stop mapping, FP cutoff isolation and coherent race probabilities.

These additions implement the agreed architecture but do **not** establish improved real-world accuracy. The uploaded project does not contain your full local historical lap, pit, weather and race-control archives. Retrain and backtest on restored local data before trusting the model.

## What's new in v0.4

1. TIF1 (FastF1-derived) lap collection writes normalized one-row-per-lap records to `data/processed/session_laps.csv` and `data/raw/session_laps.csv`.
2. Practice feature engineering separates session one-lap pace, soft-tyre best-lap pace and long-run stint pace. Long-run comparisons use a tyre compound shared by multiple drivers rather than mixing soft/medium/hard runs.
3. Constructor qualifying pace now includes rolling historic, recent-three-event, current-season and same-circuit relative gaps. Same-circuit pace is only populated from prior visits to that circuit.
4. Driver qualifying pace includes a teammate-relative gap.
5. Race-lap engineering adds lagged driver/constructor long-run pace and an observed stint lap-time slope. The slope is explicitly not treated as pure tyre degradation.
6. Future-event roster fallback uses the latest completed GP field rather than every driver ever present in the dataset.
7. DNF labels exclude disqualifications, non-starters and administrative outcomes instead of silently classifying them as mechanical/on-track DNFs.
8. New backend endpoints expose data readiness, practice pace, constructor pace components and driver pace components.
9. Pole models are trained independently for FP1, FP2 and FP3; an FP3-trained model is never reused for an earlier prediction cutoff.
10. An optional `data/processed/driver_roster.csv` supports the confirmed event-specific driver/constructor lineup before the new season's first practice session.
11. Training metadata records feature coverage so the API can identify old artifacts that never learned from the new pace features.
12. OpenF1 historical ingestion runs as a separate, rate-limited sync process. TIF1 public archive rows are the preferred completed-session lap source; OpenF1 rows are marked distinctly and fill missing event/session/driver/lap keys.
13. Open-Meteo adds archived one-day-ahead race-start forecasts for historical training and a live scheduled-race-start forecast for upcoming events. Live forecasts outside the 12–36 hour lead window represented by training are withheld from model features but remain available through the weather endpoint.
14. Race prediction adds smoothed, strictly lagged circuit/grid conversion features. The model can learn whether pole or P2/P3 starts historically convert at each circuit; no fixed Monaco or Mexico bonuses are hard-coded.
15. The OpenF1 sync now fetches the race-session `starting_grid` endpoint after qualifying and before the race when it is available. Future race predictions prefer this official grid (including published penalties) and label qualifying-position fallback rows as a proxy.
16. Training reports event-level baselines (fastest practice-time pick for pole; starting-grid order for race/podium) beside the held-out model metrics.
17. API source/readiness metadata distinguishes TIF1/FastF1-derived and OpenF1 lap coverage and exposes `/api/weather/forecast`.

These are feature-engineering and backend changes. They are **not evidence that prediction accuracy has improved**. Real-data retraining and chronological evaluation are still required.

## v0.9.3 release verification boundaries

- Backend: Python source compilation succeeded; **96 automated tests passed**; importing the FastAPI application succeeded and reported API version `0.9.3`.
- The automated test runner used Python 3.13.5. The documented client environment is Python 3.12, which still needs a workstation verification run.
- Frontend: the React source and package manifest are retained, but the Vite production build was not verified in this build environment because npm registry access timed out and the dependencies were not cached.
- Data/model: client CSV datasets are not included. The archive contains an existing `f1_models.joblib` and metrics snapshot for inspection, but that artifact is not a substitute for retraining after the v0.9.3 identity migration. Real-data migration, the cross-source audit, model retraining and chronological holdout evaluation must be completed before live predictions are represented as validated.

## Windows setup

Use Python 3.12 and a current Node.js LTS release.

### 1. Copy your local F1 data into this project

The source ZIP excludes your local CSV datasets. An existing `artifacts/f1_models.joblib` and `artifacts/metrics.json` snapshot is included, but retrain it after restoring your data and applying the v0.9.3 identity migration. Copy from your existing F1 project:

- `data/raw/`
- `data/processed/`
- Do not rely on an old `artifacts/f1_models.joblib` for this release; retrain after the v0.9.3 identity migration so the trained pipeline is built against the corrected feature joins.
- OpenF1 raw tables such as `data/processed/openf1_pit.csv`, `openf1_drivers.csv` and `openf1_race_control.csv` if already collected; otherwise run the OpenF1 sync below.

For a season-opening event before FP1, populate the optional `data/processed/driver_roster.csv` from the confirmed event entry list. Required columns are `season,round,driver_id,driver_code,driver_name,constructor_id,constructor_name`. Current-event qualifying entries (for race prediction) or practice entries (once available) take precedence over this file.

Keep `data/cache/` optional; TIF1 maintains its own local cache.

### 2. Backend environment

From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Migrate and audit identities (v0.9.3)

With the project `.venv` active, preview the migration:

```powershell
python -m src.data.backfill_practice_identity --dry-run
```

Review the driver and constructor identity status counts. Then apply the migration:

```powershell
python -m src.data.backfill_practice_identity
```

The migration writes only to `data/processed/`. It creates one `*.pre-v0.9.3.bak.csv` backup beside each table before the first write. It preserves all original provider IDs and all practice rows.

Generate the cross-source report:

```powershell
python -m src.data.validate_identities
```

Reports are saved to `reports/identity_audit/`:

- `identity_summary.json` — table coverage and per-season practice-to-race/qualifying match counts.
- `unmatched_practice_vs_race.csv` and `unmatched_practice_vs_qualifying.csv` — practice drivers absent from the corresponding result table.
- `provider_id_mismatches.csv` — cases where the same canonical event/driver identity carries different provider IDs.
- `normalized_name_variants.csv` — display-name variants that normalize to the same identity key.

Do not interpret every unmatched practice entry as a defect. Reserve and FP-only drivers can legitimately have practice data without a race result. Review the event, name, code, source and session before changing a mapping. The migration deliberately does not fuzzy-match names or use a three-letter code as a standalone identity.

After the migration and review, retrain the model from the restored local data before trusting predictions:

```powershell
python -m src.models.train_models
```

The release archive contains source code and tests, not the client's CSV history or trained model artifacts. A successful code test run does not substitute for training and chronological backtesting on the client's real data.

### 4. Start the backend

```powershell
python run.py
```

API: `http://127.0.0.1:8000`  
API docs: `http://127.0.0.1:8000/docs`

### 5. React frontend

In a second VS Code terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The Constructor Intelligence view now displays the six implemented feature pillars and their local coverage; other navigation and page structure are retained.

## Historical data workflow

Collect the historical Jolpica-compatible tables first:

```powershell
python -m src.data.build_dataset --start-year 2018 --end-year 2026 --include-practice
```

Jolpica practice classifications have incomplete coverage; they are not a replacement for lap-by-lap TIF1/FastF1-derived timing.

### Historical timing data without an F1 login

FastF1 3.9 introduced F1TV account authentication for most direct F1 data access following Formula 1's October 2026 API changes. This project therefore uses **TIF1 0.9.0** for historical session backfills: TIF1 reads published FastF1-derived session files from public CDNs and does not call the authenticated F1 data API. The historical collector does not ask for an F1 email or password. TIF1 data is available after completed sessions are published and is not a live-timing feed. See the [FastF1 authentication guide](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/accounts_auth.rst) and [TIF1 project documentation](https://pypi.org/project/tif1/) for their respective access models. Do not enter F1 credentials into the project collector.

**Important environment split:** TIF1 0.9.0 requires Pandas 3.x. The ML/API environment in `requirements.txt` stays on Pandas 2.x. Install TIF1 only in a separate collector environment, from the project root:

```powershell
py -3.12 -m venv .venv-tif1
.\.venv-tif1\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-tif1.txt
```

Then collect historical practice classifications. First test one event:

```powershell
python -m src.data.collect_tif1_practice --start-year 2025 --end-year 2025 --max-events 1
```

For the full historical practice archive, use:

```powershell
python -m src.data.collect_tif1_practice --start-year 2018 --end-year 2026
```

For lap-level session tables across a calendar range, use the compatibility batch command (also TIF1-backed and no-login):

```powershell
python -m src.data.collect_fastf1_batch --start-year 2024 --end-year 2024 --sessions FP1 FP2 FP3 R --max-events 3
python -m src.data.collect_fastf1_batch --start-year 2018 --end-year 2026 --sessions FP1 FP2 FP3 R
```

After collection, return to the ML/API environment before migration, training or running the backend:

```powershell
deactivate
.\.venv\Scripts\Activate.ps1
```

Some sprint-format weekends do not have all three traditional practice sessions; the collector reports unavailable sessions rather than fabricating them. A full ordinary-weekend collection is still required to count toward the project's complete-weekend readiness threshold. The threshold is a minimum data-coverage gate, not evidence that the resulting model is accurate. For a current weekend, collect only sessions that have finished. Run this command with `.venv-tif1` active:

```powershell
python -m src.data.collect_fastf1 --season 2026 --round <ROUND_NUMBER> --sessions FP1 FP2 FP3
```

The collector upserts by season, round, session, driver code and lap number. Re-running a collection updates the same session rows instead of appending duplicate laps.

### OpenF1 historical session data (free tier)

OpenF1 provides historical data from 2023 onwards. The free tier is not a live timing feed: data is treated as historical only after the session's live window closes. The sync process waits 35 minutes after the reported session end, applies retries/backoff, caches requests, and writes raw endpoint tables plus normalized `session_laps.csv` and `weather_observations.csv`. It respects the published free-tier limit of 3 requests/second and 30 requests/minute.

Run a one-time historical sync:

```powershell
python -m src.data.openf1_sync --start-year 2023 --end-year 2026
```

For periodic polling during a race weekend, run this in a separate terminal. It checks every 15 minutes and fetches sessions once they are in the free historical window:

```powershell
python -m src.data.openf1_sync --start-year 2023 --end-year 2026 --watch --interval-minutes 15
```

OpenF1's free historical coverage, current rate limits and live-versus-historical window are described in its [official API documentation](https://openf1.org/) and [endpoint documentation](https://openf1.org/docs/). OpenF1 is an independent/unofficial project intended for educational, research and non-commercial fan-engagement use; review its terms before commercial use.

### Historical race-start weather forecasts

The training pipeline uses the Open-Meteo Previous Runs API's `previous_day1` fields: these represent what was forecast 24 hours before the weather's valid time. Most model archives begin in January 2024, so weather-feature coverage is expected to be partial. Generate the local forecast archive after building the Jolpica calendar:

```powershell
python -m src.data.open_meteo_client --start-year 2024 --end-year 2026
```

The live API uses the Open-Meteo Forecast API at the scheduled race start (maximum forecast horizon: 16 days). Because training uses one-day-ahead forecasts, live race-weather features are only passed into the model when the forecast lead is 12–36 hours; earlier/much later forecasts remain inspectable through `GET /api/weather/forecast?season=2026&round=...` but are withheld from the estimator to reduce lead-time distribution mismatch. Open-Meteo documents its fixed 1–7 day historical forecast lead offsets in the [Previous Runs API reference](https://open-meteo.com/en/docs/previous-runs-api).

### Retrain models

Only after copying/collecting the historical data, including as many complete FP1/FP2/FP3 and R sessions as your account can access:

```powershell
python -m src.models.train_models
```

Training writes:

- `artifacts/f1_models.joblib`
- `artifacts/metrics.json`

Review `feature_coverage_fraction` for each task before trusting a prediction. The readiness endpoint only marks pace-aware training coverage ready when it finds at least 20 complete GP weekends (FP1, FP2, FP3 and R) across at least three seasons; this is a minimum data-coverage gate, not evidence of good accuracy. If practice/race lap coverage is low, the model has not learned those inputs reliably. Do not infer a trained feature from the fact that the API can calculate it at prediction time.

## API endpoints

### Core

- `GET /api/health`
- `GET /api/overview`
- `GET /api/data/readiness`
- `GET /api/data/openf1/status`
- `GET /api/weather/forecast?season=2026&round=...`
- `GET /api/events?upcoming_only=true`
- `GET /api/events/next`
- `GET /api/events/{season}/{round_number}`
- `GET /api/standings?season=2026&round=...`
- `GET /api/history/{circuit_id}`
- `GET /api/model/metrics`

### Pace intelligence

- `GET /api/practice/pace?season=2026&round=...&as_of_session=FP1|FP2|FP3`
- `GET /api/intelligence/constructor-pace?season=2026&round=...&as_of_session=FP1|FP2|FP3`
- `GET /api/intelligence/driver-performance?season=2026&round=...&as_of_session=FP1|FP2|FP3`

### Prediction gates

- `POST /api/predictions/pole` is available only before qualifying and before the race start. `as_of_session` selects the FP1/FP2/FP3 data cutoff.
- `POST /api/predictions/race` is available only after qualifying has been collected and before the scheduled race start.

Example request:

```json
{
  "season": 2026,
  "round": 21,
  "as_of_session": "FP3"
}
```

The prediction endpoints use the installed model bundle. Training now creates separate `pole_fp1`, `pole_fp2` and `pole_fp3` models; the legacy `pole` key is retained as an FP3 alias for older consumers. If the installed bundle was trained before v0.6 features, the API returns explicit warnings that it cannot use circuit conversion, forecast weather, pit-stop history or new Constructor Intelligence features. The source archive did not include your locally trained artifact; copying a legacy bundle does not upgrade it. Retrain after collecting historical pace data and the one-day-ahead weather archive.

## Reload local data while API is running

After collecting or updating CSVs:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/dev/reload-data
```

## Validate code

From project root:

```powershell
python -m compileall -q src app
python -m pytest -q
```

The current test suite uses synthetic fixtures for deterministic code-path checks. Passing tests is not a measure of real-world F1 prediction accuracy. Frontend dependency installation/build could not be verified in this packaging environment because the npm install attempt timed out; run `npm install` and `npm run build` locally if you want a production build check.

## Important methodology limitations

- Practice lap times are not fully adjusted for fuel load, engine mode, traffic, tyre programme or track evolution.
- A soft-tyre best lap is a one-lap pace proxy, not proof of a qualifying simulation.
- Long-run pace is not fuel corrected and does not isolate tyre degradation.
- Historical constructor pace can be disrupted by regulation changes, upgrades and team restructuring. Recent/current-season pace is intentionally represented separately from global history.
- Weather is represented by OpenF1 observed session weather and an Open-Meteo one-day-ahead race-start forecast. Forecast lead time and source coverage remain incomplete.
- The track effect is learned from prior circuit/grid conversion history plus circuit ID; the current version does not explicitly model first-corner geometry, tow/slipstream strength, DRS layout, altitude, track evolution or surface grip. It can learn historical Mexico P2/P3 conversion, but it does not simulate the Turn 1 slipstream mechanism.
- Pit-stop timing coverage is only available where OpenF1 pit and driver records can be joined. The current model does not yet optimise one-stop/two-stop strategies or simulate tyre choice and pit windows explicitly.
- Safety Car/VSC history influences race-outcome uncertainty only where race-control coverage exists; the model does not simulate deployment timing, restart position, or undercut/overcut decisions under neutralisation.
- Circuit effects use historical grid conversion plus circuit identity; the current version does not explicitly encode first-corner geometry, tow/slipstream strength, DRS layout, altitude, surface grip or pit-lane loss.
- Model classifier scores and normalized score shares are not calibrated probabilities. Historical backtesting must stay separate from live upcoming-event prediction.

## Data rights

Jolpica data is made available under CC BY-NC-SA 4.0 non-commercial terms. Attribute the data provider and review all upstream data terms before redistributing or using commercially. TIF1 is a community-maintained beta archive client; its README identifies FastF1-derived TracingInsights timing files and MIT-licensed library code. Review upstream data terms and attribution requirements before public or commercial deployment.


## v0.7 prediction stages and history views

- `/api/events?upcoming_only=true` feeds the live Pole Predictor and Race Predictor selectors. `/api/events?upcoming_only=false` feeds the Constructor Intelligence archive.
- `/api/circuits?first_season=2018` returns the distinct circuit directory. `/api/history/{circuit_id}?season=2018&last_season=2026` returns all available Grand Prix records for that venue, including alternate event titles at the same circuit.
- `POST /api/predictions/race` accepts `race_stage`: `before_qualifying` or `after_qualifying`. The default remains `after_qualifying` for compatibility.
- Before-qualifying mode supports a dedicated `PRE_FP` historical-only forecast or progressive FP1, FP2 and FP3 cutoffs. The selected Pole Predictor ranking is injected as the projected `grid`; qualifying position and qualifying lap time are set to missing. The response identifies `starting_grid_source` as `pole_predictor_predicted_grid` and includes `pole_predictor_grid`.
- After-qualifying mode requires qualifying results and uses the actual published starting grid when available, falling back to qualifying position only where necessary.


## v0.8 practice archive and staged pole forecasts

- Practice Intelligence reads official Jolpica practice classifications first and supplements missing driver-session rows from TIF1 (FastF1-derived)/OpenF1 lap archives. OpenF1 raw exports in `data/raw/` are normalized into the canonical `data/processed/session_laps.csv` locally at API start and before default-path model training; this performs no network requests.
- To backfill missing historical FP1/FP2/FP3 classifications from 2018 onward, run `python -m src.data.collect_fastf1_practice --start-year 2018 --end-year 2026`. Despite the retained legacy command name, v0.9 uses TIF1 public CDN archives and never requests F1TV credentials. The collector checkpoints after each session and skips sessions already containing at least 15 driver records. Results are reconstructed from each driver’s fastest valid lap, not presented as a separately published official classification. Archive availability can vary by session; sprint-format weekends may not contain all three free-practice sessions.
- After data collection, use `Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/dev/reload-data` or the dashboard refresh control so the running API reloads local files.
- v0.8 trains a dedicated `pole_pre_fp` model using historical constructor/driver/circuit form only, alongside separate FP1, FP2 and FP3 models. An FP2 forecast requires FP1+FP2 data; an FP3 forecast requires FP1+FP2+FP3 data.
- Re-train after collecting practice data with `python -m src.models.train_models`. Existing model artifacts from v0.7 do not contain the new `pole_pre_fp` model and must be replaced by the new training output.
- Constructor Intelligence dynamically hides columns with no numeric observations and displays plain-language metric definitions and interpretation notes.

## Frontend refresh — premium motorsport workspace

The React frontend has been restyled as a long-form editorial landing page and a separate premium analytics workspace. The landing page retains the title **THE GRAND PRIX PREDICTOR** (without a trailing full stop), removes the landing-page navigation strip, presents the supplied F1 concept car with a graphite material override and animated illustrative airflow/pressure-style layers, and adds motorsport editorial sections with attributed imagery. The platform workspace now uses horizontal navigation, a photo-led Race Centre, asymmetric analysis panels, circuit-country flags and driver/constructor nationality flags where a mapping is known. Existing API-backed prediction and intelligence pages are retained.

See `frontend/README.md` for run instructions and `CREDITS.md` for image / 3D asset attribution. Wikimedia photos, FlagCDN flags, Google Fonts and the model-viewer browser runtime load remotely, so the front end needs internet access for those visual assets. The airflow overlay is an illustrative animation, not a CFD calculation.
