## v0.9.3 — client release / provider-independent identity integrity

- Canonical driver keys are now based on normalized full names, independent of external provider IDs. Raw Jolpica/TIF1 IDs remain unchanged and available for provenance.
- Added `driver_name_normalized` and `constructor_name_normalized` audit fields. Driver statuses distinguish name keys with/without a source ID from unresolved ID-only records; constructor statuses are explicit and season-scoped.
- Accent and punctuation normalization supports equivalent full-name spellings without fuzzy matching. Code-only labels and ID-only rows do not receive cross-source canonical driver keys.
- Extended the offline migration to practice, race, qualifying and sprint result tables. It supports `--dry-run`, writes processed tables atomically, and keeps a pre-v0.9.3 backup before first write.
- Fixed practice feature joins to map normalized names only within the exact season and round. The previous global mapping could reuse an ID from a different season. Ambiguous mappings remain unmatched.
- Fixed practice-table merge deduplication so missing IDs do not cause separate drivers to collapse into one row. Canonical event/session/driver keys are used where a verified full name exists.
- Added `src.data.validate_identities` for source matching, per-season match rates, mismatched source IDs, unmatched practice entries and normalized display-name variants.
- Isolated TIF1 0.9.0 in `requirements-tif1.txt` and a dedicated `.venv-tif1` because its Pandas 3.x requirement conflicts with the ML/API environment's Pandas 2.x pin. Historical session collection runs in the collector environment; model training and the API run in the main environment.
- Regression tests cover provider ID changes for the same driver, accent equivalence, ID preservation, event-scoped joins, reserve/FP-only records, migration backups and audit classifications.
- Full automated suite: 96 tests passed on the available Python 3.13.5 runner; the documented Python 3.12 client environment still requires a workstation run. The frontend production build was not verified because npm registry access timed out in the build environment. The release package does not contain the client's local CSV data or trained model artifacts, so real-data audit and model retraining remain client-side release steps.

## v0.9.2 — persistent practice identity resolution

- Treats null-like source labels such as `None None`, `None`, `nan`, `null`, and code-only driver labels as missing identity metadata rather than valid full names.
- TIF1 practice merges now resolve missing driver names from unambiguous same-season driver-code history. The resolver refuses ambiguous code-to-driver mappings.
- Constructor names are filled independently and only when same-season code history identifies exactly one constructor; drivers who changed teams are not assigned a guessed team.
- No Jolpica/Ergast IDs are fabricated. Internal driver keys and season-scoped constructor keys remain explicitly separate from upstream IDs.
- Added regression tests for placeholder names, code-only fallback labels, same-season identity recovery, and ambiguous constructor history.
- Full automated suite: 86 tests passed.

## v0.9.1 — conservative practice identity layer

- Added project-owned `driver_key` and season-scoped `constructor_key` fields for TIF1/Jolpica practice data. These keys are distinct from upstream Ergast/Jolpica IDs.
- TIF1-to-Jolpica driver mapping now requires matching normalized full names; conflicting code/name matches are not assigned an upstream driver or constructor ID.
- FP-only and reserve drivers remain in practice data with nullable upstream IDs and explicit `name_key_only` identity status. Code-only identities remain unresolved.
- Added constructor alias normalization scoped by season, preserving historical constructor identities rather than collapsing team lineages.
- Added an offline migration command to enrich the existing practice CSV using unique full-name matches within the same season and round; no archive redownload is required.
- Added regression tests for reserve-driver identity, code/name conflicts, code-only records and season-scoped constructor aliases.

## v0.9 — no-login historical practice archive

- Replaced direct FastF1 3.9 session loading in historical practice/lap collectors with TIF1 public CDN archives. Existing command names are retained, but no F1 email/password prompt is triggered.
- Added TIF1 session-code mapping (FP1/FP2/FP3 and Q/SQ/S/R), driver/team metadata normalization, and data-source provenance.
- Kept checkpointing, skip-covered-session logic, and atomic CSV persistence. Unavailable sessions are logged and skipped so remaining calendar events continue.
- Updated the FastF1 compatibility client and batch collector to route completed-session data through TIF1, avoiding accidental interactive authentication prompts.
- Added the TIF1 0.9.0 collector dependency; in v0.9.3 it is isolated from the ML/API environment because of its Pandas 3.x requirement.

# ML / Backend Changelog — v0.7

## v0.7 — prediction stages and 2018+ archive views

- Live Pole Predictor and Race Predictor continue to use the future-event-only calendar endpoint.
- Constructor Intelligence now uses a separate all-events selector so viewers can inspect previous GP weekends from 2018 onward without making historical events eligible for live predictions.
- Added `/api/circuits` to list distinct venues represented in the loaded F1 calendar from 2018 onward.
- Circuit history now includes all GP event names held at a venue, rather than filtering the record to only the most recent Grand Prix title used at that venue.
- Added explicit Race Predictor modes: `before_qualifying` and `after_qualifying`.
- Before-qualifying mode uses FP3 Pole Predictor ranking as a projected grid and keeps qualifying position/time fields empty to avoid leaking qualifying information.
- After-qualifying mode requires qualifying data and uses the published grid or qualifying-position proxy.
- Added projected starting-grid position to race predictions and a visible projected-grid preview in the UI.
- Updated the season archive label to a range (for example, 2018–2026) rather than a single selected season.

# ML / Backend Changelog — v0.6

Frontend source is unchanged in this revision. This release focuses on ML feature engineering, the three-source data pipeline, race-weather forecasts and backend reporting.

## Implemented

- Retained the existing FastF1 collector as the preferred detailed timing/lap source and added a separately tracked OpenF1 historical fallback (2023 onward).
- OpenF1 sync applies a 35-minute post-session free-tier delay, request caching, retries, per-endpoint state, deduplication, raw persistence and normalized weather/lap outputs. FastF1 rows take precedence for matching event/session/driver/lap keys.
- Open-Meteo historical Previous Runs integration creates race-start weather inputs from one-day-ahead archived forecasts. Live forecasts are fetched at the scheduled race start for upcoming race prediction.
- Live race-weather features are only used at 12–36 hours lead, matching the approximate one-day-ahead training horizon. Other lead times remain available from the weather API but are withheld from the estimator.
- Race features include air temperature, humidity, precipitation, wind speed/direction and a wet-weather indicator when forecast archive coverage supports them.
- Added smoothed, lagged circuit/grid conversion features: grid-specific win/podium conversion, pole-to-win conversion, P2/P3-to-win conversion, front-three podium rate and position gain.
- Circuit conversion is learned from prior completed events and uses global grid-position priors where local samples are sparse; no manual Monaco/Mexico probability overrides are applied.
- Event-level holdout baselines are now reported with model metrics: fastest practice-time selection for pole and starting-grid order for winner/podium tasks.
- Corrected lap-source reporting so FastF1 and OpenF1 rows are distinguished, and added `GET /api/weather/forecast`.
- Added pre-race OpenF1 `starting_grid` ingestion and driver mapping. Race candidates prefer the published official grid (including grid penalties) and explicitly identify qualifying-position proxy fallbacks.
- Added transient request retry logic to Open-Meteo client and tests for source cutoffs, forecast lead eligibility, starting-grid sync/mapping and endpoint behaviour.
- 54 automated tests pass in the build environment.

## Not yet implemented or not yet validated

- Real-data retraining was not possible because the uploaded archive did not contain the user's historical race/qualifying/practice CSVs, full FastF1/OpenF1 lap archive or trained model artifacts.
- No real held-out accuracy is claimed for this build. Synthetic tests check code paths, not F1 race-prediction performance.
- Current circuit features learn historical grid conversion by circuit but do not explicitly encode first-corner length/geometry, tow/slipstream dynamics, DRS layout, altitude or overtaking opportunity.
- Practice and race lap times are not fuel-corrected, engine-mode-corrected or traffic-corrected. Long-run pace is an observed proxy, not a fully modelled tyre-degradation or race simulation.
- Weather coverage is partial. Open-Meteo's Previous Runs archive is available for most models from January 2024, with model-specific exceptions; earlier seasons may have missing weather features.
- Reliability, penalties, safety-car probability, pit strategy, undercut/overcut, tyre degradation and changing car upgrades/regulations are not fully modelled.
- Classifier probabilities are not calibrated event-level win/pole probabilities. Normalized score shares are relative rankings, not guaranteed probabilities.
- Sprint weekends are not yet treated as a separate sprint-specific prediction stage. FP1/FP2/FP3 cutoffs target the conventional qualifying workflow.
- Persistence remains local CSV plus joblib; PostgreSQL is not wired into this revision.

## Required local data before model retraining

1. Copy the existing Jolpica raw/processed tables into `data/raw/` and `data/processed/`.
2. Populate `data/processed/driver_roster.csv` for an event if the current driver/constructor entries are not yet present in practice/qualifying data.
3. Collect historical FP1, FP2, FP3 and R sessions across as many seasons as access permits using FastF1, then run the OpenF1 historical sync for 2023 onward.
4. Generate the one-day-ahead historical weather archive with `python -m src.data.open_meteo_client --start-year 2024 --end-year 2026`.
5. Run `python -m src.models.train_models`.
6. Review held-out test metrics against `test_baseline` and check feature coverage for `pole_fp1`, `pole_fp2`, `pole_fp3`, `winner`, `podium`, `position` and `dnf`.
7. Only after those checks should the newly trained bundle be considered a pace/weather/circuit-aware experimental model.

## v0.5 — priority-weighted pole prediction

- Pole inference now blends the selected FP1/FP2/FP3 trained classifier's field-relative ranking with an explicit priority-weighted evidence score.
- The hierarchy prioritises current-weekend pace (28%), constructor strength (23%), driver form (16%), teammate-relative performance (11%), circuit suitability (8%) and historical pole record (5%). Weather is supplied to the trained model as contextual input; raw weather values are not directly ranked as intrinsically good or bad for a driver.
- Pole API refuses to return a live ranking when no eligible practice classifications or lap-level practice data have been ingested. The historical-only legacy output is no longer presented as a live pole forecast.
- Prediction rows expose component scores and the prediction basis for auditability. Relative score shares remain explicitly uncalibrated.
- Frontend files are unchanged.


## v0.6 — race outcome ensemble and Constructor Intelligence

- Added `src/features/constructor_intelligence.py` with 22 lagged/current-event constructor and driver features.
- Constructor Intelligence now reports six implemented pillars: qualifying pace, race pace, long-run pace, tyre behaviour, reliability and circuit suitability. The page shows component coverage instead of `PLANNED` placeholders.
- Practice pace aggregation now separates the fastest relative gap observed across the weekend from the latest-session gap; the latter protects the model from carrying an earlier FP leader over a later-session reversal. FP1/FP2/FP3 cutoffs remain isolated.
- Added per-compound lap-time slope versus tyre age, minimum stint sample requirements, recent-five-GP constructor/driver DNF and mechanical DNF rates, same-circuit constructor finish/position-gain/DNF history, and recent OpenF1 pit-duration summaries where driver-team mapping is available.
- Added circuit and global prior Safety Car/VSC event rates from prior OpenF1 race-control messages. Sparse circuit estimates shrink toward prior global history; missing race-control data remain unavailable.
- Race inference now blends ML position/winner rankings with the agreed expert-priority hierarchy and computes field-coherent winner/podium estimates through 4,000 Monte Carlo scenarios. DNF risk and validation residual spread are incorporated.
- Safety Car/VSC rates compress simulated pace gaps and increase outcome uncertainty; wet forecast context also increases uncertainty. The simulator does not predict neutralisation deployment timing or explicitly optimise pit windows/restarts.
- Added current Constructor Intelligence metrics to the API and replaced the six planned UI placeholders with actual per-pillar coverage and constructor comparisons. Added race prediction table columns for expected finish, win, podium and DNF probabilities.
- Training accepts `openf1_pit.csv` and `openf1_drivers.csv`; these files are optional and missing coverage remains null. The OpenF1 sync already persists these raw endpoint tables.
- Added regression tests for historical-only reliability/SC/VSC features, global SC fallback, pit-stop/team mapping, FP cutoff isolation, field-coherent win/podium probabilities and Safety Car scenario uncertainty.
- 61 automated tests pass in the build environment. JSX syntax transpilation passes using the installed TypeScript compiler. A full Vite build was not run because npm dependency installation timed out in this environment.

### Known limitations

- No real-data retraining or accuracy validation has been performed: the uploaded v0.5 project did not contain your local full historical timing, practice, pit-stop, weather and race-control archives.
- Race pace and tyre age remain observed proxies without fuel/traffic/engine-mode correction. Pit-stop duration is not a strategy simulator.
- The circuit model learns historical grid conversion; no explicit Turn 1 geometry or slipstream physics model exists.
- Safety Car/VSC scenarios change field compression and outcome variance only; they do not model exact deployment, restart strategy or pit-window decisions.
- Constructor feature coverage depends on having complete and correctly mapped local data. Retrain the model after restoring and syncing those datasets.


## v0.8 — practice archive, progressive pole forecasts and metric interpretation

- Practice Intelligence now uses a historical GP selector and returns separate FP1/FP2/FP3 per-driver fastest-lap records. Official Jolpica classifications outrank FastF1/OpenF1 fastest-lap reconstructions; lap-derived records supplement missing sessions and drivers.
- Added a checkpointed FastF1 practice backfill for 2018–2026. It writes after each session, resumes from existing usable rows, logs failures and handles sprint weekends where FP2/FP3 do not exist.
- API startup and retraining now rebuild canonical OpenF1 lap/weather/grid tables from local raw and processed exports. Raw and processed endpoint data are merged with endpoint-specific deduplication so raw-only rows are not hidden by stale processed files.
- OpenF1 lap normalisation preserves drivers without an acronym using driver-number identity; practice reconstruction no longer collapses those rows.
- Pole Predictor has a dedicated historical-only `pole_pre_fp` model and separate `pole_fp1`, `pole_fp2`, `pole_fp3` models. API cutoffs require FP1; FP1+FP2; or FP1+FP2+FP3 respectively. Pre-FP input omits current practice and observed session weather.
- Before-qualifying race prediction supports PRE_FP or the selected FP cutoff and injects the Pole Predictor ranking as the grid while leaving qualifying positions/times missing.
- Constructor table labels and helper descriptions explain each available field; whole metric columns without usable numeric observations are hidden.
- Added regression tests for raw/processed OpenF1 reconciliation, acronym-missing drivers, Jolpica classification precedence, pre-FP weather cutoffs, practice reconstruction, and dedicated staged pole models.
- 73 automated tests pass. Python compilation and frontend JSX syntax transpilation pass. Full Vite build was not run because frontend dependencies are not installed in the build workspace.

### Known limitations

- The 2018–2026 historical practice archive is not bundled. Run the FastF1 backfill command locally; access and sprint-session availability vary. OpenF1's historical free tier starts in 2023.
- FastF1 collector results are per-driver fastest valid laps reconstructed from lap timing, not a separately published official classification. Official Jolpica session classifications take precedence.
- The new pre-FP / FP-stage models require retraining locally after historical practice backfill. Existing v0.7 artifacts do not contain `pole_pre_fp` or the new FP-stage feature schemas.
- Passing tests establish code behaviour, not live predictive accuracy. Pole/race performance must be assessed with chronological unseen-GP backtesting and probability calibration before describing forecasts as accurate.
