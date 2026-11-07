# Model Card — F1 Race Intelligence v0.9.3

## Intended use

Educational and portfolio use: research, analysis and pre-event forecasting experiments. This is not a professional betting model. No claim of production-grade predictive accuracy is made.

## Prediction tasks and information cutoffs

| Task | Prediction cutoff | Allowed information |
|---|---|---|
| Pole / qualifying order | After the selected practice cutoff (FP1, FP2 or FP3), before qualifying | Prior-event history, prior qualifying pace, prior race pace, and only practice laps completed by the selected cutoff |
| Race | After qualifying, before race start | Prior-event history and pace, current-event practice, current qualifying/grid, and prior race-lap pace |
| Classified position | Same as race task | Evaluated only for classified finishers |
| DNF risk | Same as race task | Evaluated only for valid starter/on-track retirement labels; administrative outcomes and non-starters are excluded |

Current-event qualifying results and grid are never inputs to the pole task. Current-event race outcomes are never inputs to an event prediction.

## Modeling philosophy

F1 competitiveness is dynamic and constructor-dependent. This project therefore treats historical finishing positions and championship points as context rather than a sufficient measure of current car speed. Pace-derived features are being introduced as separate signals:

1. **Constructor one-lap pace:** rolling constructor-relative qualifying time gaps, current-season pace, recent-three-event pace and same-circuit pace.
2. **Driver contribution:** rolling qualifying pace and qualifying gap relative to the constructor's teammate baseline.
3. **Weekend one-lap pace:** per-session practice lap-time gaps and soft-compound best-lap gaps through the specified FP cutoff.
4. **Weekend long-run pace:** per-stint median lap-time gaps for practice runs with enough valid laps.
5. **Historical race pace:** FastF1-preferred and OpenF1-fallback race-session long-run gaps and observed stint lap-time slopes, lagged to later events.
6. **Supporting context:** recent driver/constructor results and points, circuit identity/history, smoothed prior circuit/grid conversion rates, and current-event official starting grid (OpenF1 when published; qualifying position explicitly labelled as fallback) for race predictions only.
7. **Race weather:** Open-Meteo previous-day archived forecasts at the scheduled start time for historical model rows; live event forecasts are used only when they fall within the 12–36 hour training lead window. OpenF1 weather through qualifying is a separate observed-weekend context feature.

No single arbitrary 0–100 constructor score is produced by the API. Qualifying pace, race long-run pace, current weekend single-lap pace and same-circuit pace remain distinct measurements. Their usefulness and any eventual combination must be learned and validated on chronological held-out events.

## Feature definitions and limitations

- Qualifying gaps compare each driver's best available Q1/Q2/Q3 time against the event's fastest recorded time. These are relative event gaps, not absolute performance across circuits.
- Constructor qualifying pace is the event mean of the constructor's two driver gaps, then lagged. The same-circuit feature is lagged within constructor × circuit and remains missing when there is no prior visit; it is not filled from an unrelated circuit.
- Practice best-lap and soft-tyre gaps are normalized against the fastest driver in the same session. A soft-tyre lap is only a potential qualifying-pace proxy; it is not guaranteed to be a dedicated qualifying simulation.
- Practice long-run features use valid green-flag laps from stints with at least six observations and ignore the first valid lap. To reduce mixed-tyre confounding, the relative session long-run feature uses the compound with the broadest coverage across at least two drivers (with a deterministic compound priority for ties). These are robust median summaries, not fuel-corrected race simulations.
- Race long-run features use valid green-flag race laps from stints with at least six observations and ignore the first valid lap. Within each event, they retain the tyre compound with the broadest coverage across at least two drivers, avoiding direct comparison of different compounds. `stint_slope` is an observed lap-time slope against tyre life, not a pure tyre-degradation estimate. Fuel burn, traffic, track evolution, engine mode and driver management remain confounders.
- Race fastest-lap records from Jolpica are only one-lap race-session proxies. They are not described as race pace.
- FastF1 `IsAccurate` indicates basic lap timing alignment/quality checks; it does not guarantee that a lap is representative of the car's underlying pace.
- Weather, track temperature, engine modes, fuel load, traffic, tyre age/compound, upgrades, parc fermé constraints, penalties, safety cars, reliability and regulation changes can all affect results. The current feature set does not fully measure all of them.
- Circuit/grid conversion features estimate outcomes by start position at the same circuit from strictly prior races and smooth sparse samples toward overall grid-position rates. They can learn historical pole/P2/P3 conversion patterns (for example, at Monaco or Mexico) but do not directly encode first-corner geometry, tow/slipstream physics, DRS layout, track width, elevation or overtaking difficulty. No arbitrary circuit-specific bonuses are hard-coded.
- Open-Meteo Previous Runs `previous_day1` provides what was forecast 24 hours before valid time. Most forecast models are archived from January 2024; older race rows therefore usually have no archived forecast features. The live forecast endpoint supports a limited forward horizon, but its longer lead times are not fed into the model.

## Cross-source identity resolution (v0.9.3)

- `driver_key` is a project-owned key derived from a normalized full driver name. It is independent of provider-specific `driver_id` values, which remain unchanged in the source columns.
- Unicode diacritics, case and punctuation are normalized. For example, `Nico Hülkenberg` and `Nico Hulkenberg` resolve to the same name key. The implementation does not use fuzzy matching or a three-letter driver code alone.
- `constructor_key` is derived from normalized constructor names and includes the season. Historic constructor identities are not collapsed across seasons.
- Practice-to-result joins are resolved within the exact season and round. An ambiguous name-to-ID mapping is left unmatched. Practice-only and reserve drivers remain in the practice table even when no matching race result exists.
- The migration adds identity metadata to practice, race, qualifying and sprint tables without overwriting original provider IDs. The audit report separates provider ID differences from unmatched event/driver identities.
- An unmatched practice row is not automatically a data error: FP-only or reserve entries may never appear in the race or qualifying results. Review the generated event-level report rather than forcing every practice record to match.

## Leakage controls

- Historical feature values are shifted so each training row only sees strictly earlier events.
- Constructor race results are aggregated across both cars per event before rolling/lagging.
- Constructor qualifying pace is aggregated at event level before rolling/lagging.
- Same-circuit constructor pace is also lagged; the target event's own qualifying result is excluded.
- Pole models are trained separately for FP1, FP2 and FP3. Practice classifications and FastF1 laps are independently cut off for each model; at inference, `as_of_session` restricts the data to sessions completed by the requested cutoff. An FP3 model is never applied to an FP1/FP2 request.
- Race features can use the target event's qualifying/grid only after qualifying. Target-event race laps, fastest lap and race pit stops are not used.
- Data splits remain chronological by season. Validation selects models; the test season is held out for reporting. Event-level top-pick/top-three metrics are reported beside simple baselines (fastest practice time for pole, grid order for race/podium).
- The race weather training feature is an archived forecast with a fixed 24-hour lead. At inference, live forecasts outside a 12–36 hour lead are withheld from the estimator to reduce lead-time distribution shift; the raw forecast can still be retrieved from the weather endpoint.
- Future-event roster precedence is: selected-event race entries (where appropriate), current qualifying entries for race predictions, selected-event practice entries, optional curated `driver_roster.csv`, then the latest completed GP field. The historical fallback is a safety net, not a guarantee of correct driver/team assignments after an off-season transfer; populate the curated roster for a season opener before FP1.

## DNF target definition

`Finished` and `+N Lap(s)` statuses are labels of 0. On-track retirement/other non-finishing statuses are labels of 1. Disqualifications, DNS, withdrawals, non-qualifications and other administrative/non-starter statuses are excluded as missing labels. Status strings are source-taxonomy-dependent and should be audited when source data changes.

## Models and evaluation

The current model-training pipeline still compares Random Forest and XGBoost using chronological training/validation/test splits. It is a baseline model family; the new pace feature engineering is not evidence of improved accuracy by itself. Feature coverage is recorded per model task in `metrics.json`, so users can see whether the model actually had enough historic observations to learn a feature family.

Evaluation should report at least:
- Pole: event-level top-pick accuracy, top-3 recall, precision/recall and average precision.
- Race winner: event-level top-pick accuracy plus precision/recall and average precision.
- Podium: event-level top-3 recall and driver-level precision/recall.
- Position: MAE and within-one-position rate.
- DNF: precision/recall, F1 and average precision, with attention to class balance.

All results must be evaluated on event-held-out/chronological samples. Per-driver row metrics alone can exaggerate performance because the actual prediction decision is made across an entire GP field.

## Current validation status

- Synthetic/unit tests validate separate FP1/FP2/FP3 training cutoffs, practice filtering, common-compound long-run comparisons, lagging, constructor circuit pace, roster fallback/override, live prediction gates, data endpoints and basic training/inference code paths.
- These tests do **not** measure real F1 predictive performance.
- The uploaded project archive did not include the user's local race/qualifying/practice CSVs, FastF1/OpenF1 session-lap history or trained model artifacts. The pace/weather/circuit-aware models therefore have not been retrained or evaluated on those real local datasets in this build.
- Do not claim accuracy improvement until the real local data are copied in, the historical FastF1 sessions are collected, models are retrained and the held-out metrics are reviewed.

## Data attribution and terms

Historical results are retrieved through Jolpica F1. Review its CC BY-NC-SA 4.0 non-commercial terms and provide attribution when redistributing data or derived work. Review Formula 1/FastF1 data-source terms before public or commercial deployment.

## Pole inference hierarchy (v0.6)

The pole inference output combines (a) the trained cutoff-specific classifier's within-field ranking and (b) a transparent, field-relative priority score. The blend is 50/50. The priority score uses these starting design weights: current-weekend pace 28%, constructor strength 23%, driver form 16%, teammate-relative pace 11%, weather context 9%, circuit suitability 8%, and historical pole record 5%. Weather contributes neutral event context to the hierarchy and is also supplied to the supervised model; weather alone is not ranked as beneficial or harmful for a particular driver without driver/car wet-weather interaction evidence. The group weights are renormalised over groups with usable data. These are design weights, not coefficients learned from training, and neither the model score nor its relative score share is a calibrated probability.

A live pole prediction is withheld if the selected event has no eligible practice classification or lap-level practice data. This prevents the historical-only legacy baseline from being displayed as a current-weekend forecast.


## Race prediction and Constructor Intelligence (v0.6)

The race inference system combines a trained finishing-position regressor and winner classifier with the expert-priority ranking, then estimates a field-coherent finishing distribution using 4,000 deterministic-seed Monte Carlo race scenarios. DNF probability is sampled separately; in each simulation, exactly one winner and up to three podium places are allocated. Winner probabilities sum to one and podium probabilities sum to three for a full-sized field. Validation residual spread supplies the position uncertainty scale when available. These win/podium estimates are model-implied simulation frequencies, not externally calibrated probabilities.

Initial expert race-priority weights: constructor performance 20%, weekend race pace 18%, grid/circuit advantage 18%, driver form 12%, tyre/strategy 12%, reliability 7%, circuit characteristics 5% and weather context 8%. The ML and expert-priority field rankings are blended 50/50. These priorities are expert design priors; chronological holdout backtesting must determine whether they improve performance.

Safety Car/VSC history is estimated from prior OpenF1 race-control messages, using a global historical prior when no local circuit sample exists and smoothing sparse local rates. The simulator uses this rate to compress expected pace gaps and increase outcome variance in neutralisation scenarios. It does not predict Safety Car deployment timing or explicitly simulate the restart, pit window or overtaking sequence. Wet race forecasts also increase scenario variance; the forecast itself is supplied to the race ML model only when its lead time meets the historical training window.

The six Constructor Intelligence pillars expose separate metrics and coverage: (1) qualifying pace, (2) race pace, (3) long-run pace, (4) tyre behaviour, (5) reliability and (6) circuit suitability. New constructor signals include current FP latest-session one-lap gap, current-weekend long-run pace/lap/stint coverage, observed soft/medium/hard tyre-age slopes, recent linked OpenF1 pit-stop duration, recent-five-GP constructor/driver DNF rates, same-circuit results and prior SC/VSC rates. Pit-stop metrics require `openf1_pit.csv` and `openf1_drivers.csv`. Missing source coverage remains null rather than being silently converted to a measured zero.

The tyre-age slope is an observed lap-time trend and can be confounded by fuel, traffic, engine mode, track evolution and driver management. Pit duration is a historical timing proxy; a full one-stop/two-stop strategy optimiser, tyre choice model, undercut/overcut model and fuel-corrected race simulator are not implemented. Circuit/grid conversion learns historical outcomes by start position and circuit identity; physical first-corner geometry, slipstream/tow strength, DRS and track-width effects are not explicitly simulated.
