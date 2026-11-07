"""Weekend pace engineering from FastF1 lap-level data.

The module deliberately keeps single-lap pace and long-run pace separate.
Practice lap-time gaps are normalized within each session to reduce circuit and
track-evolution effects. Long-run estimates use valid, non-pit laps from stints
with enough observations; they are still proxies because fuel load, engine mode,
traffic and tyre programme are not directly known.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SESSION_ORDER = {"FP1": 1, "FP2": 2, "FP3": 3}
SESSION_FEATURES = [
    f"{session.lower()}_best_gap_pct" for session in SESSION_ORDER
] + [
    f"{session.lower()}_soft_gap_pct" for session in SESSION_ORDER
] + [
    f"{session.lower()}_longrun_gap_pct" for session in SESSION_ORDER
] + [
    "fp_best_gap_pct", "fp_latest_best_gap_pct", "fp_best_soft_gap_pct",
    "fp_latest_longrun_gap_pct", "fp_mean_longrun_gap_pct",
    "fp_valid_laps", "fp_longrun_laps", "fp_longrun_stints", "fp_sessions_completed",
]


def _seconds(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(value.total_seconds())
    except (AttributeError, TypeError, ValueError):
        try:
            return float(pd.to_timedelta(value).total_seconds())
        except (TypeError, ValueError):
            return np.nan


def _truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _normalise_session(value: Any) -> str:
    text = str(value or "").strip().upper().replace("PRACTICE", "FP")
    aliases = {"FP 1": "FP1", "FP 2": "FP2", "FP 3": "FP3", "FREE PRACTICE 1": "FP1",
               "FREE PRACTICE 2": "FP2", "FREE PRACTICE 3": "FP3"}
    return aliases.get(text, text)


def _valid_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Filter unusable laps without claiming to fully correct fuel/traffic effects."""
    frame = laps.copy()
    required = {"lap_time_seconds"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=frame.columns)
    frame["lap_time_seconds"] = pd.to_numeric(frame["lap_time_seconds"], errors="coerce")
    frame = frame[frame["lap_time_seconds"].between(40, 200, inclusive="both")].copy()

    # Prefer FastF1's own accuracy flag. It excludes in/out laps and several
    # timing/track-status anomalies, but is not a statement about representativeness.
    if "is_accurate" in frame.columns and frame["is_accurate"].notna().any():
        frame = frame[frame["is_accurate"].map(_truthy)].copy()
    else:
        for col in ("pit_in", "pit_out"):
            if col in frame.columns:
                frame = frame[~frame[col].map(_truthy)].copy()
        if "track_status" in frame.columns:
            # Conservative fallback: only laps tagged as all-green.
            frame = frame[frame["track_status"].astype(str).str.strip().eq("1")].copy()

    # Even when IsAccurate is true, keep the pace sample to all-green laps if
    # the track-status stream was recorded. This avoids mixing yellow/SC laps.
    if "track_status" in frame.columns and frame["track_status"].notna().any():
        frame = frame[frame["track_status"].astype(str).str.strip().eq("1")].copy()

    if "deleted" in frame.columns:
        frame = frame[~frame["deleted"].map(_truthy)].copy()
    if "fastf1_generated" in frame.columns:
        frame = frame[~frame["fastf1_generated"].map(_truthy)].copy()
    return frame


def _trimmed_median(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan
    if len(vals) >= 5:
        lower, upper = np.quantile(vals, [0.10, 0.90])
        vals = vals[(vals >= lower) & (vals <= upper)]
    return float(np.median(vals)) if len(vals) else np.nan


def _one_session_driver_metrics(laps: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_laps(laps)
    if valid.empty or "driver_code" not in valid.columns:
        return pd.DataFrame(columns=["driver_code", "best_lap_s", "soft_best_lap_s", "longrun_median_s", "valid_laps", "longrun_laps", "longrun_stints"])

    valid["driver_code"] = valid["driver_code"].astype(str).str.upper().str.strip()
    valid = valid[valid["driver_code"].ne("") & valid["driver_code"].ne("NAN")]
    valid["compound"] = valid.get("compound", pd.Series(index=valid.index, dtype=object)).fillna("UNKNOWN").astype(str).str.upper()
    valid["stint"] = pd.to_numeric(valid.get("stint", pd.Series(index=valid.index, dtype=float)), errors="coerce")
    valid["lap_number"] = pd.to_numeric(valid.get("lap_number", pd.Series(index=valid.index, dtype=float)), errors="coerce")

    best = valid.groupby("driver_code")["lap_time_seconds"].min().rename("best_lap_s")
    soft = valid[valid["compound"].eq("SOFT")].groupby("driver_code")["lap_time_seconds"].min().rename("soft_best_lap_s")

    stint_medians: list[dict[str, Any]] = []
    if valid["stint"].notna().any():
        for (driver, stint), group in valid.dropna(subset=["stint"]).groupby(["driver_code", "stint"]):
            group = group.sort_values("lap_number")
            # Ignore the first valid lap in a stint as a conservative warm-up exclusion.
            if len(group) >= 6:
                run = group.iloc[1:]
                median = _trimmed_median(run["lap_time_seconds"])
                if np.isfinite(median):
                    compound = group["compound"].mode().iloc[0] if not group["compound"].mode().empty else "UNKNOWN"
                    stint_medians.append({
                        "driver_code": driver, "stint": stint, "compound": compound,
                        "median_s": median, "laps": len(run),
                    })
    stints = pd.DataFrame(stint_medians)
    if stints.empty:
        longrun = pd.DataFrame(columns=["longrun_median_s", "longrun_laps", "longrun_stints", "longrun_compound"])
        longrun.index.name = "driver_code"
    else:
        # Do not compare a driver's hard-tyre run with another driver's soft-tyre
        # run. Use the compound with the broadest driver coverage in this session;
        # require at least two drivers so the resulting relative gap is meaningful.
        compound_coverage = stints.groupby("compound")["driver_code"].nunique()
        eligible = compound_coverage[compound_coverage >= 2]
        if eligible.empty:
            common_compound = None
            common_stints = stints.iloc[0:0]
        else:
            priority = {"MEDIUM": 3, "HARD": 2, "SOFT": 1, "INTERMEDIATE": 0, "WET": -1}
            common_compound = sorted(eligible.index, key=lambda item: (int(eligible.loc[item]), priority.get(str(item).upper(), -2)), reverse=True)[0]
            common_stints = stints[stints["compound"] == common_compound]
        if common_stints.empty:
            longrun = pd.DataFrame(columns=["longrun_median_s", "longrun_laps", "longrun_stints", "longrun_compound"])
            longrun.index.name = "driver_code"
        else:
            longrun = common_stints.groupby("driver_code").agg(
                longrun_median_s=("median_s", "median"),
                longrun_laps=("laps", "sum"),
                longrun_stints=("stint", "nunique"),
            )
            longrun["longrun_compound"] = common_compound

    result = pd.concat([best, soft, longrun], axis=1).reset_index()
    result["valid_laps"] = valid.groupby("driver_code").size().reindex(result["driver_code"]).to_numpy()
    for col in ("longrun_laps", "longrun_stints"):
        if col not in result:
            result[col] = 0
        result[col] = result[col].fillna(0).astype(int)
    return result


def build_event_weekend_pace_features(
    session_laps: pd.DataFrame | None,
    season: int,
    round_number: int,
    as_of_session: str = "FP3",
) -> pd.DataFrame:
    """Create driver-level features from FP sessions completed by ``as_of_session``.

    ``session_laps`` is the normalized long table written by the FastF1 collector.
    The return key is ``driver_code`` (FastF1's three-letter abbreviation).
    """
    empty = pd.DataFrame(columns=["driver_code", *SESSION_FEATURES])
    if session_laps is None or session_laps.empty:
        return empty
    cutoff = _normalise_session(as_of_session)
    if cutoff not in SESSION_ORDER:
        raise ValueError(f"as_of_session must be one of {', '.join(SESSION_ORDER)}")

    frame = session_laps.copy()
    for col in ("season", "round"):
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    frame["session"] = frame.get("session", pd.Series(index=frame.index, dtype=object)).map(_normalise_session)
    frame = frame[(frame["season"] == season) & (frame["round"] == round_number)]
    frame = frame[frame["session"].isin([s for s, order in SESSION_ORDER.items() if order <= SESSION_ORDER[cutoff]])].copy()
    if frame.empty:
        return empty

    sessions: list[pd.DataFrame] = []
    for session_name, group in frame.groupby("session", sort=False):
        metrics = _one_session_driver_metrics(group)
        if metrics.empty:
            continue
        fastest = metrics["best_lap_s"].min()
        soft_fastest = metrics["soft_best_lap_s"].min(skipna=True)
        longrun_fastest = metrics["longrun_median_s"].min(skipna=True) if "longrun_median_s" in metrics else np.nan
        metrics[f"{session_name.lower()}_best_gap_pct"] = (metrics["best_lap_s"] / fastest - 1) * 100 if pd.notna(fastest) and fastest > 0 else np.nan
        metrics[f"{session_name.lower()}_soft_gap_pct"] = (metrics["soft_best_lap_s"] / soft_fastest - 1) * 100 if pd.notna(soft_fastest) and soft_fastest > 0 else np.nan
        metrics[f"{session_name.lower()}_longrun_gap_pct"] = (metrics["longrun_median_s"] / longrun_fastest - 1) * 100 if pd.notna(longrun_fastest) and longrun_fastest > 0 else np.nan
        metrics["session"] = session_name
        sessions.append(metrics)

    if not sessions:
        return empty
    combined = pd.concat(sessions, ignore_index=True, sort=False)
    for col in SESSION_FEATURES:
        if col not in combined:
            combined[col] = np.nan
    combined["session_order"] = combined["session"].map(SESSION_ORDER)
    combined = combined.sort_values(["driver_code", "session_order"])

    agg_rows: list[dict[str, Any]] = []
    for driver, group in combined.groupby("driver_code"):
        row: dict[str, Any] = {"driver_code": driver}
        for session in SESSION_ORDER:
            g = group[group["session"].eq(session)]
            for feature in (f"{session.lower()}_best_gap_pct", f"{session.lower()}_soft_gap_pct", f"{session.lower()}_longrun_gap_pct"):
                row[feature] = g[feature].iloc[-1] if not g.empty else np.nan
        best_gaps = pd.concat([group[f"{s.lower()}_best_gap_pct"] for s in SESSION_ORDER]).dropna()
        row["fp_best_gap_pct"] = float(best_gaps.min()) if not best_gaps.empty else np.nan
        latest = group.iloc[-1]
        row["fp_latest_best_gap_pct"] = latest.get(f"{latest['session'].lower()}_best_gap_pct", np.nan)
        soft_gaps = pd.concat([group[f"{s.lower()}_soft_gap_pct"] for s in SESSION_ORDER]).dropna()
        row["fp_best_soft_gap_pct"] = float(soft_gaps.min()) if not soft_gaps.empty else np.nan
        longrun_gaps = pd.concat([group[f"{s.lower()}_longrun_gap_pct"] for s in SESSION_ORDER]).dropna()
        row["fp_latest_longrun_gap_pct"] = float(group.dropna(subset=["longrun_median_s"]).iloc[-1].get(f"{group.dropna(subset=['longrun_median_s']).iloc[-1]['session'].lower()}_longrun_gap_pct", np.nan)) if group["longrun_median_s"].notna().any() else np.nan
        row["fp_mean_longrun_gap_pct"] = float(longrun_gaps.mean()) if not longrun_gaps.empty else np.nan
        row["fp_valid_laps"] = int(group["valid_laps"].fillna(0).sum())
        row["fp_longrun_laps"] = int(group["longrun_laps"].fillna(0).sum())
        row["fp_longrun_stints"] = int(group["longrun_stints"].fillna(0).sum())
        row["fp_sessions_completed"] = int(group["session"].nunique())
        agg_rows.append(row)
    return pd.DataFrame(agg_rows).reindex(columns=["driver_code", *SESSION_FEATURES])


def summarise_constructor_pace(driver_features: pd.DataFrame, entrants: pd.DataFrame) -> pd.DataFrame:
    """Aggregate driver pace features to constructors using available drivers."""
    if driver_features.empty or entrants.empty or "driver_code" not in entrants:
        return pd.DataFrame(columns=["constructor_name"])
    entrant_cols = [c for c in ("driver_code", "constructor_id", "constructor_name") if c in entrants]
    feature_cols = [c for c in ("driver_code", *SESSION_FEATURES) if c in driver_features]
    joined = entrants[entrant_cols].merge(
        driver_features[feature_cols], on="driver_code", how="left", validate="one_to_one"
    )
    numeric = [c for c in SESSION_FEATURES if c in joined]
    if not numeric or "constructor_id" not in joined:
        return pd.DataFrame()
    aggregations = {col: (col, "mean") for col in numeric}
    metadata = {"constructor_name": ("constructor_name", "first")} if "constructor_name" in joined else {}
    return joined.groupby("constructor_id", as_index=False).agg(**metadata, **aggregations)


def build_historical_weekend_pace_table(
    session_laps: pd.DataFrame | None, as_of_session: str = "FP3"
) -> pd.DataFrame:
    """Build one training-time FP feature row per event/driver at a given cutoff."""
    columns = ["season", "round", "driver_code", *SESSION_FEATURES]
    if session_laps is None or session_laps.empty:
        return pd.DataFrame(columns=columns)
    frame = session_laps.copy()
    frame["season"] = pd.to_numeric(frame.get("season"), errors="coerce")
    frame["round"] = pd.to_numeric(frame.get("round"), errors="coerce")
    rows: list[pd.DataFrame] = []
    for (season, round_number), group in frame.dropna(subset=["season", "round"]).groupby(["season", "round"]):
        features = build_event_weekend_pace_features(group, int(season), int(round_number), as_of_session)
        if not features.empty:
            features.insert(0, "round", int(round_number))
            features.insert(0, "season", int(season))
            rows.append(features)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)

RACE_PACE_FEATURES = [
    "driver_race_longrun_gap_prev_avg_pct", "driver_race_longrun_gap_recent3_pct",
    "constructor_race_longrun_gap_prev_avg_pct", "constructor_race_longrun_gap_recent3_pct",
    "driver_race_stint_slope_prev_avg_sec_per_lap",
    "constructor_race_stint_slope_prev_avg_sec_per_lap",
    "constructor_race_valid_laps_prev_avg",
]


def build_historical_race_pace_table(
    session_laps: pd.DataFrame | None,
    race_results: pd.DataFrame | None,
    *,
    include_current_event: bool = False,
) -> pd.DataFrame:
    """Build lagged historical race long-run pace and stint-slope features.

    Uses R-session laps only. `stint_slope` is an observed lap-time slope, not a
    pure tyre-degradation estimate: fuel burn, track evolution, traffic and car
    management also affect it. `include_current_event=True` is for inference
    after the final historical race; training always uses the default shift.
    """
    columns = ["season", "round", "driver_id", "driver_code", "constructor_id", *RACE_PACE_FEATURES]
    if session_laps is None or session_laps.empty or race_results is None or race_results.empty:
        return pd.DataFrame(columns=columns)

    laps = session_laps.copy()
    laps["session"] = laps.get("session", pd.Series(index=laps.index, dtype=object)).astype(str).str.upper()
    laps = laps[laps["session"].eq("R")].copy()
    if laps.empty:
        return pd.DataFrame(columns=columns)
    valid = _valid_laps(laps)
    if "compound" in valid.columns:
        valid["compound"] = valid["compound"].fillna("UNKNOWN").astype(str).str.upper()
    # Race analysis is intentionally restricted to green-flag laps where the
    # status code is available. Do not guess a flag state when it is missing.
    if "track_status" in valid.columns and valid["track_status"].notna().any():
        valid = valid[valid["track_status"].astype(str).str.strip().eq("1")].copy()
    if valid.empty:
        return pd.DataFrame(columns=columns)

    mapping_source = race_results.copy()
    for col in ("season", "round", "driver_code", "driver_id", "constructor_id"):
        if col not in mapping_source:
            mapping_source[col] = np.nan
    mapping = mapping_source[["season", "round", "driver_code", "driver_id", "constructor_id"]].drop_duplicates(["season", "round", "driver_code"])
    valid["season"] = pd.to_numeric(valid["season"], errors="coerce")
    valid["round"] = pd.to_numeric(valid["round"], errors="coerce")
    valid["driver_code"] = valid["driver_code"].astype(str).str.upper().str.strip()
    valid = valid.dropna(subset=["season", "round", "driver_code"])
    valid = valid.merge(mapping, on=["season", "round", "driver_code"], how="left")
    valid = valid.dropna(subset=["driver_id", "constructor_id"])
    if valid.empty:
        return pd.DataFrame(columns=columns)

    # Median clean-lap pace is calculated per driver/stint, after dropping the
    # first valid lap of each stint, then aggregated over the race.
    stint_rows: list[dict[str, Any]] = []
    for (season, rnd, driver_id, driver_code, constructor_id, stint), group in valid.dropna(subset=["stint"]).groupby(
        ["season", "round", "driver_id", "driver_code", "constructor_id", "stint"]
    ):
        group = group.sort_values("lap_number")
        if len(group) < 6:
            continue
        run = group.iloc[1:]
        median = _trimmed_median(run["lap_time_seconds"])
        slope = np.nan
        if "tyre_life" in run.columns:
            age = pd.to_numeric(run["tyre_life"], errors="coerce")
            times = pd.to_numeric(run["lap_time_seconds"], errors="coerce")
            ok = age.notna() & times.notna()
            if int(ok.sum()) >= 5 and age.loc[ok].nunique() >= 4:
                slope = float(np.polyfit(age.loc[ok].to_numpy(dtype=float), times.loc[ok].to_numpy(dtype=float), 1)[0])
        if np.isfinite(median):
            stint_rows.append({
                "season": int(season), "round": int(rnd), "driver_id": driver_id,
                "driver_code": driver_code, "constructor_id": constructor_id,
                "stint": stint,
                "compound": group["compound"].mode().iloc[0] if "compound" in group and not group["compound"].mode().empty else "UNKNOWN",
                "stint_median_s": median,
                "stint_slope_sec_per_lap": slope, "valid_laps": len(run),
            })
    stints = pd.DataFrame(stint_rows)
    if stints.empty:
        return pd.DataFrame(columns=columns)

    # Calculate relative race pace on a compound shared by multiple drivers in
    # each GP. Mixed-compound stint medians are not directly comparable.
    common_compound_rows: list[pd.DataFrame] = []
    compound_priority = {"MEDIUM": 3, "HARD": 2, "SOFT": 1, "INTERMEDIATE": 0, "WET": -1}
    for (season_key, round_key), event_stints in stints.groupby(["season", "round"]):
        coverage = event_stints.groupby("compound")["driver_code"].nunique()
        eligible = coverage[coverage >= 2]
        if eligible.empty:
            continue
        selected_compound = sorted(
            eligible.index,
            key=lambda item: (int(eligible.loc[item]), compound_priority.get(str(item).upper(), -2)),
            reverse=True,
        )[0]
        common_compound_rows.append(event_stints[event_stints["compound"] == selected_compound])
    stints = pd.concat(common_compound_rows, ignore_index=True) if common_compound_rows else stints.iloc[0:0]
    if stints.empty:
        return pd.DataFrame(columns=columns)

    driver_event = stints.groupby(["season", "round", "driver_id", "driver_code", "constructor_id"], as_index=False).agg(
        driver_race_longrun_median_s=("stint_median_s", "median"),
        driver_race_stint_slope_sec_per_lap=("stint_slope_sec_per_lap", "median"),
        driver_race_valid_laps=("valid_laps", "sum"),
    )
    event_best = driver_event.groupby(["season", "round"])["driver_race_longrun_median_s"].transform("min")
    driver_event["driver_race_longrun_gap_pct"] = np.where(
        event_best.gt(0), (driver_event["driver_race_longrun_median_s"] / event_best - 1.0) * 100.0, np.nan
    )
    team_event = driver_event.groupby(["season", "round", "constructor_id"], as_index=False).agg(
        constructor_race_longrun_gap_pct=("driver_race_longrun_gap_pct", "mean"),
        constructor_race_stint_slope_sec_per_lap=("driver_race_stint_slope_sec_per_lap", "mean"),
        constructor_race_valid_laps=("driver_race_valid_laps", "sum"),
    ).sort_values(["season", "round"])

    driver_event = driver_event.sort_values(["season", "round"])
    if include_current_event:
        driver_event["_pace_gap"] = driver_event["driver_race_longrun_gap_pct"]
        driver_event["_slope"] = driver_event["driver_race_stint_slope_sec_per_lap"]
        team_event["_pace_gap"] = team_event["constructor_race_longrun_gap_pct"]
        team_event["_slope"] = team_event["constructor_race_stint_slope_sec_per_lap"]
        team_event["_laps"] = team_event["constructor_race_valid_laps"]
    else:
        driver_event["_pace_gap"] = driver_event.groupby("driver_id", sort=False)["driver_race_longrun_gap_pct"].shift(1)
        driver_event["_slope"] = driver_event.groupby("driver_id", sort=False)["driver_race_stint_slope_sec_per_lap"].shift(1)
        team_event["_pace_gap"] = team_event.groupby("constructor_id", sort=False)["constructor_race_longrun_gap_pct"].shift(1)
        team_event["_slope"] = team_event.groupby("constructor_id", sort=False)["constructor_race_stint_slope_sec_per_lap"].shift(1)
        team_event["_laps"] = team_event.groupby("constructor_id", sort=False)["constructor_race_valid_laps"].shift(1)

    driver_event["driver_race_longrun_gap_prev_avg_pct"] = driver_event.groupby("driver_id", sort=False)["_pace_gap"].transform(lambda x: x.expanding(min_periods=1).mean())
    driver_event["driver_race_longrun_gap_recent3_pct"] = driver_event.groupby("driver_id", sort=False)["_pace_gap"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    driver_event["driver_race_stint_slope_prev_avg_sec_per_lap"] = driver_event.groupby("driver_id", sort=False)["_slope"].transform(lambda x: x.expanding(min_periods=1).mean())
    team_event["constructor_race_longrun_gap_prev_avg_pct"] = team_event.groupby("constructor_id", sort=False)["_pace_gap"].transform(lambda x: x.expanding(min_periods=1).mean())
    team_event["constructor_race_longrun_gap_recent3_pct"] = team_event.groupby("constructor_id", sort=False)["_pace_gap"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    team_event["constructor_race_stint_slope_prev_avg_sec_per_lap"] = team_event.groupby("constructor_id", sort=False)["_slope"].transform(lambda x: x.expanding(min_periods=1).mean())
    team_event["constructor_race_valid_laps_prev_avg"] = team_event.groupby("constructor_id", sort=False)["_laps"].transform(lambda x: x.expanding(min_periods=1).mean())

    merged = driver_event.merge(
        team_event[["season", "round", "constructor_id", "constructor_race_longrun_gap_prev_avg_pct",
                    "constructor_race_longrun_gap_recent3_pct", "constructor_race_stint_slope_prev_avg_sec_per_lap",
                    "constructor_race_valid_laps_prev_avg"]],
        on=["season", "round", "constructor_id"], how="left", validate="many_to_one"
    )
    merged = merged.rename(columns={
        "driver_race_longrun_gap_prev_avg_pct": "driver_race_longrun_gap_prev_avg_pct",
        "driver_race_longrun_gap_recent3_pct": "driver_race_longrun_gap_recent3_pct",
    })
    return merged.reindex(columns=columns)
