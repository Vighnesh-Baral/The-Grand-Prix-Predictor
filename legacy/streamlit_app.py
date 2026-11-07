"""Streamlit dashboard for F1 Race Intelligence."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, PROCESSED_DIR
from src.data.jolpica_client import JolpicaClient, JolpicaAPIError
from src.features.build_features import build_event_prediction_frame
from src.historical.gp_history import build_gp_history
from src.historical.standings import (
    constructor_standings_frame, driver_standings_frame,
    local_constructor_standings, local_driver_standings,
)
from src.models.predict import ModelBundleError, load_model_bundle, predict_pole, predict_race

st.set_page_config(page_title="F1 Race Intelligence", page_icon="🏎️", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def load_local_table(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def get_jolpica_client() -> JolpicaClient:
    return JolpicaClient()


@st.cache_data(ttl=3600, show_spinner=False)
def get_standings(season: int, round_number: int | None, table_type: str) -> pd.DataFrame:
    client = get_jolpica_client()
    if table_type == "drivers":
        return driver_standings_frame(client.get_driver_standings(season, round_number))
    return constructor_standings_frame(client.get_constructor_standings(season, round_number))


st.markdown(
    """
    <style>
    .main-title {font-size:2.25rem;font-weight:800;letter-spacing:.02em;margin-bottom:0;}
    .subtitle {color:#8b949e;margin-top:.15rem;margin-bottom:1.1rem;}
    div[data-testid="stMetric"] {background:rgba(128,128,128,.07);padding:12px;border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="main-title">🏎️ F1 RACE INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Season-aware Formula 1 predictions, historical GP records and championship analytics</div>', unsafe_allow_html=True)

races = load_local_table("races.csv")
race_results = load_local_table("race_results.csv")
qualifying = load_local_table("qualifying_results.csv")
practice = load_local_table("practice_results.csv")
sprints = load_local_table("sprint_results.csv")

if races.empty or race_results.empty:
    st.warning("Historical data has not been downloaded into this project yet.")
    st.markdown("Run these commands from the project root to build the local dataset:")
    st.code("python -m src.data.build_dataset --start-year 2018 --end-year 2026\npython -m src.models.train_models")
    st.info("To request practice classifications where Jolpica has them, add `--include-practice` to the first command. The alpha practice endpoint has incomplete historical coverage.")
    st.stop()

races["season"] = pd.to_numeric(races["season"], errors="coerce")
races["round"] = pd.to_numeric(races["round"], errors="coerce")
seasons = sorted(races["season"].dropna().astype(int).unique().tolist())
if not seasons:
    st.error("The local race schedule contains no valid season values.")
    st.stop()

with st.sidebar:
    st.header("Event selection")
    selected_season = st.selectbox("Season", seasons, index=len(seasons) - 1)
    season_events = races[races["season"] == selected_season].sort_values("round")
    event_labels = {
        f"Round {int(row.round):02d} · {row.race_name}": int(row.round)
        for row in season_events.itertuples(index=False)
        if pd.notna(row.round)
    }
    if not event_labels:
        st.error("No events found for the selected season.")
        st.stop()
    selected_label = st.selectbox("Grand Prix", list(event_labels))
    selected_round = event_labels[selected_label]
    selected_event = season_events[season_events["round"] == selected_round].iloc[0]
    st.caption(f"{selected_event.get('circuit_name', 'Circuit not listed')} · {selected_event.get('country', '')}")
    st.divider()
    st.caption("Prediction timing")
    st.write("Qualifying mode: before qualifying")
    st.write("Race mode: after qualifying, before the race")

try:
    bundle = load_model_bundle(ARTIFACTS_DIR / "f1_models.joblib")
except ModelBundleError as exc:
    bundle = None
    model_error = str(exc)
else:
    model_error = None

pole_tab, race_tab, history_tab, standings_tab, practice_tab = st.tabs([
    "🎯 Qualifying / Pole", "🏁 Race Prediction", "📚 GP History", "🏆 Standings", "⏱️ Practice Analysis"
])

with pole_tab:
    st.subheader("Pole position prediction")
    st.caption("Pre-qualifying prediction. The model excludes the selected event's qualifying result and uses practice pace plus information from earlier rounds only.")
    if bundle is None:
        st.warning(model_error)
        st.code("python -m src.models.train_models")
    else:
        try:
            event_features = build_event_prediction_frame(
                selected_season, selected_round, race_results, qualifying, practice, races, sprints, stage="pole"
            )
            pole_predictions = predict_pole(bundle, event_features)
            if pole_predictions.empty:
                st.info("No driver candidates are available for this event.")
            else:
                first = pole_predictions.iloc[0]
                col1, col2 = st.columns(2)
                col1.metric("Model's pole favourite", first["driver_name"])
                col2.metric("Relative pole score", f"{first['pole_score_share']:.1%}")
                pole_display = pole_predictions.assign(
                    relative_pole_score_share=lambda d: d["pole_score_share"].map(lambda x: f"{x:.1%}"),
                    raw_pole_classifier_score=lambda d: d["pole_probability"].map(lambda x: f"{x:.1%}"),
                ).rename(columns={
                    "driver_name": "Driver", "constructor_name": "Constructor",
                    "relative_pole_score_share": "Relative pole score share",
                    "raw_pole_classifier_score": "Raw pole classifier score",
                })
                st.dataframe(
                    pole_display[["Driver", "Constructor", "Relative pole score share", "Raw pole classifier score"]],
                    use_container_width=True, hide_index=True,
                )
                if event_features["fp_session_count"].notna().sum() == 0:
                    st.info("No practice classifications were found for this event. This prediction therefore relies on historical driver, constructor and circuit form; it is not an FP-informed prediction.")
                st.caption("Relative pole score is each driver’s share of the model scores within this field; it is not a calibrated probability. The raw binary classifier score is shown separately.")
        except Exception as exc:
            st.error(f"Could not build a pole prediction for this event: {exc}")

with race_tab:
    st.subheader("Race outcome prediction")
    st.caption("Post-qualifying prediction. Grid and qualifying information are allowed; race outcomes from the selected event are never used as features.")
    if bundle is None:
        st.warning(model_error)
        st.code("python -m src.models.train_models")
    else:
        try:
            event_features = build_event_prediction_frame(
                selected_season, selected_round, race_results, qualifying, practice, races, sprints, stage="race"
            )
            has_quali = not qualifying.empty and ((pd.to_numeric(qualifying["season"], errors="coerce") == selected_season) & (pd.to_numeric(qualifying["round"], errors="coerce") == selected_round)).any()
            if not has_quali:
                st.warning("Qualifying results for this event are not in the local data. The model can produce a provisional form-based estimate, but it is not a proper post-qualifying prediction.")
            race_predictions = predict_race(bundle, event_features)
            display = race_predictions.copy()
            display["Expected position"] = display["expected_classified_position"].map(lambda x: f"{x:.1f}")
            for column in ("win_score_share", "win_probability", "podium_probability", "dnf_probability"):
                display[column] = display[column].map(lambda x: f"{x:.1%}")
            display = display.rename(columns={
                "predicted_order": "Predicted order", "driver_name": "Driver", "constructor_name": "Constructor",
                "win_score_share": "Win score share", "win_probability": "Raw win score", "podium_probability": "Podium estimate", "dnf_probability": "DNF risk estimate",
            })
            st.dataframe(display[["Predicted order", "Driver", "Constructor", "Expected position", "Win score share", "Raw win score", "Podium estimate", "DNF risk estimate"]], use_container_width=True, hide_index=True)
            st.caption("Expected position is conditional on being classified. Win score share is normalized within this field, not a calibrated event probability. Raw classifier scores, podium estimates and DNF risk should be treated as experimental until calibrated on larger chronological backtests.")
        except Exception as exc:
            st.error(f"Could not build race predictions for this event: {exc}")

with history_tab:
    st.subheader(f"{selected_event.get('race_name', 'Grand Prix')} · historical record")
    circuit_id = selected_event.get("circuit_id")
    history = build_gp_history(
        races, race_results, qualifying, str(circuit_id), first_season=2018,
        race_name=str(selected_event.get("race_name", "")), practice_results=practice,
    )
    if history.empty:
        st.info("No historical results found for this circuit in the local dataset.")
    else:
        st.dataframe(history.sort_values("season", ascending=False), use_container_width=True, hide_index=True)
        st.caption("Historical records are matched to the selected Grand Prix title at the selected circuit, so different events held at the same venue are not mixed together.")

with standings_tab:
    st.subheader(f"{selected_season} championship standings")
    completed_rounds = sorted(
        pd.to_numeric(
            race_results.loc[pd.to_numeric(race_results["season"], errors="coerce") == selected_season, "round"],
            errors="coerce",
        ).dropna().astype(int).unique().tolist()
    )
    snapshot_options: dict[str, int | None] = {"Final / latest available standings": None}
    for round_no in completed_rounds:
        snapshot_options[f"After Round {round_no}"] = round_no
    if not completed_rounds:
        st.info("No completed race rounds are available for this season in the local dataset.")
    snapshot_label = st.selectbox("Standings snapshot", list(snapshot_options), key="standings_snapshot")
    round_snapshot = snapshot_options[snapshot_label]
    left, right = st.columns(2)
    with left:
        st.markdown("**Drivers' Championship**")
        try:
            drivers_df = get_standings(selected_season, round_snapshot, "drivers")
            if drivers_df.empty:
                local_df = local_driver_standings(race_results, sprints, selected_season, round_snapshot)
                if local_df.empty:
                    st.info("No driver standings are available for this snapshot.")
                else:
                    st.warning("The API returned no table; showing locally calculated points, wins and podiums. Official tie-break rules may differ.")
                    st.dataframe(local_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(drivers_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            local_df = local_driver_standings(race_results, sprints, selected_season, round_snapshot)
            if local_df.empty:
                st.error(f"Could not retrieve official standings and no local snapshot is available: {exc}")
            else:
                st.warning("Live API unavailable; showing locally calculated points, wins and podiums. Official tie-break rules may differ.")
                st.dataframe(local_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Constructors' Championship**")
        try:
            constructors_df = get_standings(selected_season, round_snapshot, "constructors")
            if constructors_df.empty:
                local_df = local_constructor_standings(race_results, sprints, selected_season, round_snapshot)
                if local_df.empty:
                    st.info("No constructor standings are available for this snapshot.")
                else:
                    st.warning("The API returned no table; showing locally calculated constructor points, wins and podiums. Official tie-break rules may differ.")
                    st.dataframe(local_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(constructors_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            local_df = local_constructor_standings(race_results, sprints, selected_season, round_snapshot)
            if local_df.empty:
                st.error(f"Could not retrieve official standings and no local snapshot is available: {exc}")
            else:
                st.warning("Live API unavailable; showing locally calculated constructor points, wins and podiums. Official tie-break rules may differ.")
                st.dataframe(local_df, use_container_width=True, hide_index=True)
    st.caption("Standings are fetched for the selected season and round snapshot, not inferred from final-season points.")

with practice_tab:
    st.subheader("Free Practice classifications")
    st.caption("Practice classifications are optional and coverage depends on the source. Historical GP records include the fastest recorded driver per published practice session (FP1/FP2/FP3, or combined FP where that is what the source provides). These classifications are not full telemetry or long-run pace analysis.")
    if practice.empty:
        st.info("No practice data in the local project. Re-run ingestion with `--include-practice` to request Jolpica's alpha FP results where available.")
        st.code("python -m src.data.build_dataset --start-year 2018 --end-year 2025 --include-practice")
    else:
        p = practice[(pd.to_numeric(practice["season"], errors="coerce") == selected_season) & (pd.to_numeric(practice["round"], errors="coerce") == selected_round)]
        if p.empty:
            st.info("No FP1/FP2/FP3 or combined FP classifications are available for this event in the local data.")
        else:
            session_names = sorted(p["session"].dropna().unique().tolist())
            session_choice = st.selectbox("Practice session", session_names)
            st.dataframe(p[p["session"] == session_choice].sort_values("position"), use_container_width=True, hide_index=True)

if bundle is not None:
    st.divider()
    st.subheader("Model evaluation snapshot")
    metadata = bundle.get("metadata", {})
    data_meta = metadata.get("data", {})
    metric_cols = st.columns(3)
    metric_cols[0].metric("Training rows", f"{data_meta.get('rows', 0):,}")
    metric_cols[1].metric("Seasons", len(data_meta.get("seasons", [])))
    metric_cols[2].metric("Practice event coverage", f"{data_meta.get('practice_coverage_fraction', 0):.1%}")
    with st.expander("Validation and holdout metrics"):
        st.json(metadata.get("tasks", {}))
