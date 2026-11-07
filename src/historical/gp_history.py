"""Historical Grand Prix winners, pole sitters and podium summaries."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


def _normalise_event_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"^formula\s+1\s+", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def build_gp_history(
    races: pd.DataFrame,
    race_results: pd.DataFrame,
    qualifying_results: pd.DataFrame | None,
    circuit_id: str,
    first_season: int = 2018,
    race_name: str | None = None,
    practice_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return event history, preferring the exact GP title over circuit-only matches.

    Distinct Grands Prix can use the same circuit in one or more seasons (for
    example, a national GP and a second event at the same venue). If race_name is
    provided, only matching event titles are included to avoid mixing those series.
    """
    if races is None or races.empty or race_results is None or race_results.empty:
        return pd.DataFrame()
    season_values = pd.to_numeric(races["season"], errors="coerce")
    circuit_rows = races[(races["circuit_id"].astype(str) == str(circuit_id)) & (season_values >= first_season)].copy()
    if race_name:
        wanted = _normalise_event_name(race_name)
        exact_rows = circuit_rows[circuit_rows["race_name"].map(_normalise_event_name) == wanted]
        event_rows = exact_rows
    else:
        event_rows = circuit_rows
    if event_rows.empty:
        return pd.DataFrame()

    rr = race_results.copy()
    qr = qualifying_results.copy() if qualifying_results is not None else pd.DataFrame()
    pr = practice_results.copy() if practice_results is not None else pd.DataFrame()
    for frame in (rr, qr, pr):
        if not frame.empty:
            frame["season"] = pd.to_numeric(frame.get("season"), errors="coerce")
            frame["round"] = pd.to_numeric(frame.get("round"), errors="coerce")

    output = []
    for event in event_rows.sort_values(["season", "round"]).itertuples(index=False):
        year, round_no = int(event.season), int(event.round)
        race = rr[(rr["season"] == year) & (rr["round"] == round_no)].copy()
        quali = qr[(qr["season"] == year) & (qr["round"] == round_no)].copy() if not qr.empty else pd.DataFrame()
        if not race.empty:
            race["position_order"] = pd.to_numeric(race.get("position_order"), errors="coerce")
            podium = race[race["position_order"].between(1, 3)].sort_values("position_order")
            winner = podium[podium["position_order"] == 1].head(1)
        else:
            podium = pd.DataFrame()
            winner = pd.DataFrame()
        if not quali.empty:
            quali["qualifying_position"] = pd.to_numeric(quali.get("qualifying_position"), errors="coerce")
            pole = quali[quali["qualifying_position"] == 1].head(1)
        else:
            pole = pd.DataFrame()
        practice = pr[(pr["season"] == year) & (pr["round"] == round_no)].copy() if not pr.empty else pd.DataFrame()
        if not practice.empty:
            if "time_seconds" not in practice and "time" in practice:
                from src.data.normalize import parse_lap_time_seconds
                practice["time_seconds"] = practice["time"].map(parse_lap_time_seconds)
            practice["time_seconds"] = pd.to_numeric(practice.get("time_seconds"), errors="coerce")
            practice["position"] = pd.to_numeric(practice.get("position"), errors="coerce")

        def fastest_in(session_code: str) -> str:
            if practice.empty or "session" not in practice:
                return "Not available"
            session_rows = practice[practice["session"].astype(str).str.upper() == session_code.upper()].copy()
            if session_rows.empty:
                return "Not available"
            if session_rows["time_seconds"].notna().any():
                selected_row = session_rows.loc[session_rows["time_seconds"].idxmin()]
            elif session_rows["position"].notna().any():
                selected_row = session_rows.loc[session_rows["position"].idxmin()]
            else:
                return "Not available"
            return str(selected_row.get("driver_name") or "Not available")

        output.append({
            "season": year,
            "grand_prix": event.race_name,
            "circuit": getattr(event, "circuit_name", None),
            "winner": winner.iloc[0].get("driver_name", "Not available") if not winner.empty else "Not available",
            "winner_constructor": winner.iloc[0].get("constructor_name") if not winner.empty else None,
            "pole_sitter": pole.iloc[0].get("driver_name", "Not available") if not pole.empty else "Not available",
            "podium": " · ".join(podium.get("driver_name", pd.Series(dtype=str)).fillna("Unknown").astype(str).tolist()),
            "fp1_fastest": fastest_in("FP1"),
            "fp2_fastest": fastest_in("FP2"),
            "fp3_fastest": fastest_in("FP3"),
            "fp_combined_fastest": fastest_in("FP"),
            "race_date": getattr(event, "date", None),
        })
    return pd.DataFrame(output)
