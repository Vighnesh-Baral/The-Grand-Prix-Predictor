"""Format official Jolpica standings and derive offline standings snapshots."""
from __future__ import annotations

import pandas as pd


def driver_standings_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for item in records:
        driver = item.get("Driver", {}) or {}
        constructors = item.get("Constructors", []) or []
        rows.append({
            "position": item.get("position"),
            "driver": " ".join(filter(None, [driver.get("givenName"), driver.get("familyName")])) or driver.get("driverId"),
            "driver_id": driver.get("driverId"),
            "constructor": constructors[0].get("name") if constructors else "Unknown",
            "points": pd.to_numeric(item.get("points"), errors="coerce"),
            "wins": pd.to_numeric(item.get("wins"), errors="coerce"),
        })
    return pd.DataFrame(rows)


def constructor_standings_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for item in records:
        constructor = item.get("Constructor", {}) or {}
        rows.append({
            "position": item.get("position"),
            "constructor": constructor.get("name") or constructor.get("constructorId"),
            "constructor_id": constructor.get("constructorId"),
            "points": pd.to_numeric(item.get("points"), errors="coerce"),
            "wins": pd.to_numeric(item.get("wins"), errors="coerce"),
        })
    return pd.DataFrame(rows)


def _through_round(frame: pd.DataFrame, season: int, round_number: int | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    data["season"] = pd.to_numeric(data.get("season"), errors="coerce")
    data["round"] = pd.to_numeric(data.get("round"), errors="coerce")
    data = data[data["season"] == int(season)]
    if round_number is not None:
        data = data[data["round"] <= int(round_number)]
    return data


def local_driver_standings(
    race_results: pd.DataFrame,
    sprint_results: pd.DataFrame | None,
    season: int,
    round_number: int | None = None,
) -> pd.DataFrame:
    """Derive points/wins/podiums offline; use official API standings when possible.

    This fallback sums points recorded on race and sprint classifications. The
    official API remains authoritative for edge cases and tie-break rules.
    """
    races = _through_round(race_results, season, round_number)
    if races.empty:
        return pd.DataFrame(columns=["position", "driver", "driver_id", "constructor", "points", "wins", "podiums"])
    for col in ("points", "position_order"):
        races[col] = pd.to_numeric(races.get(col), errors="coerce")
    races["points"] = races["points"].fillna(0)
    races["position_order"] = races["position_order"]
    aggregation = races.groupby("driver_id", dropna=False).agg(
        driver=("driver_name", "last"),
        points=("points", "sum"),
        wins=("position_order", lambda x: int(x.eq(1).sum())),
        podiums=("position_order", lambda x: int(x.between(1, 3).sum())),
    ).reset_index()
    latest_team = races.sort_values(["season", "round"]).groupby("driver_id", dropna=False).tail(1)[
        ["driver_id", "constructor_name"]
    ]
    aggregation = aggregation.merge(latest_team, on="driver_id", how="left")
    aggregation = aggregation.rename(columns={"constructor_name": "constructor"})

    sprints = _through_round(sprint_results if sprint_results is not None else pd.DataFrame(), season, round_number)
    if not sprints.empty:
        sprints["points"] = pd.to_numeric(sprints.get("points"), errors="coerce").fillna(0)
        sprint_points = sprints.groupby("driver_id", dropna=False)["points"].sum().rename("sprint_points").reset_index()
        aggregation = aggregation.merge(sprint_points, on="driver_id", how="left")
        aggregation["sprint_points"] = aggregation["sprint_points"].fillna(0)
        aggregation["points"] = aggregation["points"] + aggregation["sprint_points"]
    else:
        aggregation["sprint_points"] = 0

    aggregation = aggregation.sort_values(["points", "wins", "podiums", "driver"], ascending=[False, False, False, True]).reset_index(drop=True)
    aggregation.insert(0, "position", range(1, len(aggregation) + 1))
    return aggregation[["position", "driver", "driver_id", "constructor", "points", "wins", "podiums", "sprint_points"]]


def local_constructor_standings(
    race_results: pd.DataFrame,
    sprint_results: pd.DataFrame | None,
    season: int,
    round_number: int | None = None,
) -> pd.DataFrame:
    races = _through_round(race_results, season, round_number)
    if races.empty:
        return pd.DataFrame(columns=["position", "constructor", "constructor_id", "points", "wins", "podiums"])
    for col in ("points", "position_order"):
        races[col] = pd.to_numeric(races.get(col), errors="coerce")
    races["points"] = races["points"].fillna(0)
    aggregation = races.groupby("constructor_id", dropna=False).agg(
        constructor=("constructor_name", "last"),
        points=("points", "sum"),
        wins=("position_order", lambda x: int(x.eq(1).sum())),
        podiums=("position_order", lambda x: int(x.between(1, 3).sum())),
    ).reset_index()
    sprints = _through_round(sprint_results if sprint_results is not None else pd.DataFrame(), season, round_number)
    if not sprints.empty:
        sprints["points"] = pd.to_numeric(sprints.get("points"), errors="coerce").fillna(0)
        sprint_points = sprints.groupby("constructor_id", dropna=False)["points"].sum().rename("sprint_points").reset_index()
        aggregation = aggregation.merge(sprint_points, on="constructor_id", how="left")
        aggregation["sprint_points"] = aggregation["sprint_points"].fillna(0)
        aggregation["points"] = aggregation["points"] + aggregation["sprint_points"]
    else:
        aggregation["sprint_points"] = 0
    aggregation = aggregation.sort_values(["points", "wins", "podiums", "constructor"], ascending=[False, False, False, True]).reset_index(drop=True)
    aggregation.insert(0, "position", range(1, len(aggregation) + 1))
    return aggregation[["position", "constructor", "constructor_id", "points", "wins", "podiums", "sprint_points"]]
