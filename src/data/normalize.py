"""Flatten Ergast-compatible API records into stable tabular schemas."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "", "\\N") else None
    except (TypeError, ValueError):
        return None


def _driver_parts(driver: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (
        driver.get("driverId"),
        driver.get("code"),
        " ".join(filter(None, [driver.get("givenName"), driver.get("familyName")])) or None,
    )


def _constructor_parts(constructor: dict[str, Any]) -> tuple[str | None, str | None]:
    return constructor.get("constructorId"), constructor.get("name")


def normalize_schedule(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for race in records:
        circuit = race.get("Circuit", {}) or {}
        location = circuit.get("Location", {}) or {}
        rows.append({
            "season": _as_int(race.get("season")),
            "round": _as_int(race.get("round")),
            "race_name": race.get("raceName"),
            "date": race.get("date"),
            "time": race.get("time"),
            "circuit_id": circuit.get("circuitId"),
            "circuit_name": circuit.get("circuitName"),
            "locality": location.get("locality"),
            "country": location.get("country"),
            "latitude": location.get("lat"),
            "longitude": location.get("long"),
            "url": race.get("url"),
        })
    return pd.DataFrame(rows)


def normalize_race_results(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in records:
        race = item.get("race", {}) or {}
        driver = item.get("Driver", {}) or {}
        constructor = item.get("Constructor", {}) or {}
        driver_id, driver_code, driver_name = _driver_parts(driver)
        constructor_id, constructor_name = _constructor_parts(constructor)
        time_info = item.get("Time", {}) or {}
        fastest = item.get("FastestLap", {}) or {}
        fastest_time = fastest.get("Time", {}) or {}
        rows.append({
            "season": _as_int(race.get("season")),
            "round": _as_int(race.get("round")),
            "race_name": race.get("raceName"),
            "date": race.get("date"),
            "circuit_id": (race.get("Circuit", {}) or {}).get("circuitId"),
            "driver_id": driver_id,
            "driver_code": driver_code,
            "driver_name": driver_name,
            "constructor_id": constructor_id,
            "constructor_name": constructor_name,
            "grid": _as_int(item.get("grid")),
            "position": _as_int(item.get("position")),
            "position_order": _as_int(item.get("positionOrder")),
            "points": pd.to_numeric(item.get("points"), errors="coerce"),
            "laps": _as_int(item.get("laps")),
            "status": item.get("status"),
            "time": time_info.get("time"),
            "milliseconds": _as_int(time_info.get("millis")),
            "fastest_lap_rank": _as_int(fastest.get("rank")),
            "fastest_lap_time": fastest_time.get("time"),
        })
    return pd.DataFrame(rows)


def normalize_qualifying(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in records:
        race = item.get("race", {}) or {}
        driver = item.get("Driver", {}) or {}
        constructor = item.get("Constructor", {}) or {}
        driver_id, driver_code, driver_name = _driver_parts(driver)
        constructor_id, constructor_name = _constructor_parts(constructor)
        rows.append({
            "season": _as_int(race.get("season")),
            "round": _as_int(race.get("round")),
            "race_name": race.get("raceName"),
            "date": race.get("date"),
            "circuit_id": (race.get("Circuit", {}) or {}).get("circuitId"),
            "driver_id": driver_id,
            "driver_code": driver_code,
            "driver_name": driver_name,
            "constructor_id": constructor_id,
            "constructor_name": constructor_name,
            "qualifying_position": _as_int(item.get("position")),
            "q1": item.get("Q1"),
            "q2": item.get("Q2"),
            "q3": item.get("Q3"),
        })
    return pd.DataFrame(rows)


def normalize_sprints(records: list[dict[str, Any]]) -> pd.DataFrame:
    return normalize_race_results(records)


def parse_lap_time_seconds(value: Any) -> float | None:
    """Convert a string such as '1:23.456' to seconds."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text in {"\\N", "nan", "NaT"}:
        return None
    # Jolpica alpha may return an object-like string or a plain time string.
    if ":" in text:
        parts = text.split(":")
        try:
            total = 0.0
            for part in parts:
                total = total * 60 + float(part)
            return total
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_practice(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=[
            "season", "round", "race_name", "circuit_id", "session", "position",
            "driver_id", "driver_code", "driver_name", "constructor_id",
            "constructor_name", "time", "time_seconds", "laps", "status", "source",
        ])
    if "time" not in df:
        df["time"] = None
    df["time_seconds"] = df["time"].map(parse_lap_time_seconds)
    for column in ("season", "round", "position", "laps"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
