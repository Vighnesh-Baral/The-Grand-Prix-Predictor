"""Conservative canonical identities for mixed Jolpica/TIF1 practice data.

External Ergast/Jolpica IDs remain nullable. Internal keys are project-owned
identifiers and must never be presented as upstream IDs.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


_PLACEHOLDER_NAMES = {
    "", "none", "nonenone", "nan", "null", "na", "n/a", "unknown",
    "unknowndriver", "notavailable", "notprovided", "missing", "-", "--",
}


def normalize_identity_name(value: Any) -> str:
    """Normalize a real display name; return empty for null/placeholder labels."""
    if value is None or pd.isna(value):
        return ""
    raw = str(value).strip().casefold()
    if raw in {"", "none", "nan", "null", "<na>", "n/a", "na", "unknown", "-", "--", r"\n"}:
        return ""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]", "", value.casefold())
    return "" if normalized in _PLACEHOLDER_NAMES else normalized


def resolve_season_code_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing/placeholder names from unambiguous same-season driver-code history.

    A code is resolved only when all valid rows for that code in the season agree
    on exactly one full driver name. Constructor metadata is filled independently
    and only when those rows agree on exactly one constructor name. No upstream
    IDs are created or inferred.
    """
    result = frame.copy()
    for col in ("season", "driver_code", "driver_name", "constructor_name"):
        if col not in result:
            result[col] = pd.NA
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["_code_key"] = result["driver_code"].map(normalize_identity_name)
    result["_name_key"] = result["driver_name"].map(normalize_identity_name)
    result["_team_key"] = result["constructor_name"].map(normalize_identity_name)

    valid_names = result[
        result["_code_key"].ne("")
        & result["_name_key"].ne("")
        & result["_name_key"].ne(result["_code_key"])
    ]
    name_counts = valid_names.groupby(["season", "_code_key"])["_name_key"].nunique()
    name_rows = valid_names.drop_duplicates(["season", "_code_key", "_name_key"])
    name_lookup = {
        (row["season"], row["_code_key"]): row["driver_name"]
        for _, row in name_rows.iterrows()
    }
    missing_name = result["_name_key"].eq("") | result["_name_key"].eq(result["_code_key"])
    for idx, row in result[missing_name].iterrows():
        key = (row["season"], row["_code_key"])
        if row["_code_key"] and name_counts.get(key, 0) == 1:
            result.at[idx, "driver_name"] = name_lookup[key]

    valid_teams = result[
        result["_code_key"].ne("") & result["_team_key"].ne("")
    ]
    team_counts = valid_teams.groupby(["season", "_code_key"])["_team_key"].nunique()
    team_rows = valid_teams.drop_duplicates(["season", "_code_key", "_team_key"])
    team_lookup = {
        (row["season"], row["_code_key"]): row["constructor_name"]
        for _, row in team_rows.iterrows()
    }
    for idx, row in result[result["_team_key"].eq("")].iterrows():
        key = (row["season"], row["_code_key"])
        if row["_code_key"] and team_counts.get(key, 0) == 1:
            result.at[idx, "constructor_name"] = team_lookup[key]

    return result.drop(columns=["_code_key", "_name_key", "_team_key"])


# These are display-name aliases for the same constructor entity within a season.
# The season remains part of constructor_key, so historic teams are not merged
# into today's entrant identity.
_CONSTRUCTOR_ALIASES = {
    "alphatauri": "racingbulls",
    "rbf1team": "racingbulls",
    "racingbulls": "racingbulls",
    "redbull": "redbullracing",
    "redbullracing": "redbullracing",
    "forceindia": "forceindia",
    "astonmartin": "astonmartin",
    "alfaromeoracing": "alfaromeo",
    "alfaromeo": "alfaromeo",
    "sauber": "sauber",
    "kicksauber": "sauber",
    "stakef1teamkicksauber": "sauber",
}


def _has_non_placeholder_value(series: pd.Series) -> pd.Series:
    """Return a mask for values that are present and not null-like strings."""
    text = series.astype("string").str.strip().str.casefold()
    return series.notna() & text.notna() & ~text.isin({"", "nan", "none", "null", "<na>", r"\n", "n/a", "na"})


def add_canonical_identities(frame: pd.DataFrame) -> pd.DataFrame:
    """Add provider-independent identity keys without changing source IDs.

    A verified full driver name is the cross-source identity anchor. Upstream IDs
    are deliberately *not* used as canonical keys because providers may publish
    different identifiers for the same person (for example, opaque ``driver_*``
    IDs in some Jolpica alpha practice extracts and Ergast-style IDs in results).
    Names are normalized conservatively (case, accents, punctuation only); no
    fuzzy matching or code-only identity inference is performed.
    """
    result = frame.copy()
    for column in ("season", "driver_id", "driver_code", "driver_name", "constructor_id", "constructor_name"):
        if column not in result:
            result[column] = pd.NA

    name_key = result["driver_name"].map(normalize_identity_name)
    code_key = result["driver_code"].map(normalize_identity_name)
    valid_name = name_key.ne("") & name_key.ne(code_key)

    # Full normalized names are stable across supported providers. Preserve
    # driver_id verbatim in its original column for provenance and API use.
    result["driver_name_normalized"] = name_key.where(valid_name, pd.NA)
    result["driver_key"] = ("name:" + name_key).where(valid_name, pd.NA)
    has_driver_id = _has_non_placeholder_value(result["driver_id"])
    result["driver_identity_status"] = "unresolved"
    result.loc[~valid_name & has_driver_id, "driver_identity_status"] = "unresolved_id_only"
    result.loc[valid_name & ~has_driver_id, "driver_identity_status"] = "name_key_only"
    result.loc[valid_name & has_driver_id, "driver_identity_status"] = "name_key_with_source_id"

    season = pd.to_numeric(result["season"], errors="coerce")
    team_name = result["constructor_name"].map(normalize_identity_name)
    team_alias = team_name.map(lambda value: _CONSTRUCTOR_ALIASES.get(value, value))
    valid_team = team_alias.ne("") & season.notna()
    result["constructor_name_normalized"] = team_alias.where(valid_team, pd.NA)
    result["constructor_key"] = pd.NA
    result.loc[valid_team, "constructor_key"] = (
        "season:" + season.loc[valid_team].astype(int).astype(str)
        + ":team:" + team_alias.loc[valid_team]
    )
    has_constructor_id = _has_non_placeholder_value(result["constructor_id"])
    result["constructor_identity_status"] = "unresolved"
    result.loc[~valid_team & has_constructor_id, "constructor_identity_status"] = "unresolved_id_only"
    result.loc[valid_team & ~has_constructor_id, "constructor_identity_status"] = "season_name_key_only"
    result.loc[valid_team & has_constructor_id, "constructor_identity_status"] = "season_name_key_with_source_id"
    return result
