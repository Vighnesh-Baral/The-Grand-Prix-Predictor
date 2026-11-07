
from pathlib import Path
import pandas as pd

DATA = Path("data/processed")

practice = pd.read_csv(DATA / "practice_results.csv")
qualifying = pd.read_csv(DATA / "qualifying_results.csv")
race = pd.read_csv(DATA / "race_results.csv")
races = pd.read_csv(DATA / "races.csv")
sprint = pd.read_csv(DATA / "sprint_results.csv")


def normalize_name(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def audit_join(left, right, right_name):
    print("\n" + "=" * 65)
    print(f"PRACTICE → {right_name.upper()}")

    # Only records with verified external driver IDs.
    left_valid = left.dropna(subset=["driver_id"]).copy()
    right_valid = right.dropna(subset=["driver_id"]).copy()

    keys = ["season", "round", "driver_id"]

    left_valid["driver_id"] = left_valid["driver_id"].astype(str)
    right_valid["driver_id"] = right_valid["driver_id"].astype(str)

    # Check whether the right-side join keys are unique.
    duplicate_keys = right_valid.duplicated(keys, keep=False).sum()

    left_keys = left_valid[keys].drop_duplicates()
    right_keys = right_valid[keys].drop_duplicates()

    matched = left_keys.merge(right_keys, on=keys, how="inner")
    unmatched = left_keys.merge(
        right_keys, on=keys, how="left", indicator=True
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"]

    print("Practice records with external IDs:", len(left_valid))
    print(f"{right_name} records with external IDs:", len(right_valid))
    print("Unique practice driver-event keys:", len(left_keys))
    print("Matched driver-event keys:", len(matched))
    print("Unmatched practice driver-event keys:", len(unmatched))
    print("Duplicate keys in comparison dataset:", duplicate_keys)

    if len(unmatched):
        print("\nSample unmatched practice keys:")
        print(unmatched[keys].head(15).to_string(index=False))

    # Compare names for matched IDs.
    comparison = left_valid.merge(
        right_valid,
        on=keys,
        how="inner",
        suffixes=("_practice", f"_{right_name}"),
    )

    name_left = normalize_name(comparison["driver_name_practice"])
    name_right = normalize_name(comparison[f"driver_name_{right_name}"])

    conflicts = comparison[
        (name_left != "") & (name_right != "") & (name_left != name_right)
    ]

    print("\nDriver-name conflicts for matched IDs:", len(conflicts))

    if len(conflicts):
        print(conflicts[
            keys + ["driver_name_practice", f"driver_name_{right_name}"]
        ].head(15).to_string(index=False))


print("DATASET SIZES")
for name, df in [
    ("Practice", practice),
    ("Qualifying", qualifying),
    ("Race", race),
    ("Races", races),
    ("Sprint", sprint),
]:
    print(f"{name:12}: {len(df):,} rows")

print("\nPRACTICE ID COVERAGE")
print("Missing driver IDs:", practice["driver_id"].isna().sum())
print("Missing constructor IDs:", practice["constructor_id"].isna().sum())

audit_join(practice, qualifying, "qualifying")
audit_join(practice, race, "race")

print("\n" + "=" * 65)
print("EVENT CALENDAR VALIDATION")

practice_events = practice[["season", "round"]].drop_duplicates()
race_events = races[["season", "round"]].drop_duplicates()

event_comparison = practice_events.merge(
    race_events,
    on=["season", "round"],
    how="outer",
    indicator=True,
)

print(event_comparison["_merge"].value_counts().to_string())

print("\nEvents in practice but not calendar:")
print(
    event_comparison[event_comparison["_merge"] == "left_only"]
    [["season", "round"]].to_string(index=False)
)

print("\nEvents in calendar but not practice:")
print(
    event_comparison[event_comparison["_merge"] == "right_only"]
    [["season", "round"]].to_string(index=False)
)

print("\nVALIDATION COMPLETE — NO DATA WAS MODIFIED")
