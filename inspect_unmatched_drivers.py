
from pathlib import Path
import pandas as pd

DATA = Path("data/processed")

practice = pd.read_csv(DATA / "practice_results.csv")
qualifying = pd.read_csv(DATA / "qualifying_results.csv")
race = pd.read_csv(DATA / "race_results.csv")

keys = ["season", "round", "driver_id"]

def inspect(target, label):
    left = practice.dropna(subset=["driver_id"]).copy()
    right = target.dropna(subset=["driver_id"]).copy()

    left["driver_id"] = left["driver_id"].astype(str)
    right["driver_id"] = right["driver_id"].astype(str)

    right_keys = right[keys].drop_duplicates()
    unique_practice = left.drop_duplicates(keys)

    result = unique_practice.merge(
        right_keys,
        on=keys,
        how="left",
        indicator=True
    )

    unmatched = result[result["_merge"] == "left_only"].copy()

    print("\n" + "=" * 65)
    print(f"UNMATCHED PRACTICE DRIVERS: {label}")

    print("\nCounts by season:")
    print(unmatched.groupby("season").size().to_string())

    print("\nCounts by source:")
    print(unmatched.groupby("source", dropna=False).size().to_string())

    print("\n2025 unmatched records:")
    print(
        unmatched[unmatched["season"] == 2025][
            keys + [
                "driver_code",
                "driver_name",
                "constructor_name",
                "source"
            ]
        ].to_string(index=False)
    )

    print("\nOther unmatched records:")
    print(
        unmatched[unmatched["season"] != 2025][
            keys + [
                "driver_code",
                "driver_name",
                "constructor_name",
                "source"
            ]
        ].to_string(index=False)
    )


inspect(qualifying, "QUALIFYING")
inspect(race, "RACE")

print("\n" + "=" * 65)
print("2025 QUALIFYING DRIVER ID EXAMPLES")
print(
    qualifying[qualifying["season"] == 2025][
        ["driver_id", "driver_code", "driver_name"]
    ].drop_duplicates().head(25).to_string(index=False)
)

print("\n2025 PRACTICE DRIVER ID EXAMPLES")
print(
    practice[practice["season"] == 2025][
        ["driver_id", "driver_code", "driver_name", "source"]
    ].drop_duplicates().head(25).to_string(index=False)
)

print("\nINSPECTION COMPLETE — NO DATA MODIFIED")
