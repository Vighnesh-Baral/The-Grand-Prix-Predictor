
from pathlib import Path
import shutil
import re
import pandas as pd

path = Path("data/processed/practice_results.csv")
backup = path.with_name("practice_results.pre_identity_patch.csv")

if not path.exists():
    raise FileNotFoundError(path)

df = pd.read_csv(path)

def normalize(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in {"none", "none none", "nan", "null", ""}:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())

corrections = [
    (2024, 13, "BEA", "Oliver Bearman"),
    (2024, 15, "SHW", "Robert Shwartzman"),
    (2024, 20, "OWA", "Patricio O'Ward"),
    (2024, 20, "ANT", "Andrea Kimi Antonelli"),
    (2024, 20, "BEA", "Oliver Bearman"),
    (2024, 20, "SHW", "Robert Shwartzman"),
    (2024, 20, "DRU", "Felipe Drugovich"),
]

for season, round_no, code, name in corrections:
    mask = (
        (df["season"] == season)
        & (df["round"] == round_no)
        & (df["session"] == "FP1")
        & (df["driver_code"] == code)
    )

    if mask.sum() != 1:
        raise ValueError(
            f"Expected one record for {season} R{round_no} {code}; "
            f"found {mask.sum()}"
        )

    df.loc[mask, "driver_name"] = name
    df.loc[mask, "driver_key"] = f"name:{normalize(name)}"
    df.loc[mask, "driver_identity_status"] = "name_key_only"

# Resolve 2026 Barcelona HER.
her_mask = (
    (df["season"] == 2026)
    & (df["round"] == 7)
    & (df["session"] == "FP1")
    & (df["driver_code"] == "HER")
)

if her_mask.sum() != 1:
    raise ValueError("Expected exactly one Barcelona HER record.")

known_her = df[
    (df["season"] == 2026)
    & (df["driver_code"] == "HER")
    & df["driver_name"].map(normalize).ne("")
]

if set(known_her["driver_name"].map(normalize)) != {"coltonherta"}:
    raise ValueError("HER driver identity is inconsistent.")

if set(known_her["constructor_name"].map(normalize)) != {"cadillac"}:
    raise ValueError("HER constructor identity is inconsistent.")

df.loc[her_mask, "driver_name"] = "Colton Herta"
df.loc[her_mask, "driver_key"] = "name:coltonherta"
df.loc[her_mask, "driver_identity_status"] = "name_key_only"

df.loc[her_mask, "constructor_name"] = "Cadillac"
df.loc[her_mask, "constructor_key"] = "season:2026:team:cadillac"
df.loc[her_mask, "constructor_identity_status"] = "season_name_key_only"

# Final integrity checks before writing.
if len(df) != 10279:
    raise ValueError(f"Unexpected row count: {len(df)}")

if df["driver_key"].isna().any():
    raise ValueError("Some driver keys are still missing.")

if df["constructor_key"].isna().any():
    raise ValueError("Some constructor keys are still missing.")

# Preserve the original CSV.
if not backup.exists():
    shutil.copy2(path, backup)

df.to_csv(path, index=False)

print("IDENTITY PATCH SUCCESSFUL")
print("Rows retained:", len(df))
print("Unresolved drivers:", (df["driver_identity_status"] == "unresolved").sum())
print("Unresolved constructors:", (df["constructor_identity_status"] == "unresolved").sum())
print("Backup:", backup)
