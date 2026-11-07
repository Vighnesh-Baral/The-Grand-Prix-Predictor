
from pathlib import Path
import pandas as pd

data_dir = Path("data/processed")

files = [
    "practice_results.csv",
    "qualifying_results.csv",
    "race_results.csv",
    "races.csv",
    "driver_roster.csv",
    "sprint_results.csv",
]

for filename in files:
    path = data_dir / filename

    print("\n" + "=" * 65)
    print(f"FILE: {filename}")

    if not path.exists():
        print("NOT FOUND")
        continue

    df = pd.read_csv(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns ({len(df.columns)}):")
    print(df.columns.tolist())
    print("\nSample:")
    print(df.head(2).to_string(index=False))
