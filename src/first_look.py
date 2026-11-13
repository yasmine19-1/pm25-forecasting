"""Step 1: print every column (dtype, % missing, example) of every raw CSV."""
import pandas as pd

from src import config
from src.common import detect_columns, development_rows, find_raw_file

files = sorted(
    p for p in config.RAW.rglob("*.csv")
    if "__MACOSX" not in p.parts and not p.name.startswith(".")
)
if not files:
    raise SystemExit("No CSV in data/raw/. Unzip the Kaggle download there first.")

combined = find_raw_file()
for f in files:
    df = pd.read_csv(f, low_memory=False)
    found = detect_columns(df.columns, use_overrides=f == combined)
    if found["date"] is None:
        print(f"Skipping {f.name}: no date column to verify development scope.")
        continue
    if f.name == "gee_features_daily.csv":
        found["station_id"] = None  # Its id is a GEE location, not a sensor.
    df = development_rows(df, found["date"])
    print(f"\n=== {f.name}: {len(df):,} development rows x {df.shape[1]} columns")
    print(pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_%": df.isna().mean().mul(100).round(1),
        "example": df.iloc[0] if len(df) else None,
    }).to_string())
    print("exact duplicate rows:", df.duplicated().sum())
    print("\nAuto-detected key columns (None = not found):")
    for k, v in found.items():
        print(f"  {k:<11} <- {v}")
