"""Step 1: print every column (dtype, % missing, example) of every raw CSV."""
import pandas as pd

from src import config
from src.common import detect_columns

files = sorted(
    p for p in config.RAW.rglob("*.csv")
    if "__MACOSX" not in p.parts and not p.name.startswith(".")
)
if not files:
    raise SystemExit("No CSV in data/raw/. Unzip the Kaggle download there first.")

for f in files:
    df = pd.read_csv(f, low_memory=False)
    print(f"\n=== {f.name}: {len(df):,} rows x {df.shape[1]} columns")
    print(pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_%": df.isna().mean().mul(100).round(1),
        "example": df.iloc[0],
    }).to_string())
    print("exact duplicate rows:", df.duplicated().sum())
    print("\nAuto-detected key columns (None = not found):")
    for k, v in detect_columns(df.columns).items():
        print(f"  {k:<11} <- {v}")
