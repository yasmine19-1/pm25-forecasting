"""Step 3: clean the raw data (brief section 5.1) and log every removal."""
import warnings

import pandas as pd

from src import config
from src.common import detect_columns, load_raw, parse_dates, require, to_snake

warnings.filterwarnings("ignore")
config.ensure_dirs()

df, path = load_raw()
found = detect_columns(df.columns)
require(found, ["date", "pm25"])
canonical = {"date": "date", "station_id": "station_id", "lat": "lat", "lon": "lon", "pm25": "pm25"}

# --- rename columns to lower snake_case and save the mapping ---------------
detected_raw = {v: canonical[k] for k, v in found.items() if v}
used = set(detected_raw.values())
mapping = {}
for c in df.columns:
    if c in detected_raw:
        mapping[c] = detected_raw[c]
    else:
        name, k = to_snake(c), 2
        base = name
        while name in used:
            name, k = f"{base}_{k}", k + 1
        used.add(name)
        mapping[c] = name
df = df.rename(columns=mapping)
pd.DataFrame({"raw_name": list(mapping), "clean_name": list(mapping.values())}).to_csv(
    config.PARTS / "column_mapping.csv", index=False, encoding="utf-8")

df["date"] = parse_dates(df["date"])
raw_pm = df["pm25"]
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
non_numeric = int((df["pm25"].isna() & raw_pm.notna()).sum())

log = []


def step(name, rule, before, after, note=""):
    log.append({"step": name, "rule": rule, "rows_before": before,
                "rows_removed": before - after, "rows_after": after, "note": note})
    print(f"{name:<34} removed {before - after:>8,}   left {after:>9,}   {note}")


n0 = len(df)
log.append({"step": "0_start", "rule": f"raw file {path.name}", "rows_before": n0,
            "rows_removed": 0, "rows_after": n0, "note": ""})
print(f"Start: {n0:,} rows")

# --- invalid targets, in the order required by the brief -------------------
n = len(df)
df = df[df["pm25"].notna()]
step("1_missing_pm25", "pm25 is missing", n, len(df), f"({non_numeric} were non-numeric text)")

n = len(df)
df = df[df["pm25"] > config.PM25_MIN]
step("2_pm25_nonpositive", f"pm25 <= {config.PM25_MIN}", n, len(df))

n = len(df)
df = df[df["pm25"] < config.PM25_MAX]
step("3_pm25_too_high", f"pm25 >= {config.PM25_MAX}", n, len(df))

# --- duplicates -------------------------------------------------------------
n = len(df)
df = df.drop_duplicates()
step("4_exact_duplicates", "identical in all columns, keep first", n, len(df))

key = [c for c in ("date", "lat", "lon", "station_id") if c in df.columns]
n = len(df)
dups = df[df.duplicated(key, keep=False)].copy()
if len(dups):
    dups["_group"] = dups.groupby(key, dropna=False).ngroup()
    dups[dups["_group"] < 20].sort_values("_group").to_csv(
        config.PARTS / "key_duplicates_sample.csv", index=False, encoding="utf-8")
    differ = int((dups.groupby("_group")["pm25"].nunique() > 1).sum())
    print(f"   {dups['_group'].nunique()} groups share the key {key}; {differ} of them have different PM2.5.")
    print("   Sample saved: key_duplicates_sample.csv -> look at it before trusting the rule.")
rule = config.KEY_DUP_RULE
if rule == "first":
    df = df.drop_duplicates(subset=key, keep="first")
elif rule == "mean":
    agg = {c: ("mean" if pd.api.types.is_numeric_dtype(df[c]) else "first")
           for c in df.columns if c not in key}
    df = df.groupby(key, dropna=False, sort=False).agg(agg).reset_index()[df.columns]
elif rule == "drop_all":
    df = df[~df.duplicated(key, keep=False)]
else:
    raise SystemExit("KEY_DUP_RULE must be 'first', 'mean' or 'drop_all'")
step("5_same_key_different_values", f"key {'+'.join(key)}; rule={rule}", n, len(df))

# --- rows we cannot place in time ---------------------------------------------
n = len(df)
df = df[df["date"].notna()]
step("6_unparseable_date", "date could not be parsed", n, len(df))

# --- save --------------------------------------------------------------------
df = df.sort_values([c for c in ("station_id", "date") if c in df.columns]).reset_index(drop=True)
pd.DataFrame(log).to_csv(config.PARTS / "cleaning_log.csv", index=False, encoding="utf-8")
out = config.INTERIM / "clean_full.parquet"
df.to_parquet(out, index=False)

print(f"\nSaved {out.relative_to(config.ROOT)}  ({len(df):,} rows x {df.shape[1]} columns)")
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
if "station_id" in df.columns:
    print(f"Stations: {df['station_id'].nunique()}")
print(f"PM2.5 min/median/max: {df['pm25'].min():.2f} / {df['pm25'].median():.2f} / {df['pm25'].max():.2f}")
print("Saved logs: cleaning_log.csv, column_mapping.csv in", config.PARTS.relative_to(config.ROOT))
