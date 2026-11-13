"""Step 3: clean the development period of combined_gee_pm25.csv (brief 5.1).
Rule-based steps only: nothing is learned from the data here (no imputation, scaling or outlier thresholds)."""
import warnings

import pandas as pd

from src import config
from src.common import detect_columns, load_raw, parse_dates, require, to_snake

warnings.filterwarnings("ignore")
config.ensure_dirs()

df, path = load_raw()
found = detect_columns(df.columns)
require(found, ["date", "pm25", "station_id", "lat", "lon"])
dates = parse_dates(df[found["date"]])
unknown_dates = int(dates.isna().sum())
is_dev = dates.lt(pd.Timestamp(config.TEST_START))
df = df.loc[is_dev].copy()
df[found["date"]] = dates.loc[is_dev]
canonical = {"date": "date", "station_id": "station_id", "lat": "lat", "lon": "lon", "pm25": "pm25"}

# --- standardise column names (station_id = sensor_name; GEE 'id' -> gee_id) -------------
detected_raw = {v: canonical[k] for k, v in found.items() if v}
used = set(detected_raw.values())
mapping = {}
for c in df.columns:
    if c in detected_raw:
        mapping[c] = detected_raw[c]
    elif c == "id":
        mapping[c] = "gee_id"
        used.add("gee_id")
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

raw_pm = df["pm25"]
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
non_numeric = int((df["pm25"].isna() & raw_pm.notna()).sum())

log = []


def step(name, rule, before, after, note=""):
    log.append({"step": name, "rule": rule, "rows_before": before,
                "rows_removed": before - after, "rows_after": after, "note": note})
    print(f"{name:<34} removed {before - after:>8,}   left {after:>9,}   {note}")


def drop_dups(d, subset, rule):
    if rule == "first":
        return d.drop_duplicates(subset=subset, keep="first")
    if rule == "drop_all":
        return d[~d.duplicated(subset=subset, keep=False)]
    raise SystemExit(f"Duplicate rule must be 'first' or 'drop_all', got {rule!r}")


n0 = len(df)
log.append({"step": "0_start", "rule": f"{path.name}; dated rows before {config.TEST_START}", "rows_before": n0,
            "rows_removed": 0, "rows_after": n0, "note": ""})
print(f"Start: {n0:,} dated development rows")
print(f"Rows with unknown dates excluded before cleaning: {unknown_dates:,}")

# --- invalid targets (brief 5.1: before EDA and splitting) -------------------------------
n = len(df)
df = df[df["pm25"].notna()]
step("1_missing_pm25", "pm25 is missing", n, len(df), f"({non_numeric} were non-numeric text)")

n = len(df)
df = df[df["pm25"] > config.PM25_MIN]
step("2_pm25_nonpositive", f"pm25 <= {config.PM25_MIN}", n, len(df))

n = len(df)
df = df[df["pm25"] < config.PM25_MAX]
step("3_pm25_too_high", f"pm25 >= {config.PM25_MAX}", n, len(df))

# --- duplicates ---------------------------------------------------------------------------
n = len(df)
df = df.drop_duplicates()
step("4_exact_duplicates", "identical in all columns, keep first", n, len(df))

n = len(df)
df = drop_dups(df, ["date", "lat", "lon", "station_id"], config.KEY_DUP_RULE)
step("5_key_duplicates", f"date+lat+lon+station_id; rule={config.KEY_DUP_RULE}", n, len(df))

n = len(df)
clash = df.duplicated(["station_id", "date"], keep=False)
n_groups = df[clash].groupby(["station_id", "date"]).ngroups
df[clash].sort_values(["station_id", "date"]).to_csv(
    config.PARTS / "sensor_day_duplicates.csv", index=False, encoding="utf-8")
df = drop_dups(df, ["station_id", "date"], config.SENSOR_DAY_DUP_RULE)
step("6_same_sensor_same_day", f"station_id+date; rule={config.SENSOR_DAY_DUP_RULE}", n, len(df),
     f"({n_groups} groups, saved in sensor_day_duplicates.csv)")

# --- development outputs -----------------------------------------------------------------
dev = df.sort_values(["station_id", "date"]).reset_index(drop=True)
assert not dev.duplicated(["station_id", "date"]).any()

dev.to_parquet(config.INTERIM / "clean_dev.parquet", index=False)
dev.to_csv(config.INTERIM / "clean_dev.csv", index=False, encoding="utf-8")

# --- logs: cleaning_log.csv + the team's six-column schema ---------------------------------
pd.DataFrame(log).to_csv(config.PARTS / "cleaning_log.csv", index=False, encoding="utf-8")
rows = []
for r in log:
    rows.append(("cleaning", r["step"], "", "rows_removed", r["rows_removed"], r["rule"]))
    rows.append(("cleaning", r["step"], "", "rows_after", r["rows_after"], ""))
rows.append(("date_screening", "unknown_date", "", "rows_excluded", unknown_dates,
             "Could not assign a period; excluded before development cleaning"))
rng = f"{dev['date'].min().date()} to {dev['date'].max().date()}" if len(dev) else ""
rows.append(("split", "development", "", "rows", len(dev), rng))
rows.append(("split", "development", "", "stations", int(dev["station_id"].nunique()), ""))
pd.DataFrame(rows, columns=["section", "variable", "group", "metric", "value", "note"]).to_csv(
    config.PARTS / "cleaning_counts.csv", index=False, encoding="utf-8")

# --- summary --------------------------------------------------------------------------------
print(f"\nDevelopment: {len(dev):,} rows, {dev['station_id'].nunique()} stations, {rng}")
print(f"Columns: {', '.join(dev.columns)}")
print("Saved: data/interim/clean_dev.parquet and .csv")
print("Logs in", config.PARTS.relative_to(config.ROOT), ": cleaning_log.csv, cleaning_counts.csv, column_mapping.csv")
