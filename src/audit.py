"""Step 2: dataset audit (brief section 3.1). Prints a report and saves audit.csv."""
import warnings

import pandas as pd

from src import config
from src.common import detect_columns, load_raw, parse_dates, require, save_rows

warnings.filterwarnings("ignore")

df, path = load_raw()
found = detect_columns(df.columns)
require(found, ["date", "pm25"])
rows = []


def add(section, metric, value):
    rows.append((section, metric, value))
    print(f"  {metric}: {value}")


def head(title):
    print(f"\n--- {title}")


date_c, st_c, lat_c, lon_c, pm_c = (found[k] for k in ("date", "station_id", "lat", "lon", "pm25"))

# 1. Basics -------------------------------------------------------------
head("1. Basics")
add("basics", "file", path.name)
add("basics", "rows", len(df))
add("basics", "columns", df.shape[1])
for k, v in found.items():
    add("basics", f"detected_{k}", v)
add("manual", "source", "https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin20162026")
add("manual", "license", "FILL IN from the Kaggle page (License box)")
add("manual", "pm25_units", "ug/m3 (per project brief)")

# 2. Timestamps ---------------------------------------------------------
head("2. Timestamps")
d = parse_dates(df[date_c])
add("time", "date_missing_raw", int(df[date_c].isna().sum()))
add("time", "date_unparseable", int(d.isna().sum() - df[date_c].isna().sum()))
add("time", "date_min", d.min().date() if d.notna().any() else "NA")
add("time", "date_max", d.max().date() if d.notna().any() else "NA")
add("time", "rows_with_time_of_day_not_midnight", int((d.dropna().dt.normalize() != d.dropna()).sum()))
add("time", "distinct_days", int(d.dt.normalize().nunique()))
full_years = []
for y, g in d.dropna().groupby(d.dropna().dt.year):
    months = g.dt.month.nunique()
    add("time", f"year_{y}_rows_and_months_present", f"{len(g)} rows, {months}/12 months")
    if months == 12:
        full_years.append(int(y))
last_full = max(full_years) if full_years else "none"
add("time", "proposed_test_year_(last full calendar year)", last_full)

# 3. Spatial coverage ---------------------------------------------------
head("3. Spatial coverage")
if st_c:
    add("space", "stations", int(df[st_c].nunique()))
else:
    print("  WARNING: no station-ID column found -> spatial CV by station impossible unless coordinates cluster.")
if lat_c and lon_c:
    add("space", "lat_min_max", f"{df[lat_c].min():.4f} to {df[lat_c].max():.4f}")
    add("space", "lon_min_max", f"{df[lon_c].min():.4f} to {df[lon_c].max():.4f}")
    add("space", "coords_missing", int(df[[lat_c, lon_c]].isna().any(axis=1).sum()))
    bad = ((df[lat_c].abs() > 90) | (df[lon_c].abs() > 180)).sum()
    add("space", "coords_out_of_valid_range", int(bad))
else:
    print("  WARNING: lat/lon column not found.")

# 4. Target -------------------------------------------------------------
head("4. Target (PM2.5)")
y = pd.to_numeric(df[pm_c], errors="coerce")
add("target", "pm25_missing", int(y.isna().sum()))
add("target", "pm25_non_numeric_text", int((y.isna() & df[pm_c].notna()).sum()))
add("target", f"pm25_<=_{config.PM25_MIN}", int((y <= config.PM25_MIN).sum()))
add("target", f"pm25_>=_{config.PM25_MAX}", int((y >= config.PM25_MAX).sum()))
add("target", "pm25_min_median_max", f"{y.min():.2f} / {y.median():.2f} / {y.max():.2f}")

# 5. Missing values -----------------------------------------------------
head("5. Missing values (% per column)")
for c, pct in df.isna().mean().mul(100).round(2).items():
    rows.append(("missing_pct", c, pct))
print(df.isna().mean().mul(100).round(2).sort_values(ascending=False).head(15).to_string())
print("  (full list is in audit.csv)")

# 6. Duplicates ---------------------------------------------------------
head("6. Duplicates (raw data)")
add("duplicates", "exact_duplicate_rows_beyond_first", int(df.duplicated().sum()))
key = [c for c in (date_c, lat_c, lon_c, st_c) if c]
add("duplicates", f"key_duplicate_rows_beyond_first_[{'+'.join(key)}]", int(df.duplicated(subset=key).sum()))
if st_c:
    add("duplicates", "rows_beyond_one_per_station_per_day", int(df.duplicated(subset=[st_c, date_c]).sum()))

# 7. Repeated measures per station -------------------------------------
if st_c:
    head("7. Rows per station (repeated measures)")
    n = df.groupby(st_c).size()
    add("per_station", "rows_min/median/max", f"{n.min()} / {int(n.median())} / {n.max()}")
    add("per_station", "max_over_min_ratio", round(n.max() / max(n.min(), 1), 1))
    add("per_station", "smallest_5", "; ".join(f"{k}={v}" for k, v in n.nsmallest(5).items()))
    add("per_station", "largest_5", "; ".join(f"{k}={v}" for k, v in n.nlargest(5).items()))

# 8. Station <-> coordinates reliability -------------------------------
if st_c and lat_c and lon_c:
    head("8. Station / coordinate reliability")
    pairs = df[[st_c, lat_c, lon_c]].round(4).drop_duplicates()
    many_coords = pairs.groupby(st_c).size()
    many_coords = many_coords[many_coords > 1]
    many_ids = pairs.groupby([lat_c, lon_c])[st_c].nunique()
    many_ids = many_ids[many_ids > 1]
    add("reliability", "stations_with_more_than_one_lat_lon", len(many_coords))
    add("reliability", "lat_lon_pairs_shared_by_several_station_ids", len(many_ids))
    if len(many_coords):
        add("reliability", "example_stations_multi_coords", "; ".join(map(str, many_coords.index[:5])))

# 9. Target-derived inputs ---------------------------------------------
head("9. Possible target-derived columns")
words = ("pm25", "pm2", "aqi", "categ", "lag")
by_name = [c for c in df.columns if c != pm_c and any(w in c.lower().replace(".", "").replace("_", "") for w in words)]
valid = (y > config.PM25_MIN) & (y < config.PM25_MAX)  # ignore sentinel rows so they do not distort r
num = df.select_dtypes("number").drop(columns=[pm_c], errors="ignore")[valid]
corr = num.corrwith(y[valid]).abs()
by_corr = corr[corr > 0.95].index.tolist()
add("leakage", "flagged_by_name", ", ".join(by_name) or "none")
add("leakage", "flagged_by_|corr|>0.95", ", ".join(by_corr) or "none")
print("  -> for each flagged column, check on the Kaggle page how it is defined.")

# Go / no-go hints ------------------------------------------------------
head("Go / no-go hints (you confirm by hand)")
if st_c and lat_c and lon_c:
    ok = len(many_coords) == 0 and len(many_ids) == 0
    print("  Spatial CV by station ID:", "GO (1 station = 1 coordinate pair)" if ok
          else "CHECK: some stations/coordinates are inconsistent -> coordinate clusters or drop spatial CV")
else:
    print("  Spatial CV: station/coordinate columns missing -> NO-GO unless you build clusters.")
print("  Temporal CV:", "GO" if d.notna().mean() > 0.99 else "CHECK: many unparseable dates")

out = save_rows(rows, "audit.csv")
print(f"\nSaved {out.relative_to(config.ROOT)}")
print("Post to the team: date range, test-year proposal (section 2), stations count, canonical column names.")
