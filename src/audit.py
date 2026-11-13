"""Step 2: dataset audit (brief section 3.1) for combined_gee_pm25.csv.
Read-only: prints a report and saves audit.csv.
Station = sensor_name. The GEE 'id' is only a location tag, never a station ID or a feature."""
import warnings

import pandas as pd

from src import config
from src.common import detect_columns, development_rows, load_raw, parse_dates, require, save_rows

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

df, path = load_raw()
found = detect_columns(df.columns)
require(found, ["date", "pm25", "station_id", "lat", "lon"])
df = development_rows(df, found["date"])
rows = []


def add(section, metric, value):
    rows.append((section, metric, value))
    print(f"  {metric}: {value}")


def head(title):
    print(f"\n--- {title}")


date_c, st_c, lat_c, lon_c, pm_c = (found[k] for k in ("date", "station_id", "lat", "lon", "pm25"))
gee_c = "id" if "id" in df.columns else None

# 1. Basics -------------------------------------------------------------
head("1. Basics")
add("basics", "file", path.name)
add("basics", "scope", f"Development dates before {config.TEST_START}")
add("basics", "rows", len(df))
add("basics", "columns", df.shape[1])
add("basics", "column_names", ", ".join(df.columns))
for k, v in found.items():
    add("basics", f"detected_{k}", v)
add("manual", "source", "https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026")
add("manual", "license", "Reported upstream terms: OpenAQ CC BY 4.0 where applicable (see provider terms); Copernicus Sentinel open access; NASA Earth science open data. Combined Kaggle release license text: unverified.")
add("manual", "units", "pm25 ug/m3; temperature_celsius degC; pressure_mb hPa; wind_u and wind_v m/s; NO2, CO, O3 mol/m2 (Sentinel-5P); AOD dimensionless (MODIS) - per the Kaggle dataset description")

# 2. Timestamps ---------------------------------------------------------
head("2. Timestamps")
d = parse_dates(df[date_c])
add("time", "date_min", d.min().date() if d.notna().any() else "NA")
add("time", "date_max", d.max().date() if d.notna().any() else "NA")
add("time", "distinct_days", int(d.dt.normalize().nunique()))
if "datetime_utc" in df.columns:
    dt = parse_dates(df["datetime_utc"])
    both = d.notna() & dt.notna()
    add("time", "datetime_utc_unparseable", int(dt.isna().sum()))
    add("time", "datetime_utc_not_midnight", int((dt[both] != dt[both].dt.normalize()).sum()))
    add("time", "date_vs_datetime_utc_day_mismatch", int((d[both].dt.normalize() != dt[both].dt.normalize()).sum()))
for yr, g in d.dropna().groupby(d.dropna().dt.year):
    months = g.dt.month.nunique()
    add("time", f"year_{yr}_rows_and_months_present", f"{len(g)} rows, {months}/12 months")

# 3. Spatial coverage ---------------------------------------------------
head("3. Spatial coverage")
add("space", "sensor_names", int(df[st_c].nunique()))
add("space", "sensor_name_missing", int(df[st_c].isna().sum()))
if gee_c:
    add("space", "gee_ids", int(df[gee_c].nunique()))
add("space", "lat_min_max", f"{df[lat_c].min():.4f} to {df[lat_c].max():.4f}")
add("space", "lon_min_max", f"{df[lon_c].min():.4f} to {df[lon_c].max():.4f}")
add("space", "coords_missing", int(df[[lat_c, lon_c]].isna().any(axis=1).sum()))
bad = ((df[lat_c].abs() > 90) | (df[lon_c].abs() > 180)).sum()
add("space", "coords_out_of_valid_range", int(bad))

# 4. Target -------------------------------------------------------------
head("4. Target (PM2.5)")
y = pd.to_numeric(df[pm_c], errors="coerce")
valid = (y > config.PM25_MIN) & (y < config.PM25_MAX)
yv = y[valid]
add("target", "pm25_missing", int(y.isna().sum()))
add("target", "pm25_non_numeric_text", int((y.isna() & df[pm_c].notna()).sum()))
add("target", f"pm25_<=_{config.PM25_MIN}", int((y <= config.PM25_MIN).sum()))
add("target", f"pm25_>=_{config.PM25_MAX}", int((y >= config.PM25_MAX).sum()))
add("target", "valid_rows_after_sentinel_rule", int(valid.sum()))
add("target", "valid_min_median_max", f"{yv.min():.2f} / {yv.median():.2f} / {yv.max():.2f}")
add("target", "valid_share_>25", f"{(yv > 25).mean() * 100:.1f}%")
add("target", "valid_share_>50", f"{(yv > 50).mean() * 100:.1f}%")
add("target", "valid_skewness", round(float(yv.skew()), 2))

# 5. Missing values -----------------------------------------------------
head("5. Missing values (% per column, raw)")
for c, pct in df.isna().mean().mul(100).round(2).items():
    rows.append(("missing_pct", c, pct))
print(df.isna().mean().mul(100).round(2).sort_values(ascending=False).head(15).to_string())
print("  (full list is in audit.csv)")

head("5b. AOD / NO2 / CO / O3 missing % by year (valid-PM2.5 rows)")
sat = [c for c in ("AOD", "NO2", "CO", "O3") if c in df.columns]
if sat:
    miss = df.loc[valid, sat].isna().groupby(d[valid].dt.year).mean().mul(100).round(1)
    print(miss.to_string())
    for yr, r in miss.iterrows():
        for c, v in r.items():
            rows.append(("missing_pct_by_year", f"{c}_{int(yr)}", v))

# 6. Duplicates ---------------------------------------------------------
head("6. Duplicates (raw data)")
add("duplicates", "exact_duplicate_rows_beyond_first", int(df.duplicated().sum()))
key = [date_c, lat_c, lon_c, st_c]
add("duplicates", "key_duplicate_rows_beyond_first_[date+lat+lon+sensor_name]", int(df.duplicated(subset=key).sum()))
add("duplicates", "rows_beyond_one_per_sensor_per_day", int(df.duplicated(subset=[st_c, date_c]).sum()))
if gee_c:
    add("duplicates", "rows_beyond_one_per_gee_id_per_day", int(df.duplicated(subset=[gee_c, date_c]).sum()))

head("6b. Repeated sensor-day groups (valid-PM2.5 rows only)")
v = df[valid]
g = v[v.duplicated([st_c, date_c], keep=False)].copy()
add("duplicates", "valid_rows_in_repeated_sensor_day_groups", len(g))
if len(g):
    g["_pm"] = y[g.index]
    g["_loc"] = g[lat_c].round(4).astype(str) + "_" + g[lon_c].round(4).astype(str)
    grp = g.groupby([st_c, date_c])
    add("duplicates", "repeated_sensor_day_groups", grp.ngroups)
    add("duplicates", "groups_with_different_pm25", int((grp["_pm"].nunique() > 1).sum()))
    add("duplicates", "groups_with_different_coordinates", int((grp["_loc"].nunique() > 1).sum()))
    if gee_c:
        add("duplicates", "groups_with_different_gee_id", int((grp[gee_c].nunique() > 1).sum()))
    top = grp.size().reset_index(name="n").groupby(st_c).size().nlargest(5)
    add("duplicates", "sensors_with_most_repeated_groups", "; ".join(f"{k}={n}" for k, n in top.items()))

# 7. Rows and coverage per sensor ---------------------------------------
head("7. Rows and day-coverage per sensor_name (valid-PM2.5 rows)")
vv = v.assign(_d=d[valid].dt.normalize())
per = vv.groupby(st_c).agg(rows=(pm_c, "size"), days=("_d", "nunique"), start=("_d", "min"), end=("_d", "max"))
per["span"] = (per["end"] - per["start"]).dt.days + 1
per["coverage"] = per["days"] / per["span"]
add("per_sensor", "rows_min/median/max", f"{per['rows'].min()} / {int(per['rows'].median())} / {per['rows'].max()}")
add("per_sensor", "days_min/median/max", f"{per['days'].min()} / {int(per['days'].median())} / {per['days'].max()}")
add("per_sensor", "day_coverage_min/median/max", f"{per['coverage'].min():.2f} / {per['coverage'].median():.2f} / {per['coverage'].max():.2f}")
add("per_sensor", "sensors_with_fewer_than_100_days", int((per["days"] < 100).sum()))
add("per_sensor", "smallest_5_by_days", "; ".join(f"{k}={r}" for k, r in per["days"].nsmallest(5).items()))
add("per_sensor", "largest_5_by_days", "; ".join(f"{k}={r}" for k, r in per["days"].nlargest(5).items()))

# 8. Sensor / coordinate / GEE-id reliability ---------------------------
head("8. Sensor / coordinate / GEE-id reliability")
loc = df[[st_c, lat_c, lon_c] + ([gee_c] if gee_c else [])].copy()
loc[lat_c], loc[lon_c] = loc[lat_c].round(4), loc[lon_c].round(4)
pairs = loc[[st_c, lat_c, lon_c]].drop_duplicates()
multi_loc = pairs.groupby(st_c).size()
multi_loc = multi_loc[multi_loc > 1]
add("reliability", "sensors_with_more_than_one_lat_lon_(rounded_4dp)", len(multi_loc))
gmax, gmin = df.groupby(st_c)[[lat_c, lon_c]].max(), df.groupby(st_c)[[lat_c, lon_c]].min()
spread = (gmax - gmin).max(axis=1).sort_values(ascending=False)
far = spread[spread > 0.05]
add("reliability", "sensors_with_coordinate_spread_>0.05_deg", len(far))
add("reliability", "sensors_with_coordinate_spread_>0.25_deg", int((spread > 0.25).sum()))
if len(far):
    add("reliability", "largest_spreads_deg", "; ".join(f"{k}={s:.3f}" for k, s in far.head(8).items()))
shared = pairs.groupby([lat_c, lon_c])[st_c].nunique()
add("reliability", "lat_lon_pairs_shared_by_several_sensor_names", int((shared > 1).sum()))
if gee_c:
    ids_per_sensor = loc.groupby(st_c)[gee_c].nunique()
    ids_per_sensor = ids_per_sensor[ids_per_sensor > 1]
    sensors_per_id = loc.groupby(gee_c)[st_c].nunique()
    sensors_per_id = sensors_per_id[sensors_per_id > 1]
    coords_per_id = loc[[gee_c, lat_c, lon_c]].drop_duplicates().groupby(gee_c).size()
    add("reliability", "sensors_with_more_than_one_gee_id", len(ids_per_sensor))
    if len(ids_per_sensor):
        add("reliability", "those_sensors_(n_ids)", "; ".join(f"{k}={n}" for k, n in ids_per_sensor.items()))
    add("reliability", "gee_ids_shared_by_several_sensors", len(sensors_per_id))
    add("reliability", "gee_ids_with_more_than_one_lat_lon", int((coords_per_id > 1).sum()))

# 9. Target-derived inputs ---------------------------------------------
head("9. Possible target-derived columns")
words = ("pm25", "pm2", "aqi", "categ", "lag")
by_name = [c for c in df.columns if c != pm_c and any(w in c.lower().replace(".", "").replace("_", "") for w in words)]
num = df.select_dtypes("number").drop(columns=[pm_c], errors="ignore")[valid]
corr = num.corrwith(y[valid]).abs()
by_corr = corr[corr > 0.95].index.tolist()
add("leakage", "flagged_by_name", ", ".join(by_name) or "none")
add("leakage", "flagged_by_|corr|>0.95", ", ".join(by_corr) or "none")

# Go / no-go hints ------------------------------------------------------
head("Go / no-go hints (you confirm by hand)")
print("  Spatial CV grouped by sensor_name:",
      "GO" if len(far) == 0 else f"CHECK: {len(far)} sensor_name(s) span several locations (section 8)")
print("  Temporal CV:", "GO" if d.notna().mean() > 0.99 else "CHECK: many unparseable dates")

out = save_rows(rows, "audit.csv")
print(f"\nSaved {out.relative_to(config.ROOT)}")
