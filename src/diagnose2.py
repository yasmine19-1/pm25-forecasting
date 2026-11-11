"""Second read-only diagnostic: repeated sensor-days, coordinate timelines,
GEE date regularity, anatomy of the rows combined_gee_pm25.csv lost.

Run:  python -m src.diagnose2 | tee /tmp/diagnose2_output.txt
Nothing is modified. Each section is wrapped so one failure does not stop the rest.
"""
import traceback
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from src import config

warnings.filterwarnings("ignore")
A_FILE, B_FILE, C_FILE = "pm25.csv", "gee_features_daily.csv", "combined_gee_pm25.csv"
THRESH_DEG = 0.25
FEATS = ["temperature_celsius", "pressure_mb", "wind_u", "wind_v", "NO2", "CO", "O3", "AOD"]


def find(name):
    hits = [p for p in config.RAW.rglob(name) if "__MACOSX" not in p.parts]
    if not hits:
        raise SystemExit(f"{name} not found under data/raw/")
    return hits[0]


def day(s):
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def title(t):
    print("\n" + "=" * 8, t)


def safe(fn):
    try:
        fn()
    except Exception as e:
        print(f"  !! section failed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


A = pd.read_csv(find(A_FILE), low_memory=False)
B = pd.read_csv(find(B_FILE), low_memory=False)
C = pd.read_csv(find(C_FILE), low_memory=False)
for df in (A, B, C):
    df["date"] = day(df["date"])
A["coord"] = A["latitude"].round(6).astype(str) + "," + A["longitude"].round(6).astype(str)
_cache = {}


def funnel():
    """Same reason-coding as diagnose.py, plus the distance of each row's nearest GEE point."""
    if "a" in _cache:
        return _cache["a"]
    a = A.copy()
    y = pd.to_numeric(a["pm25"], errors="coerce")
    a["invalid"] = y.isna() | (y <= 0) | (y >= 500)
    a["repeated_key"] = a.groupby(["sensor_name", "date"])["pm25"].transform("size") > 1
    sites = a[["sensor_name", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    loc = B.drop_duplicates("id")[["id", "lat", "lon"]].reset_index(drop=True)
    dm = np.sqrt((sites["latitude"].values[:, None] - loc["lat"].values[None, :]) ** 2 +
                 (sites["longitude"].values[:, None] - loc["lon"].values[None, :]) ** 2)
    sites["gee_id"] = loc["id"].values[dm.argmin(axis=1)]
    sites["dist_deg"] = dm.min(axis=1)
    a = a.merge(sites, on=["sensor_name", "latitude", "longitude"], how="left")
    bk = B.drop_duplicates(["id", "date"]).copy()
    bk["b_complete"] = bk[FEATS].notna().all(axis=1)
    a = a.merge(bk[["id", "date", "b_complete"]].rename(columns={"id": "gee_id"}), on=["gee_id", "date"], how="left")
    a["b_exists"] = a["b_complete"].notna()
    a["in_C"] = pd.MultiIndex.from_frame(a[["sensor_name", "date"]]).isin(
        pd.MultiIndex.from_frame(C[["sensor_name", "date"]]))
    a["reason"] = np.select(
        [a["invalid"], a["dist_deg"] > THRESH_DEG, ~a["b_exists"], a["b_exists"] & (a["b_complete"] == False)],
        ["1 invalid pm25", "3 no GEE id within 0.25 deg", "4 no GEE row on that date", "5 GEE row, NaN features"],
        default="6 complete join possible")
    _cache["a"] = a
    return a


def d1_repeats():
    title("1. Repeated (sensor_name, date) in A: which sensors, same or different coordinates?")
    size = A.groupby(["sensor_name", "date"])["pm25"].transform("size")
    dup = A[size > 1]
    gp = dup.groupby(["sensor_name", "date"]).agg(
        n=("pm25", "size"), ncoord=("coord", "nunique"), lo=("pm25", "min"), hi=("pm25", "max")).reset_index()
    gp["diff"] = gp["hi"] - gp["lo"]
    print(f"  groups: {len(gp)} | rows involved: {len(dup)} | groups whose rows share ONE coordinate pair: "
          f"{int((gp['ncoord'] == 1).sum())} | groups spanning >1 coordinate pair: {int((gp['ncoord'] > 1).sum())}")
    summ = gp.groupby("sensor_name").agg(
        groups=("date", "size"), max_rows_in_group=("n", "max"),
        groups_diff_coords=("ncoord", lambda s: int((s > 1).sum())),
        median_pm_diff=("diff", "median"), max_pm_diff=("diff", "max")).sort_values("groups", ascending=False)
    print(f"  sensors with repeats: {len(summ)}")
    print(summ.head(15).round(2).to_string())
    print(f"  top-3 sensors account for {summ['groups'].head(3).sum() / len(gp):.0%} of all repeated groups")


def d2_coord_timelines():
    title("2. Sensors with more than one coordinate pair: timeline of each pair")
    g = A.groupby(["sensor_name", "latitude", "longitude"])["date"].agg(first="min", last="max", rows="size").reset_index()
    n = g.groupby("sensor_name").size()
    for s in n[n > 1].index:
        sub = g[g["sensor_name"] == s].sort_values("first")
        print(f"\n  {s}")
        dates = {}
        for r in sub.itertuples():
            print(f"    ({r.latitude:.6f}, {r.longitude:.6f})  {r.first.date()} -> {r.last.date()}  rows={r.rows}")
            dates[(r.latitude, r.longitude)] = set(A.loc[(A["sensor_name"] == s) & (A["latitude"] == r.latitude)
                                                          & (A["longitude"] == r.longitude), "date"])
        for p, q in combinations(dates, 2):
            print(f"    dates present under BOTH ({p[0]:.4f},{p[1]:.4f}) and ({q[0]:.4f},{q[1]:.4f}): {len(dates[p] & dates[q])}")


def d3_date_regularity():
    title("3. Is the GEE file daily? Date gaps per id (B) and per sensor (A)")
    b = B.sort_values(["id", "date"])
    diff = b.groupby("id")["date"].diff().dt.days.dropna()
    print("  B: gap between consecutive dates of the same id (days) -> share of all gaps")
    print("    " + ", ".join(f"{k}d: {(diff == k).mean():.1%}" for k in (1, 2, 3)) +
          f", 4-7d: {diff.between(4, 7).mean():.1%}, >7d: {(diff > 7).mean():.1%}")
    per = b.groupby("id")["date"].agg(first="min", last="max", rows="size")
    per["span"] = (per["last"] - per["first"]).dt.days + 1
    per["coverage_%"] = per["rows"] / per["span"] * 100
    print(f"  B per id: span in days min/median/max = {per['span'].min()} / {int(per['span'].median())} / {per['span'].max()}")
    print(f"  B per id: rows/span (%) min/median/max = {per['coverage_%'].min():.0f} / {per['coverage_%'].median():.0f} / {per['coverage_%'].max():.0f}")
    print("  B first-date year per id:", per["first"].dt.year.value_counts().sort_index().to_dict())
    print("  B last-date year per id: ", per["last"].dt.year.value_counts().sort_index().to_dict())
    a = A.drop_duplicates(["sensor_name", "date"]).sort_values(["sensor_name", "date"])
    da = a.groupby("sensor_name")["date"].diff().dt.days.dropna()
    print("  A: gap between consecutive dates of the same sensor -> share: "
          f"1d {(da == 1).mean():.1%}, 2d {(da == 2).mean():.1%}, 3-7d {da.between(3, 7).mean():.1%}, >7d {(da > 7).mean():.1%}")


def d4_lost_complete_rows():
    title("4. Rows that COULD be joined completely but are missing from C")
    a = funnel()
    r6 = a[a["reason"] == "6 complete join possible"]
    print("  distance to nearest GEE point (deg), rows in C vs not in C:")
    print(r6.groupby("in_C")["dist_deg"].describe(percentiles=[.5, .9, .99]).round(4).to_string())
    miss = r6[~r6["in_C"]]
    print(f"\n  missing rows: {len(miss):,} | sensors involved: {miss['sensor_name'].nunique()}")
    top = miss.groupby("sensor_name").size().sort_values(ascending=False).head(10)
    tot = r6.groupby("sensor_name").size()
    print("  top sensors (missing rows / that sensor's joinable rows):")
    for s, n in top.items():
        print(f"    {s}: {n} / {tot[s]}")
    print(f"  of the missing rows, those whose (sensor_name, date) is repeated in A: {int(miss['repeated_key'].sum()):,} of {len(miss):,}"
          f"  (for comparison, joinable rows that ARE in C: {r6.loc[r6['in_C'], 'repeated_key'].mean():.1%} repeated)")
    wholly = set(miss["sensor_name"]) - set(C["sensor_name"])
    print(f"  sensors with joinable rows but NO row at all in C: {len(wholly)}")
    print("  missing rows by year:", miss.groupby(miss["date"].dt.year).size().to_dict())


def d5_no_gee_row_by_year():
    title("5. Share of each year's rows by reason (A rows, all sensors)")
    a = funnel()
    ct = pd.crosstab(a["date"].dt.year, a["reason"], normalize="index").mul(100).round(1)
    print(ct.to_string())


def d6_s5p_missing_by_year():
    title("6. NO2 / CO / O3 missingness in B by year (%)")
    m = B[["NO2", "CO", "O3"]].isna().groupby(B["date"].dt.year).mean().mul(100).round(1)
    print(m.to_string())


for fn in (d1_repeats, d2_coord_timelines, d3_date_regularity, d4_lost_complete_rows,
           d5_no_gee_row_by_year, d6_s5p_missing_by_year):
    safe(fn)
print("\nDone. Nothing was modified. Paste this output back.")
