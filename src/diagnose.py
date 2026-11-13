"""Read-only diagnostics for the three raw CSVs. Changes nothing on disk.

Run:  python -m src.diagnose
Then paste the whole printed output back. Each section is wrapped so that one
failure does not stop the others.
"""
import traceback
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from src import config
from src.common import read_development_csv

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


def hav_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def title(t):
    print("\n" + "=" * 8, t)


def safe(fn):
    try:
        fn()
    except Exception as e:  # keep going, report what failed
        print(f"  !! section failed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


A = read_development_csv(find(A_FILE))
B = read_development_csv(find(B_FILE))
C = read_development_csv(find(C_FILE))
for df in (A, B, C):
    df["date"] = day(df["date"])


def s1_overview():
    title("1. Overview of each file")
    for name, df in (("A pm25", A), ("B gee_features_daily", B), ("C combined", C)):
        print(f"\n{name}: {len(df):,} rows x {df.shape[1]} cols | dates {df['date'].min().date()} -> {df['date'].max().date()}"
              f" | unparseable dates: {int(df['date'].isna().sum())}")
        print("  exact duplicate rows:", int(df.duplicated().sum()))
        miss = df.isna().mean().mul(100).round(2)
        print("  columns with missing %:", miss[miss > 0].to_dict() or "none")
    for name, df in (("A", A), ("C", C)):
        y = pd.to_numeric(df["pm25"], errors="coerce")
        print(f"\n{name} pm25: missing={int(y.isna().sum())}  <=0: {int((y <= 0).sum())}  >=500: {int((y >= 500).sum())}"
              f"  min/median/max = {y.min():.2f} / {y.median():.2f} / {y.max():.2f}")


def s2_dropna_hypothesis():
    title("2. Did the author drop rows with missing GEE variables? (B vs C missingness, %)")
    cmp = pd.DataFrame({"B_missing_%": B[FEATS].isna().mean() * 100,
                        "C_missing_%": C[FEATS].isna().mean() * 100}).round(2)
    print(cmp.to_string())
    print(f"  share of C rows with all 8 features present: {C[FEATS].notna().all(axis=1).mean() * 100:.1f}%")
    print(f"  share of B rows with all 8 features present: {B[FEATS].notna().all(axis=1).mean() * 100:.1f}%")


def s3_sensor_sites():
    title("3. A (pm25.csv): sensors, coordinates, duplicates")
    print("  unique sensor names:", A["sensor_name"].nunique(),
          "| unique (lat, lon):", A[["latitude", "longitude"]].drop_duplicates().shape[0])
    g = A.groupby(["sensor_name", "latitude", "longitude"])["date"].agg(["min", "max", "size"]).reset_index()
    n = g.groupby("sensor_name").size()
    multi = n[n > 1].index
    print(f"  sensors with more than one coordinate pair: {len(multi)}")
    rows = []
    for s in multi:
        sub = g[g["sensor_name"] == s]
        la, lo = sub["latitude"].mean(), sub["longitude"].mean()
        spread = hav_m(sub["latitude"], sub["longitude"], la, lo).max()
        rows.append((s, len(sub), round(float(spread)), sub["min"].min().date(), sub["max"].max().date()))
    if rows:
        out = pd.DataFrame(rows, columns=["sensor", "n_coord_pairs", "max_spread_m", "first", "last"])
        print(out.sort_values("max_spread_m", ascending=False).head(15).to_string(index=False))
        print(f"  of these, spread > 1 km (name possibly shared by distant sites): {(out['max_spread_m'] > 1000).sum()}")
    shared = A.groupby(["latitude", "longitude"])["sensor_name"].nunique()
    print("  coordinate pairs used by several sensor names:", int((shared > 1).sum()))
    dd = A[A.duplicated(["sensor_name", "date"], keep=False)]
    if len(dd):
        k = dd.groupby(["sensor_name", "date"])["pm25"].nunique()
        print(f"  (sensor_name, date) groups with >1 row: {len(k)} | same pm25 in all rows: {(k == 1).sum()} | different pm25: {(k > 1).sum()}")
    else:
        print("  no repeated (sensor_name, date)")


def s4_gee_ids():
    title("4. B (gee_features_daily.csv): ids and coordinates")
    print("  unique ids:", B["id"].nunique())
    ncoord = B.groupby("id")[["lat", "lon"]].nunique().max(axis=1)
    print("  ids with more than one lat/lon:", int((ncoord > 1).sum()))
    r = B.groupby("id").size()
    print("  rows per id (min / median / max):", r.min(), int(r.median()), r.max())
    print("  repeated (id, date):", int(B.duplicated(["id", "date"]).sum()))
    loc = B.drop_duplicates("id")[["id", "lat", "lon"]]
    cl = C.drop_duplicates("id")[["id", "latitude", "longitude"]].merge(loc, on="id", how="left")
    d = hav_m(cl["latitude"], cl["longitude"], cl["lat"], cl["lon"])
    print(f"  C sensor coordinates vs B location of the same id (metres): median={np.nanmedian(d):.0f}, max={np.nanmax(d):.0f}")
    print("  -> ~0 m means GEE points were created at sensor coordinates; large means a separate GEE point set.")


def s5_multi_id_sensors():
    title("5. C: sensors with more than one id")
    n_id = C.groupby("sensor_name")["id"].nunique()
    multi = n_id[n_id > 1].index
    print(f"  unique ids: {C['id'].nunique()} | unique sensor names: {C['sensor_name'].nunique()} | sensors with >1 id: {len(multi)}")
    Bi = {i: g.drop_duplicates("date").set_index("date")[FEATS] for i, g in B.groupby("id")}
    for s in multi:
        sub = C[C["sensor_name"] == s]
        for i, j in combinations(sorted(sub["id"].unique()), 2):
            a, b = sub[sub["id"] == i], sub[sub["id"] == j]
            dist = hav_m(a["latitude"].iloc[0], a["longitude"].iloc[0], b["latitude"].iloc[0], b["longitude"].iloc[0])
            ov = sorted(set(a["date"]) & set(b["date"]))
            pm = ""
            if ov:
                pa = a.drop_duplicates("date").set_index("date")["pm25"].loc[ov]
                pb = b.drop_duplicates("date").set_index("date")["pm25"].loc[ov]
                pm = f" | pm25 on overlap days: {pa.round(2).tolist()} vs {pb.round(2).tolist()}"
            common = Bi[i].index.intersection(Bi[j].index) if i in Bi and j in Bi else []
            same = np.nan
            if len(common):
                same = np.isclose(Bi[i].loc[common], Bi[j].loc[common], equal_nan=True).all(axis=1).mean()
            gap = (b["date"].min() - a["date"].max()).days
            print(f"  {s}: ids {i}/{j} | {dist:.0f} m apart | {a['date'].min().date()}..{a['date'].max().date()} then "
                  f"{b['date'].min().date()}..{b['date'].max().date()} (gap {gap} d) | overlap days {len(ov)}{pm}"
                  f" | B dates in common {len(common)}, GEE values identical on {same:.0%} of them")


def s6_attrition():
    title("6. Where do the A rows go? (reason-coded, checked against C by sensor_name + date)")
    a = A.copy()
    y = pd.to_numeric(a["pm25"], errors="coerce")
    a["invalid_pm25"] = y.isna() | (y <= 0) | (y >= 500)
    a["exact_dup"] = a.duplicated(keep="first")
    sites = a[["sensor_name", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    loc = B.drop_duplicates("id")[["id", "lat", "lon"]].reset_index(drop=True)
    dm = np.sqrt((sites["latitude"].values[:, None] - loc["lat"].values[None, :]) ** 2 +
                 (sites["longitude"].values[:, None] - loc["lon"].values[None, :]) ** 2)
    sites["gee_id"] = loc["id"].values[dm.argmin(axis=1)]
    sites["dist_deg"] = dm.min(axis=1)
    print(f"  A sites (sensor+coords): {len(sites)} | nearest-id distance (deg) median={sites['dist_deg'].median():.3f}"
          f" max={sites['dist_deg'].max():.3f} | beyond {THRESH_DEG} deg: {(sites['dist_deg'] > THRESH_DEG).sum()}")
    print(f"  distinct nearest ids used: {sites['gee_id'].nunique()} | sites sharing a nearest id: "
          f"{int(sites.duplicated('gee_id', keep=False).sum())}")
    a = a.merge(sites, on=["sensor_name", "latitude", "longitude"], how="left")
    bk = B.drop_duplicates(["id", "date"]).copy()
    bk["b_complete"] = bk[FEATS].notna().all(axis=1)
    bk = bk[["id", "date", "b_complete"]].rename(columns={"id": "gee_id"})
    a = a.merge(bk, on=["gee_id", "date"], how="left")
    a["b_exists"] = a["b_complete"].notna()
    a["in_C"] = pd.MultiIndex.from_frame(a[["sensor_name", "date"]]).isin(
        pd.MultiIndex.from_frame(C[["sensor_name", "date"]]))
    a["reason"] = np.select(
        [a["invalid_pm25"], a["exact_dup"], a["dist_deg"] > THRESH_DEG, ~a["b_exists"], a["b_exists"] & (a["b_complete"] == False)],
        ["1 invalid pm25 (teacher rule)", "2 exact duplicate row", "3 no GEE id within 0.25 deg",
         "4 no GEE row on that date", "5 GEE row exists but has NaN features"],
        default="6 complete join possible")
    print("\n  rows of A by reason (columns: not in C / in C):")
    print(pd.crosstab(a["reason"], a["in_C"]).to_string())
    Cm = C.merge(sites[["sensor_name", "latitude", "longitude", "gee_id"]], on=["sensor_name", "latitude", "longitude"], how="left")
    print(f"\n  C rows whose (sensor_name, date) is not a valid A row: "
          f"{int((~pd.MultiIndex.from_frame(C[['sensor_name','date']]).isin(pd.MultiIndex.from_frame(a.loc[~a['invalid_pm25'], ['sensor_name','date']]))).sum())}")
    print(f"  C rows whose id differs from our nearest-id match: {int((Cm['id'] != Cm['gee_id']).sum())} of {len(Cm):,}")
    sensors_missing = set(A["sensor_name"]) - set(C["sensor_name"])
    print(f"  sensors in A but absent from C: {len(sensors_missing)} (rows: {int(A['sensor_name'].isin(sensors_missing).sum()):,})")


def s7_time_coverage():
    title("7. Coverage by year (matters for choosing the test year)")
    ya = A.dropna(subset=["date"]).groupby(A["date"].dt.year).agg(rows=("pm25", "size"), sensors=("sensor_name", "nunique"))
    yc = C.dropna(subset=["date"]).groupby(C["date"].dt.year).agg(C_rows=("pm25", "size"), C_sensors=("sensor_name", "nunique"))
    out = ya.join(yc, how="outer")
    out["C_share_%"] = (out["C_rows"] / out["C_rows"].sum() * 100).round(1)
    print(out.to_string())
    first = C.groupby("sensor_name")["date"].min().dt.year.value_counts().sort_index()
    print("\n  C: number of sensors by year of first record:", first.to_dict())
    n = C.groupby("sensor_name").size()
    print("  C rows per sensor (min / median / max):", n.min(), int(n.median()), n.max())


def s8_date_consistency():
    title("8. date vs datetime_utc")
    for name, df in (("B", B), ("C", C)):
        d2 = day(df["datetime_utc"])
        print(f"  {name}: rows where date != date part of datetime_utc: {int((d2 != df['date']).sum())}")
        t = pd.to_datetime(df["datetime_utc"], errors="coerce", utc=True)
        print(f"     distinct hours of day in datetime_utc: {sorted(t.dt.hour.dropna().unique().tolist())[:6]}")


for fn in (s1_overview, s2_dropna_hypothesis, s3_sensor_sites, s4_gee_ids,
           s5_multi_id_sensors, s6_attrition, s7_time_coverage, s8_date_consistency):
    safe(fn)
print("\nDone. Nothing was modified. Paste this output back.")
