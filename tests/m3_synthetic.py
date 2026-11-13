"""Synthetic development frame with the SAME columns as data/interim/clean_dev.parquet.

Only for the tests and quick checks (the real parquet is not in git).
Mimics what matters for the Pipeline: station/date identifiers, Sentinel-5P
columns that are completely missing in the first years, AOD zeros, a skewed
PM2.5, and station coverage that changes over time.
"""
import numpy as np
import pandas as pd


def make_dev_frame(n_stations=14, seed=42):
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(n_stations):
        start = pd.Timestamp("2017-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
        days = pd.date_range(start, "2024-12-31", freq="D")
        days = days[rng.random(len(days)) < rng.uniform(0.15, 0.5)]   # irregular coverage
        n = len(days)
        doy = days.day_of_year.to_numpy()
        temp = 18 + 8 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 3, n)
        pres = 997 + rng.normal(0, 8, n)
        wu, wv = rng.normal(0, 3, n), rng.normal(0, 3, n)
        speed = np.hypot(wu, wv)
        aod = np.abs(0.15 + 0.08 * rng.normal(size=n) + 0.05 * np.sin(2 * np.pi * doy / 365))
        aod[rng.random(n) < 0.02] = 0.0                                # failed retrievals
        no2 = 9e-5 + 2e-5 * rng.normal(size=n) + 3e-5 * aod
        co = 0.03 + 0.004 * rng.normal(size=n) + 0.01 * aod
        o3 = 0.14 + 0.01 * rng.normal(size=n)
        pm = np.exp(1.8 + 1.5 * aod + 0.01 * temp - 0.08 * speed + 0.4 * rng.normal(size=n))
        df = pd.DataFrame({
            "date": days, "datetime_utc": days, "gee_id": 1000 + s,
            "lat": 36.0 + 0.4 * s, "lon": -5.0 + 0.6 * s,
            "temperature_celsius": temp, "pressure_mb": pres, "wind_u": wu, "wind_v": wv,
            "no2": no2, "co": co, "o3": o3, "aod": aod, "pm25": np.clip(pm, 0.5, 300),
            "station_id": f"STATION {s:02d}"})
        parts.append(df)
    d = pd.concat(parts).sort_values(["date", "station_id"]).reset_index(drop=True)
    early = d["date"] < "2019-01-01"                                   # S5P absent in the first years
    gaps = rng.random(len(d)) < 0.07
    for c in ("no2", "co", "o3"):
        d.loc[early | gaps, c] = np.nan
    return d
