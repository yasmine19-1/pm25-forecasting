
import pandas as pd
import os

data = pd.read_parquet("clean_dev.parquet")
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date", "station_id"])

# One PM2.5 median per station and day
daily = (
    data.groupby(["station_id", "date"])["pm25"]
    .median()
    .reset_index()
    .sort_values(["station_id", "date"])
)

results = []

for station, group in daily.groupby("station_id"):
    group = group.reset_index(drop=True)

    # Split into uninterrupted daily segments
    segment_ids = group["date"].diff().dt.days.ne(1).cumsum()

    for segment_id, segment in group.groupby(segment_ids):
        series = segment["pm25"].dropna().reset_index(drop=True)

        # Need enough observations to examine a 7-day lag
        if len(series) < 15:
            continue

        correlation = series.corr(series.shift(7))

        results.append({
            "station_id": station,
            "segment_id": int(segment_id),
            "observations": len(series),
            "lag_days": 7,
            "correlation": correlation
        })

weekly_df = pd.DataFrame(results)

os.makedirs("results/B04-01", exist_ok=True)

weekly_df.to_csv(
    "results/B04-01/member4_weekly_cycle_test.csv",
    index=False
)

print("Segments tested:", len(weekly_df))
print("Mean lag-7 correlation:", weekly_df["correlation"].mean())
print("Median lag-7 correlation:", weekly_df["correlation"].median())
print("Results saved successfully.")
