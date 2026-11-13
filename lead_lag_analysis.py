
import pandas as pd
from src import config

data = pd.read_parquet(config.INTERIM / "clean_dev.parquet")
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date", "station_id"])

variables = [
    "aod", "no2", "co", "o3",
    "wind_u", "wind_v",
    "temperature_celsius", "pressure_mb"
]
variables = [v for v in variables if v in data.columns]

# One daily median per station
daily = (
    data.groupby(["station_id", "date"])[["pm25"] + variables]
    .median()
    .reset_index()
    .sort_values(["station_id", "date"])
)

results = []

for station, group in daily.groupby("station_id"):
    group = group.reset_index(drop=True)

    # Split wherever consecutive dates are not exactly one day apart
    segment_ids = group["date"].diff().dt.days.ne(1).cumsum()

    for segment_id, segment in group.groupby(segment_ids):
        if len(segment) < 15:
            continue

        for variable in variables:
            for lag in range(-5, 6):
                # Positive lag: predictor values from earlier days
                predictor = segment[variable].shift(lag)
                paired = pd.concat(
                    [segment["pm25"], predictor], axis=1
                ).dropna()

                if len(paired) < 10:
                    continue

                correlation = paired.iloc[:, 0].corr(paired.iloc[:, 1])

                results.append({
                    "station_id": station,
                    "segment_id": int(segment_id),
                    "variable": variable,
                    "lag_days": lag,
                    "observations": len(paired),
                    "correlation": correlation
                })

detail = pd.DataFrame(results)

# Summarise correlations across station segments
summary = (
    detail.groupby(["variable", "lag_days"])["correlation"]
    .agg(["count", "median", "mean"])
    .reset_index()
    .rename(columns={
        "count": "segments",
        "median": "median_correlation",
        "mean": "mean_correlation"
    })
)

config.PARTS.mkdir(parents=True, exist_ok=True)

detail.to_csv(
    config.PARTS / "member4_lead_lag_detail.csv", index=False
)
summary.to_csv(
    config.PARTS / "member4_lead_lag_summary.csv", index=False
)

print("Detailed results:", len(detail))
print("Summary rows:", len(summary))
print("\nSummary:")
print(summary.to_string(index=False))
print("\nLead-lag analysis completed.")
