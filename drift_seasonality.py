
import pandas as pd
from src import config

data = pd.read_parquet(config.INTERIM / "clean_dev.parquet")

data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date", "station_id"])

config.PARTS.mkdir(parents=True, exist_ok=True)

variables = [
    "pm25", "aod", "no2", "co", "o3",
    "temperature_celsius", "pressure_mb", "wind_u", "wind_v"
]

variables = [col for col in variables if col in data.columns]

data["month_period"] = data["date"].dt.to_period("M").astype(str)
data["year_period"] = data["date"].dt.year

threshold = data["pm25"].quantile(0.90)

results = []

for period_column, period_name in [
    ("month_period", "monthly"),
    ("year_period", "yearly")
]:
    for (station, period), group in data.groupby(
        ["station_id", period_column]
    ):
        for variable in variables:
            values = group[variable].dropna()

            if len(values) == 0:
                continue

            results.append({
                "section": period_name,
                "variable": variable,
                "group": f"{station} - {period}",
                "metric": "median",
                "value": values.median(),
                "note": f"n={len(values)}"
            })

            results.append({
                "section": period_name,
                "variable": variable,
                "group": f"{station} - {period}",
                "metric": "IQR",
                "value": values.quantile(0.75) - values.quantile(0.25),
                "note": f"n={len(values)}"
            })

        pm = group["pm25"].dropna()

        if len(pm) > 0:
            results.append({
                "section": period_name,
                "variable": "pm25",
                "group": f"{station} - {period}",
                "metric": "high_pm25_frequency_percent",
                "value": (pm > threshold).mean() * 100,
                "note": f"threshold={threshold:.3f}; n={len(pm)}"
            })

drift_df = pd.DataFrame(results)

drift_df.to_csv(
    config.PARTS / "member4_drift_summary.csv",
    index=False
)

print("PM2.5 threshold:", threshold)
print("Drift rows generated:", len(drift_df))
print("Drift analysis completed.")
