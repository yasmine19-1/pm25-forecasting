
import pandas as pd
from src import config

# Load the cleaned dataset
data = pd.read_parquet(config.INTERIM / "clean_dev.parquet")
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date", "station_id"])

config.PARTS.mkdir(parents=True, exist_ok=True)

# Variables to analyse
variables = [
    "pm25", "aod", "no2", "co", "o3",
    "temperature_celsius", "pressure_mb", "wind_u", "wind_v"
]
variables = [col for col in variables if col in data.columns]

# Aggregate to one daily median per station
daily = (
    data.groupby(["station_id", "date"])[variables]
    .median()
    .reset_index()
)

# Extract temporal information
daily["day_of_week"] = daily["date"].dt.day_name()
daily["month"] = daily["date"].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"

daily["season"] = daily["month"].apply(get_season)

# Calculate patterns
results = []

# Weekly patterns
for (station, day), group in daily.groupby(["station_id", "day_of_week"]):
    for variable in variables:
        values = group[variable].dropna()
        if len(values) > 0:
            results.append({
                "section": "weekly",
                "variable": variable,
                "group": f"{station} - {day}",
                "metric": "median",
                "value": values.median(),
                "note": f"n={len(values)}"
            })

# Monthly patterns
for (station, month), group in daily.groupby(["station_id", "month"]):
    for variable in variables:
        values = group[variable].dropna()
        if len(values) > 0:
            results.append({
                "section": "monthly",
                "variable": variable,
                "group": f"{station} - month {month}",
                "metric": "median",
                "value": values.median(),
                "note": f"n={len(values)}"
            })

# Seasonal patterns
for (station, season), group in daily.groupby(["station_id", "season"]):
    for variable in variables:
        values = group[variable].dropna()
        if len(values) > 0:
            results.append({
                "section": "seasonal",
                "variable": variable,
                "group": f"{station} - {season}",
                "metric": "median",
                "value": values.median(),
                "note": f"n={len(values)}"
            })

patterns = pd.DataFrame(results)

# Save outputs
daily.to_csv(config.PARTS / "member4_daily_values.csv", index=False)
patterns.to_csv(config.PARTS / "member4_temporal_patterns.csv", index=False)

print("Daily rows:", len(daily))
print("Pattern rows:", len(patterns))
print("Files saved successfully.")
