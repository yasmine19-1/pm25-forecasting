
import pandas as pd
from src import config

data = pd.read_parquet(config.INTERIM / "clean_dev.parquet")

data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date", "station_id"])

results = []

for station, group in data.groupby("station_id"):
    group = group.sort_values("date")

    dates = pd.DatetimeIndex(group["date"].dt.normalize().unique()).sort_values()
    gaps = dates.to_series().diff().dt.days.dropna()

    first_date = dates.min()
    last_date = dates.max()
    expected_days = (last_date - first_date).days + 1

    duplicate_timestamps = (
        group.duplicated(subset=["datetime_utc"]).sum()
        if "datetime_utc" in group.columns
        else None
    )

    results.append({
        "station_id": station,
        "rows": len(group),
        "first_date": first_date,
        "last_date": last_date,
        "unique_dates": len(dates),
        "missing_calendar_days": expected_days - len(dates),
        "daily_sampling_percent": (
            (gaps == 1).mean() * 100 if len(gaps) else None
        ),
        "average_gap_days": gaps.mean() if len(gaps) else None,
        "maximum_gap_days": gaps.max() if len(gaps) else None,
        "irregular_intervals": int((gaps != 1).sum()),
        "duplicate_timestamps": duplicate_timestamps
    })

report = pd.DataFrame(results)

config.PARTS.mkdir(parents=True, exist_ok=True)

report.to_csv(
    config.PARTS / "member4_gap_report.csv",
    index=False
)

print("Stations analysed:", len(report))
print("Gap report saved successfully.")
print(report.head().to_string(index=False))
