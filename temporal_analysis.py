
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

data = pd.read_parquet("clean_dev.parquet")

data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date", "station_id"])
data = data.sort_values(["station_id", "date"])

os.makedirs("results/B04-01", exist_ok=True)

daily = (
    data.groupby(["station_id", "date"])["pm25"]
    .median()
    .reset_index()
    .sort_values(["station_id", "date"])
)

diagnostic_rows = []

for station, group in daily.groupby("station_id"):
    group = group.reset_index(drop=True)
    segment_ids = group["date"].diff().dt.days.ne(1).cumsum()

    for segment_id, segment in group.groupby(segment_ids):
        series = segment["pm25"].dropna().reset_index(drop=True)

        if len(series) < 30:
            continue

        note = f"segment={segment_id}; observations={len(series)}"

        try:
            acf_values = acf(series, nlags=14, fft=False)
            pacf_values = pacf(series, nlags=14, method="ywm")

            for lag in range(1, 15):
                diagnostic_rows.append({
                    "section": "temporal_dependence",
                    "variable": "pm25",
                    "group": station,
                    "metric": f"acf_lag_{lag}",
                    "value": acf_values[lag],
                    "note": note
                })

                diagnostic_rows.append({
                    "section": "temporal_dependence",
                    "variable": "pm25",
                    "group": station,
                    "metric": f"pacf_lag_{lag}",
                    "value": pacf_values[lag],
                    "note": note
                })

            lb = acorr_ljungbox(series, lags=[7, 14], return_df=True)

            for lag in [7, 14]:
                diagnostic_rows.append({
                    "section": "temporal_dependence",
                    "variable": "pm25",
                    "group": station,
                    "metric": f"ljung_box_pvalue_lag_{lag}",
                    "value": lb.loc[lag, "lb_pvalue"],
                    "note": note
                })

            adf = adfuller(series, autolag="AIC")

            for metric, value in [
                ("adf_statistic", adf[0]),
                ("adf_pvalue", adf[1])
            ]:
                diagnostic_rows.append({
                    "section": "temporal_dependence",
                    "variable": "pm25",
                    "group": station,
                    "metric": metric,
                    "value": value,
                    "note": note
                })

        except Exception as error:
            print("Diagnostic error:", station, error)

diagnostics = pd.DataFrame(
    diagnostic_rows,
    columns=["section", "variable", "group", "metric", "value", "note"]
)

diagnostics.to_csv(
    "results/B04-01/Exploratory_temporal_member4.csv",
    index=False
)

print("Temporal diagnostics generated:", len(diagnostics))
print("Columns:", diagnostics.columns.tolist())
print(diagnostics.head().to_string(index=False))
print("Analysis completed.")
