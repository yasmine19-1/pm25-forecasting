
import pandas as pd
import numpy as np


def build_features(data, forecast_horizon_days=1, high_aod_threshold=None):

    if forecast_horizon_days < 1:
        raise ValueError("Forecast horizon must be at least 1 day.")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "station_id"])

    variables = [
        "pm25", "aod", "no2", "co", "o3",
        "temperature_celsius", "pressure_mb", "wind_u", "wind_v"
    ]
    variables = [v for v in variables if v in data.columns]

    daily = (
        data.groupby(["station_id", "date"])[variables]
        .median()
        .reset_index()
        .sort_values(["station_id", "date"])
    )

    feature_parts = []

    for station, group in daily.groupby("station_id"):
        group = group.sort_values("date").reset_index(drop=True)
        segment_ids = group["date"].diff().dt.days.ne(1).cumsum()

        for segment_id, segment in group.groupby(segment_ids):
            segment = segment.copy().reset_index(drop=True)

            day_of_year = segment["date"].dt.dayofyear
            segment["day_of_year_sin"] = np.sin(
                2 * np.pi * day_of_year / 365.25
            )
            segment["day_of_year_cos"] = np.cos(
                2 * np.pi * day_of_year / 365.25
            )

            for variable in variables:
                previous = segment[variable].shift(1)
                two_days_ago = segment[variable].shift(2)

                segment[f"{variable}_lag_1"] = previous
                segment[f"{variable}_lag_3"] = segment[variable].shift(3)

                segment[f"{variable}_rolling_mean_3"] = (
                    previous.rolling(3, min_periods=3).mean()
                )
                segment[f"{variable}_rolling_std_3"] = (
                    previous.rolling(3, min_periods=3).std()
                )
                segment[f"{variable}_rolling_max_3"] = (
                    previous.rolling(3, min_periods=3).max()
                )

                segment[f"{variable}_difference_1"] = (
                    previous - two_days_ago
                )

                denominator = two_days_ago.abs().replace(0, np.nan)

                segment[f"{variable}_rate_of_change_1"] = (
                    (previous - two_days_ago) / denominator
                )

            if "aod" in segment.columns and high_aod_threshold is not None:
                high_aod = segment["aod"] > high_aod_threshold
                days_since = []
                last_high_position = None

                for position, is_high in enumerate(high_aod):
                    if last_high_position is None:
                        days_since.append(np.nan)
                    else:
                        days_since.append(position - last_high_position)

                    if is_high:
                        last_high_position = position

                segment["days_since_last_high_aod_day"] = days_since

            segment["target_pm25"] = segment["pm25"].shift(
                -forecast_horizon_days
            )

            # Identifiers are kept for grouping, not prediction.
            segment["station_id"] = station
            segment["segment_id"] = int(segment_id)

            feature_parts.append(segment)

    if not feature_parts:
        return pd.DataFrame()

    return pd.concat(feature_parts, ignore_index=True)


if __name__ == "__main__":
    import os

    data = pd.read_parquet("clean_dev.parquet")
    features = build_features(data, forecast_horizon_days=1)

    os.makedirs("results/B04-01", exist_ok=True)
    features.to_csv(
        "results/B04-01/member4_features_preview.csv",
        index=False
    )

    print("Feature rows:", len(features))
    print("Feature columns:", len(features.columns))
    print("Rate-of-change features:", sum(
        "_rate_of_change_1" in column for column in features.columns
    ))
    print("Feature file saved.")
    
