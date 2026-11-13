
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

    metadata = [c for c in ["gee_id", "lat", "lon"] if c in data.columns]
    daily_rows = []

    for (station, date), day in data.groupby(["station_id", "date"], sort=True):
        row = {"station_id": station, "date": date}

        for variable in variables:
            row[variable] = day[variable].median()

        mixed_metadata = False
        for column in metadata:
            values = day[column].dropna().unique()
            row[column] = values[0] if len(values) else np.nan
            if len(values) > 1:
                mixed_metadata = True

        row["_mixed_metadata"] = mixed_metadata
        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows).sort_values(
        ["station_id", "date"]
    ).reset_index(drop=True)

    feature_parts = []

    for station, group in daily.groupby("station_id", sort=True):
        group = group.sort_values("date").reset_index(drop=True)

        segment_ids = []
        current_segment = 0

        for i in range(len(group)):
            if i > 0:
                date_gap = (group.loc[i, "date"] - group.loc[i - 1, "date"]).days != 1

                metadata_changed = any(
                    not (
                        (pd.isna(group.loc[i, col]) and pd.isna(group.loc[i - 1, col]))
                        or group.loc[i, col] == group.loc[i - 1, col]
                    )
                    for col in metadata
                )

                mixed_boundary = (
                    group.loc[i, "_mixed_metadata"]
                    or group.loc[i - 1, "_mixed_metadata"]
                )

                if date_gap or metadata_changed or mixed_boundary:
                    current_segment += 1

            segment_ids.append(current_segment)

        group["_segment_id"] = segment_ids

        for segment_id, segment in group.groupby("_segment_id"):
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

                segment[f"{variable}_difference_1"] = previous - two_days_ago

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

            segment["station_id"] = station
            segment["segment_id"] = int(segment_id)

            segment = segment.drop(columns=["_mixed_metadata", "_segment_id"])
            feature_parts.append(segment)

    if not feature_parts:
        return pd.DataFrame()

    return pd.concat(feature_parts, ignore_index=True)
