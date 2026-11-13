import pandas as pd
import numpy as np

from src import config


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

            day_of_year = segment["date"].dt.day_of_year
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


def main():
    data = pd.read_parquet(config.INTERIM / "clean_dev.parquet")
    features = build_features(data, forecast_horizon_days=1)

    config.PARTS.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        config.PARTS / "member4_features_preview.csv",
        index=False
    )

    print("Feature rows:", len(features))
    print("Feature columns:", len(features.columns))
    print("Rate-of-change features:", sum(
        "_rate_of_change_1" in column for column in features.columns
    ))
    print("Feature file saved.")



def rolling_forecast_features(data):
    """Align calendar and observed prior-day PM2.5 to each target date.

    This development-only preparation assumes yesterday's measurement is
    available at each rolling forecast origin, including within validation.
    It learns no parameters. It is unsuitable for fixed-origin or cold-start
    validation. The current target and same-day measurements stay out of X.
    """
    dates = pd.to_datetime(data["date"], errors="raise")
    cutoff = pd.Timestamp(config.TEST_START)
    keys = data.assign(date=dates)
    if (dates.isna().any() or not dates.lt(cutoff).all()
            or not dates.eq(dates.dt.normalize()).all()
            or keys["station_id"].isna().any()
            or keys.duplicated(["station_id", "date"]).any()):
        raise ValueError("Expected unique, normalized pre-holdout station-days.")
    ordered = keys.sort_values(["station_id", "date"])
    group = ordered.groupby("station_id", sort=False)
    consecutive = group["date"].diff().eq(pd.Timedelta(days=1))
    for column in ("gee_id", "lat", "lon"):
        if column in ordered:
            previous = group[column].shift()
            consecutive &= ordered[column].notna() & ordered[column].eq(previous)
    lag = group["pm25"].shift().where(consecutive)
    phase = 2 * np.pi * (ordered["date"].dt.day_of_year - 1) / 365.25
    predictors = pd.DataFrame({
        "day_of_year_sin": np.sin(phase),
        "day_of_year_cos": np.cos(phase),
        "pm25_lag_1": lag,
        "pm25_lag_1_missing": lag.isna().astype(int),
    })
    return predictors.reindex(data.index)


if __name__ == "__main__":
    main()
