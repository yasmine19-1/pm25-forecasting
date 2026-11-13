"""Check calendar and previous-day PM2.5 through the shared temporal folds."""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.features import rolling_forecast_features
from src.folds import load_folds
from src.m3_common import DEV_FILE, PARTS, add, load_dev, save_parts
from src.pipeline import build_pipeline


def main():
    dev = load_dev(DEV_FILE)
    splits = load_folds(PARTS.parent / "folds_temporal.json", dev)
    features = rolling_forecast_features(dev)
    if not features.index.equals(dev.index):
        raise ValueError("Forecast features must preserve development row alignment.")
    target = dev["pm25"]
    calendar = ["day_of_year_sin", "day_of_year_cos"]
    rows = []
    for number, (train, val) in enumerate(splits, 1):
        group = f"temporal/fold{number}"
        add(rows, "5.3", "pm25_lag_1", group, "validation_available_share",
            features.iloc[val]["pm25_lag_1"].notna().mean())
        for name, columns in (
            ("calendar_only", calendar),
            ("calendar_plus_pm25_lag_1", list(features.columns)),
        ):
            X = features[columns]
            pipeline = build_pipeline(Ridge(), scaled=False, clip_outliers=False)
            pipeline.fit(X.iloc[train], target.iloc[train])
            predictions = pipeline.predict(X.iloc[val])
            note = "Development smoke check; rolling daily observations assumed available"
            add(rows, "5.3", name, group, "rmse",
                root_mean_squared_error(target.iloc[val], predictions), note)
            add(rows, "5.3", name, group, "mae",
                mean_absolute_error(target.iloc[val], predictions), note)
    print(save_parts(rows, "rolling_pipeline_check.csv"))


if __name__ == "__main__":
    main()
