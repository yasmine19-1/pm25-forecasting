"""Compare predictor clipping on identical development validation rows."""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src import config
from src.folds import load_folds
from src.m3_common import load_dev
from src.pipeline import build_pipeline


def main():
    dev = load_dev()
    predictors, target = dev.drop(columns=config.TARGET), dev[config.TARGET]
    rows = []
    results_dir = config.PARTS.parent
    for scheme in ("temporal", "spatial", "standard"):
        folds = load_folds(results_dir / f"folds_{scheme}.json", dev)
        for fold, (train, val) in enumerate(folds, start=1):
            for enabled in (True, False):
                pipeline = build_pipeline(Ridge(), clip_outliers=enabled)
                pipeline.fit(predictors.iloc[train], target.iloc[train])
                predictions = pipeline.predict(predictors.iloc[val])
                actual = target.iloc[val]
                rows.append({
                    "scheme": scheme,
                    "fold": fold,
                    "clipping_enabled": enabled,
                    "validation_rows": len(val),
                    "mae": mean_absolute_error(actual, predictions),
                    "rmse": root_mean_squared_error(actual, predictions),
                    "r2": r2_score(actual, predictions),
                })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(results_dir / "clipping_comparison.csv", index=False, encoding="utf-8")
    print(comparison.groupby(["scheme", "clipping_enabled"])[["rmse", "mae", "r2"]].mean().round(4))


if __name__ == "__main__":
    main()
