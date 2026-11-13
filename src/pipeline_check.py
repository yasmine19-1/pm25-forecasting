"""Run the full Pipeline once through the shared shared folds.

This is a smoke run, NOT the Phase 2 model comparison: two cheap models, no tuning.
It shows that
  * the Pipeline fits and predicts on all three fold files (temporal, spatial, standard)
  * each step is fitted on the training rows of the fold only (checked on every fold)
  * the selected features and their number are logged for every fold (brief 5.3.2)

Run from the repo root, once folds_*.json exist:
    python -m src.pipeline_check

Writes:
    results/<GROUP>/parts/m3_pipeline_check.csv       (shared schema: RMSE / MAE per fold)
    results/<GROUP>/parts/m3_selected_features.csv    (scheme, fold, model, method, n_input, n_selected, selected)
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src import config
from src.folds import load_folds
from src.m3_common import DEV_FILE, PARTS, add, load_dev, save_parts
from src.pipeline import build_pipeline
from src.selection import FeatureSelector, describe_selection

SEC = "5.3"
SCHEMES = ["temporal", "spatial", "standard"]


class RowSpy(BaseEstimator, TransformerMixin):
    """Pass-through step that records the rows it was fitted on."""

    def fit(self, X, y=None):
        self.fit_index_ = np.asarray(X.index)
        return self

    def transform(self, X):
        return X


def candidates():
    # name, model, selector (the same selector settings for every fold)
    return {
        "lasso_scaled_ftest6": (Lasso(alpha=0.1, random_state=config.SEED), FeatureSelector("f_test", k=6)),
        "extratrees_tree6": (ExtraTreesRegressor(100, random_state=config.SEED, n_jobs=1),
                             FeatureSelector("tree", k=6, random_state=config.SEED)),
    }


def main():
    dev = load_dev(DEV_FILE)
    X, y = dev.drop(columns=config.TARGET), dev[config.TARGET]
    rows, selected_log = [], []
    for scheme in SCHEMES:
        path = PARTS.parent / f"folds_{scheme}.json"
        if not path.exists():
            print(f"skip {scheme}: {path.name} not found")
            continue
        for number, (train, val) in enumerate(load_folds(path, dev), start=1):
            for name, (model, selector) in candidates().items():
                pipe = build_pipeline(model, selector=selector)
                pipe.steps.insert(0, ("spy", RowSpy()))
                t0 = time.perf_counter()
                pipe.fit(X.iloc[train], y.iloc[train])
                pred = pipe.predict(X.iloc[val])
                secs = time.perf_counter() - t0
                # leakage guard: the Pipeline was fitted on exactly the training rows
                assert set(pipe["spy"].fit_index_) == set(X.index[train])
                assert not set(pipe["spy"].fit_index_) & set(X.index[val])
                group = f"{scheme}/fold{number}"
                add(rows, SEC, name, group, "rmse", root_mean_squared_error(y.iloc[val], pred))
                add(rows, SEC, name, group, "mae", mean_absolute_error(y.iloc[val], pred))
                add(rows, SEC, name, group, "n_train", len(train))
                add(rows, SEC, name, group, "n_val", len(val))
                add(rows, SEC, name, group, "seconds", secs)
                selected_log.append(dict(scheme=scheme, fold=number, model=name,
                                         **describe_selection(pipe["select"])))
            print(f"{scheme} fold {number}: {len(train)} train / {len(val)} val  ok")
    save_parts(rows, "m3_pipeline_check.csv")
    pd.DataFrame(selected_log).to_csv(PARTS / "m3_selected_features.csv", index=False)
    print("written: m3_pipeline_check.csv, m3_selected_features.csv")


if __name__ == "__main__":
    main()
