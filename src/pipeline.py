"""Pipeline assembly and fold-local preprocessing (brief 5.3).

Order of the steps:

    features  ->  drop_ids  ->  outliers  ->  impute  ->  scale  ->  select  ->  model

Every step that learns something (clipping limits, medians, scaler, selected
columns) does it in fit(). Because they all sit in one sklearn Pipeline, a
cross-validation loop refits them on each training fold, and the validation
rows are only ever transformed with what was learned on the training rows.

Usage (X = the development frame rows of a fold, y = its pm25):

    pipe = build_pipeline(Lasso(random_state=42))          # scaling picked automatically
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)

    build_pipeline(ExtraTreesRegressor(random_state=42), selector="tree")
    pipe.set_params(outliers__enabled=False)               # disable predictor clipping

Pass the full development rows, ids included. The `drop_ids` step removes the
station id, coordinates, dates, GEE id and the target itself before anything is
learned, so they can never become model inputs (ground rule 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src import config
from src.features import TemporalFeatures
from src.imputation import S5PImputer
from src.outliers import OutlierClipper
from src.scaling import TrainFoldScaler, needs_scaling
from src.selection import FeatureSelector

NEVER_FEATURES = tuple(config.NON_FEATURE_COLS) + (config.TARGET, "target_pm25", "segment_id")


class DropNonFeatures(TransformerMixin, BaseEstimator):
    """Removes ids, coordinates, dates and the target from X. Rows are never touched."""

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.kept_ = [c for c in X.columns if c not in NEVER_FEATURES]
        if not self.kept_:
            raise ValueError("no predictor columns left after dropping ids and target")
        return self

    def transform(self, X):
        check_is_fitted(self, "kept_")
        return self._as_frame(X)[self.kept_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "kept_")
        return np.asarray(self.kept_, dtype=object)

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))


def temporal_step(scheme, history, scenario="next_day"):
    """TemporalFeatures set up for one fold scheme ("temporal", "spatial" or "standard").

    Next-day scenario: PM2.5 lags are allowed, except for the spatial scheme, where the held-out
    stations have no PM2.5 history at forecast time (cold-start rule in the methodology).
    `history` is the development frame; lags only ever look at earlier days.
    """
    if scenario == "next_day":
        lags = scheme != "spatial"
    else:
        lags = None          # same_day: the transformer decides (no PM2.5 lags)
    return TemporalFeatures(scenario=scenario, history=history, use_target_lags=lags)


def build_pipeline(model, *, scaled=None, scaler="standard", selector=None,
                   features=None, clip_outliers=True, imputer=None):
    """Assemble the full Pipeline around `model`.

    scaled        True / False, or None (default) to decide from the model family
                  (SVR, KNN, neural nets, regularised linear -> scaled; trees -> not)
    scaler        "standard" or "robust", used only when scaled
    selector      None (keep every column), a method name ("f_test", "mutual_info",
                  "rfe", "l1", "tree") or a FeatureSelector instance
    features      the temporal transformer, if any. It runs first and receives the full
                  frame (it needs station_id and date to build past-only lags) and must
                  return the same rows in the same order.
    clip_outliers False switches the clipper off from the start
    imputer       None = the default S5PImputer(). Pass S5PImputer(s5p_cols=...) to also get a
                  0/1 missing indicator for other columns, e.g. the lag columns, which are
                  often missing and otherwise get the training median without a trace
    """
    if scaled is None:
        scaled = needs_scaling(model)
    if isinstance(selector, str):
        selector = FeatureSelector(method=selector, random_state=config.SEED)

    steps = []
    if features is not None:
        steps.append(("features", features))
    steps += [
        ("drop_ids", DropNonFeatures()),
        ("outliers", OutlierClipper(enabled=clip_outliers)),
        ("impute", imputer if imputer is not None else S5PImputer()),
    ]
    if scaled:
        steps.append(("scale", TrainFoldScaler(kind=scaler)))
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", model))
    return Pipeline(steps)
