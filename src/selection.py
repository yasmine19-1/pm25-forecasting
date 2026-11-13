"""Feature selection step for the Pipeline (brief 5.3.2).

FeatureSelector(method=..., k=...)
    filter    "f_test"        ANOVA F-test of each column against PM2.5
              "mutual_info"   mutual information with PM2.5
    wrapper   "rfe"           recursive feature elimination with a Ridge model
    embedded  "l1"            columns with a non-zero Lasso coefficient
              "tree"          Extra Trees importance

All scores are computed in fit(), so inside a Pipeline they only use the
training fold. After fit():
    selected_features_   names of the kept columns (original column order)
    n_selected_          how many
    scores_              one score per input column (higher = more useful)

The brief asks to "record selected features and their number": use
describe_selection(selector) to get both as one row for the log.

Ranking methods keep the k best columns. "l1" keeps whatever Lasso keeps
(k is ignored there) but never fewer than min_features.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE, f_regression, mutual_info_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.utils.validation import check_is_fitted

METHODS = ("f_test", "mutual_info", "rfe", "l1", "tree")


class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, method="f_test", k=8, alpha=0.01, min_features=1, random_state=42):
        self.method = method
        self.k = k                      # number of columns kept by the ranking methods
        self.alpha = alpha              # Lasso penalty (method "l1" only)
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        X = self._as_frame(X)
        y = np.asarray(y, dtype=float).ravel()
        names = list(X.columns)
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        k = max(1, min(int(self.k), len(names)))

        if self.method in ("rfe", "l1"):
            # these two depend on the size of the coefficients, so rank on a standardised
            # copy; the mean and std come from the training rows only
            sd = X.std(ddof=0).replace(0, 1.0)
            Z = ((X - X.mean()) / sd).to_numpy()
        else:
            Z = X.to_numpy(dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant columns give 0/0 in the F-test
            if self.method == "f_test":
                scores = np.nan_to_num(f_regression(Z, y)[0])
            elif self.method == "mutual_info":
                scores = mutual_info_regression(Z, y, random_state=self.random_state)
            elif self.method == "tree":
                tree = ExtraTreesRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                           random_state=self.random_state, n_jobs=1).fit(Z, y)
                scores = tree.feature_importances_
            elif self.method == "rfe":
                rfe = RFE(Ridge(alpha=1.0), n_features_to_select=k, step=1).fit(Z, y)
                scores = -rfe.ranking_.astype(float)       # rank 1 (kept) becomes the highest score
            else:  # l1
                lasso = Lasso(alpha=self.alpha, max_iter=10000, random_state=self.random_state).fit(Z, y)
                scores = np.abs(lasso.coef_)

        self.scores_ = pd.Series(scores, index=names)
        if self.method == "l1":
            keep = scores > 0
            need = min(self.min_features, len(names))
            if keep.sum() < need:                           # e.g. very small training fold
                keep = np.zeros(len(names), dtype=bool)
                keep[np.argsort(-scores, kind="stable")[:need]] = True
        else:
            keep = np.zeros(len(names), dtype=bool)
            keep[np.argsort(-scores, kind="stable")[:k]] = True   # stable: ties keep column order
        self.support_ = keep
        self.selected_features_ = [n for n, f in zip(names, keep) if f]
        self.n_selected_ = len(self.selected_features_)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        X = self._as_frame(X)
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(f"columns seen in fit are missing: {sorted(missing)}")
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))


def describe_selection(selector):
    """One log entry: how many columns were kept and which ones."""
    check_is_fitted(selector, "selected_features_")
    return {"method": selector.method,
            "n_selected": selector.n_selected_,
            "n_input": selector.n_features_in_,
            "selected": "; ".join(selector.selected_features_)}
