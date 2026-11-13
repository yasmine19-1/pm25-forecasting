"""Outlier step for the Pipeline (§5.3.3).

OutlierClipper
    * learns lower/upper limits per column in fit() = on the training fold only:
      Q1 - k*IQR and Q3 + k*IQR (k = 3 by default), never wider than the
      physical domain limits
    * transform() clips (winsorises) every row to those limits, unchanged for
      validation and test: rows are never dropped, so X and y stay aligned
    * zero_as_missing: columns where an exact 0 is a failed retrieval (AOD)
      are set to NaN; put S5PImputer after this step to fill them
    * enabled=False disables clipping but retains zero-to-missing normalization, for the with-vs-without
      comparison of MAE, RMSE and R² in Phase 2:
          pipe.set_params(outliers__enabled=False)
      or in a grid: {"outliers__enabled": [True, False]}

The target (PM2.5) is not in X, so it is never clipped here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DOMAIN_LIMITS = {
    "temperature_celsius": (-40, 55),
    "pressure_mb": (800, 1085),
    "wind_u": (-40, 40),
    "wind_v": (-40, 40),
    "no2": (0, 1e-3),
    "co": (0, 0.2),
    "o3": (0, 0.5),
    "aod": (0, 5),
}


class OutlierClipper(TransformerMixin, BaseEstimator):
    def __init__(self, enabled=True, k=3.0, cols=None,
                 zero_as_missing=("aod",), domain_limits=None):
        self.enabled = enabled
        self.k = k
        self.cols = cols                    # None = every numeric column
        self.zero_as_missing = zero_as_missing
        self.domain_limits = domain_limits  # None = DOMAIN_LIMITS

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.lower_, self.upper_ = {}, {}
        if not self.enabled:
            return self
        dom = DOMAIN_LIMITS if self.domain_limits is None else self.domain_limits
        cols = self.cols or X.select_dtypes("number").columns
        # skip 0/1 indicator columns
        cols = [c for c in cols if c in X.columns and X[c].nunique(dropna=True) > 2]
        Xz = self._zero_to_nan(X)
        for c in cols:
            s = Xz[c].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            lo, hi = q1 - self.k * iqr, q3 + self.k * iqr
            if c in dom:
                lo, hi = max(lo, dom[c][0]), min(hi, dom[c][1])
            self.lower_[c], self.upper_[c] = lo, hi
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        X = self._zero_to_nan(self._as_frame(X))
        for c in self.lower_:
            X[c] = X[c].clip(self.lower_[c], self.upper_[c])
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "lower_")
        return np.asarray(self.feature_names_in_, dtype=object)

    def report(self, X):
        """Number of values clipped per column (for logging in Phase 2)."""
        check_is_fitted(self, "lower_")
        X = self._zero_to_nan(self._as_frame(X))
        return pd.Series({c: int(((X[c] < self.lower_[c]) | (X[c] > self.upper_[c])).sum())
                          for c in self.lower_}, name="n_clipped")

    def _zero_to_nan(self, X):
        X = X.copy()  # never modify the caller's DataFrame
        for c in self.zero_as_missing or ():
            if c in X.columns:
                X[c] = X[c].mask(X[c] == 0)
        return X

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        names = getattr(self, "feature_names_in_", None)
        return pd.DataFrame(X, columns=names)
