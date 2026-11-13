"""Imputation step for the Pipeline (§5.3.1).

S5PImputer
    * median imputation, medians learned in fit() = on the training fold only
    * one 0/1 missingness indicator per Sentinel-5P column (no2_missing, ...)
    * any other numeric column with gaps is also median-imputed (no indicator)
    * categorical columns (if any are passed): mode, or an explicit "missing"
      category
    * no forward-fill, ever (it would leak across time and stations)

Works on pandas DataFrames and keeps column names, so later steps (scaler,
selector, feature logging) see readable names.

Order in the Pipeline: OutlierClipper -> S5PImputer -> scaler -> selector
(the clipper turns impossible values such as AOD == 0 into NaN, the imputer
then fills them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

S5P_DEFAULT = ("no2", "co", "o3")


class S5PImputer(TransformerMixin, BaseEstimator):
    def __init__(self, s5p_cols=S5P_DEFAULT, add_indicator=True,
                 categorical_cols=(), categorical_strategy="missing_category"):
        self.s5p_cols = s5p_cols
        self.add_indicator = add_indicator
        self.categorical_cols = categorical_cols
        self.categorical_strategy = categorical_strategy  # or "mode"

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        cat = [c for c in self.categorical_cols if c in X.columns]
        num = [c for c in X.columns if c not in cat]
        self.numeric_cols_ = num
        self.categorical_cols_ = cat
        self.s5p_cols_ = [c for c in self.s5p_cols if c in X.columns]

        self.medians_ = X[num].median(numeric_only=True)
        # A column entirely missing in this training fold: fall back to 0 and
        # rely on the indicator rather than crash.
        self.medians_ = self.medians_.fillna(0.0)

        self.fill_cat_ = {}
        for c in cat:
            if self.categorical_strategy == "mode" and X[c].notna().any():
                self.fill_cat_[c] = X[c].mode(dropna=True).iloc[0]
            else:
                self.fill_cat_[c] = "missing"
        return self

    def transform(self, X):
        check_is_fitted(self, "medians_")
        X = self._as_frame(X).copy()
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"columns seen in fit are missing: {sorted(missing_cols)}")
        X = X[list(self.feature_names_in_)]

        if self.add_indicator:
            ind = {f"{c}_missing": X[c].isna().astype(np.int8) for c in self.s5p_cols_}
        X[self.numeric_cols_] = X[self.numeric_cols_].fillna(self.medians_)
        for c in self.categorical_cols_:
            X[c] = X[c].astype(object).where(X[c].notna(), self.fill_cat_[c])
        if self.add_indicator:
            X = pd.concat([X, pd.DataFrame(ind, index=X.index)], axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "medians_")
        names = list(self.feature_names_in_)
        if self.add_indicator:
            names += [f"{c}_missing" for c in self.s5p_cols_]
        return np.asarray(names, dtype=object)

    # ------------------------------------------------------------------
    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        names = getattr(self, "feature_names_in_", None)
        return pd.DataFrame(X, columns=names)
