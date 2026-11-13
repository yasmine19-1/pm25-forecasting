"""Member 3 - scaling step for the Pipeline (brief 5.3.2).

TrainFoldScaler
    * StandardScaler or RobustScaler ("standard" / "robust"), learned in fit(),
      so inside a Pipeline it only ever sees the training fold
    * 0/1 indicator columns (no2_missing, ...) are left as they are
    * keeps the DataFrame and its column names, so the selector and the
      feature log downstream still read "aod", not "x3"

needs_scaling(model)
    True for the model families that are sensitive to the scale of the inputs.
    Trees and boosting do not need it, so they get an unscaled Pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

# Brief: "SVR, KNN, neural networks and PCA". Ridge, Lasso, ElasticNet and Huber are added
# because their penalty depends on the scale of the columns. Team can still remove them.
SCALE_SENSITIVE = {
    "Ridge", "Lasso", "ElasticNet", "HuberRegressor",
    "KNeighborsRegressor", "SVR",
    "MLPRegressor", "LSTM", "GRU", "BiLSTM", "CNN1D", "Transformer", "PatchTST",
    "CNNLSTM", "PCA",
}


def needs_scaling(model):
    """Accepts an estimator, a class or a plain name such as "SVR"."""
    if isinstance(model, str):
        name = model
    elif isinstance(model, type):
        name = model.__name__
    else:
        name = type(model).__name__
    return name in SCALE_SENSITIVE


class TrainFoldScaler(TransformerMixin, BaseEstimator):
    def __init__(self, kind="standard"):
        self.kind = kind  # "standard" or "robust"

    def fit(self, X, y=None):
        X = self._as_frame(X)
        if self.kind not in ("standard", "robust"):
            raise ValueError("kind must be 'standard' or 'robust'")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        # indicators (only 0 and 1) are not scaled
        self.scaled_cols_ = [c for c in X.columns if X[c].nunique(dropna=True) > 2]
        self.scaler_ = StandardScaler() if self.kind == "standard" else RobustScaler()
        if self.scaled_cols_:
            self.scaler_.fit(X[self.scaled_cols_])
        return self

    def transform(self, X):
        check_is_fitted(self, "scaler_")
        X = self._as_frame(X)[list(self.feature_names_in_)].copy()
        if self.scaled_cols_:
            X[self.scaled_cols_] = self.scaler_.transform(X[self.scaled_cols_])
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "scaler_")
        return np.asarray(self.feature_names_in_, dtype=object)

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))
