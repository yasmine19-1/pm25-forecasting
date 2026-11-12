"""Member 3 - leakage tests for the Pipeline (brief 5.3 and 6.1).   Run:  pytest tests/

What is checked
  1. every fitted step sees training-fold rows only (spy step inside the Pipeline)
  2. predicting on validation rows never changes what the steps learned
  3. the learned numbers (clip limits, medians, scaler, F-scores) equal what you get
     from the training rows alone, and differ from what the full data would give
  4. station id, coordinates, dates and the target never reach the model
  5. the same holds with the real shared fold files, when the real parquet exists
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from m3_synthetic import make_dev_frame
from src import config
from src.pipeline import NEVER_FEATURES, build_pipeline
from src.selection import FeatureSelector


# --- helpers -------------------------------------------------------------------------
class RowSpy(BaseEstimator, TransformerMixin):
    """Pass-through step that remembers the index of every frame it was FITTED on."""
    seen = []

    def fit(self, X, y=None):
        RowSpy.seen.append(np.asarray(X.index))
        return self

    def transform(self, X):
        return X


class ColumnRecorder(BaseEstimator, RegressorMixin):
    """Final 'model' that stores the columns it receives and predicts the training mean."""

    def fit(self, X, y):
        self.columns_ = list(X.columns)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


@pytest.fixture(scope="module")
def dev():
    return make_dev_frame()


def state_of(pipe):
    """Everything the steps learned, as plain numbers, so two pipelines can be compared."""
    clip, imp, sca = pipe["outliers"], pipe["impute"], pipe["scale"]
    return {"lower": dict(clip.lower_), "upper": dict(clip.upper_), "medians": imp.medians_.to_dict(),
            "mean": sca.scaler_.mean_.tolist(), "scale": sca.scaler_.scale_.tolist(),
            "selected": list(pipe["select"].selected_features_),
            "scores": pipe["select"].scores_.to_dict()}


def all_splits(dev):
    X = dev.drop(columns="pm25")
    groups = dev["station_id"]
    return {
        "temporal": list(TimeSeriesSplit(5, gap=1).split(X)),
        "spatial": list(GroupKFold(5).split(X, groups=groups)),
        "standard": list(KFold(5, shuffle=True, random_state=42).split(X)),
    }


# --- 1. every fitted step sees training rows only -----------------------------------------
@pytest.mark.parametrize("scheme", ["temporal", "spatial", "standard"])
def test_fit_only_sees_training_rows(dev, scheme):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    for train, val in all_splits(dev)[scheme]:
        pipe = build_pipeline(Lasso(random_state=42), selector="f_test")
        pipe.steps.insert(0, ("spy", RowSpy()))        # the spy sits in front of every learning step
        RowSpy.seen = []
        pipe.fit(X.iloc[train], y.iloc[train])
        pipe.predict(X.iloc[val])
        assert len(RowSpy.seen) == 1                    # fitted once, on the training fold
        assert set(RowSpy.seen[0]) == set(X.index[train])
        assert not set(RowSpy.seen[0]) & set(X.index[val])


def test_negative_control_global_fit_is_different(dev):
    """If someone scaled on all rows, the numbers would change: the test above would catch it."""
    X = dev.drop(columns="pm25")
    train, _ = all_splits(dev)["temporal"][0]
    cols = ["temperature_celsius", "pressure_mb"]
    assert not np.isclose(X.iloc[train][cols].mean().to_numpy(), X[cols].mean().to_numpy()).all()


# --- 2. validation rows never change what was learned ----------------------------------------
def test_predicting_does_not_change_fitted_state(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    train, val = all_splits(dev)["temporal"][2]
    pipe = build_pipeline(Lasso(random_state=42), selector="f_test").fit(X.iloc[train], y.iloc[train])
    before = state_of(pipe)
    wild = X.iloc[val].copy()
    wild[["no2", "co", "o3"]] = np.nan                  # all Sentinel-5P missing
    wild["temperature_celsius"] = 1e6                   # absurd values
    wild["aod"] = 0.0
    pipe.predict(wild)
    pipe.predict(X.iloc[val])
    assert state_of(pipe) == before


# --- 3. learned numbers come from the training rows ---------------------------------------
def test_scaler_clipper_imputer_selector_use_train_statistics(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    train, val = all_splits(dev)["temporal"][3]
    pipe = build_pipeline(Lasso(random_state=42), selector="f_test").fit(X.iloc[train], y.iloc[train])

    feats = [c for c in X.columns if c not in NEVER_FEATURES]
    Xt = X.iloc[train][feats].copy()
    # clipper: limits equal to the ones of an independent clipper fitted on the training rows
    from src.outliers import OutlierClipper
    ref = OutlierClipper().fit(Xt)
    assert pipe["outliers"].lower_ == ref.lower_ and pipe["outliers"].upper_ == ref.upper_
    # imputer: medians of the clipped training rows
    clipped = ref.transform(Xt)
    assert np.allclose(pipe["impute"].medians_[feats], clipped[feats].median().fillna(0.0)[feats])
    # selector: F-scores of the imputed + scaled training rows only
    Z = pipe[:-2].transform(X.iloc[train])
    expected = np.nan_to_num(f_regression(Z.to_numpy(), y.iloc[train].to_numpy())[0])
    assert np.allclose(pipe["select"].scores_.to_numpy(), expected)
    # validation rows are transformed with the TRAINING mean and scale
    sc = pipe["scale"]
    imputed_val = pipe[:-3].transform(X.iloc[val])
    cols = sc.scaled_cols_
    manual = (imputed_val[cols] - sc.scaler_.mean_) / sc.scaler_.scale_
    assert np.allclose(manual.to_numpy(), sc.transform(imputed_val)[cols].to_numpy())
    assert np.allclose(sc.scaler_.mean_, pipe[:-3].transform(X.iloc[train])[cols].mean().to_numpy())


def test_selector_scores_ignore_validation_rows(dev):
    """A column that is a perfect copy of the target on the validation rows only must not get a high score."""
    X, y = dev.drop(columns="pm25").copy(), dev["pm25"]
    train, val = all_splits(dev)["temporal"][4]
    rng = np.random.default_rng(0)
    X["planted"] = rng.normal(size=len(X))
    X.iloc[val, X.columns.get_loc("planted")] = y.iloc[val].to_numpy()   # leak that exists only in val
    feats = [c for c in X.columns if c not in NEVER_FEATURES]
    pipe = build_pipeline(Ridge(), selector=FeatureSelector("f_test", k=3)).fit(X.iloc[train], y.iloc[train])
    train_only = f_regression(pipe[:-2].transform(X.iloc[train]).to_numpy(), y.iloc[train].to_numpy())[0]
    assert np.allclose(pipe["select"].scores_.to_numpy(), np.nan_to_num(train_only))
    # control: had the selector seen all rows, the planted column would score far higher
    full = pd.Series(f_regression(X[feats].fillna(X[feats].median()).to_numpy(), y.to_numpy())[0], index=feats)
    assert full["planted"] > 20 * max(pipe["select"].scores_["planted"], 1.0)


# --- 4. identifiers and the target never reach the model --------------------------------------
@pytest.mark.parametrize("scaled", [True, False])
def test_ids_coordinates_dates_target_never_reach_the_model(dev, scaled):
    X, y = dev, dev["pm25"]                              # the target column is even left IN the frame
    for train, val in all_splits(dev)["spatial"]:
        pipe = build_pipeline(ColumnRecorder(), scaled=scaled, selector="tree")
        pipe.fit(X.iloc[train], y.iloc[train])
        cols = set(pipe["model"].columns_)
        assert not cols & set(NEVER_FEATURES)
        assert not cols & {"station_id", "lat", "lon", "latitude", "longitude", "date"}
        # spatial folds: no station is in both sides
        assert not set(dev.iloc[train]["station_id"]) & set(dev.iloc[val]["station_id"])


def test_rows_are_never_dropped_or_reordered(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    train, val = all_splits(dev)["standard"][0]
    pipe = build_pipeline(Ridge(), selector="f_test").fit(X.iloc[train], y.iloc[train])
    out = pipe[:-1].transform(X.iloc[val])
    assert len(out) == len(val) and list(out.index) == list(X.index[val])
    assert out.isna().sum().sum() == 0


# --- 5. the real shared folds (skipped when the parquet is not on this machine) ------------------
REAL = config.INTERIM / "clean_dev.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="data/interim/clean_dev.parquet not present")
@pytest.mark.parametrize("scheme", ["temporal", "spatial", "standard"])
def test_real_folds_fit_on_train_only(scheme):
    from src.folds import load_folds
    real = pd.read_parquet(REAL)
    X, y = real.drop(columns="pm25"), real["pm25"]
    for train, val in load_folds(config.PARTS.parent / f"folds_{scheme}.json", real):
        pipe = build_pipeline(Lasso(random_state=42), selector="f_test")
        pipe.steps.insert(0, ("spy", RowSpy()))
        RowSpy.seen = []
        pipe.fit(X.iloc[train], y.iloc[train])
        pipe.predict(X.iloc[val])
        assert set(RowSpy.seen[0]) == set(X.index[train])
        assert not set(RowSpy.seen[0]) & set(X.index[val])
