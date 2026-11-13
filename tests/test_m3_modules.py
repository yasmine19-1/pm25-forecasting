"""Tests for scaling.py, selection.py and the build_pipeline options.   Run:  pytest tests/"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from m3_synthetic import make_dev_frame
from src.pipeline import build_pipeline
from src.scaling import TrainFoldScaler, needs_scaling
from src.selection import METHODS, FeatureSelector, describe_selection


@pytest.fixture(scope="module")
def dev():
    return make_dev_frame()


# --- scaling ------------------------------------------------------------------------------
def test_needs_scaling_by_family():
    assert all(needs_scaling(m) for m in (SVR(), KNeighborsRegressor(), Lasso(), "MLPRegressor"))
    assert not any(needs_scaling(m) for m in (ExtraTreesRegressor(), "XGBRegressor", "LGBMRegressor", "CatBoostRegressor"))


@pytest.mark.parametrize("kind", ["standard", "robust"])
def test_scaler_uses_training_statistics_and_skips_indicators(kind):
    rng = np.random.default_rng(1)
    tr = pd.DataFrame({"a": rng.normal(10, 3, 100), "b": rng.normal(0, 1, 100), "no2_missing": rng.integers(0, 2, 100)})
    va = pd.DataFrame({"a": [1000.0, 2000.0], "b": [5.0, -5.0], "no2_missing": [1, 0]})
    s = TrainFoldScaler(kind).fit(tr)
    out = s.transform(va)
    assert s.scaled_cols_ == ["a", "b"]
    assert list(out["no2_missing"]) == [1, 0]
    if kind == "standard":
        assert np.isclose(out["a"].iloc[0], (1000 - tr["a"].mean()) / tr["a"].std(ddof=0))
        assert np.allclose(s.transform(tr)[["a", "b"]].mean(), 0, atol=1e-9)
    else:
        assert np.isclose(s.scaler_.center_[0], tr["a"].median())
    assert list(out.columns) == list(va.columns)


def test_scaler_rejects_unknown_kind():
    with pytest.raises(ValueError):
        TrainFoldScaler("minmax").fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))


# --- selection ----------------------------------------------------------------------------
def make_xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    y = 3 * X["a"] - 2 * X["b"] + 0.1 * rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("method", METHODS)
def test_every_method_finds_the_real_signal(method):
    X, y = make_xy()
    sel = FeatureSelector(method, k=2, alpha=0.1).fit(X, y)
    assert {"a", "b"} <= set(sel.selected_features_)
    assert sel.n_selected_ == len(sel.selected_features_)
    assert list(sel.transform(X).columns) == sel.selected_features_
    row = describe_selection(sel)
    assert row["n_selected"] == sel.n_selected_ and row["n_input"] == 6


@pytest.mark.parametrize("method", METHODS)
def test_selection_is_reproducible(method):
    X, y = make_xy()
    a = FeatureSelector(method, k=3).fit(X, y)
    b = FeatureSelector(method, k=3).fit(X, y)
    assert a.selected_features_ == b.selected_features_


@pytest.mark.parametrize("method", METHODS)
def test_constant_columns_do_not_break_selection(method):
    """First temporal fold: Sentinel-5P all missing -> imputer makes constant columns."""
    X, y = make_xy()
    X["no2"] = 0.0
    X["no2_missing"] = 1
    sel = FeatureSelector(method, k=3).fit(X, y)
    assert sel.n_selected_ >= 1
    assert "no2" not in sel.selected_features_ or method == "l1"


def test_l1_never_returns_zero_features():
    X, y = make_xy()
    sel = FeatureSelector("l1", alpha=1e6, min_features=2).fit(X, y)   # penalty so big every coef is 0
    assert sel.n_selected_ == 2


def test_selector_rejects_unknown_method():
    X, y = make_xy()
    with pytest.raises(ValueError):
        FeatureSelector("magic").fit(X, y)


# --- build_pipeline -------------------------------------------------------------------------
def test_scaling_decided_by_model_family(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    scaled = build_pipeline(SVR()).fit(X.iloc[:300], y.iloc[:300])
    unscaled = build_pipeline(ExtraTreesRegressor(10, random_state=42)).fit(X.iloc[:300], y.iloc[:300])
    assert "scale" in scaled.named_steps and "scale" not in unscaled.named_steps
    assert "scale" in build_pipeline(ExtraTreesRegressor(), scaled=True).named_steps


def test_outlier_switch_reaches_the_clipper(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    pipe = build_pipeline(Ridge())
    pipe.set_params(outliers__enabled=False).fit(X.iloc[:500], y.iloc[:500])
    assert pipe["outliers"].enabled is False
    assert build_pipeline(Ridge(), clip_outliers=False)["outliers"].enabled is False


@pytest.mark.parametrize("selector", [None, "f_test", "mutual_info", "rfe", "l1", "tree"])
@pytest.mark.parametrize("model", [Ridge(), ExtraTreesRegressor(20, random_state=42)])
def test_every_variant_fits_and_predicts_on_the_hardest_fold(dev, selector, model):
    """First temporal fold: training rows have no Sentinel-5P value at all."""
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    train = np.where(X["date"] < "2018-06-01")[0]
    val = np.where(X["date"] >= "2018-06-02")[0][:200]
    assert dev.iloc[train][["no2", "co", "o3"]].isna().all().all()
    pred = build_pipeline(model, selector=selector).fit(X.iloc[train], y.iloc[train]).predict(X.iloc[val])
    assert np.isfinite(pred).all() and len(pred) == len(val)
