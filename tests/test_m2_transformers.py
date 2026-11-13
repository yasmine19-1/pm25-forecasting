"""Tests for imputation.py and outliers.py.  Run:  pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imputation import S5PImputer  # noqa: E402
from outliers import OutlierClipper  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "no2": rng.normal(9e-5, 2e-5, n), "co": rng.normal(.03, .004, n),
        "o3": rng.normal(.14, .01, n), "aod": rng.gamma(2, .07, n),
        "temperature_celsius": rng.normal(18, 6, n),
    })
    X.loc[:19, ["no2", "co", "o3"]] = np.nan
    X.loc[25, "aod"] = 0.0
    return X.iloc[:150].copy(), X.iloc[150:].copy()


def test_imputer_learns_from_train_only(data):
    tr, va = data
    imp = S5PImputer().fit(tr)
    va_extreme = va.copy()
    va_extreme.loc[:, "no2"] = np.nan
    va_extreme.iloc[0, va_extreme.columns.get_loc("co")] = 99.0
    out = imp.transform(va_extreme)
    assert np.allclose(out["no2"], tr["no2"].median())
    assert imp.medians_["co"] == tr["co"].median()


def test_imputer_indicators_and_no_nan(data):
    tr, _ = data
    out = S5PImputer().fit_transform(tr)
    assert out.isna().sum().sum() == 0
    assert out["no2_missing"].sum() == 20
    assert list(out.columns[-3:]) == ["no2_missing", "co_missing", "o3_missing"]
    assert list(S5PImputer().fit(tr).get_feature_names_out()) == list(out.columns)


def test_imputer_no_forward_fill(data):
    tr, _ = data
    out = S5PImputer().fit_transform(tr)
    # every imputed value is the median, not the previous row's value
    assert out.loc[:19, "no2"].nunique() == 1


def test_clipper_train_limits_applied_unchanged(data):
    tr, va = data
    clip = OutlierClipper().fit(tr)
    va2 = va.copy()
    va2.iloc[0, va2.columns.get_loc("temperature_celsius")] = 500.0
    out = clip.transform(va2)
    assert out["temperature_celsius"].iloc[0] == pytest.approx(clip.upper_["temperature_celsius"])
    assert len(out) == len(va2)  # rows are never dropped


def test_disabling_clipping_keeps_aod_normalization(data):
    tr, va = data
    clip = OutlierClipper(enabled=False).fit(tr)
    va = va.copy()
    va.iloc[0, va.columns.get_loc("aod")] = 0.0
    expected = va.assign(aod=va.aod.mask(va.aod.eq(0)))
    pd.testing.assert_frame_equal(clip.transform(va), expected)


def test_clipper_aod_zero_becomes_nan_and_input_untouched(data):
    tr, _ = data
    before = tr.copy()
    out = OutlierClipper().fit(tr).transform(tr)
    assert np.isnan(out.loc[25, "aod"])
    pd.testing.assert_frame_equal(tr, before)


def test_in_pipeline_and_switchable(data):
    tr, va = data
    y = tr["temperature_celsius"] * 0.5
    X = tr.drop(columns="temperature_celsius")
    pipe = Pipeline([("outliers", OutlierClipper()), ("impute", S5PImputer()),
                     ("model", LinearRegression())])
    pipe.fit(X, y)
    pipe.predict(va.drop(columns="temperature_celsius"))
    p2 = clone(pipe).set_params(outliers__enabled=False)
    p2.fit(X, y)
    assert p2.named_steps["outliers"].lower_ == {}
