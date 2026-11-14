"""Tests for the Pipeline together with TemporalFeatures (next-day scenario).   Run:  pytest tests/

What is checked
  1. the whole Pipeline, features step included, is fitted on training rows only
  2. the high-AOD threshold comes from the training rows
  3. ids, labels and same-day predictors never reach the model
  4. spatial folds get no PM2.5 lags, temporal folds do
  5. END TO END: changing days after day d0 cannot change a prediction made for a day up to d0
     (and changing days before d0 does, so the test is sensitive)
  6. the optional imputer shows up in the Pipeline
  7. the same holds on the real shared folds, when the real parquet exists
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from src import config
from src.imputation import S5PImputer
from src.pipeline import build_pipeline, temporal_step


def continuous_frame(n_stations=6, seed=7):
    """Daily data for 2020-2021, a few missing days per station, columns as in clean_dev.parquet."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(n_stations):
        days = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        days = days[rng.random(len(days)) > 0.04]
        n = len(days)
        aod = np.abs(0.15 + 0.08 * rng.normal(size=n))
        aod[rng.random(n) < 0.02] = 0.0
        parts.append(pd.DataFrame({
            "date": days, "datetime_utc": days, "gee_id": 100 + s, "lat": 36.0 + s, "lon": -5.0 + s,
            "temperature_celsius": 18 + 8 * np.sin(2 * np.pi * days.day_of_year / 365) + rng.normal(0, 2, n),
            "pressure_mb": 1000 + rng.normal(0, 6, n),
            "wind_u": rng.normal(0, 3, n), "wind_v": rng.normal(0, 3, n),
            "no2": 9e-5 + 2e-5 * rng.normal(size=n), "co": 0.03 + 0.004 * rng.normal(size=n),
            "o3": 0.14 + 0.01 * rng.normal(size=n), "aod": aod,
            "pm25": np.exp(2 + 1.2 * aod + 0.4 * rng.normal(size=n)),
            "station_id": f"ST{s}"}))
    d = pd.concat(parts).sort_values(["date", "station_id"]).reset_index(drop=True)
    d.loc[rng.random(len(d)) < 0.08, ["no2", "co", "o3"]] = np.nan
    return d


class RowSpy(BaseEstimator, TransformerMixin):
    seen = []

    def fit(self, X, y=None):
        RowSpy.seen.append(np.asarray(X.index))
        return self

    def transform(self, X):
        return X


class Recorder(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.columns_ = list(X.columns)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


@pytest.fixture(scope="module")
def dev():
    return continuous_frame()


def splits(dev):
    X = dev.drop(columns="pm25")
    return {"temporal": list(TimeSeriesSplit(4, gap=1).split(X)),
            "spatial": list(GroupKFold(3).split(X, groups=dev["station_id"])),
            "standard": list(KFold(3, shuffle=True, random_state=42).split(X))}


# --- 1 and 2 -------------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", ["temporal", "spatial", "standard"])
def test_every_step_including_features_is_fitted_on_training_rows_only(dev, scheme):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    for train, val in splits(dev)[scheme]:
        pipe = build_pipeline(Ridge(), selector="f_test", features=temporal_step(scheme, dev))
        pipe.steps.insert(0, ("spy", RowSpy()))
        RowSpy.seen = []
        pipe.fit(X.iloc[train], y.iloc[train])
        pipe.predict(X.iloc[val])
        assert len(RowSpy.seen) == 1 and set(RowSpy.seen[0]) == set(X.index[train])
        assert not set(RowSpy.seen[0]) & set(X.index[val])


def test_high_aod_threshold_comes_from_training_rows(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    train, _ = splits(dev)["temporal"][0]
    pipe = build_pipeline(Ridge(), features=temporal_step("temporal", dev)).fit(X.iloc[train], y.iloc[train])
    aod = X.iloc[train]["aod"].mask(X.iloc[train]["aod"] == 0).dropna()
    full = X["aod"].mask(X["aod"] == 0).dropna()
    assert np.isclose(pipe["features"].high_aod_threshold_, aod.quantile(0.9))
    assert not np.isclose(pipe["features"].high_aod_threshold_, full.quantile(0.9))


# --- 3 and 4 -------------------------------------------------------------------------------
def test_model_never_sees_ids_labels_or_same_day_predictors(dev):
    X, y = dev, dev["pm25"]                              # even with the target left in the frame
    train, _ = splits(dev)["temporal"][1]
    pipe = build_pipeline(Recorder(), features=temporal_step("temporal", dev)).fit(X.iloc[train], y.iloc[train])
    cols = set(pipe["model"].columns_)
    for banned in ("station_id", "date", "datetime_utc", "gee_id", "lat", "lon", "pm25",
                   "target_pm25", "segment_id", "aod", "no2", "co", "o3",
                   "temperature_celsius", "pressure_mb", "wind_u", "wind_v"):
        assert banned not in cols
    assert "pm25_lag_1" in cols and "aod_lag_1" in cols


def test_spatial_has_no_pm25_lags_but_temporal_has(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    for scheme, expected in (("spatial", False), ("temporal", True), ("standard", True)):
        train, _ = splits(dev)[scheme][0]
        pipe = build_pipeline(Recorder(), features=temporal_step(scheme, dev)).fit(X.iloc[train], y.iloc[train])
        has = any(c.startswith("pm25_") for c in pipe["model"].columns_)
        assert has is expected, scheme


def test_same_rows_same_order_through_the_features_step(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    train, val = splits(dev)["temporal"][2]
    pipe = build_pipeline(Ridge(), selector="f_test", features=temporal_step("temporal", dev)).fit(X.iloc[train], y.iloc[train])
    out = pipe[:-1].transform(X.iloc[val])
    assert list(out.index) == list(X.index[val]) and out.isna().sum().sum() == 0


# --- 5. no future leakage, end to end -------------------------------------------------------------
def _pipe(dev_history):
    return build_pipeline(Ridge(), selector=None, features=temporal_step("temporal", dev_history))


def test_changing_the_future_cannot_change_earlier_predictions(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    train = np.flatnonzero((dev["date"] < "2021-03-01").to_numpy())
    val = np.flatnonzero((dev["date"] >= "2021-03-02").to_numpy())
    d0 = pd.Timestamp("2021-06-30")
    early = val[(dev["date"].iloc[val] <= d0).to_numpy()]           # days we predict
    columns = ["pm25", "aod", "no2", "co", "o3", "temperature_celsius", "pressure_mb"]

    future = dev.copy()                                              # days AFTER d0 changed wildly
    future.loc[future["date"] > d0, columns] = future.loc[future["date"] > d0, columns] * 10
    past = dev.copy()                                                # days BEFORE d0 (in April) changed
    april = (past["date"] > "2021-04-01") & (past["date"] < "2021-04-30")
    past.loc[april, columns] = past.loc[april, columns] * 10

    base = _pipe(dev).fit(X.iloc[train], y.iloc[train]).predict(X.iloc[early])
    after_future_change = _pipe(future).fit(X.iloc[train], y.iloc[train]).predict(X.iloc[early])
    after_past_change = _pipe(past).fit(X.iloc[train], y.iloc[train]).predict(X.iloc[early])

    assert np.allclose(base, after_future_change)                    # the future has no say
    assert not np.allclose(base, after_past_change)                  # the past does: the test can fail


# --- 6. optional imputer -----------------------------------------------------------------------------
def test_optional_imputer_adds_flags_for_lag_columns(dev):
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    train, val = splits(dev)["temporal"][1]
    imputer = S5PImputer(s5p_cols=("no2", "co", "o3", "pm25_lag_1"))
    pipe = build_pipeline(Recorder(), features=temporal_step("temporal", dev), imputer=imputer)
    pipe.fit(X.iloc[train], y.iloc[train])
    assert "pm25_lag_1_missing" in pipe["model"].columns_
    default = build_pipeline(Recorder(), features=temporal_step("temporal", dev)).fit(X.iloc[train], y.iloc[train])
    assert "pm25_lag_1_missing" not in default["model"].columns_


# --- 7. real shared folds (skipped when the parquet is not on this machine) ----------------------------
REAL = config.INTERIM / "clean_dev.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="data/interim/clean_dev.parquet not present")
@pytest.mark.parametrize("scheme", ["temporal", "spatial", "standard"])
def test_real_folds_with_temporal_features_fit_on_train_only(scheme):
    from src.folds import load_folds
    real = pd.read_parquet(REAL)
    X, y = real.drop(columns="pm25"), real["pm25"]
    for train, val in load_folds(config.PARTS.parent / f"folds_{scheme}.json", real):
        pipe = build_pipeline(Ridge(), selector="f_test", features=temporal_step(scheme, real))
        pipe.steps.insert(0, ("spy", RowSpy()))
        RowSpy.seen = []
        pipe.fit(X.iloc[train], y.iloc[train])
        pipe.predict(X.iloc[val])
        assert set(RowSpy.seen[0]) == set(X.index[train])
        assert not set(RowSpy.seen[0]) & set(X.index[val])
