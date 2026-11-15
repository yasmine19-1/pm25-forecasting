"""Tests for the two baselines reported by pipeline_check.py.   Run:  pytest tests/

  baseline_train_mean   always the mean of the training fold
  baseline_persistence  yesterday's PM2.5 of the same station, training mean when yesterday is missing
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

from src.pipeline_check import baseline_predictions
from test_m3_temporal_integration import continuous_frame


def setup():
    dev = continuous_frame()
    X, y = dev.drop(columns="pm25"), dev["pm25"]
    return dev, X, y


def test_train_mean_uses_training_rows_only():
    dev, X, y = setup()
    train, val = list(TimeSeriesSplit(4, gap=1).split(X))[1]
    pred, share = baseline_predictions(dev, X, y, train, val, "temporal")["baseline_train_mean"]
    assert share is None and len(pred) == len(val)
    assert np.allclose(pred, y.iloc[train].mean())
    assert not np.isclose(y.iloc[train].mean(), y.mean())


def test_persistence_is_yesterday_of_the_same_station():
    dev, X, y = setup()
    train, val = list(TimeSeriesSplit(4, gap=1).split(X))[2]
    pred, share = baseline_predictions(dev, X, y, train, val, "temporal")["baseline_persistence"]
    known = {(r.station_id, r.date): r.pm25 for r in dev.itertuples()}
    mean = y.iloc[train].mean()
    expected = np.array([known.get((r.station_id, r.date - pd.Timedelta(days=1)), mean)
                         for r in dev.iloc[val].itertuples()])
    assert np.allclose(pred, expected)
    has_yesterday = np.mean([(r.station_id, r.date - pd.Timedelta(days=1)) in known for r in dev.iloc[val].itertuples()])
    assert np.isclose(share, has_yesterday) and 0.8 < share < 1.0   # a few days are missing in the fake data


def test_no_persistence_baseline_for_spatial_folds():
    dev, X, y = setup()
    train, val = next(iter(GroupKFold(3).split(X, groups=dev["station_id"])))
    out = baseline_predictions(dev, X, y, train, val, "spatial")
    assert list(out) == ["baseline_train_mean"]
