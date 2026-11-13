"""Member 4 - tests for the TemporalFeatures pipeline step.

Run from the repo root:   python -m unittest discover -s tests -v     (or: pytest tests/)
The last test needs Member 3's pipeline and synthetic frame; it is skipped
until their files are merged into your branch.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features import TemporalFeatures, build_features

try:
    from m3_synthetic import make_dev_frame
    from src.pipeline import build_pipeline
    HAVE_PIPELINE = True
except ImportError:
    HAVE_PIPELINE = False


def small_frame():
    days = pd.date_range("2024-01-01", periods=12, freq="D").delete([5])   # day 6 missing
    n = len(days)
    return pd.DataFrame({
        "date": days,
        "station_id": "A",
        "gee_id": 1, "lat": 36.0, "lon": -5.0,
        "pm25": np.arange(10.0, 10.0 + n),
        "aod": np.linspace(0.1, 0.5, n),
        "no2": np.linspace(1e-5, 2e-5, n),
    })


class TestTemporalFeatures(unittest.TestCase):

    def test_same_rows_same_order_same_index(self):
        df = small_frame().sample(frac=1, random_state=1)           # shuffled rows
        out = TemporalFeatures().fit(df).transform(df)
        self.assertTrue(out.index.equals(df.index))
        self.assertTrue((out["date"].to_numpy() == df["date"].to_numpy()).all())
        self.assertEqual(len(out), len(df))

    def test_no_target_or_segment_columns(self):
        out = TemporalFeatures().fit(small_frame()).transform(small_frame())
        self.assertNotIn("target_pm25", out.columns)
        self.assertNotIn("segment_id", out.columns)

    def test_lag_values_and_gap(self):
        df = small_frame()
        out = TemporalFeatures().fit(df).transform(df).set_index("date")
        self.assertEqual(out.loc["2024-01-03", "pm25_lag_1"], 11.0)        # day 2
        self.assertTrue(pd.isna(out.loc["2024-01-07", "pm25_lag_1"]))      # day 6 missing
        self.assertTrue(pd.isna(out.loc["2024-01-07", "pm25_lag_3"]))      # would cross the gap

    def test_features_use_only_the_past(self):
        df = small_frame()
        before = TemporalFeatures().fit(df).transform(df)
        changed = df.copy()
        last = changed.index[-3:]                                          # change the last 3 days
        changed.loc[last, ["pm25", "aod", "no2"]] *= 100
        after = TemporalFeatures().fit(df).transform(changed)
        cols = [c for c in before.columns if c.endswith(("_lag_1", "_lag_3")) or "rolling" in c]
        untouched = before.index[:-3]                                      # rows before the change
        pd.testing.assert_frame_equal(before.loc[untouched, cols], after.loc[untouched, cols])

    def test_history_gives_validation_rows_their_lags(self):
        df = small_frame()
        val = df.iloc[7:].drop(columns="pm25")                             # 9 Jan onwards, no target
        alone = TemporalFeatures(use_target_lags=False).fit(val).transform(val)
        self.assertNotIn("pm25_lag_1", alone.columns)
        with_history = TemporalFeatures(history=df).fit(val).transform(val).set_index("date")
        previous_day = df.set_index("date").loc["2024-01-08", "pm25"]
        self.assertEqual(with_history.loc["2024-01-09", "pm25_lag_1"], previous_day)
        self.assertEqual(len(with_history), len(val))

    def test_threshold_is_learned_on_training_rows_only(self):
        df = small_frame()
        train, full = df.iloc[:6], df
        t_train = TemporalFeatures().fit(train).high_aod_threshold_
        t_full = TemporalFeatures().fit(full).high_aod_threshold_
        self.assertAlmostEqual(t_train, train["aod"].quantile(0.9))
        self.assertNotAlmostEqual(t_train, t_full)

    def test_next_day_removes_same_day_predictors(self):
        df = small_frame()
        out = TemporalFeatures(scenario="next_day").fit(df).transform(df)
        self.assertNotIn("aod", out.columns)
        self.assertNotIn("no2", out.columns)
        self.assertIn("aod_lag_1", out.columns)

    def test_same_day_keeps_predictors_and_has_no_pm25_lags(self):
        df = small_frame()
        out = TemporalFeatures(scenario="same_day").fit(df).transform(df)
        self.assertIn("aod", out.columns)
        self.assertFalse([c for c in out.columns if c.startswith("pm25_")])

    def test_missing_pm25_history_is_an_error(self):
        df = small_frame().drop(columns="pm25")
        with self.assertRaises(ValueError):
            TemporalFeatures().fit(df).transform(df)

    def test_fast_version_keeps_horizon_check(self):
        with self.assertRaises(ValueError):
            build_features(small_frame(), forecast_horizon_days=0)


@unittest.skipUnless(HAVE_PIPELINE, "needs Member 3's pipeline.py and m3_synthetic.py")
class TestInsideMember3Pipeline(unittest.TestCase):

    def test_pipeline_runs_and_model_sees_no_ids_or_target(self):
        from sklearn.base import BaseEstimator, RegressorMixin

        class Recorder(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                self.columns_ = list(X.columns)
                self.mean_ = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_)

        dev = make_dev_frame(n_stations=6)
        X, y = dev.drop(columns="pm25"), dev["pm25"]
        train = np.flatnonzero((dev["date"] < "2022-01-01").to_numpy())
        val = np.flatnonzero((dev["date"] >= "2022-01-02").to_numpy())

        pipe = build_pipeline(Recorder(), features=TemporalFeatures(history=dev))
        pipe.fit(X.iloc[train], y.iloc[train])
        prediction = pipe.predict(X.iloc[val])

        self.assertEqual(len(prediction), len(val))
        columns = pipe["model"].columns_
        for banned in ["station_id", "date", "lat", "lon", "gee_id", "pm25",
                       "target_pm25", "segment_id", "aod", "no2"]:
            self.assertNotIn(banned, columns)
        self.assertIn("pm25_lag_1", columns)
        self.assertIn("days_since_last_high_aod_day", columns)


if __name__ == "__main__":
    unittest.main()
