
import unittest
import pandas as pd
import numpy as np

from src.features import build_features


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-05",
                "2024-01-06"
            ]),
            "station_id": ["Station A"] * 5,
            "pm25": [10, 20, 30, 40, 50],
            "aod": [0.1, 0.2, 0.3, 0.4, 0.5],
            "no2": [5, 6, 7, 8, 9]
        })

    def test_output_has_expected_rows(self):
        result = build_features(self.data, forecast_horizon_days=1)
        self.assertEqual(len(result), 5)

    def test_lag_uses_previous_day(self):
        result = build_features(self.data, forecast_horizon_days=1)
        row = result[result["date"] == pd.Timestamp("2024-01-02")].iloc[0]
        self.assertEqual(row["pm25_lag_1"], 10)

    def test_features_do_not_cross_date_gaps(self):
        result = build_features(self.data, forecast_horizon_days=1)
        row = result[result["date"] == pd.Timestamp("2024-01-05")].iloc[0]
        self.assertTrue(pd.isna(row["pm25_lag_1"]))

    def test_target_is_future_pm25(self):
        result = build_features(self.data, forecast_horizon_days=1)
        row = result[result["date"] == pd.Timestamp("2024-01-01")].iloc[0]
        self.assertEqual(row["target_pm25"], 20)

    def test_target_does_not_cross_date_gaps(self):
        result = build_features(self.data, forecast_horizon_days=1)
        row = result[result["date"] == pd.Timestamp("2024-01-03")].iloc[0]
        self.assertTrue(pd.isna(row["target_pm25"]))

    def test_invalid_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            build_features(self.data, forecast_horizon_days=0)

    def test_aod_recency_requires_training_threshold(self):
        result = build_features(self.data, forecast_horizon_days=1)
        self.assertNotIn("days_since_last_high_aod_day", result.columns)

        result_with_threshold = build_features(
            self.data,
            forecast_horizon_days=1,
            high_aod_threshold=0.25
        )
        self.assertIn(
            "days_since_last_high_aod_day",
            result_with_threshold.columns
        )


    def test_features_do_not_cross_metadata_changes(self):
        data = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-01", "2024-01-02", "2024-01-03"
            ]),
            "station_id": ["Station A"] * 3,
            "lat": [36.0, 36.0, 37.0],
            "pm25": [10, 20, 30],
            "aod": [0.1, 0.2, 0.3]
        })

        result = build_features(data, forecast_horizon_days=1)
        row = result[result["date"] == pd.Timestamp("2024-01-03")].iloc[0]
        self.assertTrue(pd.isna(row["pm25_lag_1"]))


def test_rolling_features_preserve_alignment_and_reset_at_boundaries():
    from src.features import rolling_forecast_features

    data = pd.DataFrame({
        "station_id": ["a", "a", "a", "a", "b"],
        "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-01",
                                "2024-01-05", "2024-01-02"]),
        "gee_id": [2, 1, 1, 2, 1],
        "pm25": [30, 20, 10, 50, 100],
        "aod": [.3, .2, .1, .5, .9],
    }, index=[7, 6, 2, 9, 13])
    result = rolling_forecast_features(data)
    assert result.index.equals(data.index)
    assert set(result.columns) == {
        "day_of_year_sin", "day_of_year_cos", "pm25_lag_1", "pm25_lag_1_missing",
    }
    assert result.loc[6, "pm25_lag_1"] == 10
    assert result.loc[[7, 2, 9, 13], "pm25_lag_1"].isna().all()
    assert result["pm25_lag_1_missing"].tolist() == [1, 0, 1, 1, 1]


def test_pipeline_drops_exploratory_future_target():
    from src.pipeline import DropNonFeatures

    preview = pd.DataFrame({"pm25": [10], "target_pm25": [99],
                            "segment_id": [1], "aod": [.3]})
    assert DropNonFeatures().fit_transform(preview).columns.tolist() == ["aod"]


def test_rolling_features_use_only_the_previous_days_label():
    from src.features import rolling_forecast_features

    observations = pd.DataFrame({
        "station_id": ["a"] * 4,
        "date": pd.date_range("2024-01-01", periods=4),
        "pm25": [10.0, 20.0, 30.0, 40.0],
    })
    baseline = rolling_forecast_features(observations)
    # Days 2 through 4 are validation dates. Changing day 3 can affect day 4
    # under the rolling assumption, but it cannot affect day 3 or earlier.
    changed = observations.copy()
    changed.loc[2, "pm25"] = 300.0
    updated = rolling_forecast_features(changed)
    pd.testing.assert_frame_equal(baseline.iloc[:3], updated.iloc[:3])
    assert updated.loc[2, "pm25_lag_1"] == 20.0
    assert updated.loc[3, "pm25_lag_1"] == 300.0


if __name__ == "__main__":
    unittest.main()
