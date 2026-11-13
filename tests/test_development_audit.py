import pandas as pd

from src.common import development_rows, detect_columns


def test_development_filter_excludes_holdout_and_unknown_dates():
    raw = pd.DataFrame({
        "date": ["2024-12-31", "2025-01-01", "2026-01-01", "invalid", None],
        "pm25": [1, 2, 3, 4, 5],
    })
    pd.testing.assert_frame_equal(development_rows(raw, "date"), raw.iloc[:1])


def test_auxiliary_schema_does_not_use_combined_file_overrides():
    found = detect_columns(["date", "id", "lat", "lon", "AOD"], use_overrides=False)
    assert found["lat"] == "lat"
    assert found["pm25"] is None
