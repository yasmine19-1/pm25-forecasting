import runpy

import pandas as pd

from src import config
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


def test_cleaner_uses_only_dated_development_rows(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / config.RAW_FILE
    source.write_text(
        "date,latitude,longitude,sensor_name,pm25\n"
        "2024-12-30,36,3,dev,10\n"
        "2024-12-31,36,3,dev,0\n"
        "2025-01-01,36,3,reserved,0\n"
        "2025-01-01,36,3,reserved,0\n"
        "2026-01-01,36,3,set-aside,900\n"
        "unknown,36,3,undated,0\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    interim, parts = tmp_path / "interim", tmp_path / "results" / "parts"
    reserved = tmp_path / "final_test"
    reserved.mkdir()
    existing = reserved / "final_test.parquet"
    existing.write_bytes(b"reserved file must stay untouched")
    for name, path in (
        ("ROOT", tmp_path), ("RAW", raw), ("INTERIM", interim),
        ("PARTS", parts), ("FINAL_TEST", reserved),
    ):
        monkeypatch.setattr(config, name, path)

    runpy.run_module("src.clean", run_name="__main__")

    dev = pd.read_parquet(interim / "clean_dev.parquet")
    assert dev["pm25"].tolist() == [10]
    assert dev["date"].tolist() == [pd.Timestamp("2024-12-30")]
    log = pd.read_csv(parts / "cleaning_log.csv").set_index("step")
    assert log.loc["0_start", "rows_before"] == 2
    assert log.loc["2_pm25_nonpositive", "rows_removed"] == 1
    counts = pd.read_csv(parts / "cleaning_counts.csv")
    assert counts.loc[counts.section.eq("date_screening"), "value"].tolist() == [1]
    assert counts.loc[counts.section.eq("split"), "variable"].unique().tolist() == ["development"]
    assert source.read_bytes() == original
    assert list(reserved.iterdir()) == [existing]
    assert existing.read_bytes() == b"reserved file must stay untouched"
