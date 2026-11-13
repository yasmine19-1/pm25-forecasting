"""Tests for the within-station check in stratified.py.   Run:  pytest tests/"""
import numpy as np
import pandas as pd

from src.stratified import pattern, verdict, within_station_slopes


def stations_that_differ_but_have_no_daily_link(seed=0):
    """Station level drives both AOD and PM2.5 (think altitude), but inside a station they are unrelated."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(6):
        n = 300
        parts.append(pd.DataFrame({
            "station_id": f"S{i}",
            "pressure_mb": 940 + 20 * i + rng.normal(0, 2, n),
            "aod": 0.1 + 0.1 * i + rng.normal(0, 0.02, n),
            "pm25": 5 + 5 * i + rng.normal(0, 1, n)}))
    return pd.concat(parts, ignore_index=True)


def test_station_differences_are_not_mistaken_for_a_daily_link():
    d = stations_that_differ_but_have_no_daily_link()
    pooled_slope = np.polyfit(d["aod"], d["pm25"], 1)[0]
    inside, counts, n_stations = within_station_slopes(d, "pressure_mb")
    assert pooled_slope > 30                      # looks like a strong link when stations are mixed
    assert all(abs(s) < 5 for s in inside)        # but there is none inside a station
    assert n_stations == 6 and sum(counts) == len(d)


def test_real_daily_link_survives_the_within_station_check():
    rng = np.random.default_rng(1)
    d = stations_that_differ_but_have_no_daily_link()
    d["pm25"] = d["pm25"] + 20 * (d["aod"] - d.groupby("station_id")["aod"].transform("mean")) + rng.normal(0, 0.1, len(d))
    inside, _, _ = within_station_slopes(d, "pressure_mb")
    assert all(15 < s < 25 for s in inside)


def test_thin_stations_are_left_out():
    d = stations_that_differ_but_have_no_daily_link()
    thin = d[d["station_id"] == "S0"].head(40)
    d = pd.concat([d[d["station_id"] != "S0"], thin])
    _, _, n_stations = within_station_slopes(d, "pressure_mb")
    assert n_stations == 5


def test_pattern_wording():
    assert pattern([1, 2, 3]) == "rises steadily"
    assert pattern([3, 2, 1]) == "falls steadily"
    assert pattern([5, 2, 4]) == "is lowest in the middle third"
    assert pattern([2, 5, 3]) == "is highest in the middle third"
    assert "too few rows" in pattern([1, np.nan, 3])


def test_verdict_wording():
    assert "persists" in verdict([10, 8, 2], [9, 7, 3])
    assert "weaker" in verdict([10, 8, 2], [5, 4, 3])
    assert "other way" in verdict([10, 8, 2], [3, 4, 6])
    assert verdict([1, 2, 3], [np.nan, 1, 2]) == ""
