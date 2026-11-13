"""Member 4: past-only temporal features.

Two things live here:

* build_features(...)   the feature table (one row per station and day). Fast,
                        vectorised version of the original function; the numbers
                        match the original on unchanged station segments.
* TemporalFeatures      a scikit-learn step that plugs into Member 3's pipeline:
                        build_pipeline(model, features=TemporalFeatures(history=available_history))
                        It returns the SAME rows in the SAME order as it receives,
                        with the new columns added.

Rules kept everywhere: a feature for day t only uses days before t, no feature
crosses a missing-date gap or a change of station metadata, and anything learned
from data (the high-AOD threshold) is learned in fit() on the training rows only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src import config

VARIABLES = [
    "pm25", "aod", "no2", "co", "o3",
    "temperature_celsius", "pressure_mb", "wind_u", "wind_v",
]
KEYS = ["station_id", "date"]
METADATA = ["gee_id", "lat", "lon"]


def build_features(data, forecast_horizon_days=1, high_aod_threshold=None):
    if forecast_horizon_days < 1:
        raise ValueError("Forecast horizon must be at least 1 day.")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "station_id"])

    variables = [v for v in VARIABLES if v in data.columns]
    metadata = [c for c in METADATA if c in data.columns]

    if data.empty:
        return pd.DataFrame()

    # --- one row per station and day (median of the day's rows) -----------------
    grouped = data.groupby(KEYS, sort=True)
    daily = grouped[variables].median()
    mixed = pd.Series(False, index=daily.index)
    for column in metadata:
        daily[column] = grouped[column].first()           # first non-missing value
        mixed = mixed | (grouped[column].nunique() > 1)   # two different values that day
    daily["_mixed"] = mixed
    daily = daily.reset_index().sort_values(KEYS).reset_index(drop=True)

    # --- segments: a new one after a date gap, a metadata change, a mixed day ---
    station = daily["station_id"]
    new_station = station.ne(station.shift(1))
    date_gap = daily["date"].diff().dt.days.ne(1)
    changed = pd.Series(False, index=daily.index)
    for column in metadata:
        current, previous = daily[column], daily[column].shift(1)
        same = current.notna() & current.eq(previous).fillna(False)
        changed = changed | ~same
    mixed_boundary = daily["_mixed"] | daily["_mixed"].shift(1, fill_value=False)
    breaks = new_station | date_gap | changed | mixed_boundary
    segment = breaks.cumsum()                              # unique over the whole table
    daily["segment_id"] = (
        segment - segment.groupby(station).transform("min")
    ).astype(int)                                          # 0, 1, 2 ... per station
    groups = daily.groupby(segment)

    # --- calendar features (known in advance) -----------------------------------
    day_of_year = daily["date"].dt.day_of_year
    new_columns = {
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
    }

    # --- past-only features, one variable at a time -----------------------------
    for variable in variables:
        s1 = groups[variable].shift(1)
        s2 = groups[variable].shift(2)
        s3 = groups[variable].shift(3)
        window = np.vstack([s1.to_numpy(float), s2.to_numpy(float), s3.to_numpy(float)])

        new_columns[f"{variable}_lag_1"] = s1
        new_columns[f"{variable}_lag_3"] = s3
        new_columns[f"{variable}_rolling_mean_3"] = pd.Series(window.mean(axis=0), index=daily.index)
        new_columns[f"{variable}_rolling_std_3"] = pd.Series(window.std(axis=0, ddof=1), index=daily.index)
        new_columns[f"{variable}_rolling_max_3"] = pd.Series(window.max(axis=0), index=daily.index)
        new_columns[f"{variable}_difference_1"] = s1 - s2
        new_columns[f"{variable}_rate_of_change_1"] = (s1 - s2) / s2.abs().replace(0, np.nan)

    # --- days since the last high-AOD day (threshold learned on training rows) --
    if "aod" in variables and high_aod_threshold is not None:
        position = groups.cumcount()
        high_position = position.where(daily["aod"] > high_aod_threshold)
        last_high = high_position.groupby(segment).ffill()
        last_before_today = last_high.groupby(segment).shift(1)
        new_columns["days_since_last_high_aod_day"] = position - last_before_today

    # --- the next-day target, kept only for the standalone preview --------------
    target = groups["pm25"].shift(-forecast_horizon_days) if "pm25" in variables else np.nan
    new_columns["target_pm25"] = target

    result = pd.concat([daily.drop(columns=["_mixed", "segment_id"]),
                        pd.DataFrame(new_columns)], axis=1)
    result["segment_id"] = daily["segment_id"]
    return result


class TemporalFeatures(TransformerMixin, BaseEstimator):
    """Adds past-only features to the rows it receives (same rows, same order).

    Parameters
    ----------
    scenario : "next_day" (default) or "same_day"
        next_day  predict day t from what is known at the end of day t-1:
                  the raw same-day predictors (aod, no2, ...) are removed and
                  only their past values (lags, rolling stats) stay.
        same_day  keep the same-day predictors; PM2.5 lags are NOT built.
    use_target_lags : None, True or False
        PM2.5 lag features. None = True for next_day, False for same_day.
        Set False for held-out-station (spatial) experiments.
        same_day requires False or None.
    history : DataFrame or None
        History available at the forecast origin. For rolling temporal validation,
        earlier validation observations may be used only when they would
        already have been released. Supply fold-specific history for other
        experiments and exclude held-out stations for cold-start evaluation.
    aod_quantile : float
        high-AOD threshold = this quantile of the TRAINING aod values.
    """

    def __init__(self, scenario="next_day", use_target_lags=None, history=None,
                 aod_quantile=0.9):
        self.scenario = scenario
        self.use_target_lags = use_target_lags
        self.history = history
        self.aod_quantile = aod_quantile

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        if self.scenario not in ("next_day", "same_day"):
            raise ValueError("scenario must be 'next_day' or 'same_day'")
        if self.scenario == "same_day" and self.use_target_lags:
            raise ValueError("Same-day nowcasting cannot use PM2.5 lags.")
        X = self._check(X)
        self.use_target_lags_ = (self.scenario == "next_day"
                                 if self.use_target_lags is None else bool(self.use_target_lags))
        aod = X["aod"].mask(X["aod"] == 0).dropna() if "aod" in X.columns else pd.Series(dtype=float)
        self.high_aod_threshold_ = float(aod.quantile(self.aod_quantile)) if len(aod) else None
        return self

    def transform(self, X):
        check_is_fitted(self, "high_aod_threshold_")
        X = self._check(X)
        base = self._base_frame(X)

        table = build_features(base, forecast_horizon_days=1,
                               high_aod_threshold=self.high_aod_threshold_)
        keep_out = set(KEYS) | set(VARIABLES) | set(METADATA) | {"target_pm25", "segment_id"}
        new_cols = [c for c in table.columns if c not in keep_out]
        if not self.use_target_lags_:
            new_cols = [c for c in new_cols if not c.startswith("pm25_")]
        elif not base["pm25"].notna().any():
            raise ValueError("PM2.5 lag features need pm25 history: pass history=<development frame> "
                             "or set use_target_lags=False")

        keys = pd.DataFrame({"station_id": X["station_id"].to_numpy(),
                             "date": self._day(X["date"]).to_numpy()})
        merged = keys.merge(table[KEYS + new_cols], on=KEYS, how="left")
        if len(merged) != len(X):
            raise ValueError("feature table is not unique per station and day")

        inputs = KEYS + METADATA
        if self.scenario == "same_day":
            inputs += [column for column in VARIABLES if column != "pm25"]
        result = X.loc[:, X.columns.isin(inputs)].copy()
        for column in new_cols:
            result[column] = merged[column].to_numpy()
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _day(dates):
        return pd.to_datetime(dates, errors="coerce").dt.normalize()

    def _check(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TemporalFeatures needs a DataFrame with station_id and date")
        missing = [c for c in KEYS if c not in X.columns]
        if missing:
            raise ValueError(f"columns missing: {missing}")
        return X

    def _base_frame(self, X):
        """Rows the features are computed from: history (if any) plus X."""
        parts = [p for p in (self.history, X) if p is not None]
        base = pd.concat(parts, ignore_index=True, sort=False) if len(parts) > 1 else X.copy()
        base["date"] = self._day(base["date"])
        base = base.drop_duplicates(subset=KEYS, keep="first") if len(parts) > 1 else base
        if "pm25" not in base.columns:
            base["pm25"] = np.nan
        if "aod" in base.columns:                       # AOD == 0 is a failed retrieval
            base["aod"] = base["aod"].mask(base["aod"] == 0)
        return base


def main():
    data = pd.read_parquet(config.INTERIM / "clean_dev.parquet")
    features = build_features(data, forecast_horizon_days=1)

    config.PARTS.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        config.PARTS / "member4_features_preview.csv",
        index=False
    )

    print("Feature rows:", len(features))
    print("Feature columns:", len(features.columns))
    print("Rate-of-change features:", sum(
        "_rate_of_change_1" in column for column in features.columns
    ))
    print("Feature file saved.")



def rolling_forecast_features(data):
    """Align calendar and observed prior-day PM2.5 to each target date.

    This development-only preparation assumes yesterday's measurement is
    available at each rolling forecast origin, including within validation.
    It learns no parameters. It is unsuitable for fixed-origin or cold-start
    validation. The current target and same-day measurements stay out of X.
    """
    dates = pd.to_datetime(data["date"], errors="raise")
    cutoff = pd.Timestamp(config.TEST_START)
    keys = data.assign(date=dates)
    if (dates.isna().any() or not dates.lt(cutoff).all()
            or not dates.eq(dates.dt.normalize()).all()
            or keys["station_id"].isna().any()
            or keys.duplicated(["station_id", "date"]).any()):
        raise ValueError("Expected unique, normalized pre-holdout station-days.")
    ordered = keys.reset_index(drop=True).sort_values(KEYS)
    group = ordered.groupby("station_id", sort=False)
    consecutive = group["date"].diff().eq(pd.Timedelta(days=1))
    for column in ("gee_id", "lat", "lon"):
        if column in ordered:
            previous = group[column].shift()
            consecutive &= ordered[column].notna() & ordered[column].eq(previous)
    lag = group["pm25"].shift().where(consecutive)
    phase = 2 * np.pi * (ordered["date"].dt.day_of_year - 1) / 365.25
    predictors = pd.DataFrame({
        "day_of_year_sin": np.sin(phase),
        "day_of_year_cos": np.cos(phase),
        "pm25_lag_1": lag,
        "pm25_lag_1_missing": lag.isna().astype(int),
    })
    predictors = predictors.sort_index()
    predictors.index = data.index
    return predictors


if __name__ == "__main__":
    main()
