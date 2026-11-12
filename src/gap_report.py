"""Audit development-date coverage without reading or changing the final test data.

Run: python -m src.gap_report
Candidate history lengths describe availability only, not selected model features.
"""
import numpy as np
import pandas as pd

from src import config


HISTORY_DAYS = (1, 2, 3, 7)


def station_metrics(station, dates):
    dates = pd.Series(pd.to_datetime(dates)).sort_values().reset_index(drop=True)
    unique = dates.drop_duplicates().reset_index(drop=True)
    gaps = unique.diff().dt.days.dropna()
    span = (unique.iloc[-1] - unique.iloc[0]).days + 1
    runs = unique.groupby(unique.diff().dt.days.ne(1).cumsum()).size()
    row = {
        "station_id": station,
        "number_of_rows": len(dates),
        "first_date": unique.iloc[0].strftime("%Y-%m-%d"),
        "last_date": unique.iloc[-1].strftime("%Y-%m-%d"),
        "unique_dates": len(unique),
        "history_span_days": span,
        "missing_calendar_days": span - len(unique),
        "calendar_coverage_percentage": 100 * len(unique) / span,
        "one_day_interval_percentage": 100 * gaps.eq(1).mean() if len(gaps) else np.nan,
        "average_gap_days": gaps.mean(),
        "maximum_gap_days": gaps.max(),
        "longest_missing_streak_days": int(gaps.max() - 1) if len(gaps) else 0,
        "irregular_intervals": int(gaps.ne(1).sum()),
        "duplicate_timestamps": len(dates) - len(unique),
        "continuous_daily_runs": len(runs),
        "longest_continuous_run_days": int(runs.max()),
    }
    for k in HISTORY_DAYS:
        row[f"usable_rows_with_{k}d_contiguous_history"] = int((runs - k).clip(lower=0).sum())
    return row


def compare_original(original, enhanced):
    if original.station_id.duplicated().any():
        raise ValueError("Original report contains duplicate station IDs.")
    old = original.set_index("station_id")
    new = enhanced.set_index("station_id")
    rows = []
    for station in sorted(set(old.index) | set(new.index)):
        if station not in old.index or station not in new.index:
            rows.append((station, "station_presence", station in old.index,
                         station in new.index, False))
            continue
        for column in old.columns:
            target = "one_day_interval_percentage" if column == "daily_sampling_percentage" else column
            before = old.at[station, column]
            if target not in new.columns:
                rows.append((station, column, before, None, False))
                continue
            after = new.at[station, target]
            if column in ("first_date", "last_date"):
                match = str(before) == str(after)
            else:
                match = bool(np.isclose(float(before), float(after), atol=0.005001,
                                        rtol=0, equal_nan=True))
            rows.append((station, column, before, after, match))
    return pd.DataFrame(rows, columns=["station_id", "original_metric", "original_value",
                                       "recomputed_value", "matches_with_rounding"])


def main():
    source = config.INTERIM / "clean_dev.parquet"
    df = pd.read_parquet(source, columns=["station_id", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df.empty or df[["station_id", "date"]].isna().any().any():
        raise ValueError("Development data must contain non-null station IDs and dates.")
    if (df.date >= pd.Timestamp(config.TEST_START)).any():
        raise ValueError("Input contains dates in or after the reserved test period.")
    if not df.date.eq(df.date.dt.normalize()).all():
        raise ValueError("Expected normalized daily timestamps.")
    if df.duplicated(["station_id", "date"]).any():
        raise ValueError("Resolve duplicate station-days before measuring usable history.")
    report = pd.DataFrame([station_metrics(s, g.date)
                           for s, g in df.groupby("station_id", sort=True)])
    original = pd.read_csv(config.ROOT / "member4_gap_report.csv")
    verification = compare_original(original, report)
    config.PARTS.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.PARTS / "gap_report_enhanced.csv", index=False, float_format="%.4f")
    verification.to_csv(config.PARTS / "gap_report_verification.csv", index=False,
                        float_format="%.6f")
    mismatches = int((~verification.matches_with_rounding).sum())
    note = f"""# Development station coverage

Reproduce with `python -m src.gap_report` from the repository root.
Input: `data/interim/clean_dev.parquet`; {len(df):,} observations across {len(report)} stations.
Dates: {df.date.min().date()} to {df.date.max().date()}. The script rejects dates on or
after {config.TEST_START}; it never reads or rewrites the final-test files.

## Original report verification

Compared all original metrics against the development data, allowing 0.005001
absolute rounding error for numeric values: {len(verification)} checks, {mismatches} mismatches.
See `gap_report_verification.csv` for each comparison. The original
`member4_gap_report.csv` is preserved. Matching values verify the apparent metric
definition against this dataset, rather than recovering the original author's code.

## Metric definitions

- `history_span_days`: calendar days from first through last observation, inclusive.
- `calendar_coverage_percentage`: 100 * unique observed days / history span.
  This does not measure coverage over the entire development period; a short station
  history can still have 100% coverage. Read it alongside dates and observation count.
- `one_day_interval_percentage`: percentage of adjacent observed-date intervals
  equal to one day. This is the clarified name for the original
  `daily_sampling_percentage`. Undefined (blank) for a station with one observation.
- `average_gap_days` and `maximum_gap_days`: elapsed days between observed dates.
  A maximum gap of 10 days means 9 missing days between its endpoints.
- `longest_missing_streak_days`: maximum gap minus one, or zero for a single day.
- `irregular_intervals`: number of adjacent observed-date intervals longer than one day.
- `continuous_daily_runs`: number of uninterrupted sequences, including isolated days.
- `longest_continuous_run_days`: length of the longest such sequence.
- `usable_rows_with_Kd_contiguous_history`: observed target days with all K preceding
  calendar days observed at the same station. For a run of length L, contributes
  max(L-K, 0). Candidates K = 1, 2, 3, 7 are diagnostics, not approved feature choices.
  These counts require an uninterrupted history, not merely an observation at t-K.
  They measure date availability only: predictor missingness, forecast-time availability,
  and fold boundaries can further reduce model-ready examples. No values are imputed.

## Interpretation and next decision

No station is removed or marked eligible automatically. Agree on station-retention
rules using counts, span, coverage, consecutive history and the forecast scenario;
document reasons before generating shared folds. Do not choose rules using test scores.
This report describes the cleaned merged dataset: absent dates do not by themselves
prove the underlying station was offline. Calendar coverage excludes time outside
each station's first and last observations.
"""
    (config.PARTS / "gap_report_notes.md").write_text(note, encoding="utf-8")
    print(f"Saved enhanced report: {len(report)} stations, {len(df):,} rows.")
    print(f"Original report verification: {len(verification)} checks, {mismatches} mismatches.")
    print(f"Outputs: {config.PARTS}")


if __name__ == "__main__":
    main()
