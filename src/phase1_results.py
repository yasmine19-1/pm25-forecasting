"""Consolidate the Phase 1 contributor results under the common CSV schema."""
import pandas as pd

from src import config
from src.m3_common import COLUMNS, add

DATA_PARTS = (
    "audit", "cleaning_counts", "station_eligibility_summary", "m2_eda",
    "m2_review", "m2_s5p_stability", "m3_relationships", "m3_stratified",
    "m3_pipeline_check", "rolling_pipeline_check",
)
TEMPORAL_PARTS = (
    "Exploratory_temporal_member4", "member4_drift_summary",
    "member4_temporal_patterns", "thursday_temporal_summary",
)


def read_part(name):
    path = config.PARTS / f"{name}.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != COLUMNS or frame.empty:
        raise ValueError(f"Unexpected result schema or empty input: {path}")
    if frame[["section", "variable", "metric"]].eq("").any().any():
        raise ValueError(f"Missing result identity: {path}")
    return frame


def main():
    parts = config.PARTS
    data = [read_part(name) for name in DATA_PARTS]
    temporal = [read_part(name) for name in TEMPORAL_PARTS]
    rows = []
    extremes = pd.read_csv(parts / "m2_extremes_investigation.csv")
    for verdict, count in extremes["verdict"].value_counts().items():
        add(rows, "5.3.3", "pm25", verdict, "n_flagged", count,
            "Neighbor and adjacent-day screening; retain plausible extremes")
    selected = pd.read_csv(parts / "m3_selected_features.csv")
    for row in selected.itertuples(index=False):
        group = f"{row.scheme}/fold{row.fold}"
        add(rows, "5.3.2", row.model, group, "n_selected", row.n_selected,
            f"{row.method}; selected from {row.n_input}: {row.selected}")
    data.append(pd.DataFrame(rows, columns=COLUMNS))

    rows = []
    gaps = pd.read_csv(parts / "gap_report_enhanced.csv").set_index("station_id")
    for station, values in gaps.iterrows():
        for metric, value in values.items():
            add(rows, "timestamp_quality", "station_day", station, metric, value,
                "Development only; coverage is measured from first to last observed day")
    weekly = pd.read_csv(parts / "member4_weekly_cycle_test.csv")
    for row in weekly.itertuples(index=False):
        add(rows, "weekly_cycle", "pm25", row.station_id, "pearson_lag_7",
            row.correlation, f"segment={row.segment_id}; observations={row.observations}; descriptive")
    add(rows, "weekly_cycle", "pm25", "all_segments", "median_pearson_lag_7",
        weekly.correlation.median(), f"{len(weekly)} segments; association alone cannot establish a weekly cycle")
    diagnostics = temporal[0]
    for station in gaps.index:
        if station not in set(diagnostics.group):
            add(rows, "temporal_dependence", "pm25", station, "diagnostics_available",
                0, "No diagnostics produced under the 30-consecutive-observation rule")
    lead_lag = pd.read_csv(parts / "member4_lead_lag_summary.csv")
    for row in lead_lag.itertuples(index=False):
        add(rows, "lead_lag_segments", row.variable, f"lag_{row.lag_days}d",
            "median_segment_pearson", row.median_correlation,
            f"{row.segments} segments; positive lag means past predictor; descriptive")
    temporal.append(pd.DataFrame(rows, columns=COLUMNS))

    for filename, frames in (
        ("Exploratory_Data.csv", data),
        ("Exploratory_temporal.csv", temporal),
    ):
        merged = pd.concat(frames, ignore_index=True).drop_duplicates()
        path = parts.parent / filename
        merged.to_csv(path, index=False, encoding="utf-8")
        print(f"{path}: {len(merged):,} rows")


if __name__ == "__main__":
    main()
