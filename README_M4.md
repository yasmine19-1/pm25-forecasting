# Member 4 — Temporal Analysis and Feature Engineering

## 1. Completed work

Member 4 analysed the temporal structure of the PM2.5 dataset and prepared time-based features for forecasting.

### Timestamp quality
- Audited timestamp regularity for each station.
- Identified missing calendar dates, irregular intervals and duplicate timestamps.
- Generated a station-level gap report.

### Temporal dependence
- Calculated ACF and PACF.
- Performed Ljung-Box tests at lags 7 and 14.
- Performed Augmented Dickey-Fuller tests on sufficiently long continuous segments.

### Drift and seasonality
- Summarised monthly and yearly PM2.5 distributions using median and IQR.
- Calculated the frequency of high PM2.5 observations.
- Compared daily, weekly, monthly and seasonal patterns.

### Weekly cycle
- Evaluated lag-7 PM2.5 correlations within continuous daily segments.
- Results do not show a strong, consistent weekly cycle across all tested segments.
- A seven-day lag should be evaluated during temporal validation.

### Lead-lag analysis
- Evaluated correlations between PM2.5 and atmospheric predictors at lags from -5 to +5 days.
- Positive lags represent predictor observations from earlier days.
- Negative lags use future predictor observations and must not be used as forecasting features.

## 2. Feature engineering

Feature implementation: src/features.py and features.py.

Features include day-of-year sine and cosine, past-value lags, rolling mean, standard deviation and maximum, first differences, rate of change, and the future PM2.5 target.

Feature calculations are separated by station and continuous date segments. They do not calculate lags across missing-date gaps or detected station metadata changes. PM2.5 lag features require a forecast horizon of at least one day.

## 3. Tests

Run from the project root:

    python -m unittest discover -s tests -v

The eight feature tests passed in the current Colab environment.

## 4. Main outputs

All Member 4 outputs are stored in B04-01/.

- Exploratory_temporal_M4.csv: combined Member 4 results using the shared six-column schema.
- Exploratory_temporal_member4.csv: temporal dependence diagnostics.
- member4_gap_report.csv: timestamp quality by station.
- member4_drift_summary.csv: monthly and yearly drift summaries.
- member4_temporal_patterns.csv: daily, weekly, monthly and seasonal summaries.
- member4_weekly_cycle_test.csv: lag-7 correlations.
- member4_lead_lag_summary.csv: aggregated lead-lag correlations.
- member4_lead_lag_detail.csv: segment-level lead-lag correlations.
- member4_daily_values.csv: daily station-level values.
- member4_features_preview.csv: preview of engineered features.

## 5. Limitations and pending integration

- The seven-day cycle assessment is exploratory.
- Lead-lag correlations are descriptive and do not establish causality.
- Sparse station segments should be interpreted cautiously.
- The feature transformer still needs to be tested with Member 3's actual pipeline.
- Validation must preserve past-only information and must not allow features to cross station gaps.
- Historical PM2.5 handling and validation-fold context must be confirmed during integration.

## 6. Handoff

Exploratory_temporal_M4.csv is the Member 4 contribution for Member 1 to merge into the shared Exploratory_temporal.csv.

Retain the detailed CSV files as supporting outputs.