# Temporal validation and lag decision

Phase 1 · 13 Nov 2026 · Historical decision note

## Decisions

| Question | Decision |
|---|---|
| Temporal CV | **Go.** Expanding windows, one-day gap (`folds_temporal.json`). |
| Spatial CV | **Go** (the shared decision, `folds_spatial.json`): stations stay grouped, and the four stations with two coordinate pairs are kept in one group. |
| Forecast scenario | **Next-day**: predict day t from what is known at the end of day t-1. PM2.5 lags allowed. |
| Same-day and spatial experiments | **No PM2.5 lags** (cold-start rule from the methodology). |
| Lags and windows | PM2.5 and every predictor: lag 1, lag 3, 3-day rolling mean, standard deviation and maximum, first difference, rate of change. Plus sine and cosine of day of year. 7-day features: **not built**. |
| Stations kept | All 70 (the shared retained-station list). |
| Final test period | 2025 (2025-01-01 to 2025-12-31), reserved. Development data: 2016-11-25 to 2024-12-31 (31,282 station-days). |

## Evidence

- **Persistence.** Day-to-day dependence is clear. Median within-station Spearman between consecutive days is 0.695 across 20,496 pairs. In the station-segment ACF, lag 1 has a median of 0.51, and lag 7 is negative (median -0.09).
- **No weekly cycle.** The lag-7 correlation has a median of -0.09 over 123 continuous segments from 59 stations, and only 43% are positive. A 7-day period is not supported.
- **Lead-lag** (median correlation with PM2.5, positive lag = predictor from earlier days, 123 segments):

| Variable | Lag 1 | Lag 2 | Lag 3 | Note |
|---|---|---|---|---|
| Temperature | 0.25 | 0.06 | -0.07 | same-day 0.45, not usable next-day |
| Ozone | -0.26 | -0.15 | -0.09 | strongest next-day-usable lag |
| Wind v | 0.20 | 0.11 | 0.05 | |
| Pressure | 0.15 | 0.21 | 0.13 | best at lag 2 (not built) |
| NO₂ | 0.14 | 0.03 | -0.09 | |
| Wind u | -0.11 | -0.10 | -0.22 | best at lag 3 |
| CO | 0.06 | -0.02 | -0.14 | best at lag 3 |
| AOD | 0.02 | -0.01 | 0.04 | same-day 0.23, not usable next-day |

  Correlations are modest (|r| ≤ 0.27 at usable lags) and descriptive only. Negative lags use future data and are never features.
- **Seasonality.** Pooled monthly medians of PM2.5 are highest in February (9.9) and January (8.9), then August (8.8) and December (8.6), and lowest in April (6.4) and May (6.5). Sine and cosine of day of year capture one yearly cycle.
- **Drift.** Yearly medians vary (2019: 6.7, 2020: 5.7, 2023: 9.0, 2024: 8.0), but station coverage changes a lot (252 station-days in 2018, 8,684 in 2024), so pooled drift mixes station composition with time.

## Limits

- **The series are fragmented.** There are 10,786 continuous daily segments, with a median length of 2 days and a maximum of 32. Lag 1 is available for 65.5% of rows. Lag 3 and the 3-day windows exist for only 32.6%. Lags never cross a gap, so the rest are missing and the imputer fills them (training medians).
- **Formal diagnostics cover only 6 segments** (the only ones with at least 30 consecutive days). In those, Ljung-Box at lag 7 rejects no autocorrelation in 3 of 6, and ADF rejects a unit root in 2 of 6 (median p = 0.13). Stationarity is inconclusive.
- **Unverified:** actual data release times. Any predictor used in a next-day forecast must be available at the forecast origin.

## Implementation

`src/features.py`: `TemporalFeatures(history=dev)` inside `build_pipeline(model, features=...)`. It returns the same rows in the same order, drops same-day predictors in the next-day scenario, treats AOD = 0 as missing, and learns the high-AOD threshold (training 90th percentile) in `fit()`. Tests: `tests/test_features.py`, `tests/test_temporal_features.py`.
