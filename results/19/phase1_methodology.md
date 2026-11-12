# Phase 1 methodology

## Development cohort and data quality

Use the supplied merged dataset and keep all 70 development stations
(31,282 observations, 2016-11-25 to 2024-12-31). The 2025 final test and
the later 2026 records are reserved outside development. Cleaning removes
missing PM2.5 and values outside the strict interval (0, 500), checks duplicate
records, and removes both observations in conflicting station-date groups.
The audit and cleaning counts are in `parts/`.

All development rows matched the corresponding station/date/coordinate and GEE
ID/date records in the supplied sources. Each assigned GEE position matched a
nearest candidate for its day. This verifies the supplied merge against the
supplied sources. Four stations have two recorded coordinate pairs and GEE IDs,
with changes of about 13–179 metres. Their physical cause is unresolved and
both periods stay in the same station group. The 100-day cohort is reserved
for sensitivity comparisons. Missing dates limit individual analyses.

## Evaluation and features

The shared temporal folds expand over the calendar, hold each date together and
reserve a full day between training and validation. Spatial folds hold out whole
stations. Shuffled folds use seed 42 for diagnostic comparison. Spatial scores
measure generalization to held-out station identities. Temporal scores cover
observed target dates and reflect changing station coverage across years.

Test one-day PM2.5 persistence only for rolling next-day forecasts at monitored
stations, assuming yesterday's observation is available. The development data
contain 20,496 consecutive-day pairs. Median within-station Spearman correlation
is 0.695 and all 68 estimable stations have a positive value. This is descriptive
evidence for a feature comparison on shared validation rows. Keep histories
within a station and across consecutive days with consistent metadata. Exclude
target lags in same-day and cold-start spatial experiments. Actual data release
times are unverified. Any predictor used in a next-day forecast must be available
at the forecast origin. Day-of-year sine and cosine are known in advance.
Longer lag windows remain exploratory given the available continuous history.

## Missing values and extreme observations

Use training-fold medians, with one missingness indicator per Sentinel-5P
predictor and no forward filling. The first shared temporal training fold has
2,339 rows and zero observed NO₂, CO or O₃ values. Its correlations are undefined,
so the five-fold stability criterion for KNN cannot be evaluated. The imputer
uses a fixed zero placeholder for missing entries in an entirely missing training
column and sets the corresponding indicator to 1. Observed validation values are
preserved. This keeps the feature schema consistent, but the first fold cannot
learn a relationship for these predictors or their indicators from observed variation.

Fit predictor clipping within each training fold before imputation. The proposed
treatment uses training IQR bounds constrained by configured domain limits, flags
zero AOD as missing, and has a switch for the Phase 2 comparison. Extreme PM2.5
values within the cleaning interval are retained. Descriptive EDA found a median
of 7.7 µg/m³, with 2.7% of observations above 25 and 0.29% above 50 µg/m³.
The 268 extreme flags are provisional screens for corroboration.

## Phase 1 work still open

Complete predictor distributions by station and corroborate suspected retrieval
errors and pollution episodes with nearby stations or independent evidence.
The available gap summaries, lag correlations and temporal profiles do not
provide formal ACF/PACF, stationarity or residual-autocorrelation diagnostics.
Complete applicable diagnostics with attention to the short, irregular series.
Finish the predictor relationships and feature-selection analysis, assemble and
verify the full Pipeline, and consolidate the required `Exploratory_Data.csv`
and `Exploratory_temporal.csv`. The clipping-on/off score comparison belongs to
Phase 2. This methodology is shared project documentation. The brief specifies
the exploratory CSVs, source code and reproducibility information as deliverables.
