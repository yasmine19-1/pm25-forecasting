# Phase 1 methodology

## Development cohort and data quality

Use the supplied merged dataset and keep all 70 development stations
(31,282 observations, 2016-11-25 to 2024-12-31). The 2025 final test and
the later 2026 records are reserved outside development. Cleaning removes
missing PM2.5 and values outside the strict interval (0, 500), checks duplicate
records, and removes both observations in conflicting station-date groups.
The audit and cleaning counts are in `parts/`. Raw exploratory statistics
use dates before the holdout cutoff. Cleaning also logs row counts and date
ranges for the separately reserved periods.

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
The 268 extreme flags are screening candidates. Member 2 added station
predictor distributions and reviewed extremes against nearby stations and
adjacent days. The resulting verdicts do not independently confirm causes.

## Phase 1 deliverables and limits

`Exploratory_Data.csv` combines the audit, cleaning, station eligibility,
descriptive EDA, predictor relationships, feature selection, and Pipeline
checks. `Exploratory_temporal.csv` combines station coverage, temporal profiles,
drift and seasonality, and lead–lag results. Both use the shared six-column schema.

The ACF, PACF, Ljung–Box and ADF calculations use uninterrupted PM2.5 segments
with at least 30 observations. Six stations have eligible diagnostic segments.
The CSV explicitly marks the other 64 stations without estimates. The lag-7
screen covers 123 segments and has a median Pearson correlation of about
−0.090. It does not establish weekly seasonality. Short segments limit formal
inference, and the tests apply to the PM2.5 series rather than model residuals.

The original feature preview includes a shifted target for exploratory use.
The rolling comparison instead constructs calendar features and exact
previous-day observed PM2.5 before slicing the existing temporal folds.
It preserves row order, resets at date gaps and metadata changes, and keeps
the target and same-day measurements out of predictors. Both variants use the
same validation rows and fit preprocessing on the training fold. Adding the
lag reduced MAE in five of five folds and RMSE in four of five. This is a
development smoke check under the rolling availability assumption.
Phase 2 still covers model comparison and the clipping-on/off experiment.

The regenerated tables were checked with CPython 3.14.7 on macOS arm64.
The shared folds reproduce byte for byte. Smoke-test runtimes are machine
specific, and the regenerated ExtraTrees scores differ slightly from the
historical table. The combined dataset license and operational release times
still require source confirmation.
