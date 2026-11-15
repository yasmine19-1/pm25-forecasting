# Member 3 - notes for report sections 5 to 7

Numbers come from the real development data (31,282 station-days, 70 stations, before 2025).
Source files are in results/19/parts/ (m3_*.csv, m3_*_notes.md) and results/19/figures/ (m3_*.png).
Rewrite these lines in your own words before using them in the report.

## The 10 lines

1. No pair of predictors has an absolute Pearson or Spearman correlation above 0.9, and no VIF is above 10 (computed on 28,829 rows with no missing predictor). No column needs to be removed for collinearity. Pairs are reported, never dropped.
2. PCA needs 6 of the 8 predictors' components for 80% of the variance and 7 for 90%, so the predictors are not redundant.
3. Each single predictor is only weakly related to PM2.5. The strongest Spearman correlations are NO2 (+0.22), pressure (+0.17) and AOD (+0.15).
4. PM2.5 and AOD (n = 31,277): Pearson 0.07, Spearman 0.15, slope 6.5 ug/m3 per AOD unit. The gap between the two suggests the skewed target dilutes the linear measure.
5. By season the link is strongest in winter (DJF, Spearman 0.24) and weakest in summer (JJA, 0.15). By station (67 of 70 with enough data) the median is 0.18 (IQR 0.09 to 0.26) and 96% are positive.
6. Weather changes the PM2.5-AOD slope. With the thirds made inside each station (station means removed), the slope by pressure is 11.6 / 7.7 / 4.0, so AOD tells less about PM2.5 as pressure rises, and this persists inside stations. Temperature (8.9 / 10.9 / 6.6) and wind speed (4.2 / 6.4 / 6.8) show weaker patterns, partly explained by differences between stations.
7. The Pipeline order is: temporal features, drop ids, outlier clipping, imputation, scaling (only scale-sensitive models), feature selection, model. Everything learned is fitted on the training fold only.
8. Leakage tests pass on the real shared folds (temporal, spatial, standard): every fitted step sees training rows only, ids and the target never reach the model, and changing days after a date cannot change a prediction for an earlier day.
9. Five selection methods are available (F-test, mutual information, RFE, L1, tree importance). The chosen features and their number are logged per fold in m3_selected_features.csv.
10. Smoke run with cheap, untuned models (average RMSE in ug/m3): on temporal folds Lasso went from 9.22 to 8.28 when the temporal features were added; Extra Trees from 9.55 to 9.39; on spatial folds (no PM2.5 lags) there was no clear gain. [FILL AFTER THE RUN: baselines - training mean ___ and yesterday's PM2.5 ___ on temporal folds.] This is not the Phase 2 comparison.

## Limits to state

- All relationships are associations on development data, not causal effects.
- No p-values or confidence intervals: rows of one station are not independent.
- The smoke run used one untuned configuration per model; the "with temporal features" models also kept 10 columns instead of 6, so the gain cannot be credited to the features alone.
- Shuffled (standard) folds are diagnostic only: neighbouring days fall on both sides of the split, which flatters lag features.
- Sentinel-5P is missing in the first temporal fold, so selected features there cannot include a useful S5P column.
- Lag columns are often missing (lag 1 for 65.5% of rows, lag 3 for 32.6%) and get the training median without a missing flag in the default pipeline.

## Questions raised with the team

- Should the default imputer add missing flags for lag columns? (optional via `imputer=`, default unchanged)
- Should the outlier clipper act on the lag columns, including PM2.5 history?
- Ridge was added to the scaled-model list by a teammate; keep it?

## Figure list

| Figure | Section | Use |
|---|---|---|
| m3_corr_pearson.png, m3_corr_spearman.png | 5.1 | correlations between predictors and with PM2.5 |
| m3_pca_scree.png | 5.1 | variance explained by each component |
| m3_tsne.png | 5.1 | exploratory picture only, not evidence of predictive value |
| m3_pm25_aod_by_season.png | 5.1 | PM2.5 against AOD in each season |
| m3_pm25_aod_by_station.png | 5.1 | station-by-station PM2.5-AOD correlation |
| m3_pm25_aod_by_weather.png | 5.1 | slope by weather third, pooled (top) and inside stations (bottom) |

## Tables

| File | Content |
|---|---|
| m3_relationships.csv | correlations, pairs, VIF, PCA variance and loadings |
| m3_stratified.csv | PM2.5-AOD link by season, station and weather |
| m3_pipeline_check.csv | RMSE and MAE per fold, models and baselines |
| m3_selected_features.csv | selected features and their number, per fold and model |
