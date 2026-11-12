# Member 2 - notes for report §5

## Target (PM2.5) - §5.1

- Strong right skew (skewness 19.11, excess kurtosis 666.6); median
  7.7 µg/m³ vs mean 9.3. log1p brings skewness to
  -0.14.
- 2.7% of station-days exceed 25 µg/m³ and 0.29%
  exceed 50 µg/m³: high-pollution days are rare events.
- Seasonal medians: {'DJF': 9.1, 'JJA': 7.9, 'MAM': 6.7, 'SON': 7.7} (Kruskal-Wallis p = 4.9e-105).
- The top 1 % of days carries 69% of the total squared deviation but
  only 11% of the absolute deviation. Consequences: RMSE will be
  dominated by a handful of extreme days, MAE much less so; a model can have a
  good MAE and still miss every episode. Report RMSE and MAE side by side and
  add an extreme-event metric (e.g. MAE or recall on days > 25 µg/m³).
- Option for Phase 2: train on log1p(PM2.5) and back-transform, compared with
  the raw target inside the same folds.

## Missingness - §5.1, §5.3.1

- Only the Sentinel-5P columns have gaps: NO₂ 7.7%, CO
  7.7%, O₃ 7.8%. Every other column,
  including the target, is complete.
- The three are almost always missing together (2407 of
  2448 rows with any gap).
- **The gaps are structural, not random.** Every missing value falls between
  2016-11-25 and 2018-09-25; the first row with Sentinel-5P data is 2018-11-11.
  Sentinel-5P was launched in October 2017 and its products only start in
  2018, which matches these dates; after 2018-11-11 there are 0 missing rows. See `m2_missing_by_yearmonth.png`.
- Because of that, the month and station patterns are artefacts: month
  10 (13.3% missing) and the worst stations
  (FR03030 55%, FR03006 47%, FR03043 46%) are just the ones with
  many pre-2018 rows.
- Missingness vs other variables (Mann-Whitney): PM2.5 p = 0.27,
  AOD p = 0.1, temperature p = 2.6e-12.
  Gaps are not linked to the target.

### Decision for the Thursday gate (§5.3.1)
- **Option A (recommended): start the development data at 2018-11-11.** It drops
  2448 rows (7.8%) and leaves essentially no missing
  values. Imputing two years of medians would invent pollutant data for a
  period with no satellite, and the missingness indicator would just become a
  "before 2018" flag. With TimeSeriesSplit (5 folds) the first training
  fold would be about 47% imputed rows.
- **Option B: keep all rows.** Then median imputation fitted on the training
  fold + one indicator per S5P column, no forward-fill. `imputation.py`
  supports both options unchanged.
- KNN imputation is not worth it here: there is no partially observed period
  to learn from. The stability check stays in the repo for completeness.

## Outliers - §5.1, §5.3.3

- Domain limits: no value is physically impossible after Member 1's cleaning,
  except 5 AOD values exactly at 0 (suspicious
  retrievals) and 23 AOD values above 1 (median PM2.5 on those rows
  14.4 µg/m³: likely dust/cloud, not local pollution).
- Univariate rules flag very different numbers (see `n_flagged` rows): the
  1.5×IQR rule flags a lot on skewed variables (PM2.5, AOD) and is too
  aggressive; 3×IQR or robust z (MAD) is more sensible. Pressure is a good
  warning: 1.5×IQR flags thousands of rows that are simply high-altitude
  stations (pressure is multimodal), while 3×IQR flags none.
- PM2.5 above 3×IQR (> 34.4 µg/m³): 268 rows. Classification:
  {'episode_likely': 144, 'isolated_suspect': 122, 'undetermined_few_stations': 2}. "episode_likely" = several stations high the same day or high AOD;
  "isolated_suspect" = one station spikes alone with ordinary AOD.
  List in `m2_pm25_extremes.csv`.
- Isolation Forest and LOF (1 % each) agree on 95
  predictor rows.

### Outlier policy (proposal for the team)
1. Never remove target values inside the Pipeline (rows can't be dropped in an
   sklearn transformer, and real episodes are exactly what we must predict).
   The cleaning rule 0 < PM2.5 < 500 stays the only target filter.
2. Predictors: winsorise (clip) at limits learned on the training fold only:
   default Q1 − 3×IQR / Q3 + 3×IQR, applied unchanged to validation and test.
   AOD == 0 is treated as missing.
3. `outliers.py` has `enabled=True/False` so Phase 2 compares MAE, RMSE, R²
   with and without the treatment in the same folds.
4. "isolated_suspect" PM2.5 days are reported, not deleted; a sensitivity run
   without them can be added in Phase 2 if the team wants.
