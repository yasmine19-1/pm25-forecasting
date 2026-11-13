# Member 3: relationships, scaling, feature selection and the Pipeline

Brief sections §5.1 (relationships), §5.3, §5.3.2. Seed 42; every output comes from a script.

## Files

| File | What it does |
|---|---|
| `src/m3_common.py` | small helpers: load the development parquet, shared CSV schema, AOD == 0 rule |
| `src/relationships.py` | Pearson/Spearman, pairs above 0.9, VIF, PCA, t-SNE (descriptive only) |
| `src/stratified.py` | PM2.5 against AOD by season, by station and by temperature/pressure/wind (pooled and inside each station) |
| `src/scaling.py` | `TrainFoldScaler` (standard or robust) and `needs_scaling(model)` |
| `src/selection.py` | `FeatureSelector`: ANOVA F, mutual information, RFE, L1, tree importance |
| `src/pipeline.py` | `build_pipeline(model, ...)`: joins Member 2's steps, Member 4's features, scaler and selector |
| `src/pipeline_check.py` | runs the Pipeline through the three shared fold files and logs the selected features |
| `tests/test_m3_leakage.py` | leakage tests (every fitted step sees training rows only, no ids in the model) |
| `tests/test_m3_modules.py` | tests for the scaler, the five selection methods and the Pipeline options |
| `tests/test_m3_stratified.py` | tests for the within-station check (station differences are not mistaken for a weather effect) |
| `tests/m3_synthetic.py`, `tests/conftest.py` | a fake development frame with the real columns, used by the tests |

Nothing from Members 1, 2 or 4 was changed.

## Run (from the repo root, after `python -m src.clean` and `python -m src.folds`)

```bash
python -m src.relationships      # add --no-tsne to skip the slow figure
python -m src.stratified
python -m src.pipeline_check
pytest tests/
```

Outputs, all in `results/<GROUP_DIR>/`:

| File | Content |
|---|---|
| `parts/m3_relationships.csv`, `parts/m3_stratified.csv` | rows in the shared schema, for Member 1 to merge into `Exploratory_Data.csv` |
| `parts/m3_pipeline_check.csv` | RMSE/MAE per fold of the smoke run (shared schema) |
| `parts/m3_selected_features.csv` | for every fold and model: method, number of columns kept, their names (§5.3.2) |
| `parts/m3_relationships_notes.md`, `parts/m3_stratified_notes.md` | auto-written sentences for report section 5; check the numbers before pasting |
| `figures/m3_*.png` | correlation heatmaps, PCA scree, t-SNE, PM2.5-AOD by season/station/weather |

## The Pipeline

```python
from src.pipeline import build_pipeline

pipe = build_pipeline(model)                       # scaling decided from the model family
pipe = build_pipeline(model, selector="tree")      # "f_test" | "mutual_info" | "rfe" | "l1" | "tree"
pipe = build_pipeline(model, scaled=True, scaler="robust")
pipe.set_params(outliers__enabled=False)           # Member 2's with/without switch

for train, val in load_folds(path, dev):           # dev = the whole clean_dev.parquet
    pipe.fit(dev.iloc[train].drop(columns="pm25"), dev["pm25"].iloc[train])
    pred = pipe.predict(dev.iloc[val].drop(columns="pm25"))
```

Steps: `features -> drop_ids -> outliers -> impute -> scale -> select -> model`.

* Pass the full rows. `drop_ids` removes `station_id`, `gee_id`, `lat`, `lon`, `date`,
  `datetime_utc` and `pm25` before anything is learned (ground rule 2).
* `scale` is only added for SVR, KNN, neural nets and regularised linear models (`needs_scaling`).
  Trees and boosting get no scaler.
* Build a fresh pipeline for every fold (or `sklearn.base.clone` it). Never fit once and reuse.

### For Member 4 (`features.py`)

Pass your transformer as `build_pipeline(model, features=MyFeatures())`. It runs first and
receives the full frame, so it can still use `station_id` and `date`. It must return the same
rows in the same order, with lags built from past values only and no lag across a gap.
Nothing else needs to change.

## Decisions to agree with the team

1. **Scaled families.** The brief names SVR, KNN, neural networks and PCA. I also scale Lasso,
   ElasticNet and Huber because their penalty depends on column scale. One set in `scaling.py`
   (`SCALE_SENSITIVE`); edit it if the team disagrees.
2. **AOD == 0** is treated as a failed retrieval (missing) in the relationship analysis, the same
   rule as Member 2's `OutlierClipper`.
3. **Stratified analysis** skips stations with fewer than 30 valid PM2.5/AOD pairs. For each weather
   variable the thirds are made twice: pooled over all rows, and inside each station (stations with
   at least 90 rows, station means removed). Comparing the two shows whether a pattern is weather or
   just differences between stations such as altitude. The notes describe the three slopes in words
   instead of relying on one linear interaction coefficient.
4. **Correlated pairs are only reported.** No column is dropped by the analysis (ground rule 3).
5. **First temporal fold**: Sentinel-5P is entirely missing there (see `phase1_methodology.md`).
   The imputer makes those columns constant, and `FeatureSelector` handles constant columns
   (tested). Selected features in that fold simply cannot include a useful S5P column.
6. **UMAP** is not used (not in `requirements.txt`); t-SNE covers the exploratory picture.

## Limits

* The scripts were developed and tested on a synthetic frame with the real column names, because
  the real parquet is not in git. Numbers in `results/` come from your own run.
* `pipeline_check.py` is a smoke run with two cheap models, not the Phase 2 comparison.
* The leakage tests on the real fold files are skipped automatically when
  `data/interim/clean_dev.parquet` is missing; they run on any machine that has it.
