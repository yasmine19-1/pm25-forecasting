# PM2.5 Forecasting - Master 1 mini project (2026-2027)

Daily surface PM2.5 forecasting on the Kaggle "Air Quality - Mediterranean Basin (2016-2026)" dataset.

## Setup (macOS / Linux)

Use CPython 3.14 (verified with 3.14.7). If an existing environment uses
Anaconda Python 3.13 and pytest crashes while importing readline, create a
fresh environment with CPython 3.14 in a new directory and activate it.
Use the active environment for all commands below.
```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.config        # creates data/ and results/ folders
```

## Phase 1 - data audit and cleaning

Set `GROUP_DIR` in [src/config.py](src/config.py) to the assigned course folder code.
The current setting is `19`. All paths below use that setting for `<GROUP_DIR>`.
1. Download the [Kaggle dataset](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026) and place `combined_gee_pm25.csv`, `pm25.csv` and `gee_features_daily.csv` directly in `data/raw/`.
   Phase 1 analyzes observations dated before 2025. For the 2025 final test and the 2026 set-aside, the cleaner reports raw row counts and date ranges. The workflow leaves the original input files unchanged and writes only development datasets.
2. `python -m src.first_look`   - list columns, types, missing values
3. `python -m src.audit`        - dataset audit -> `results/<GROUP_DIR>/parts/audit.csv`
4. `python -m src.clean`        - screens dates before PM2.5 cleaning and duplicate checks, then writes `data/interim/clean_dev.parquet`, `clean_dev.csv`, and development cleaning logs
5. `python -m src.gap_report`   - development-only station coverage -> `results/<GROUP_DIR>/parts/gap_report_enhanced.csv`, metric verification and explanatory notes; preserves the original gap report
6. `python -m src.station_eligibility` - metadata evidence table and acceptance checks; the >=100-day flag is sensitivity only
7. `python -m src.thursday_evidence` - source-matching verification, development-only lag diagnostics and retained-station list; see `results/<GROUP_DIR>/phase1_methodology.md` for the documented decisions and limitations

## Shared cross-validation folds

From the repository root, with the cleaned development parquet and retained-station list in place:

```bash
python -m src.folds
python -m src.folds --verify
```

The files are under `results/<GROUP_DIR>/`. Each contains five `train` and `val`
position lists, the ordered `[station_id, date]` keys, the development parquet's
SHA-256, and the splitter settings. Positions refer to the original parquet order.
Verification checks the split rules and compares the regenerated JSON byte for byte.

- `folds_temporal.json` uses expanding training windows on a daily calendar. Each date stays together and one calendar day is reserved between training and validation.
- `folds_spatial.json` groups by station ID so all GEE IDs of a station stay together.
- `folds_standard.json` uses shuffled K-fold with seed 42 for diagnostic comparison only.

For Pipeline integration, load the positions before sorting, filtering, or removing
the identity columns. Use the same positions to select predictors and targets.

```python
import pandas as pd
from src import config
from src.folds import load_folds

dev = pd.read_parquet(config.INTERIM / "clean_dev.parquet")
splits = load_folds(config.PARTS.parent / "folds_temporal.json", dev)
for train, val in splits:
    train_rows, val_rows = dev.iloc[train], dev.iloc[val]
    # Select scenario-appropriate predictors and fit a fresh Pipeline on train_rows.
```

Select the spatial file for the held-out-station experiment. Feature construction
must preserve row alignment and follow the forecast-time rules in
`results/<GROUP_DIR>/phase1_methodology.md`. Exclude the target and `config.NON_FEATURE_COLS`
from the model input.

### Fold-local preprocessing

Import `OutlierClipper` from `src.outliers` and `S5PImputer` from
`src.imputation`. Put `("outliers", OutlierClipper())` before
`("impute", S5PImputer())`, followed by the model-specific scaling and
selection steps. Fit the Pipeline on each fold's training predictors and target,
then use the fitted Pipeline on its validation predictors. The imputer uses
training medians and adds one missingness indicator per Sentinel-5P predictor.
If a predictor is entirely missing in training, its missing entries use a fixed
zero placeholder with the indicator set to 1. See the methodology for the
first temporal fold's limitation.

## Reproduce the Phase 1 deliverables

The final tables are `Exploratory_Data.csv`
and `Exploratory_temporal.csv` in `results/<GROUP_DIR>/`.
From the repository root, after running the audit and cleaning steps and generating the shared folds, run the full set of result producers:

```bash
(
    set -e
    python src/m2_eda.py
    python src/m2_review.py
    python src/m2_s5p_stability.py
    python -m src.relationships
    python -m src.stratified
    python -m src.pipeline_check
    python -m src.clipping_check
    for script in temporal_analysis temporal_patterns weekly_cycle lead_lag_analysis drift_seasonality; do
        python "${script}.py"
    done
    python -m src.m4_figures
    python -m src.rolling_check
    python -m src.phase1_results
    python -m src.folds --verify
    python -m pytest -q
)
```

The consolidation reads the generated parts directory. Fold verification compares
JSON bytes. Model timings vary by machine. Record Python and platform alongside
ExtraTrees results. The earlier audit observed score differences across runs with
the same package version pins. Their detailed outputs remain available separately.
The rolling check uses `src.features.rolling_forecast_features` on the development
data before applying the shared temporal indices. It is a Phase 1 diagnostic
under the assumption that each previous day's PM2.5 is available at the next
forecast origin. It can use an earlier validation observation under that
assumption. Actual release times and permission for within-fold updates in
scored evaluation still need confirmation. Keep the exploratory feature preview
out of the Pipeline because it contains future labels and same-day measurements.

For a comparison with the rolling features, keep the same fold positions and
rows, including rows with no prior-day observation. Compare calendar-only
columns with the full rolling frame, which has a missing-lag indicator. For
same-day or cold-start experiments, omit target-history features. For
preprocessing comparisons, `impute__add_indicator=False` disables the Sentinel-5P
indicators, and `outliers__enabled=False` disables clipping while retaining the
AOD zero-to-missing conversion. To train on log1p PM2.5, wrap the pipeline in
`TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)`
and score predictions in the original units. For same-day feature ablations,
remove AOD, the three Sentinel-5P columns, or the four weather columns from
both training and validation inputs before building the pipeline. If using
`TemporalFeatures`, also remove the ablated source columns from its history
so it cannot regenerate derived features from them.

## Forecast history examples

Run these examples from the repository root after generating the shared folds.
Each example fits preprocessing on the training rows and uses the stored
validation positions.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src import config
from src.features import VARIABLES, TemporalFeatures, rolling_forecast_features
from src.folds import load_folds
from src.pipeline import build_pipeline

dev = pd.read_parquet(config.INTERIM / "clean_dev.parquet")
y = dev[config.TARGET]
X = dev.drop(columns=config.TARGET)
results = config.PARTS.parent
train, val = load_folds(results / "folds_temporal.json", dev)[0]

# Rolling next-day diagnostic. Assume yesterday's observations are released
# Before today's forecast. Earlier validation dates may supply a prior value.
rolling = rolling_forecast_features(dev)
rolling_predictions = (
    build_pipeline(Ridge())
    .fit(rolling.iloc[train], y.iloc[train])
    .predict(rolling.iloc[val])
)

# Fixed origin at the end of the last observed training day.
# Freeze observed values at that cutoff. Keep validation station/date keys
# And static location metadata so features align with the scored rows.
cutoff = dev["date"].iloc[train].max()
history = dev.iloc[train].loc[lambda rows: rows["date"].le(cutoff)]
fixed_inputs = X.iloc[val].copy()
for column in fixed_inputs.columns.intersection(VARIABLES):
    fixed_inputs[column] = np.nan
fixed = build_pipeline(Ridge(), features=TemporalFeatures(history=history))
fixed_predictions = fixed.fit(X.iloc[train], y.iloc[train]).predict(fixed_inputs)

# Cold start at held-out stations. The same-day inputs must be available at
# Prediction time. Keep only training-station history and disable PM2.5 lags.
train, val = load_folds(results / "folds_spatial.json", dev)[0]
cold = build_pipeline(
    Ridge(),
    features=TemporalFeatures(
        scenario="same_day", use_target_lags=False, history=X.iloc[train],
    ),
)
cold_predictions = cold.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[val])
```

The rolling example is a Phase 1 diagnostic. Actual release times are unverified.
Confirm the within-fold observation policy before using rolling scores for model
selection. In the fixed-origin example, later validation rows contribute no observed
targets or weather/satellite values to history.

## Paired predictor clipping comparison

`python -m src.clipping_check` writes `results/<GROUP_DIR>/clipping_comparison.csv`.
The two default Ridge fits per fold use the same validation rows. Clipping
thresholds are fitted per training fold. The AOD zero-to-missing rule is shared
by both runs. The table reports unweighted means across each scheme's five folds.

| Scheme | Clipping | RMSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| Temporal | On | 9.1867 | 4.8137 | 0.0334 |
| Temporal | Off | 9.1965 | 4.8201 | 0.0306 |
| Spatial | On | 9.7219 | 4.8115 | 0.0518 |
| Spatial | Off | 9.7364 | 4.8162 | 0.0486 |
| Standard | On | 9.9630 | 4.7531 | 0.0464 |
| Standard | Off | 9.9740 | 4.7578 | 0.0442 |

## Design decisions and switches

| Decision | Current default | Implementation | How to switch |
| --- | --- | --- | --- |
| Station retention | All 70 stations. The ≥100-observed-day sensitivity cohort contains 63 stations and 30,963 rows. | [Station eligibility](src/station_eligibility.py#L143), [retention decision](src/thursday_evidence.py#L66) | Needs code change. Rebuild the cohort and its folds in a separate experiment. |
| No prior-day PM2.5 value | Keep all 10,786 rows without a usable lag and flag them with `pm25_lag_1_missing`. | [Rolling features](src/features.py#L230) | Argument for the general transformer (`use_target_lags=False`), or use the calendar-only columns for the rolling diagnostic. A different row policy needs code change. |
| Sentinel-5P column entirely missing in training | Fill missing values with zero. Keep observed validation values and add a missingness indicator. | [Fold-local imputer](src/imputation.py#L47) | Changing the placeholder needs code change. The `add_indicator` argument controls the indicators. |
| Predictor clipping | Enabled. Fit thresholds on each training fold before imputation. | [Outlier clipper](src/outliers.py#L46), [Pipeline](src/pipeline.py#L63) | Argument `clip_outliers=False` or `outliers__enabled=False`. |
| Extreme PM2.5 | Flag 268 values above Q3 + 3×IQR and keep them in development. Episode and data-quality labels are provisional, based on corroboration from nearby stations, adjacent days, AOD and repeated readings. | [Extreme screen](src/m2_eda.py#L378), [Corroboration review](src/m2_review.py#L120) | Changing retention of flagged extremes needs code change. |

The development data run from 2016-11-25 through 2024-12-31. The earlier
recommendation to start after Sentinel-5P coverage began is superseded. The
first-pass `parts/m2_eda.csv` has 684 rows. The canonical `figures/` directory
contains 42 figures, including the 19-figure initial exploratory subset. The
verified test suite passes 94 tests.

Upstream terms referenced for the audit are CC BY 4.0 for applicable OpenAQ
records (check provider terms), open access for Copernicus Sentinel data, and
NASA's Earth science open-data policy. The combined Kaggle release's own license
text is unverified. [OpenAQ licenses](https://docs.openaq.org/resources/licenses) ·
[Copernicus Sentinel license](https://cds.climate.copernicus.eu/licences/ec-sentinel) ·
[NASA Earthdata policy](https://www.earthdata.nasa.gov/engage/open-data-services-software).

## Historical / non-canonical files

- The root `member4_gap_report.csv` and the `B04-01/` temporal diagnostics,
  drift, patterns, lead–lag, weekly-cycle and gap CSVs are historical copies.
  Use the consolidated tables and generated `parts/` outputs under the
  configured group directory.
- The two large observation-level historical files in `B04-01/` are
  `member4_features_preview.csv` (22.7 MB) and `member4_daily_values.csv`
  (6.0 MB). The feature preview contains a shifted target.
- There are six historical figure copies outside the canonical `figures/`
  directory. Both `src/` and `results/<GROUP_DIR>/` contain a copy of each:
  `m4_acf_pm25.png`, `m4_lead_lag.png` and `m4_seasonal_profile.png`.
- The decision-gate note preserves the exploratory discussion. The Phase 1
  methodology records the final station, imputation and lag choices.

## Conventions
- Random seed: `SEED = 42` (see `src/config.py`)
- Canonical column names: `date, station_id, lat, lon, pm25`, everything else lower snake_case
- Data files are not committed; they are rebuilt by running the scripts.
