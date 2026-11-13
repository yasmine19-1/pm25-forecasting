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
1. Download the [Kaggle dataset](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026) and place `combined_gee_pm25.csv`, `pm25.csv` and `gee_features_daily.csv` directly in `data/raw/`.
   The raw inspection commands summarize only dated observations before 2025. The cleaner writes the reserved periods to separate files.
2. `python -m src.first_look`   - list columns, types, missing values
3. `python -m src.audit`        - dataset audit -> `results/<GROUP_ID>/parts/audit.csv`
4. `python -m src.clean`        - cleaning -> `data/interim/clean_dev.parquet`, reserved test files, and cleaning logs
5. `python -m src.gap_report`   - development-only station coverage -> `results/<GROUP_ID>/parts/gap_report_enhanced.csv`, metric verification and explanatory notes; preserves the original gap report
6. `python -m src.station_eligibility` - metadata evidence table and acceptance checks; the >=100-day flag is sensitivity only
7. `python -m src.thursday_evidence` - source-matching verification, development-only lag diagnostics and retained-station list; see `results/19/phase1_methodology.md` for the documented decisions and limitations

## Shared cross-validation folds

From the repository root, with the cleaned development parquet and retained-station list in place:

```bash
python -m src.folds
python -m src.folds --verify
```

The files are under `results/<GROUP_ID>/`. Each contains five `train` and `val`
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
`results/19/phase1_methodology.md`. Exclude the target and `config.NON_FEATURE_COLS`
from the model input.

### Preprocessing in Member 3's Pipeline

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

The final tables are [Exploratory_Data.csv](results/19/Exploratory_Data.csv)
and [Exploratory_temporal.csv](results/19/Exploratory_temporal.csv).
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
data before applying the shared temporal indices. It assumes prior-day PM2.5
has been observed at prediction time. Keep the exploratory feature preview out
of the Pipeline because it contains future labels and same-day measurements.

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

## Conventions
- Random seed: `SEED = 42` (see `src/config.py`)
- Canonical column names: `date, station_id, lat, lon, pm25`, everything else lower snake_case
- Data files are not committed; they are rebuilt by running the scripts.
