# PM2.5 Forecasting - Master 1 mini project (2026-2027)

Daily surface PM2.5 forecasting on the Kaggle "Air Quality - Mediterranean Basin (2016-2026)" dataset.

## Setup (macOS / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.config        # creates data/ and results/ folders
```

## Phase 1 - data audit and cleaning
1. Download the dataset from Kaggle and unzip it into `data/raw/`.
2. `python -m src.first_look`   - list columns, types, missing values
3. `python -m src.audit`        - dataset audit -> `results/<GROUP_ID>/parts/audit.csv`
4. `python -m src.clean`        - cleaning -> `data/interim/clean_dev.parquet`, reserved test files, and cleaning logs
5. `python -m src.gap_report`   - development-only station coverage -> `results/<GROUP_ID>/parts/gap_report_enhanced.csv`, metric verification and explanatory notes; preserves the original gap report
6. `python -m src.station_eligibility` - metadata evidence table and acceptance checks; the >=100-day flag is sensitivity only
7. `python -m src.thursday_evidence` - source-matching verification, development-only lag diagnostics and retained-station list; see `results/19/phase1_methodology.md` for the documented decisions and limitations

## Shared cross-validation folds

From the repository root, with the cleaned development parquet and retained-station list in place:

```bash
.venv/bin/python -m src.folds
.venv/bin/python -m src.folds --verify
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
From the repository root, with the existing development parquet and shared folds:

```bash
(
    set -e
    for script in temporal_analysis temporal_patterns weekly_cycle lead_lag_analysis drift_seasonality; do
        .venv/bin/python "${script}.py"
    done
    .venv/bin/python -m src.rolling_check
    .venv/bin/python -m src.phase1_results
    .venv/bin/python -m src.folds --verify
)
```

The consolidation uses the saved audit, Member 2 EDA and Member 3 results
in the parts directory. Their detailed outputs remain available separately.
The rolling check uses `src.features.rolling_forecast_features` on the development
data before applying the shared temporal indices. It assumes prior-day PM2.5
has been observed at prediction time. Keep the exploratory feature preview out
of the Pipeline because it contains future labels and same-day measurements.

## Conventions
- Random seed: `SEED = 42` (see `src/config.py`)
- Canonical column names: `date, station_id, lat, lon, pm25`, everything else lower snake_case
- Data files are not committed; they are rebuilt by running the scripts.
