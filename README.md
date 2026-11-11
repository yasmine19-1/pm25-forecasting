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
4. `python -m src.clean`        - cleaning -> `data/interim/clean_full.parquet` + `cleaning_log.csv`

## Conventions
- Random seed: `SEED = 42` (see `src/config.py`)
- Canonical column names: `date, station_id, lat, lon, pm25`, everything else lower snake_case
- Data files are not committed; they are rebuilt by running the scripts.
