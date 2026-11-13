# Member 2: descriptive EDA, missingness, outliers

Brief sections §5.1, §5.3.1, §5.3.3. Seed 42; every output comes from a script.

## Files

| File | What it does | Due |
|---|---|---|
| `src/m2_config.py` | seed, paths, group folder 19 (separate file so it never overwrites the team config) | — |
| `src/m2_eda.py` | descriptive stats, figures, target analysis, missingness, outlier inspection | Thu 18:00 |
| `src/imputation.py` | `S5PImputer`: median + one indicator per S5P column, fitted on the training fold | Fri 12:00 |
| `src/outliers.py` | `OutlierClipper`: Q1−3×IQR / Q3+3×IQR clipping learned on the training fold, `enabled` switch, AOD = 0 → NaN | Fri 12:00 |
| `src/m2_s5p_stability.py` | NO₂/CO/O₃ correlation stability across training folds (KNN gate) | Fri 18:00 |
| `tests/test_m2_transformers.py` | 7 tests: train-only fit, no forward-fill, switch, no row dropped, works in a Pipeline | — |

## Run

```bash
python src/m2_eda.py            # -> results/<GROUP_DIR>/parts/m2_eda.csv, m2_notes.md, figures/m2_*.png
python src/m2_s5p_stability.py  # rerun once Member 1 publishes folds_temporal.json
pytest tests/
```

Expects the clean development dataset at `data/interim/clean_dev.parquet`.
Install the pinned dependencies from `requirements.txt`. See the shared methodology at `results/<GROUP_DIR>/phase1_methodology.md` for the imputation decision and limitations.

## For Member 3 (Pipeline)

```python
from src.outliers import OutlierClipper
from src.imputation import S5PImputer

Pipeline([
    ("outliers", OutlierClipper()),   # first: turns AOD == 0 into NaN
    ("impute",   S5PImputer()),       # then fills NaN + adds no2_missing, co_missing, o3_missing
    # features / scaler / selector / model ...
])
```

Both steps take and return DataFrames with column names. With-vs-without comparison for Phase 2: `pipe.set_params(outliers__enabled=False)`.
Station ID, coordinates and dates must not be in `X` (ground rule 2).
