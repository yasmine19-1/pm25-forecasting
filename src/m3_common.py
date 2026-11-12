"""Member 3 - small helpers shared by relationships.py and stratified.py.

Nothing in here learns anything from the data that a model would use: the
scripts of Member 3 that run on the whole development set are descriptive only.
"""
import pandas as pd

from src import config

DEV_FILE = config.INTERIM / "clean_dev.parquet"
PARTS = config.PARTS
FIGURES = config.PARTS.parent / "figures"
COLUMNS = ["section", "variable", "group", "metric", "value", "note"]  # shared schema (ground rule 5)


def load_dev(path=None):
    """Read the development dataset published by Member 1 (test year already removed)."""
    path = path or DEV_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m src.clean` first.")
    return pd.read_parquet(path)


def zero_aod_to_nan(df):
    """Same convention as Member 2's OutlierClipper: AOD == 0 is a failed retrieval."""
    df = df.copy()
    if "aod" in df.columns:
        df["aod"] = df["aod"].mask(df["aod"] == 0)
    return df


def predictor_cols(df):
    """Numeric columns that can be model inputs: no ids, no coordinates, no dates, no target."""
    banned = set(config.NON_FEATURE_COLS) | {config.TARGET}
    cols = [c for c in df.select_dtypes("number").columns if c not in banned]
    return [c for c in cols if df[c].nunique(dropna=True) > 1]  # a constant column has no correlation


def add(rows, section, variable, group, metric, value, note=""):
    rows.append(dict(section=section, variable=variable, group=group,
                     metric=metric, value=value, note=note))


def save_parts(rows, filename):
    PARTS.mkdir(parents=True, exist_ok=True)
    path = PARTS / filename
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, encoding="utf-8")
    return path


def save_notes(lines, filename):
    PARTS.mkdir(parents=True, exist_ok=True)
    path = PARTS / filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
