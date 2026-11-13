"""Is the NO2/CO/O3 correlation structure stable across training folds?
(§5.3.1, gate for KNN imputation)

Uses the shared folds_temporal.json if present, otherwise a provisional
TimeSeriesSplit on the complete calendar with a one-day gap. Provisional
results cannot approve KNN.
Expected fold file format (adapt load_folds if the project uses another one):
    [{"train": [row positions...], "val": [row positions...]}, ...]
    or {"folds": [...same...]}

Rule: stable if, for each pair, the Spearman correlation varies by less than
0.15 across training folds (max - min).
Writes results/<GROUP>/parts/m2_s5p_stability.csv
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from m2_config import DATA_PATH, PARTS, RESULTS, S5P_COLS

TOL = 0.15


def load_folds(df):
    path = RESULTS / "folds_temporal.json"
    if path.exists():
        folds = json.loads(path.read_text())
        folds = folds["folds"] if isinstance(folds, dict) else folds
        return [np.asarray(f["train"]) for f in folds], path.name, True
    dates = pd.to_datetime(df["date"]).dt.normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    tss = TimeSeriesSplit(n_splits=5, gap=1)
    return [np.flatnonzero(dates.isin(calendar[tr])) for tr, _ in tss.split(calendar)], "provisional calendar TimeSeriesSplit", False


def main():
    df = pd.read_parquet(DATA_PATH, columns=["date", *S5P_COLS])
    trains, source, final = load_folds(df)
    rows = []
    availability = []
    for i, tr in enumerate(trains):
        training = df.iloc[tr][S5P_COLS]
        for column, count in training.count().items():
            availability.append(dict(section="5.3.1", variable=column, group=f"train_fold_{i}",
                                     metric="observed_training_rows", value=int(count),
                                     note=f"out of {len(training)} rows; folds: {source}"))
        sub = training.dropna()
        c = sub.corr(method="spearman")
        for a, b in combinations(S5P_COLS, 2):
            rows.append(dict(section="5.3.1", variable=f"{a}~{b}", group=f"train_fold_{i}",
                             metric="spearman", value=c.loc[a, b],
                             note=f"n={len(sub)}; folds: {source}"))
    out = pd.DataFrame(rows)
    verdicts = []
    for pair, g in out.groupby("variable"):
        complete = len(g) == 5 and np.isfinite(g.value).all()
        spread = g.value.max() - g.value.min() if complete else np.nan
        verdicts.append(dict(section="5.3.1", variable=pair, group="all_folds",
                             metric="spearman_spread", value=spread,
                             note=("stable" if spread < TOL else "NOT stable") if complete else "Unavailable in at least one required fold"))
    out = pd.concat([out, pd.DataFrame(verdicts)], ignore_index=True)
    allowed = final and all(v["value"] < TOL for v in verdicts)
    out.loc[len(out)] = dict(section="5.3.1", variable="s5p", group="all_folds",
                             metric="knn_imputation_allowed", value=int(allowed),
                             note=f"Requires five estimable shared folds and spread < {TOL}; folds: {source}")
    PARTS.mkdir(parents=True, exist_ok=True)
    out = pd.concat([out, pd.DataFrame(availability)], ignore_index=True)
    out.to_csv(PARTS / "m2_s5p_stability.csv", index=False)
    print(pd.DataFrame(verdicts)[["variable", "value", "note"]].to_string(index=False))
    print("KNN imputation allowed:", allowed, f"(folds: {source})")


if __name__ == "__main__":
    main()
