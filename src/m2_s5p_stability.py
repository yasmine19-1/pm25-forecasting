"""Member 2 - is the NO2/CO/O3 correlation structure stable across training folds?
(§5.3.1, gate for KNN imputation)

Uses Member 1's folds_temporal.json if present, otherwise a provisional
TimeSeriesSplit(5, gap = 1 day) on the date-sorted data.
Expected fold file format (adapt load_folds if Member 1 uses another one):
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
        return [np.asarray(f["train"]) for f in folds], f"{path.name}"
    order = np.argsort(df["date"].values, kind="stable")
    # gap counted in rows: about one day of rows across all stations
    gap = int(df.groupby("date").size().median())
    tss = TimeSeriesSplit(n_splits=5, gap=gap)
    return [order[tr] for tr, _ in tss.split(order)], "provisional TimeSeriesSplit"


def main():
    df = pd.read_parquet(DATA_PATH)
    trains, source = load_folds(df)
    rows = []
    for i, tr in enumerate(trains):
        sub = df.iloc[tr][S5P_COLS].dropna()
        c = sub.corr(method="spearman")
        for a, b in combinations(S5P_COLS, 2):
            rows.append(dict(section="5.3.1", variable=f"{a}~{b}", group=f"train_fold_{i}",
                             metric="spearman", value=c.loc[a, b],
                             note=f"n={len(sub)}; folds: {source}"))
    out = pd.DataFrame(rows)
    verdicts = []
    for pair, g in out.groupby("variable"):
        spread = g.value.max() - g.value.min()
        verdicts.append(dict(section="5.3.1", variable=pair, group="all_folds",
                             metric="spearman_spread", value=spread,
                             note="stable" if spread < TOL else "NOT stable"))
    out = pd.concat([out, pd.DataFrame(verdicts)], ignore_index=True)
    stable = all(v["value"] < TOL for v in verdicts)
    out.loc[len(out)] = dict(section="5.3.1", variable="s5p", group="all_folds",
                             metric="knn_imputation_allowed", value=int(stable),
                             note=f"tolerance {TOL}")
    PARTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(PARTS / "m2_s5p_stability.csv", index=False)
    print(pd.DataFrame(verdicts)[["variable", "value", "note"]].to_string(index=False))
    print("KNN imputation allowed:", stable, f"(folds: {source})")


if __name__ == "__main__":
    main()
