"""Member 3 - relationships between the variables (brief section 5.1).

Pearson and Spearman correlations, predictor pairs with |r| > 0.9, VIF, PCA
(variance and loadings) and a t-SNE picture.

DESCRIPTIVE ONLY. Nothing here drops or changes a column that a model uses.
Correlated pairs are reported; if a column has to go, that happens inside the
training fold of the Pipeline (ground rule 3).

Run from the repo root:
    python -m src.relationships            # add --no-tsne to skip the slow plot

Writes:
    results/<GROUP>/parts/m3_relationships.csv        (shared schema)
    results/<GROUP>/parts/m3_relationships_notes.md   (text for the report)
    results/<GROUP>/figures/m3_corr_pearson.png, m3_corr_spearman.png,
                            m3_pca_scree.png, m3_tsne.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src import config
from src.m3_common import (FIGURES, add, load_dev, predictor_cols, save_notes,
                           save_parts, zero_aod_to_nan)

SEC = "5.1"
HIGH_R = 0.9


def vif_table(df, cols):
    """VIF_i = i-th diagonal element of the inverse correlation matrix.

    Uses only the rows where every predictor is present (listwise) so that the
    matrix is a proper correlation matrix. Returns (vif Series, rows used).
    """
    complete = df[cols].dropna()
    if len(complete) < 10 * len(cols):
        return None, len(complete)
    corr = complete.corr().to_numpy()
    inv = np.linalg.pinv(corr)  # pinv so that a perfectly collinear pair gives a huge value, not a crash
    return pd.Series(np.diag(inv), index=cols), len(complete)


def heatmap(corr, title, path):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.75 * n + 2.5, 0.75 * n + 1.5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(n), corr.index)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tsne_plot(df, cols, path, n=3000):
    """Exploratory picture only (the brief says t-SNE is not proof of predictive value)."""
    sample = df.sample(min(n, len(df)), random_state=config.SEED)
    Z = StandardScaler().fit_transform(sample[cols].fillna(sample[cols].median()))
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=config.SEED).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.log1p(sample[config.TARGET]), s=5, cmap="viridis")
    fig.colorbar(sc, label="log(1 + PM2.5)")
    ax.set_title(f"t-SNE of the predictors ({len(sample)} random rows, exploratory)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(tsne=True, data_path=None):
    df = zero_aod_to_nan(load_dev(data_path))
    X = predictor_cols(df)
    cols = X + [config.TARGET]
    rows, notes = [], []
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1) correlations (pairwise complete, because Sentinel-5P has gaps)
    mats = {"pearson": df[cols].corr(method="pearson"),
            "spearman": df[cols].corr(method="spearman")}
    for name, m in mats.items():
        for v in X:
            add(rows, SEC, v, name, "r_with_pm25", m.loc[v, config.TARGET])
        heatmap(m, f"{name.capitalize()} correlation (development data)", FIGURES / f"m3_corr_{name}.png")

    # 2) predictor pairs above 0.9 in absolute value: reported, never dropped
    flagged = []
    for name, m in mats.items():
        for i, a in enumerate(X):
            for b in X[i + 1:]:
                r = m.loc[a, b]
                add(rows, SEC, f"{a}|{b}", name, "r_pair", r,
                    "above 0.9 in absolute value" if abs(r) > HIGH_R else "")
                if abs(r) > HIGH_R:
                    flagged.append((name, a, b, r))
    add(rows, SEC, "predictor_pairs", "all", "n_pairs_above_0.9", len({(a, b) for _, a, b, _ in flagged}))

    # 3) VIF
    vif, n_vif = vif_table(df, X)
    if vif is None:
        add(rows, SEC, "vif", "all", "not_computed", n_vif, "too few rows without a missing predictor")
    else:
        for v, val in vif.items():
            add(rows, SEC, v, "all", "vif", val, f"listwise rows used: {n_vif}")

    # 4) PCA on standardised predictors (median fill is for this description only)
    Z = StandardScaler().fit_transform(df[X].fillna(df[X].median()))
    pca = PCA(random_state=config.SEED).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    for k, (e, c) in enumerate(zip(pca.explained_variance_ratio_, cum), start=1):
        add(rows, SEC, f"PC{k}", "all", "explained_variance_ratio", e, f"cumulative {c:.4f}")
    for k in range(min(3, len(X))):
        for v, load in zip(X, pca.components_[k]):
            add(rows, SEC, v, f"PC{k + 1}", "loading", load)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, label="per component")
    ax.plot(range(1, len(cum) + 1), cum, "k.-", label="cumulative")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "m3_pca_scree.png", dpi=150)
    plt.close(fig)

    # 5) t-SNE
    if tsne:
        tsne_plot(df, X, FIGURES / "m3_tsne.png")

    path = save_parts(rows, "m3_relationships.csv")

    # text for the report
    r_target = mats["spearman"].loc[X, config.TARGET].sort_values(key=abs, ascending=False)
    notes.append("# Member 3 - relationships (auto-generated, check the numbers before pasting)\n")
    notes.append(f"Development rows: {len(df)}. Predictors: {', '.join(X)}.\n")
    notes.append("Strongest Spearman correlations with PM2.5: " +
                 ", ".join(f"{v} ({r:+.2f})" for v, r in r_target.head(3).items()) + ".")
    if flagged:
        pairs = sorted({(a, b) for _, a, b, _ in flagged})
        notes.append("Predictor pairs above 0.9 in absolute value (reported, not dropped): " +
                     ", ".join(f"{a}-{b}" for a, b in pairs) + ".")
    else:
        notes.append("No predictor pair has an absolute Pearson or Spearman correlation above 0.9.")
    if vif is not None:
        high = vif[vif > 10]
        notes.append("VIF above 10: " + (", ".join(f"{v} ({x:.1f})" for v, x in high.items()) or "none") +
                     f" (computed on {n_vif} rows with no missing predictor).")
    n80 = int(np.searchsorted(cum, 0.80) + 1)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    notes.append(f"PCA: {n80} components explain 80% and {n90} explain 90% of the standardised variance.")
    notes.append("Meaning for the models: these are descriptive findings. Any column removed because of "
                 "collinearity is decided inside each training fold (ground rule 3).")
    save_notes(notes, "m3_relationships_notes.md")
    print(f"{len(rows)} rows -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-tsne", action="store_true", help="skip the t-SNE figure")
    ap.add_argument("--data", default=None, help="other parquet than data/interim/clean_dev.parquet")
    args = ap.parse_args()
    from pathlib import Path
    main(tsne=not args.no_tsne, data_path=Path(args.data) if args.data else None)
