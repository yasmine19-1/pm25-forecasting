"""Member 2 - descriptive EDA, target analysis, missingness and outlier inspection.

Brief sections: 5.1, 5.3.1, 5.3.3.
Everything here is DESCRIPTIVE and runs on the development data only (test
period already removed by Member 1). No threshold computed here is used to
transform data: the Pipeline learns its own thresholds per training fold.

Run from the repo root:  python src/m2_eda.py
Writes: results/<GROUP>/parts/m2_eda.csv  (shared schema)
        results/<GROUP>/figures/m2_*.png
        results/<GROUP>/parts/m2_notes.md  (interpretations to paste in the report)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from m2_config import (DATA_PATH, FIGURES, PARTS, SEED, TARGET, S5P_COLS,
                    NUMERIC_COLS, PM25_THRESHOLDS)

ROWS: list[dict] = []
NOTES: list[str] = []

# Physically plausible ranges (domain limits). Values outside = sensor/retrieval error.
DOMAIN_LIMITS = {
    "temperature_celsius": (-40, 55),
    "pressure_mb": (800, 1085),
    "wind_u": (-40, 40),
    "wind_v": (-40, 40),
    "no2": (0, 1e-3),      # mol/m2 tropospheric column
    "co": (0, 0.2),        # mol/m2
    "o3": (0, 0.5),        # mol/m2
    "aod": (0, 5),
    "pm25": (0, 500),      # cleaning rule already applied by Member 1
}
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
PLOT_VARS = ["pm25", "aod", "no2", "co", "o3", "temperature_celsius",
             "pressure_mb", "wind_u", "wind_v"]


def add(section, variable, group, metric, value, note=""):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    ROWS.append(dict(section=section, variable=variable, group=group,
                     metric=metric, value=value, note=note))


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"m2_{name}.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["season"] = df["month"].map(SEASONS)
    df["wind_speed"] = np.hypot(df["wind_u"], df["wind_v"])  # descriptive only
    return df


# --------------------------------------------------------------------------
def descriptive(df):
    for v in NUMERIC_COLS + ["wind_speed"]:
        s = df[v].dropna()
        q1, q2, q3 = s.quantile([.25, .5, .75])
        for m, val in [("n", s.size), ("min", s.min()), ("q1", q1),
                       ("median", q2), ("q3", q3), ("max", s.max()),
                       ("mean", s.mean()), ("sd", s.std()),
                       ("skewness", s.skew()), ("kurtosis_excess", s.kurt())]:
            add("5.1", v, "all", m, val,
                "wind_speed derived from u,v for description only" if v == "wind_speed" else "")

    add("5.1", "dataset", "all", "n_rows", len(df))
    add("5.1", "dataset", "all", "n_stations", df.station_id.nunique())
    add("5.1", "dataset", "all", "n_gee_ids", df.gee_id.nunique(),
        "more gee_ids than stations: some stations map to several GEE cells")
    add("5.1", "dataset", "all", "date_min", str(df.date.min().date()))
    add("5.1", "dataset", "all", "date_max", str(df.date.max().date()))

    per_station = df.groupby("station_id").agg(
        n=("date", "size"), first=("date", "min"), last=("date", "max"))
    for st, r in per_station.iterrows():
        add("5.1", "station_id", st, "n_rows", int(r.n),
            f"{r['first'].date()} to {r['last'].date()}")
    desc = per_station.n.describe()
    for m in ["min", "50%", "max"]:
        add("5.1", "station_id", "all", f"rows_per_station_{m.replace('%', '')}", desc[m])
    small = per_station[per_station.n < 30]
    add("5.1", "station_id", "all", "stations_under_30_rows", len(small),
        "; ".join(small.index))
    for y, n in df.year.value_counts().sort_index().items():
        add("5.1", "year", str(y), "n_rows", n)


# --------------------------------------------------------------------------
def figures(df):
    # histograms
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, v in zip(axes.ravel(), PLOT_VARS):
        ax.hist(df[v].dropna(), bins=60, color="#4C72B0", edgecolor="white", lw=.3)
        ax.set_title(v)
    savefig(fig, "hist_all")

    # PM2.5 raw vs log
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df.pm25, bins=100, color="#C44E52")
    ax[0].set_title("PM2.5 (µg/m³)")
    for t in PM25_THRESHOLDS:
        ax[0].axvline(t, ls="--", c="k", lw=.8)
    ax[1].hist(np.log1p(df.pm25), bins=80, color="#C44E52")
    ax[1].set_title("log(1 + PM2.5)")
    savefig(fig, "pm25_hist_log")

    # boxplots (standardised so they share an axis)
    z = (df[PLOT_VARS] - df[PLOT_VARS].median()) / (
        df[PLOT_VARS].quantile(.75) - df[PLOT_VARS].quantile(.25))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.boxplot([z[v].dropna() for v in PLOT_VARS], tick_labels=PLOT_VARS,
               flierprops=dict(markersize=2, alpha=.3))
    ax.set_ylabel("(x - median) / IQR")
    ax.set_title("Boxplots on a robust common scale")
    plt.setp(ax.get_xticklabels(), rotation=30)
    savefig(fig, "boxplots_robust")

    # by month
    for v in ["pm25", "aod", "no2", "co", "o3", "temperature_celsius"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        data = [df.loc[df.month == m, v].dropna() for m in range(1, 13)]
        ax.boxplot(data, tick_labels=range(1, 13), showfliers=False)
        ax.set_xlabel("month")
        ax.set_title(f"{v} by month (fliers hidden)")
        savefig(fig, f"{v}_by_month")

    # PM2.5 by station, ordered by median
    order = df.groupby("station_id").pm25.median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.boxplot([df.loc[df.station_id == s, "pm25"] for s in order],
               tick_labels=order, orientation="horizontal", showfliers=False)
    ax.set_xlabel("PM2.5 (µg/m³), fliers hidden")
    ax.tick_params(axis="y", labelsize=6)
    savefig(fig, "pm25_by_station")

    # rows per station (categorical count plot)
    cnt = df.station_id.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 14))
    ax.barh(cnt.index, cnt.values, color="#55A868")
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("rows")
    ax.set_title("Observations per station")
    savefig(fig, "rows_per_station")


# --------------------------------------------------------------------------
def target(df):
    y = df[TARGET]
    sk, ku = y.skew(), y.kurt()
    add("5.1", TARGET, "all", "skewness", sk)
    add("5.1", TARGET, "all", "kurtosis_excess", ku)
    add("5.1", TARGET, "all", "skewness_log1p", np.log1p(y).skew(),
        "log1p transform reduces skew")
    for t in PM25_THRESHOLDS:
        add("5.1", TARGET, "all", f"share_above_{t}", (y > t).mean())
        add("5.1", TARGET, "all", f"n_above_{t}", int((y > t).sum()))
    for p in [.9, .95, .99, .999]:
        add("5.1", TARGET, "all", f"p{p*100:g}", y.quantile(p))

    for grp_col in ["season", "month", "year"]:
        g = df.groupby(grp_col)[TARGET]
        for k, s in g:
            add("5.1", TARGET, f"{grp_col}={k}", "median", s.median())
            add("5.1", TARGET, f"{grp_col}={k}", "mean", s.mean())
            for t in PM25_THRESHOLDS:
                add("5.1", TARGET, f"{grp_col}={k}", f"share_above_{t}", (s > t).mean())

    # Kruskal-Wallis: does PM2.5 differ by season?
    h, p = stats.kruskal(*[s.values for _, s in df.groupby("season")[TARGET]])
    add("5.1", TARGET, "season", "kruskal_H", h)
    add("5.1", TARGET, "season", "kruskal_p", p)

    # Monthly median line
    m = df.groupby("month")[TARGET].agg(["median", lambda s: s.quantile(.9)])
    m.columns = ["median", "p90"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m.index, m["median"], marker="o", label="median")
    ax.plot(m.index, m["p90"], marker="s", label="90th percentile")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month"); ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(); ax.set_title("Seasonal profile of PM2.5")
    savefig(fig, "pm25_seasonal_profile")

    # Share of error budget from extremes: how much of the total squared
    # deviation comes from the top 1 % of days (illustrates RMSE sensitivity)
    dev2 = (y - y.mean()) ** 2
    top = y >= y.quantile(.99)
    share_sq = dev2[top].sum() / dev2.sum()
    dev1 = (y - y.median()).abs()
    share_abs = dev1[top].sum() / dev1.sum()
    add("5.1", TARGET, "top1pct", "share_of_squared_deviation", share_sq,
        "how much the top 1% of days weigh in an RMSE-type loss")
    add("5.1", TARGET, "top1pct", "share_of_absolute_deviation", share_abs,
        "same for an MAE-type loss")

    seas = df.groupby("season")[TARGET].median().round(1).to_dict()
    NOTES.append(f"""## Target (PM2.5) - §5.1

- Strong right skew (skewness {sk:.2f}, excess kurtosis {ku:.1f}); median
  {y.median():.1f} µg/m³ vs mean {y.mean():.1f}. log1p brings skewness to
  {np.log1p(y).skew():.2f}.
- {(y > 25).mean():.1%} of station-days exceed 25 µg/m³ and {(y > 50).mean():.2%}
  exceed 50 µg/m³: high-pollution days are rare events.
- Seasonal medians: {seas} (Kruskal-Wallis p = {p:.2g}).
- The top 1 % of days carries {share_sq:.0%} of the total squared deviation but
  only {share_abs:.0%} of the absolute deviation. Consequences: RMSE will be
  dominated by a handful of extreme days, MAE much less so; a model can have a
  good MAE and still miss every episode. Report RMSE and MAE side by side and
  add an extreme-event metric (e.g. MAE or recall on days > 25 µg/m³).
- Option for Phase 2: train on log1p(PM2.5) and back-transform, compared with
  the raw target inside the same folds.
""")


# --------------------------------------------------------------------------
def missingness(df):
    for v in NUMERIC_COLS:
        add("5.3.1", v, "all", "missing_share", df[v].isna().mean())
        add("5.3.1", v, "all", "missing_n", int(df[v].isna().sum()))

    miss = df[S5P_COLS].isna()
    add("5.3.1", "s5p", "all", "all_three_missing_n", int(miss.all(axis=1).sum()))
    add("5.3.1", "s5p", "all", "any_missing_n", int(miss.any(axis=1).sum()))
    pat = miss.astype(int).astype(str).agg("".join, axis=1).value_counts()
    for k, n in pat.items():
        add("5.3.1", "s5p", "pattern_no2co_o3=" + k, "n_rows", int(n),
            "1 = missing, order no2,co,o3")

    by_month = df.groupby("month")[S5P_COLS].apply(lambda d: d.isna().mean())
    by_year = df.groupby("year")[S5P_COLS].apply(lambda d: d.isna().mean())
    by_station = df.groupby("station_id")[S5P_COLS].apply(lambda d: d.isna().mean())
    for name, tab in [("month", by_month), ("year", by_year), ("station", by_station)]:
        for k, r in tab.iterrows():
            for v in S5P_COLS:
                add("5.3.1", v, f"{name}={k}", "missing_share", r[v])

    # Is missingness related to the target or weather? (MAR check)
    any_m = miss.any(axis=1)
    p_mw = {}
    for v in ["pm25", "aod", "temperature_celsius"]:
        a, b = df.loc[any_m, v], df.loc[~any_m, v]
        u, p = stats.mannwhitneyu(a, b)
        p_mw[v] = p
        add("5.3.1", v, "s5p_missing_vs_present", "median_when_missing", a.median())
        add("5.3.1", v, "s5p_missing_vs_present", "median_when_present", b.median())
        add("5.3.1", v, "s5p_missing_vs_present", "mannwhitney_p", p)

    # heatmaps
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(by_month.T.values, aspect="auto", cmap="Reds", vmin=0)
    ax.set_yticks(range(3), S5P_COLS); ax.set_xticks(range(12), range(1, 13))
    ax.set_xlabel("month"); fig.colorbar(im, label="share missing")
    ax.set_title("Sentinel-5P missingness by month")
    savefig(fig, "missing_by_month")

    ym = df.assign(ym=df.date.dt.to_period("M")).groupby("ym")[S5P_COLS] \
        .apply(lambda d: d.isna().mean())
    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(ym.T.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_yticks(range(3), S5P_COLS)
    step = max(1, len(ym) // 24)
    ax.set_xticks(range(0, len(ym), step), [str(p) for p in ym.index[::step]], rotation=60)
    fig.colorbar(im, label="share missing")
    ax.set_title("Sentinel-5P missingness by year-month")
    savefig(fig, "missing_by_yearmonth")

    st = by_station.sort_values("no2")
    fig, ax = plt.subplots(figsize=(5, 15))
    im = ax.imshow(st.values, aspect="auto", cmap="Reds", vmin=0)
    ax.set_xticks(range(3), S5P_COLS)
    ax.set_yticks(range(len(st)), st.index, fontsize=6)
    fig.colorbar(im, label="share missing")
    ax.set_title("Missingness by station")
    savefig(fig, "missing_by_station")

    gap_first = str(df.loc[any_m, "date"].min().date())
    gap_last = str(df.loc[any_m, "date"].max().date())
    s5p_first = str(df.loc[~any_m, "date"].min().date())
    miss_after = int(any_m[df.date >= s5p_first].sum())
    n_before = int((df.date < s5p_first).sum())
    first_fold = any_m.loc[df.sort_values("date").index][: len(df) // 6]
    first_fold_share = first_fold.mean()
    add("5.3.1", "s5p", "all", "first_missing_date", gap_first)
    add("5.3.1", "s5p", "all", "last_missing_date", gap_last)
    add("5.3.1", "s5p", "all", "first_observed_date", s5p_first,
        "Sentinel-5P products start here; gaps are structural")
    add("5.3.1", "s5p", "all", "missing_after_first_observed", miss_after)
    add("5.3.1", "dataset", "all", "rows_before_s5p_start", n_before)
    worst_m = by_month.no2.idxmax(); best_m = by_month.no2.idxmin()
    worst_s = by_station.no2.sort_values(ascending=False).head(3)
    NOTES.append(f"""## Missingness - §5.1, §5.3.1

- Only the Sentinel-5P columns have gaps: NO₂ {df.no2.isna().mean():.1%}, CO
  {df.co.isna().mean():.1%}, O₃ {df.o3.isna().mean():.1%}. Every other column,
  including the target, is complete.
- The three are almost always missing together ({int(miss.all(axis=1).sum())} of
  {int(miss.any(axis=1).sum())} rows with any gap).
- **The gaps are structural, not random.** Every missing value falls between
  {gap_first} and {gap_last}; the first row with Sentinel-5P data is {s5p_first}.
  Sentinel-5P was launched in October 2017 and its products only start in
  2018, which matches these dates; after {s5p_first} there are {miss_after} missing rows. See `m2_missing_by_yearmonth.png`.
- Because of that, the month and station patterns are artefacts: month
  {worst_m} ({by_month.no2.max():.1%} missing) and the worst stations
  ({', '.join(f'{s} {v:.0%}' for s, v in worst_s.items())}) are just the ones with
  many pre-{s5p_first[:4]} rows.
- Missingness vs other variables (Mann-Whitney): PM2.5 p = {p_mw['pm25']:.2g},
  AOD p = {p_mw['aod']:.2g}, temperature p = {p_mw['temperature_celsius']:.2g}.
  Gaps are not linked to the target.

### Decision for the Thursday gate (§5.3.1)
- **Option A (recommended): start the development data at {s5p_first}.** It drops
  {n_before} rows ({n_before/len(df):.1%}) and leaves essentially no missing
  values. Imputing two years of medians would invent pollutant data for a
  period with no satellite, and the missingness indicator would just become a
  "before 2018" flag. With TimeSeriesSplit (5 folds) the first training
  fold would be about {first_fold_share:.0%} imputed rows.
- **Option B: keep all rows.** Then median imputation fitted on the training
  fold + one indicator per S5P column, no forward-fill. `imputation.py`
  supports both options unchanged.
- KNN imputation is not worth it here: there is no partially observed period
  to learn from. The stability check stays in the repo for completeness.
""")


# --------------------------------------------------------------------------
def outliers(df):
    X = df[PLOT_VARS].copy()
    for v in PLOT_VARS:
        s = X[v].dropna()
        lo, hi = DOMAIN_LIMITS[v]
        add("5.3.3", v, "domain", "n_outside_limits", int(((s < lo) | (s > hi)).sum()),
            f"limits [{lo}, {hi}]")
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        add("5.3.3", v, "iqr1.5", "n_flagged", int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()))
        add("5.3.3", v, "iqr3", "n_flagged", int(((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).sum()))
        z = (s - s.mean()) / s.std()
        add("5.3.3", v, "z>3", "n_flagged", int((z.abs() > 3).sum()))
        mad = stats.median_abs_deviation(s, scale="normal")
        rz = (s - s.median()) / mad if mad > 0 else s * 0
        add("5.3.3", v, "robust_z>3.5", "n_flagged", int((rz.abs() > 3.5).sum()))

    # multivariate: complete rows, standardised
    cols = [c for c in PLOT_VARS if c != "pm25"]
    Xc = X[cols].dropna()
    Xs = (Xc - Xc.median()) / (Xc.quantile(.75) - Xc.quantile(.25))
    iso = IsolationForest(contamination=0.01, random_state=SEED).fit(Xs)
    iso_flag = iso.predict(Xs) == -1
    lof_flag = LocalOutlierFactor(n_neighbors=35, contamination=0.01).fit_predict(Xs) == -1
    add("5.3.3", "predictors", "isolation_forest_1pct", "n_flagged", int(iso_flag.sum()))
    add("5.3.3", "predictors", "lof_1pct", "n_flagged", int(lof_flag.sum()))
    add("5.3.3", "predictors", "iso_and_lof", "n_flagged", int((iso_flag & lof_flag).sum()))
    flagged = Xc.index[iso_flag & lof_flag]
    for v in cols:
        add("5.3.3", v, "iso_and_lof", "median_flagged", Xc.loc[flagged, v].median())
        add("5.3.3", v, "iso_and_lof", "median_all", Xc[v].median())

    # PM2.5: sensor error or real episode?
    # A real episode is usually regional: other stations the same day are also
    # high, or AOD is high. An isolated spike with low AOD and calm neighbours
    # is more likely a sensor problem.
    q1, q3 = df.pm25.quantile([.25, .75]); hi = q3 + 3 * (q3 - q1)
    day_med = df.groupby("date").pm25.transform("median")
    day_n = df.groupby("date").pm25.transform("size")
    aod_p90 = df.aod.quantile(.9)
    ext = df[df.pm25 > hi].assign(day_median=day_med, day_n=day_n)
    regional = (ext.day_median > df.pm25.quantile(.75)) & (ext.day_n >= 3)
    aod_high = ext.aod > aod_p90
    ext["class"] = np.select([regional | aod_high, ext.day_n < 3],
                             ["episode_likely", "undetermined_few_stations"],
                             "isolated_suspect")
    add("5.3.3", "pm25", "iqr3_upper", "threshold", hi)
    for k, n in ext["class"].value_counts().items():
        add("5.3.3", "pm25", f"extreme_{k}", "n", int(n),
            "regional = same-day median across stations > overall Q3; or AOD > P90")
    ext.sort_values("pm25", ascending=False)[
        ["date", "station_id", "pm25", "aod", "day_median", "day_n", "class"]
    ].to_csv(PARTS / "m2_pm25_extremes.csv", index=False)

    # AOD extremes
    big_aod = df[df.aod > 1]
    add("5.3.3", "aod", "aod>1", "n", len(big_aod))
    add("5.3.3", "aod", "aod>1", "median_pm25_same_row", big_aod.pm25.median(),
        "if PM2.5 is not elevated, high AOD is likely cloud contamination / dust aloft")
    add("5.3.3", "aod", "aod==0", "n", int((df.aod == 0).sum()), "exact zero: suspicious retrieval")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"episode_likely": "#C44E52", "isolated_suspect": "#4C72B0",
              "undetermined_few_stations": "#999999"}
    ax.scatter(df.aod, df.pm25, s=3, c="#DDDDDD", label="all")
    for k, g in ext.groupby("class"):
        ax.scatter(g.aod, g.pm25, s=12, c=colors[k], label=k)
    ax.set_xscale("symlog", linthresh=.1); ax.set_yscale("log")
    ax.set_xlabel("AOD"); ax.set_ylabel("PM2.5"); ax.legend(fontsize=7)
    ax.set_title("PM2.5 extremes: episode or sensor error?")
    savefig(fig, "pm25_extremes_vs_aod")

    counts = ext["class"].value_counts().to_dict()
    NOTES.append(f"""## Outliers - §5.1, §5.3.3

- Domain limits: no value is physically impossible after Member 1's cleaning,
  except {int((df.aod == 0).sum())} AOD values exactly at 0 (suspicious
  retrievals) and {len(big_aod)} AOD values above 1 (median PM2.5 on those rows
  {big_aod.pm25.median():.1f} µg/m³: likely dust/cloud, not local pollution).
- Univariate rules flag very different numbers (see `n_flagged` rows): the
  1.5×IQR rule flags a lot on skewed variables (PM2.5, AOD) and is too
  aggressive; 3×IQR or robust z (MAD) is more sensible. Pressure is a good
  warning: 1.5×IQR flags thousands of rows that are simply high-altitude
  stations (pressure is multimodal), while 3×IQR flags none.
- PM2.5 above 3×IQR (> {hi:.1f} µg/m³): {len(ext)} rows. Classification:
  {counts}. "episode_likely" = several stations high the same day or high AOD;
  "isolated_suspect" = one station spikes alone with ordinary AOD.
  List in `m2_pm25_extremes.csv`.
- Isolation Forest and LOF (1 % each) agree on {int((iso_flag & lof_flag).sum())}
  predictor rows.

### Outlier policy (proposal for the team)
1. Never remove target values inside the Pipeline (rows can't be dropped in an
   sklearn transformer, and real episodes are exactly what we must predict).
   The cleaning rule 0 < PM2.5 < 500 stays the only target filter.
2. Predictors: winsorise (clip) at limits learned on the training fold only:
   default Q1 − 3×IQR / Q3 + 3×IQR, applied unchanged to validation and test.
   AOD == 0 is treated as missing.
3. `outliers.py` has `enabled=True/False` so Phase 2 compares MAE, RMSE, R²
   with and without the treatment in the same folds.
4. "isolated_suspect" PM2.5 days are reported, not deleted; a sensitivity run
   without them can be added in Phase 2 if the team wants.
""")


# --------------------------------------------------------------------------
def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    df = load()
    descriptive(df)
    figures(df)
    target(df)
    missingness(df)
    outliers(df)
    out = pd.DataFrame(ROWS, columns=["section", "variable", "group", "metric", "value", "note"])
    out.to_csv(PARTS / "m2_eda.csv", index=False)
    (PARTS / "m2_notes.md").write_text("# Member 2 - notes for report §5\n\n" + "\n".join(NOTES))
    print(f"{len(out)} rows -> {PARTS / 'm2_eda.csv'}")
    print(f"{len(list(FIGURES.glob('m2_*.png')))} figures -> {FIGURES}")


if __name__ == "__main__":
    main()
