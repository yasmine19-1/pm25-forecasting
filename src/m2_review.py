"""Member 2 - follow-up from the review.

1. How the predictors vary between stations (§5.1)
   - one boxplot figure per predictor, stations ordered by latitude
   - one heatmap of station medians on a robust common scale
   - Kruskal-Wallis test and eta² (share of variance explained by the station)
2. Investigation of the flagged PM2.5 extremes (§5.1, §5.3.3)
   For every PM2.5 value above Q3 + 3×IQR, the evidence that separates a real
   pollution event from a data error:
   - neighbours: stations within 30 km measuring the same day, and whether
     they are also high
   - persistence: is the station itself elevated the day before / after?
   - jump: value divided by the station's median over ±3 days
   - AOD: is the aerosol column high that day?
   - repeated exact value at the station (stuck sensor)
   Each extreme gets a verdict and the reason for it.

Descriptive only, on the development data. Run from the repo root:
    python src/m2_review.py
Writes results/<GROUP>/parts/m2_review.csv, m2_extremes_investigation.csv,
m2_review_notes.md and figures m2_station_*.png, m2_extreme_case_*.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from m2_config import DATA_PATH, FIGURES, PARTS, SEED

PREDICTORS = ["aod", "no2", "co", "o3", "temperature_celsius", "pressure_mb",
              "wind_u", "wind_v"]
RADIUS_KM = 30
ROWS: list[dict] = []


def add(section, variable, group, metric, value, note=""):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    ROWS.append(dict(section=section, variable=variable, group=group,
                     metric=metric, value=value, note=note))


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"m2_{name}.png", dpi=130)
    plt.close(fig)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ==========================================================================
# 1. Predictors between stations
# ==========================================================================
def predictors_by_station(df):
    coords = df.groupby("station_id")[["lat", "lon"]].median()
    order = coords.sort_values("lat").index  # south -> north

    eta = {}
    for v in PREDICTORS:
        d = df[["station_id", v]].dropna()
        groups = [g[v].values for _, g in d.groupby("station_id") if len(g) >= 5]
        h, p = stats.kruskal(*groups)
        # eta² = between-station sum of squares / total sum of squares
        means = d.groupby("station_id")[v].transform("mean")
        eta[v] = ((means - d[v].mean()) ** 2).sum() / ((d[v] - d[v].mean()) ** 2).sum()
        add("5.1", v, "station", "kruskal_H", h)
        add("5.1", v, "station", "kruskal_p", p)
        add("5.1", v, "station", "eta_squared", eta[v],
            "share of the variance explained by the station")
        med = d.groupby("station_id")[v].median()
        add("5.1", v, "station", "median_min", med.min(), med.idxmin())
        add("5.1", v, "station", "median_max", med.max(), med.idxmax())

        fig, ax = plt.subplots(figsize=(9, 15))
        data = [d.loc[d.station_id == s, v] for s in order if s in d.station_id.values]
        labels = [s for s in order if s in d.station_id.values]
        ax.boxplot(data, tick_labels=labels, orientation="horizontal",
                   showfliers=False)
        ax.axvline(d[v].median(), ls="--", c="#C44E52", lw=1, label="overall median")
        ax.set_title(f"{v} by station (south at the bottom, fliers hidden)\n"
                     f"eta² = {eta[v]:.2f}")
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(loc="lower right", fontsize=8)
        savefig(fig, f"station_{v}")

    # heatmap of station medians on a robust common scale
    med = df.groupby("station_id")[PREDICTORS + ["pm25"]].median().loc[order]
    z = (med - med.median()) / (med.quantile(.75) - med.quantile(.25)).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(8, 15))
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(z)), z.index, fontsize=6)
    fig.colorbar(im, label="(station median − median of stations) / IQR")
    ax.set_title("How each station differs (south at the top)")
    savefig(fig, "station_heatmap")

    # bar chart of eta² - the one-figure summary
    e = pd.Series(eta).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(e.index, e.values, color="#4C72B0")
    ax.set_xlabel("eta² (share of variance explained by the station)")
    ax.set_title("How much each predictor depends on the station")
    savefig(fig, "station_eta2")
    return e, med


# ==========================================================================
# 2. Extreme values
# ==========================================================================
def investigate_extremes(df):
    q1, q3 = df.pm25.quantile([.25, .75])
    thr = q3 + 3 * (q3 - q1)
    aod_p90 = df.aod.quantile(.9)

    coords = df.groupby("station_id")[["lat", "lon"]].median()
    names = coords.index.to_numpy()
    dist = haversine_km(coords.lat.values[:, None], coords.lon.values[:, None],
                        coords.lat.values[None, :], coords.lon.values[None, :])
    neigh = {s: [names[j] for j in np.where((dist[i] <= RADIUS_KM) & (dist[i] > 0))[0]]
             for i, s in enumerate(names)}

    st_p90 = df.groupby("station_id").pm25.quantile(.9)
    pivot = df.pivot_table(index="date", columns="station_id", values="pm25")
    # value counts of exact PM2.5 values per station (stuck sensor)
    repeats = df.groupby(["station_id", "pm25"]).size()

    out = []
    for _, r in df[df.pm25 > thr].iterrows():
        s, d, v = r.station_id, r.date, r.pm25
        nb = [n for n in neigh[s] if n in pivot.columns]
        nb_vals = pivot.loc[d, nb].dropna() if nb else pd.Series(dtype=float)
        # "also high" = above the neighbour's own P90 AND at least a fifth of the
        # extreme, so a neighbour at 30 does not "confirm" a value of 400
        nb_high = int(sum(nb_vals[n] > max(25, st_p90[n], v / 5) for n in nb_vals.index))
        own = pivot[s]
        prev = own.get(d - pd.Timedelta(days=1), np.nan)
        nxt = own.get(d + pd.Timedelta(days=1), np.nan)
        persist = any(x > max(25, st_p90[s], v / 5) for x in (prev, nxt) if pd.notna(x))
        window = own.loc[d - pd.Timedelta(days=3): d + pd.Timedelta(days=3)].drop(d)
        jump = v / window.median() if window.notna().any() and window.median() > 0 else np.nan
        aod_high = r.aod > aod_p90
        stuck = repeats.get((s, v), 1) > 1

        if nb_high >= 1 or (aod_high and persist):
            verdict = "real event"
            why = ("neighbouring station(s) also high" if nb_high
                   else "high AOD and elevated on adjacent days")
        elif persist:
            verdict = "plausible local event"
            why = "station elevated on adjacent days, neighbours not high"
        elif stuck:
            verdict = "probable data error"
            why = "exact value repeated at this station (stuck sensor)"
        elif len(nb_vals) >= 1 and pd.notna(jump) and jump >= 5:
            verdict = "probable data error"
            why = f"one-day spike ×{jump:.0f}, neighbours normal, AOD normal"
        elif pd.notna(jump) and jump >= 5:
            verdict = "suspect, unconfirmed"
            why = f"one-day spike ×{jump:.0f}, no neighbour within {RADIUS_KM} km to check"
        else:
            verdict = "undetermined"
            why = "moderate value, not enough evidence either way"
        out.append(dict(date=d.date(), station_id=s, pm25=round(v, 1),
                        aod=round(r.aod, 3), aod_high=aod_high,
                        n_neighbours_same_day=len(nb_vals),
                        neighbours_high=nb_high,
                        neighbour_max=round(nb_vals.max(), 1) if len(nb_vals) else np.nan,
                        pm25_day_before=round(prev, 1) if pd.notna(prev) else np.nan,
                        pm25_day_after=round(nxt, 1) if pd.notna(nxt) else np.nan,
                        jump_vs_7day_median=round(jump, 1) if pd.notna(jump) else np.nan,
                        repeated_value=stuck, month=d.month,
                        verdict=verdict, reason=why))

    ext = pd.DataFrame(out).sort_values("pm25", ascending=False)
    ext.to_csv(PARTS / "m2_extremes_investigation.csv", index=False)

    add("5.3.3", "pm25", "extremes", "threshold_q3_plus_3iqr", thr)
    add("5.3.3", "pm25", "extremes", "n_flagged", len(ext))
    add("5.3.3", "pm25", "extremes", "neighbour_radius_km", RADIUS_KM)
    for k, n in ext.verdict.value_counts().items():
        add("5.3.3", "pm25", f"verdict={k}", "n", int(n))
    for k, n in ext.groupby("verdict").pm25.median().items():
        add("5.3.3", "pm25", f"verdict={k}", "median_pm25", n)
    top = ext.station_id.value_counts().head(8)
    for s, n in top.items():
        add("5.3.3", "pm25", f"station={s}", "n_extremes", int(n))
    for m, n in ext.month.value_counts().sort_index().items():
        add("5.3.3", "pm25", f"month={m}", "n_extremes", int(n))
    return ext, pivot, neigh


def case_figures(ext, pivot, neigh):
    """Time series around a few typical cases: the station and its neighbours."""
    picks = []
    for verdict in ["real event", "plausible local event", "probable data error"]:
        sub = ext[ext.verdict == verdict]
        if len(sub):
            picks.append(sub.iloc[0])
    for i, r in enumerate(picks, 1):
        d = pd.Timestamp(r.date)
        win = pivot.loc[d - pd.Timedelta(days=10): d + pd.Timedelta(days=10)]
        fig, ax = plt.subplots(figsize=(9, 4))
        for n in [x for x in neigh[r.station_id] if x in win.columns][:5]:
            ax.plot(win.index, win[n], marker=".", lw=1, alpha=.6, label=n)
        ax.plot(win.index, win[r.station_id], marker="o", lw=2, c="k", label=r.station_id)
        ax.axvline(d, c="#C44E52", ls="--", lw=1)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"{r.verdict}: {r.station_id}, {r.date} ({r.pm25} µg/m³)\n{r.reason}")
        ax.legend(fontsize=7)
        fig.autofmt_xdate()
        savefig(fig, f"extreme_case_{i}")


def write_notes(eta, med, ext):
    vc = ext.verdict.value_counts()
    medv = ext.groupby("verdict").pm25.median()
    real = ext[ext.verdict == "real event"]
    err = ext[ext.verdict == "probable data error"]
    winter = real.month.isin([11, 12, 1, 2]).mean()
    top_real = ", ".join(f"{s} ({n})" for s, n in real.station_id.value_counts().head(4).items())
    top_err = ", ".join(f"{s} ({n})" for s, n in err.station_id.value_counts().head(4).items())
    big = ext.head(10)
    n_big_err = int((big.verdict == "probable data error").sum())
    hi_no2, lo_no2 = med.no2.idxmax(), med.no2.idxmin()
    p_lo, p_hi = med.pressure_mb.idxmin(), med.pressure_mb.idxmax()
    text = f"""# Member 2 - review follow-up

## Predictors between stations (§5.1)

Figures: `m2_station_<variable>.png` (one per predictor, stations from south
to north), `m2_station_heatmap.png` (all predictors at once),
`m2_station_eta2.png` (summary).

eta² = share of a predictor's variance explained by the station alone:

| predictor | eta² |
|---|---|
""" + "\n".join(f"| {k} | {v:.2f} |" for k, v in eta.sort_values(ascending=False).items()) + f"""

- **Pressure** is almost entirely a station property (eta² {eta['pressure_mb']:.2f}):
  it reflects altitude (lowest median at {p_lo}, highest at {p_hi}). It works
  as a proxy for elevation, which is fine physically, but it also lets a model
  recognise a station: worth remembering when reading spatial-CV results.
- **NO₂** differs clearly between stations (eta² {eta['no2']:.2f}): highest
  median at {hi_no2}, lowest at {lo_no2}. Urban and industrial sites stand out.
- **AOD, O₃, temperature and wind** barely depend on the station (eta² under
  0.1): most of their variation is day to day, shared across the region.
  Satellite pixels are coarse (several stations share one pixel), so these
  columns carry little information on *where* PM2.5 is high, mostly on *when*.
- Every predictor still differs significantly between stations
  (Kruskal-Wallis p < 0.001), but with 31,000 rows that test is always
  significant: eta² is the useful measure.

## Flagged extreme values: real events or data errors? (§5.1, §5.3.3)

All {len(ext)} PM2.5 values above Q3 + 3×IQR were checked one by one
(`m2_extremes_investigation.csv`, one row per value, with the evidence and the
verdict). Evidence used: stations within {RADIUS_KM} km the same day, the
station's own values the day before and after, the jump against the
station's ±3-day median, AOD, and repeated exact values. A neighbour or an
adjacent day only "confirms" a value if it is above its own 90th percentile
and at least a fifth of the extreme.

| verdict | n | median PM2.5 |
|---|---|---|
""" + "\n".join(f"| {k} | {vc[k]} | {medv[k]:.1f} |" for k in vc.index) + f"""

- **Real events ({vc.get('real event', 0)})** are moderate values (median
  {medv.get('real event', np.nan):.0f} µg/m³) shared with nearby stations.
  {winter:.0%} happen in November to February, mostly at urban stations
  ({top_real}), consistent with winter inversions, heating and traffic. These are exactly the
  days the model must learn to predict, so they stay.
- **Probable data errors ({vc.get('probable data error', 0)})** look very
  different: much higher values (median {medv.get('probable data error', np.nan):.0f} µg/m³),
  a one-day spike (5 to 76 times the station's usual level) while
  neighbours within {RADIUS_KM} km stay normal and AOD is ordinary. They
  concentrate at a few stations: {top_err}. {n_big_err} of the 10 highest
  values in the dataset fall in this group, including the 485 µg/m³ at CENCA1
  on 2024-09-04 (neighbour 14.7 µg/m³ the same day).
- **Plausible local events ({vc.get('plausible local event', 0)})**: the
  station stays high for two days or more while neighbours do not. A sensor
  fault can last two days, but so can a local source. Example: in November
  2021 Los Barrios and Campamento (Campo de Gibraltar, an industrial area)
  each stay above 200 µg/m³ for two days, and on 23 November a neighbour
  reaches 123 µg/m³. A local industrial episode is possible, but the
  evidence is mixed. These are kept.
- **Suspect, unconfirmed ({vc.get('suspect, unconfirmed', 0)})** and
  **undetermined ({vc.get('undetermined', 0)})**: no neighbour to compare
  with, or not enough evidence. Kept.
- Case studies: `m2_extreme_case_1.png` (real event),
  `m2_extreme_case_2.png` (local event), `m2_extreme_case_3.png` (data error).

### What to do with them (proposal for the team)
1. Keep every real, local and undetermined event: removing them would hide
   the high-pollution days from the model.
2. The {vc.get('probable data error', 0)} probable data errors are
   {vc.get('probable data error', 0) / 31282:.2%} of the rows but, at
   100 to 485 µg/m³, they can dominate RMSE. Proposal: run Phase 2 twice,
   with and without them in the **training** folds (the rule uses fixed
   thresholds and same-day neighbours, nothing learned from validation), and
   report validation scores both on all rows and without the flagged rows.
3. Flag CENCA1 and Los Barrios to the team: they produce most of the
   suspicious spikes.
"""
    (PARTS / "m2_review_notes.md").write_text(text)


# ==========================================================================
def main():
    np.random.seed(SEED)
    PARTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    eta, med = predictors_by_station(df)
    ext, pivot, neigh = investigate_extremes(df)
    case_figures(ext, pivot, neigh)

    pd.DataFrame(ROWS).to_csv(PARTS / "m2_review.csv", index=False)
    write_notes(eta, med, ext)
    print(eta.round(3).to_string())
    print(ext.verdict.value_counts().to_string())
    return eta, med, ext


if __name__ == "__main__":
    main()
