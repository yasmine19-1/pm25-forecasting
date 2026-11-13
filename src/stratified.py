"""PM2.5 against AOD, by season, by station and by weather (brief 5.1).

Questions answered:
  * how strongly is PM2.5 linked to AOD overall, in each season and at each station?
  * does the PM2.5-AOD link change with temperature, pressure or wind?

DESCRIPTIVE ONLY, on the development data. Rows with AOD missing or AOD == 0
(failed retrieval, consistent with the predictor clipper) are left out of this analysis.

Run from the repo root:   python -m src.stratified

Writes:
    results/<GROUP>/parts/m3_stratified.csv         (shared schema)
    results/<GROUP>/parts/m3_stratified_notes.md    (interpretation for the report)
    results/<GROUP>/figures/m3_pm25_aod_by_season.png, m3_pm25_aod_by_station.png,
                            m3_pm25_aod_by_weather.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.m3_common import FIGURES, add, load_dev, save_notes, save_parts, zero_aod_to_nan

SEC = "5.1"
TARGET = config.TARGET
MIN_PAIRS = 30            # fewer valid PM2.5/AOD pairs than this: the station is too thin to trust
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
WEATHER = ["temperature_celsius", "pressure_mb", "wind_speed"]
BIN_LABELS = ["low", "middle", "high"]


def link(x, y):
    """Pearson r, Spearman rho and the OLS slope of y on x (PM2.5 is skewed, so both correlations)."""
    if len(x) < 3 or x.nunique() < 2:
        return dict(n=len(x), pearson=np.nan, spearman=np.nan, slope=np.nan)
    return dict(n=len(x),
                pearson=stats.pearsonr(x, y)[0],
                spearman=stats.spearmanr(x, y)[0],
                slope=np.polyfit(x, y, 1)[0])


def add_link(rows, variable, group, res, note=""):
    for metric in ("n", "pearson", "spearman", "slope"):
        add(rows, SEC, variable, group, f"aod_{metric}", res[metric], note if metric == "n" else "")


def interaction(d, met):
    """pm25 ~ aod + met + aod*met on z-scores. Returns the three coefficients.

    Standard errors are not reported: rows of one station are not independent,
    so naive p-values would be too optimistic.
    """
    z = lambda s: (s - s.mean()) / s.std()
    a, m = z(d["aod"]), z(d[met])
    A = np.column_stack([np.ones(len(d)), a, m, a * m])
    beta = np.linalg.lstsq(A, d[TARGET].to_numpy(), rcond=None)[0]
    return dict(aod=beta[1], weather=beta[2], interaction=beta[3])


def within_station_slopes(d, met):
    """PM2.5-on-AOD slope in the low/middle/high third of `met`, with the thirds made INSIDE each station.

    Station means of AOD and PM2.5 are removed inside each third, so only day-to-day variation
    within a station counts. Stations with fewer than 3 * MIN_PAIRS rows are left out (thirds too thin).
    Returns (slopes, rows per third, number of stations used).
    """
    size = d.groupby("station_id")[met].transform("size")
    g = d[size >= 3 * MIN_PAIRS].copy()
    g["third"] = g.groupby("station_id")[met].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    slopes, counts = [], []
    for k in range(3):
        s = g[g["third"] == k]
        dx = s["aod"] - s.groupby("station_id")["aod"].transform("mean")
        dy = s[TARGET] - s.groupby("station_id")[TARGET].transform("mean")
        den = (dx ** 2).sum()
        slopes.append((dx * dy).sum() / den if den > 0 else np.nan)
        counts.append(len(s))
    return slopes, counts, g["station_id"].nunique()


def pattern(slopes):
    """Plain-words shape of three slopes (low, middle, high)."""
    a, b, c = slopes
    if np.isnan(slopes).any():
        return "cannot be described (too few rows)"
    if a < b < c:
        return "rises steadily"
    if a > b > c:
        return "falls steadily"
    if b < min(a, c):
        return "is lowest in the middle third"
    if b > max(a, c):
        return "is highest in the middle third"
    return "has no clear order"


def verdict(pooled, inside):
    """Does the pooled high-versus-low difference survive when the thirds are made inside stations?"""
    dp, di = pooled[2] - pooled[0], inside[2] - inside[0]
    if np.isnan(dp) or np.isnan(di):
        return ""
    if dp * di <= 0:
        return ("Inside stations the high-versus-low difference points the other way, so the pooled "
                "pattern mostly reflects differences between stations.")
    if abs(di) >= 0.5 * abs(dp):
        return "The pattern persists inside stations."
    return "Inside stations the pattern is weaker, so part of it reflects differences between stations."


def main():
    df = zero_aod_to_nan(load_dev())
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.month.map(SEASONS)
    df["wind_speed"] = np.hypot(df["wind_u"], df["wind_v"])  # derived for this description only
    d = df.dropna(subset=["aod", TARGET]).copy()
    rows, notes = [], ["# PM2.5 and AOD (auto-generated, check the numbers before pasting)\n"]
    FIGURES.mkdir(parents=True, exist_ok=True)

    # overall
    overall = link(d["aod"], d[TARGET])
    add_link(rows, "all", "overall", overall, f"{len(df) - len(d)} rows without usable AOD left out")
    notes.append(f"Overall (n = {overall['n']}): Pearson {overall['pearson']:.2f}, "
                 f"Spearman {overall['spearman']:.2f}, slope {overall['slope']:.1f} ug/m3 per AOD unit.\n")

    # by season
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    seas = {}
    ymax = d[TARGET].quantile(0.995)
    for ax, s in zip(axes.ravel(), ["DJF", "MAM", "JJA", "SON"]):
        g = d[d["season"] == s]
        seas[s] = link(g["aod"], g[TARGET])
        add_link(rows, s, "season", seas[s])
        ax.scatter(g["aod"], g[TARGET], s=3, alpha=0.15)
        if len(g) > 2 and g["aod"].nunique() > 1:
            c = np.polyfit(g["aod"], g[TARGET], 1)
            xs = np.linspace(g["aod"].min(), g["aod"].quantile(0.995), 50)
            ax.plot(xs, np.polyval(c, xs), "r-")
        ax.set_title(f"{s}  (n={len(g)}, rho={seas[s]['spearman']:.2f})")
        ax.set_ylim(0, ymax)
    fig.supxlabel("AOD")
    fig.supylabel("PM2.5 (ug/m3)")
    fig.tight_layout()
    fig.savefig(FIGURES / "m3_pm25_aod_by_season.png", dpi=150)
    plt.close(fig)
    best = max(seas, key=lambda s: np.nan_to_num(seas[s]["spearman"], nan=-9))
    worst = min(seas, key=lambda s: np.nan_to_num(seas[s]["spearman"], nan=9))
    notes.append("By season (Spearman): " + ", ".join(f"{s} {seas[s]['spearman']:.2f}" for s in seas) +
                 f". The link is strongest in {best} and weakest in {worst}.\n")

    # by station
    per = {}
    for st, g in d.groupby("station_id"):
        if len(g) >= MIN_PAIRS:
            per[st] = link(g["aod"], g[TARGET])
            add_link(rows, str(st), "station", per[st])
    st_df = pd.DataFrame(per).T.sort_values("spearman")
    n_skipped = d["station_id"].nunique() - len(st_df)
    add(rows, SEC, "stations_used", "station", "count", len(st_df), f"at least {MIN_PAIRS} PM2.5/AOD pairs")
    add(rows, SEC, "stations_skipped", "station", "count", n_skipped, f"fewer than {MIN_PAIRS} pairs")
    if len(st_df):
        q1, med, q3 = st_df["spearman"].quantile([.25, .5, .75])
        share_pos = (st_df["spearman"] > 0).mean()
        for k, v in (("median", med), ("q1", q1), ("q3", q3), ("share_positive", share_pos)):
            add(rows, SEC, "station_spearman", "station", k, v)
        notes.append(f"By station ({len(st_df)} stations with at least {MIN_PAIRS} pairs, {n_skipped} skipped): "
                     f"median Spearman {med:.2f} (IQR {q1:.2f} to {q3:.2f}); "
                     f"{share_pos:.0%} of the stations have a positive link.\n")
        fig, ax = plt.subplots(figsize=(7, max(4, 0.16 * len(st_df))))
        ax.barh(range(len(st_df)), st_df["spearman"], color=np.where(st_df["spearman"] > 0, "tab:blue", "tab:red"))
        ax.set_yticks(range(len(st_df)), st_df.index, fontsize=6)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("Spearman correlation of PM2.5 with AOD")
        fig.tight_layout()
        fig.savefig(FIGURES / "m3_pm25_aod_by_station.png", dpi=150)
        plt.close(fig)

    # weather: split each variable in three equal-sized groups and compare the PM2.5-AOD link.
    # Top row = all rows pooled. Bottom row = the same split made INSIDE each station, so that
    # differences between stations (altitude, local climate) cannot pass for a weather effect.
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharey=True)
    for j, met in enumerate(WEATHER):
        bins = pd.qcut(d[met], 3, labels=BIN_LABELS, duplicates="drop")
        pooled = []
        for lab in bins.cat.categories:
            g = d[bins == lab]
            res = link(g["aod"], g[TARGET])
            lo, hi = g[met].min(), g[met].max()
            add_link(rows, met, f"{lab} ({lo:.3g} to {hi:.3g})", res)
            pooled.append(res["slope"])
        co = interaction(d, met)
        for k, v in co.items():
            add(rows, SEC, met, "aod_x_weather_model", f"std_coef_{k}", v,
                "pm25 on z(aod), z(weather) and their product; descriptive, no p-values (station clustering)")
        inside, counts, n_st = within_station_slopes(d, met)
        for lab, sl, n in zip(BIN_LABELS, inside, counts):
            add(rows, SEC, met, f"{lab} (within station)", "aod_slope", sl,
                f"tertiles made inside each of {n_st} stations; station means removed")
            add(rows, SEC, met, f"{lab} (within station)", "aod_n", n)
        axes[0, j].bar(list(bins.cat.categories), pooled)
        axes[0, j].set_title(f"{met}\npooled, interaction coef {co['interaction']:+.2f}")
        axes[1, j].bar(BIN_LABELS, inside, color="tab:green")
        axes[1, j].set_title("same split inside each station")
        axes[1, j].set_xlabel("tertile of the weather variable")
        notes.append(f"{met}: pooled slope of PM2.5 on AOD in the low/middle/high tertile = "
                     + " / ".join(f"{x:.1f}" for x in pooled) + f", it {pattern(pooled)}. "
                     + "Inside stations: " + " / ".join(f"{x:.1f}" for x in inside)
                     + f", it {pattern(inside)}. {verdict(pooled, inside)} "
                     f"(Single AOD x {met} coefficient {co['interaction']:+.2f}; one linear number can hide a curved shape.)")
    axes[0, 0].set_ylabel("slope of PM2.5 on AOD (pooled)")
    axes[1, 0].set_ylabel("slope inside stations")
    fig.tight_layout()
    fig.savefig(FIGURES / "m3_pm25_aod_by_weather.png", dpi=150)
    plt.close(fig)

    notes.append("\nReading guide: these are associations on development data, not causal effects. "
                 "AOD is a column-integrated satellite value while PM2.5 is measured at the surface, "
                 "so boundary-layer conditions (temperature, pressure, wind) are expected to change the link.")
    path = save_parts(rows, "m3_stratified.csv")
    save_notes(notes, "m3_stratified_notes.md")
    print(f"{len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
