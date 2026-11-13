"""The three figures for report sections 5 to 7.

Reads the CSV files generated in the shared parts directory and writes three
PNG files to the corresponding figures directory.

Run from the repo root:   python -m src.m4_figures
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

INPUT = config.PARTS
OUTPUT = config.PARTS.parent / "figures"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def seasonal_profile():
    daily = pd.read_csv(INPUT / "member4_daily_values.csv", parse_dates=["date"])
    daily = daily.dropna(subset=["pm25"])
    monthly = daily.groupby(daily["date"].dt.month)["pm25"]
    yearly = daily.groupby(daily["date"].dt.year)["pm25"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})

    median = monthly.median()
    left.bar(range(1, 13), median.reindex(range(1, 13)), color="#4c78a8")
    left.vlines(range(1, 13), monthly.quantile(.25).reindex(range(1, 13)),
                monthly.quantile(.75).reindex(range(1, 13)), color="black", lw=1.2)
    left.set_xticks(range(1, 13), MONTHS)
    left.set_ylabel("PM2.5 (µg/m³)")
    left.set_title("Monthly median (bar) and interquartile range (line)")

    y_median, y_count = yearly.median(), yearly.size()
    right.plot(y_median.index, y_median.values, marker="o", color="#e45756")
    for year in y_median.index:
        right.annotate(f"n={y_count[year]:,}", (year, y_median[year]), textcoords="offset points",
                       xytext=(0, 7), ha="center", fontsize=7)
    right.set_xlabel("Year")
    right.set_ylabel("Median PM2.5 (µg/m³)")
    right.set_title("Yearly median (n = station-days)")
    right.margins(y=0.2)

    fig.suptitle("PM2.5 seasonality and drift, all stations pooled (development data)", y=1.02)
    fig.tight_layout()
    save(fig, "m4_seasonal_profile.png")


def acf_figure():
    table = pd.read_csv(INPUT / "Exploratory_temporal_member4.csv")
    acf = table[table["metric"].str.startswith("acf_lag_")].copy()
    acf["lag"] = acf["metric"].str.replace("acf_lag_", "", regex=False).astype(int)
    segments = acf["group"] + " | " + acf["note"]
    n_segments = segments.nunique()
    typical_length = acf["note"].str.extract(r"observations=(\d+)")[0].astype(int).median()
    band = 1.96 / np.sqrt(typical_length)

    fig, ax = plt.subplots(figsize=(8, 4))
    for _, part in acf.groupby(segments):
        ax.plot(part["lag"], part["value"], color="grey", alpha=.5, lw=1, marker="o", ms=3)
    median = acf.groupby("lag")["value"].median()
    ax.plot(median.index, median.values, color="#e45756", lw=2.5, marker="o", label="median over segments")
    ax.axhline(0, color="black", lw=.8)
    ax.axhline(band, color="blue", ls="--", lw=1, label=f"approx. 95% band (n≈{int(typical_length)})")
    ax.axhline(-band, color="blue", ls="--", lw=1)
    ax.axvline(7, color="green", ls=":", lw=1.5, label="lag 7 (weekly cycle?)")
    ax.set_xticks(range(1, 15))
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"PM2.5 autocorrelation, only {n_segments} segments with ≥ 30 consecutive days")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "m4_acf_pm25.png")


def lead_lag_figure():
    summary = pd.read_csv(INPUT / "member4_lead_lag_summary.csv")
    table = summary[summary["lag_days"].between(0, 3)].pivot(
        index="variable", columns="lag_days", values="median_correlation")
    table = table.reindex(table[1].abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.2
    colors = {0: "#bbbbbb", 1: "#4c78a8", 2: "#f58518", 3: "#54a24b"}
    for i, lag in enumerate([0, 1, 2, 3]):
        ax.bar(np.arange(len(table)) + (i - 1.5) * width, table[lag], width,
               color=colors[lag], hatch="//" if lag == 0 else None, edgecolor="white",
               label="lag 0 (same day, not usable for next-day)" if lag == 0 else f"lag {lag}")
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(np.arange(len(table)), table.index, rotation=20)
    ax.set_ylabel("Median correlation with PM2.5")
    ax.set_title("Lead-lag: predictor observed k days before PM2.5 (median over station segments)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "m4_lead_lag.png")


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUTPUT / name)


if __name__ == "__main__":
    seasonal_profile()
    acf_figure()
    lead_lag_figure()
