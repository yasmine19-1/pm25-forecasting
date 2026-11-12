"""Generate and verify the shared development folds."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from src import config

SOURCE = config.INTERIM / "clean_dev.parquet"
RETAINED = config.PARTS / "retained_stations.csv"
RESULTS = config.PARTS.parent
KEYS = ["station_id", "date"]
N_SPLITS = 5
GAP_DAYS = 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def row_ids(frame):
    dates = pd.to_datetime(frame["date"])
    stations = frame["station_id"]
    require(len(frame) > 0 and dates.notna().all() and stations.notna().all()
            and stations.astype(str).str.strip().ne("").all(),
            "Development row keys must be present and nonblank.")
    require(dates.eq(dates.dt.normalize()).all(), "Expected normalized calendar dates.")
    require(dates.lt(pd.Timestamp(config.TEST_START)).all(), "Found a date in or after the holdout.")
    keys = list(zip(stations.astype(str), dates.dt.strftime("%Y-%m-%d")))
    require(len(set(keys)) == len(keys), "Development station/date keys must be unique.")
    return [[station, day] for station, day in keys]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_folds(path, frame):
    """Return train/validation positions after checking the data and frame row order."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(payload["schema_version"] == 1, "Unsupported fold schema.")
    require(payload["dataset"]["sha256"] == sha256(SOURCE),
            "Fold file was generated from a different development parquet.")
    require(payload["row_ids"] == row_ids(frame),
            "Frame station/date keys or order differ from the fold file.")
    verify(payload["scheme"], payload["folds"], frame)
    return [(np.asarray(fold["train"]), np.asarray(fold["val"])) for fold in payload["folds"]]


def verify(scheme, folds, frame):
    require(scheme in {"temporal", "spatial", "standard"}, "Unknown fold scheme.")
    require(len(folds) == N_SPLITS, "Expected five folds.")
    dates = pd.to_datetime(frame["date"])
    stations = frame["station_id"]
    require(dates.lt(pd.Timestamp(config.TEST_START)).all(), "Holdout dates are forbidden.")
    validation_counts = np.zeros(len(frame), dtype=int)
    previous_validation_end = None
    for fold in folds:
        train, val = np.asarray(fold["train"]), np.asarray(fold["val"])
        for positions in (train, val):
            require(positions.ndim == 1 and positions.size > 0 and positions.dtype.kind in "iu"
                    and positions.min() >= 0 and positions.max() < len(frame)
                    and np.unique(positions).size == positions.size,
                    f"{scheme}: invalid, empty, or repeated row positions.")
        require(np.intersect1d(train, val).size == 0, f"{scheme}: train/validation overlap.")
        validation_counts[val] += 1
        if scheme == "temporal":
            first = pd.Timestamp(fold["validation_calendar_start"])
            last = pd.Timestamp(fold["validation_calendar_end"])
            gap = pd.Timedelta(days=GAP_DAYS)
            require(dates.iloc[val].min() - dates.iloc[train].max() > gap,
                    "Temporal: observed dates lack a full intervening calendar day.")
            require(np.array_equal(train, np.flatnonzero(dates.lt(first - gap)))
                    and np.array_equal(val, np.flatnonzero(dates.between(first, last))),
                    "Temporal: split does not contain all rows on its designated dates.")
            require(previous_validation_end is None or first > previous_validation_end,
                    "Temporal: validation blocks are out of order or overlap.")
            previous_validation_end = last
        else:
            require(len(train) + len(val) == len(frame), f"{scheme}: incomplete partition.")
            if scheme == "spatial":
                require(set(stations.iloc[train]).isdisjoint(stations.iloc[val]),
                        "Spatial: a station appears on both sides.")
    require((validation_counts <= 1).all(), f"{scheme}: repeated validation rows across folds.")
    if scheme != "temporal":
        require((validation_counts == 1).all(), f"{scheme}: validation coverage is incomplete.")


def generate():
    frame = pd.read_parquet(SOURCE, columns=KEYS)
    keys = row_ids(frame)
    retained = pd.read_csv(RETAINED, usecols=["station_id", "retain", "n_obs"], keep_default_na=False)
    require(retained["station_id"].is_unique and retained["retain"].isin([True, False]).all(),
            "Retained-station list needs unique IDs and boolean retain flags.")
    expected = retained.loc[retained["retain"]].set_index("station_id")["n_obs"].sort_index()
    actual = frame.groupby("station_id").size().sort_index()
    require(expected.to_dict() == actual.to_dict(),
            "Development station IDs or counts differ from the retained-station decision.")

    dates = pd.to_datetime(frame["date"])
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    temporal = []
    for train_days, val_days in TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP_DAYS).split(calendar):
        temporal.append({
            "train": np.flatnonzero(dates.isin(calendar[train_days])).tolist(),
            "val": np.flatnonzero(dates.isin(calendar[val_days])).tolist(),
            "validation_calendar_start": calendar[val_days[0]].date().isoformat(),
            "validation_calendar_end": calendar[val_days[-1]].date().isoformat(),
        })

    rows = np.arange(len(frame))
    spatial = GroupKFold(n_splits=N_SPLITS).split(rows, groups=frame["station_id"])
    standard = KFold(n_splits=N_SPLITS, shuffle=True, random_state=config.SEED).split(rows)
    splits = {
        "temporal": temporal,
        "spatial": [{"train": tr.tolist(), "val": va.tolist()} for tr, va in spatial],
        "standard": [{"train": tr.tolist(), "val": va.tolist()} for tr, va in standard],
    }
    settings = {
        "temporal": {"splitter": "TimeSeriesSplit", "calendar_gap_days": GAP_DAYS},
        "spatial": {"splitter": "GroupKFold", "group_column": "station_id"},
        "standard": {"splitter": "KFold", "shuffle": True, "seed": config.SEED, "diagnostic_only": True},
    }
    dataset = {
        "path": SOURCE.relative_to(config.ROOT).as_posix(),
        "sha256": sha256(SOURCE),
        "rows": len(frame),
        "stations": len(actual),
        "first_date": dates.min().date().isoformat(),
        "last_date": dates.max().date().isoformat(),
        "test_start": config.TEST_START,
        "retained_stations_path": RETAINED.relative_to(config.ROOT).as_posix(),
        "retained_stations_sha256": sha256(RETAINED),
        "row_id_columns": KEYS,
        "row_order": "zero-based positions in the original development parquet",
    }
    payloads = {}
    for scheme, folds in splits.items():
        verify(scheme, folds, frame)
        payloads[scheme] = {
            "schema_version": 1,
            "scheme": scheme,
            "settings": {**settings[scheme], "n_splits": N_SPLITS, "scikit_learn": sklearn.__version__},
            "dataset": dataset,
            "row_ids": keys,
            "folds": folds,
        }
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Compare saved files with regenerated splits.")
    args = parser.parse_args()
    for scheme, payload in generate().items():
        path = RESULTS / f"folds_{scheme}.json"
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if args.verify:
            require(path.read_bytes() == encoded, f"{path.name} differs from the regenerated file.")
        else:
            temporary = path.with_suffix(".json.tmp")
            temporary.write_bytes(encoded)
            temporary.replace(path)
        print(f"PASS {path.name}: five folds, " + ("byte-identical regeneration" if args.verify else "split checks passed"))
        for number, fold in enumerate(payload["folds"], 1):
            print(f"  {number}: {len(fold['train']):,} train / {len(fold['val']):,} validation")


if __name__ == "__main__":
    main()
