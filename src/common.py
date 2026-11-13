"""Helpers shared by the first-look, audit and clean scripts."""
import re
import sys

import pandas as pd

from src import config

CANDIDATES = {
    "date": ["date", "datetime", "timestamp", "time", "day"],
    "station_id": ["stationid", "station", "stationcode", "stationname", "siteid", "site", "id"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "long", "lng", "longitude"],
    "pm25": ["pm25", "pm25ugm3", "pm25concentration", "pm25value"],
}


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def to_snake(name):
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    return s.strip("_").lower()


def find_raw_file():
    if config.RAW_FILE:
        path = config.RAW / config.RAW_FILE
        if not path.exists():
            sys.exit(f"RAW_FILE {path} not found.")
        return path
    files = sorted(
        p for p in config.RAW.rglob("*.csv")
        if "__MACOSX" not in p.parts and not p.name.startswith(".")
    )
    if not files:
        sys.exit("No CSV found in data/raw/. Did you unzip the Kaggle download there?")
    if len(files) > 1:
        names = "\n  ".join(str(f.relative_to(config.ROOT)) for f in files)
        sys.exit(
            "More than one CSV in data/raw:\n  " + names +
            "\nIf PM2.5 and the satellite/weather variables are in separate files, "
            "send me the output of src.first_look so I can write the merge step "
            "with the real column names."
        )
    return files[0]


def load_raw():
    path = find_raw_file()
    return pd.read_csv(path, low_memory=False), path


def development_rows(frame, date_column):
    """Exclude rows whose dates cannot be confirmed to precede the holdout."""
    dates = parse_dates(frame[date_column])
    return frame.loc[dates.lt(pd.Timestamp(config.TEST_START))].copy()


def read_development_csv(path, date_column="date"):
    return development_rows(pd.read_csv(path, low_memory=False), date_column)


def detect_columns(columns, *, use_overrides=True):
    normed = {c: norm(c) for c in columns}
    found = {}
    for key, cands in CANDIDATES.items():
        override = config.COLUMN_OVERRIDES.get(key) if use_overrides else None
        if override:
            if override not in normed:
                sys.exit(f"COLUMN_OVERRIDES['{key}'] = '{override}' is not a column in the raw file.")
            found[key] = override
            continue
        match = next((c for cand in cands for c, n in normed.items() if n == cand), None)
        if key == "pm25" and match is None:
            bad = ("lag", "aqi", "categ", "roll", "mean", "pm10", "diff")
            loose = [c for c, n in normed.items() if "pm25" in n and not any(b in n for b in bad)]
            if len(loose) == 1:
                match = loose[0]
        found[key] = match
    return found


def require(found, keys):
    missing = [k for k in keys if not found.get(k)]
    if missing:
        sys.exit(
            f"Could not find a raw column for: {', '.join(missing)}.\n"
            "Run `python -m src.first_look`, then put the real column name in "
            "COLUMN_OVERRIDES in src/config.py."
        )


def parse_dates(s):
    d = pd.to_datetime(s, format=config.DATE_FORMAT, errors="coerce", utc=True)
    return d.dt.tz_localize(None)


def save_rows(rows, filename):
    config.ensure_dirs()
    out = pd.DataFrame(rows, columns=["section", "variable", "value"])
    out.insert(2, "group", "")
    out.insert(3, "metric", "value")
    out["note"] = ""
    path = config.PARTS / filename
    out.to_csv(path, index=False, encoding="utf-8")
    return path
