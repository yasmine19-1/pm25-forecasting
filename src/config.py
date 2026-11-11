"""Shared settings for the whole project. Edit GROUP_ID first."""
from pathlib import Path

SEED = 42
GROUP_ID = "19"  # <- replace with your real group ID, e.g. "B03-02"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
FINAL_TEST = ROOT / "data" / "final_test"
PARTS = ROOT / "results" / GROUP_ID / "parts"

# Which CSV in data/raw to use. None = auto (works only if there is exactly one CSV).
RAW_FILE = None

# If a script picks the wrong raw column, write the real name here,
# e.g. {"pm25": "PM2.5 (ug/m3)"}. Leave None to auto-detect.
COLUMN_OVERRIDES = {"date": None, "station_id": None, "lat": None, "lon": None, "pm25": None}

# Date parsing. Leave None unless dates parse badly (e.g. "%d/%m/%Y").
DATE_FORMAT = None

# Cleaning rules from the project brief (section 5.1)
PM25_MIN = 0    # remove PM2.5 <= 0
PM25_MAX = 500  # remove PM2.5 >= 500

# Rows with the same date + lat + lon + station_id but DIFFERENT values:
#   "first"    keep the first row
#   "mean"     average the numeric columns, keep the first value of the others
#   "drop_all" remove the whole group
KEY_DUP_RULE = "first"


def ensure_dirs():
    for d in (RAW, INTERIM, FINAL_TEST, PARTS):
        d.mkdir(parents=True, exist_ok=True)
    (PARTS / ".gitkeep").touch()


if __name__ == "__main__":
    ensure_dirs()
    print("Folders ready under", ROOT)
