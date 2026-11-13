"""Shared settings for the whole project. Edit GROUP_DIR first."""
from pathlib import Path

SEED = 42
GROUP_DIR = "19"  # <- replace with your real group ID, e.g. "B03-02"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
FINAL_TEST = ROOT / "data" / "final_test"
PARTS = ROOT / "results" / GROUP_DIR / "parts"

# Which CSV in data/raw to use. None = auto (works only if there is exactly one CSV).
RAW_FILE = "combined_gee_pm25.csv"


# If a script picks the wrong raw column, write the real name here,
# e.g. {"pm25": "PM2.5 (ug/m3)"}. Leave None to auto-detect.
COLUMN_OVERRIDES = {"date": "date", "station_id": "sensor_name",
                    "lat": "latitude", "lon": "longitude", "pm25": "pm25"}

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


# Final test period: proposed after the audit; the team confirms.
# Development data = rows BEFORE TEST_START. Rows AFTER TEST_END are locked away too
# (training on them would use data later than the test period).
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# Same sensor_name + same date but different GEE id/coordinates (audit: 2 groups):
#   "drop_all" removes every row of the group (no arbitrary pick)   "first" keeps the first row
SENSOR_DAY_DUP_RULE = "drop_all"

# Never model inputs (brief 5.4: station ID defines groups only; coordinates excluded)
TARGET = "pm25"
NON_FEATURE_COLS = ["date", "datetime_utc", "station_id", "gee_id", "lat", "lon"]


def ensure_dirs():
    for d in (RAW, INTERIM, PARTS):
        d.mkdir(parents=True, exist_ok=True)
    (PARTS / ".gitkeep").touch()


if __name__ == "__main__":
    ensure_dirs()
    print("Folders ready under", ROOT)
