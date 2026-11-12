"""Shared settings. Member 2 settings, kept separate so it never overwrites the team config."""
from pathlib import Path

SEED = 42
GROUP = "19"  # group folder used in the team repo (results/19)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "clean_dev.parquet"
RESULTS = ROOT / "results" / GROUP
PARTS = RESULTS / "parts"
FIGURES = RESULTS / "figures"

TARGET = "pm25"
S5P_COLS = ["no2", "co", "o3"]  # Sentinel-5P, the only columns with gaps
NUMERIC_COLS = [
    "temperature_celsius", "pressure_mb", "wind_u", "wind_v",
    "no2", "co", "o3", "aod", "pm25",
]
# Never predictive features (ground rule 2)
ID_COLS = ["station_id", "gee_id", "lat", "lon", "date", "datetime_utc"]

# WHO 2021 daily guideline is 15; brief asks for 25 and 50 thresholds
PM25_THRESHOLDS = [25, 50]
