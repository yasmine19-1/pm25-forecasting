"""Shared settings. EDA settings, kept separate so it never overwrites the team config."""
if __package__:
    from . import config
else:
    import config

SEED = config.SEED
DATA_PATH = config.PROCESSED / "clean_dev.parquet"
PARTS = config.PARTS
FOLDS = config.FOLDS
FIGURES = config.FIGURES

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
