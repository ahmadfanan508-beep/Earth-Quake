from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"



# TODO (Task 4 and 5): fill these in after you have explored the data.

SEED = 42
TARGET = "big_quake"

NUMERIC: list[str] = [
    "lon",
    "lat",
    "depth_km",
    "nst",
    "dmin",
    "rms",
    "gap",
    "tsunami",
    "tz",
    "hour",
    "dayofweek",
    "update_lag_hours",
    "is_reviewed",
    "nst_missing",
    "abs_lat",
    "is_shallow",
]

NOMINAL: list[str] = [
    "region",
    "status",
    "magType",
    "net",
]

LEAKY: list[str] = [
    "title",
    "sig",
    "mmi",
    "cdi",
    "felt",
    "alert",
]

