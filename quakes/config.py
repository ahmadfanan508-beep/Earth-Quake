from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

FALLBACK_PATH = (
    ROOT / "data" / "fallback" / "all_week.geojson"
)

STREAM_DIR = ROOT / "data" / "stream"

PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42

TARGET = "big_quake"


# Numeric features used by the preprocessing pipeline.
NUMERIC = [
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


# Categorical features that will be one-hot encoded.
NOMINAL = [
    "region",
    "status",
    "magType",
    "net",
]


# Columns that contain information derived directly from
# magnitude or later information and therefore must not be
# used as predictive features.
LEAKY = [
    "title",
    "sig",
    "mmi",
    "cdi",
    "felt",
    "alert",
]