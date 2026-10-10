
"""Task 4: feature engineering."""
import pandas as pd

from . import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add UTC hour and day of week."""
    out = df.copy()

    if "time" in out.columns:
        timestamps = pd.to_datetime(out["time"], utc=True, errors="coerce")
        out["hour"] = timestamps.dt.hour
        out["dayofweek"] = timestamps.dt.dayofweek

    return out


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add update delay and data-quality indicators."""
    out = df.copy()

    if "time" in out.columns and "updated" in out.columns:
        event_time = pd.to_datetime(out["time"], utc=True, errors="coerce")
        update_time = pd.to_datetime(out["updated"], utc=True, errors="coerce")
        out["update_lag_hours"] = (
            update_time - event_time
        ).dt.total_seconds() / 3600

    if "status" in out.columns:
        out["is_reviewed"] = (
            out["status"].fillna("").astype(str).str.strip().str.lower()
            == "reviewed"
        ).astype(int)
    else:
        out["is_reviewed"] = 0

    if "nst" in out.columns:
        out["nst_missing"] = out["nst"].isna().astype(int)
    else:
        out["nst_missing"] = 1

    return out


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute latitude and shallow-event indicator."""
    out = df.copy()

    if "lat" in out.columns:
        out["abs_lat"] = pd.to_numeric(out["lat"], errors="coerce").abs()

    if "depth_km" in out.columns:
        depth = pd.to_numeric(out["depth_km"], errors="coerce")
        out["is_shallow"] = (depth < 70).astype(int)

    return out


def group_rare(s: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep the top-k frequent categories and group the rest as Other."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    counts = s.value_counts(dropna=False)
    common_values = counts.head(top_k).index
    return s.where(s.isin(common_values), "Other")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label events with magnitude >= 4.5 as big quakes."""
    out = df.copy()

    if "mag" not in out.columns:
        raise KeyError("Required target source column 'mag' is missing")

    magnitude = pd.to_numeric(out["mag"], errors="coerce")
    out[config.TARGET] = (magnitude >= 4.5).astype(int)

    return out


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns designated as leakage risks in config."""
    columns_to_drop = [
        column for column in config.LEAKY if column in df.columns
    ]
    return df.drop(columns=columns_to_drop).copy()
