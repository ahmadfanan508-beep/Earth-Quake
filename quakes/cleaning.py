"""Task 3: cleaning."""

import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert time and updated from epoch milliseconds to UTC datetimes."""

    df = df.copy()

    for column in ["time", "updated"]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                unit="ms",
                utc=True,
                errors="coerce",
            )

    return df


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per id using the greatest updated value."""

    df = df.copy()

    if df.empty:
        return df

    return (
        df.sort_values("updated")
        .drop_duplicates("id", keep="last")
        .reset_index(drop=True)
    )


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where type is earthquake."""

    df = df.copy()

    if "type" not in df.columns:
        return df.iloc[0:0].copy()

    df["type"] = (
        df["type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return df[df["type"] == "earthquake"].reset_index(drop=True)


def extract_region(place: pd.Series) -> pd.Series:
    """Extract the text after the last comma.

    Missing places become 'Unknown'.
    """

    return (
        place.fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.split(",")
        .str[-1]
        .str.strip()
        .replace("", "Unknown")
    )


def iqr_outlier_mask(
    s: pd.Series,
    k: float = 1.5,
) -> pd.Series:
    """Return True for values outside the IQR bounds."""

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return (s < lower) | (s > upper)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where magnitude is missing."""

    return df.dropna(subset=["mag"]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete cleaning pipeline."""

    df = df.copy()

    df = epoch_ms_to_datetime(df)

    df = dedupe_latest(df)

    df = keep_earthquakes(df)

    df = drop_missing_target(df)

    if "place" in df.columns:
        df["region"] = extract_region(df["place"])
    else:
        df["region"] = "Unknown"

    return df.reset_index(drop=True)