
"""Task 3: cleaning."""
import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert epoch milliseconds to UTC datetimes."""
    out = df.copy()

    for column in ("time", "updated"):
        if column in out.columns:
            out[column] = pd.to_datetime(
                out[column], unit="ms", utc=True, errors="coerce"
            )

    return out


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest updated row for each event ID."""
    if "id" not in df.columns or "updated" not in df.columns:
        return df.copy()

    out = df.copy()
    out = out.sort_values("updated")
    out = out.drop_duplicates(subset="id", keep="last")
    return out.reset_index(drop=True)


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose type is earthquake, ignoring case and spaces."""
    if "type" not in df.columns:
        return df.copy()

    event_type = df["type"].fillna("").astype(str).str.strip().str.lower()
    return df.loc[event_type == "earthquake"].copy()


def extract_region(place: pd.Series) -> pd.Series:
    """Extract text after the last comma; use Unknown for missing places."""
    def get_region(value):
        if pd.isna(value) or not str(value).strip():
            return "Unknown"

        value = str(value).strip()
        if "," in value:
            return value.rsplit(",", 1)[1].strip() or "Unknown"
        return value

    return place.apply(get_region)


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Mark values outside the IQR bounds."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return ((s < lower) | (s > upper)).fillna(False)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose magnitude is missing."""
    if "mag" not in df.columns:
        return df.copy()

    return df.dropna(subset=["mag"]).copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean records and add the region feature."""
    out = epoch_ms_to_datetime(df)
    out = dedupe_latest(out)
    out = keep_earthquakes(out)
    out = drop_missing_target(out)

    if "place" in out.columns:
        out["region"] = extract_region(out["place"])
    else:
        out["region"] = "Unknown"

    return out.reset_index(drop=True)
