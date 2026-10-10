"""Task 1: fetch the USGS GeoJSON feed and turn it into a DataFrame."""

import pandas as pd
import requests


BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """GET {BASE}/{feed}.geojson and return the parsed dict.

    Use a timeout and raise_for_status(). A misspelled feed name returns
    HTTP 200 with a plain-text body, so raise a clear error if the body
    is not valid JSON.
    """
    url = f"{BASE}/{feed}.geojson"

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            f"USGS feed '{feed}' did not return valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"USGS feed '{feed}' returned an unexpected response."
        )

    return payload


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """Convert USGS GeoJSON into one DataFrame row per earthquake.

    Columns:
        id
        every key inside properties
        lon
        lat
        depth_km
    """
    features = payload.get("features", [])

    rows = []

    for feature in features:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []

        row = {
            "id": feature.get("id"),
        }

        # Add every field from the properties object.
        row.update(properties)

        # GeoJSON earthquake coordinates are:
        # [longitude, latitude, depth_km]
        row["lon"] = coordinates[0] if len(coordinates) > 0 else None
        row["lat"] = coordinates[1] if len(coordinates) > 1 else None
        row["depth_km"] = coordinates[2] if len(coordinates) > 2 else None

        rows.append(row)

    return pd.DataFrame(rows)