
"""Task 1: fetch the USGS GeoJSON feed and convert it to a DataFrame."""
import requests
import pandas as pd

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """Fetch a USGS GeoJSON feed."""
    url = f"{BASE}/{feed}.geojson"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            f"USGS response for feed '{feed}' was not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("USGS response must be a JSON object.")

    if "features" not in payload:
        raise ValueError(
            f"USGS response for feed '{feed}' has no 'features' field."
        )

    return payload


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """Convert GeoJSON features into one row per event."""
    rows = []

    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []

        row = {
            "id": feature.get("id"),
            **properties,
            "lon": coordinates[0] if len(coordinates) > 0 else None,
            "lat": coordinates[1] if len(coordinates) > 1 else None,
            "depth_km": coordinates[2] if len(coordinates) > 2 else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)
