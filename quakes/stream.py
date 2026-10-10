
"""Task 2: simulate streaming by polling a feed on a timer."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return unseen (id, updated) records and update the seen set."""
    if "id" not in df.columns or "updated" not in df.columns:
        raise KeyError("DataFrame must contain 'id' and 'updated' columns")

    unseen_indices = []

    for index, row in df.iterrows():
        key = (row["id"], row["updated"])
        if key not in seen:
            unseen_indices.append(index)
            seen.add(key)

    return df.loc[unseen_indices].copy()


def run_stream(
    feed: str,
    interval_s: int,
    duration_s: int,
    out_path: str,
) -> None:
    """Poll a feed repeatedly and append new events as JSON Lines."""
    if interval_s < 1:
        raise ValueError("interval_s must be at least 1")
    if duration_s < 0:
        raise ValueError("duration_s cannot be negative")

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    start = time.monotonic()

    while time.monotonic() - start < duration_s:
        try:
            payload = fetch_feed(feed)
            df = geojson_to_df(payload)
            new_rows = filter_unseen(df, seen)

            if not new_rows.empty:
                new_rows.to_json(
                    output,
                    orient="records",
                    lines=True,
                    date_format="iso",
                    mode="a",
                )

            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{stamp}] {len(new_rows)} new / {len(df)} fetched")

        except Exception as exc:
            # A failed poll should not stop future polling.
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{stamp}] Poll failed: {exc}")

        remaining = duration_s - (time.monotonic() - start)
        if remaining <= 0:
            break

        time.sleep(min(interval_s, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll a USGS earthquake feed and save new events."
    )
    parser.add_argument("--feed", default="all_hour")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--duration", type=int, default=2400)
    parser.add_argument(
        "--out", default="data/stream/stream.jsonl"
    )

    args = parser.parse_args()
    run_stream(args.feed, args.interval, args.duration, args.out)


if __name__ == "__main__":
    main()
