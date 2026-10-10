"""Task 2: simulate streaming by polling a feed on a timer."""

import argparse
from datetime import datetime
from pathlib import Path
import time

import pandas as pd

from .fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return only rows whose (id, updated) pair is not in `seen`.

    New pairs are added to `seen` in place.
    """
    new_rows = []

    for _, row in df.iterrows():
        pair = (row["id"], row["updated"])

        if pair not in seen:
            seen.add(pair)
            new_rows.append(row)

    if not new_rows:
        return df.iloc[0:0].copy()

    return pd.DataFrame(new_rows).reset_index(drop=True)


def run_stream(
    feed: str,
    interval_s: int,
    duration_s: int,
    out_path: str,
) -> None:
    """Poll a feed and append unseen events to a JSONL file."""

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    start_time = time.monotonic()

    while time.monotonic() - start_time < duration_s:
        try:
            payload = fetch_feed(feed)
            df = geojson_to_df(payload)

            fetched_count = len(df)

            new_df = filter_unseen(df, seen)
            new_count = len(new_df)

            if new_count > 0:
                with out_file.open("a", encoding="utf-8") as file:
                    new_df.to_json(
                        file,
                        orient="records",
                        lines=True,
                    )

            timestamp = datetime.now().strftime("%H:%M:%S")

            print(
                f"[{timestamp}] "
                f"{new_count} new / {fetched_count} fetched"
            )

        except Exception as exc:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] poll failed: {exc}")

        elapsed = time.monotonic() - start_time
        remaining = duration_s - elapsed

        if remaining <= 0:
            break

        time.sleep(min(interval_s, remaining))


def main() -> None:
    """Parse command-line arguments and start the stream."""

    parser = argparse.ArgumentParser(
        description="Poll the USGS earthquake feed."
    )

    parser.add_argument(
        "--feed",
        default="all_hour",
        help="USGS feed name.",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between polls.",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=2400,
        help="Total streaming duration in seconds.",
    )

    parser.add_argument(
        "--out",
        default="data/stream/stream.jsonl",
        help="Output JSONL file.",
    )

    args = parser.parse_args()

    run_stream(
        feed=args.feed,
        interval_s=args.interval,
        duration_s=args.duration,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()