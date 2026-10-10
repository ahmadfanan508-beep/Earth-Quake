"""Task 5: run the whole pipeline end to end."""

import json
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .cleaning import clean, dedupe_latest
from .features import (
    add_location_features,
    add_quality_features,
    add_target,
    add_time_features,
    drop_leaky_columns,
    group_rare,
)
from .fetch import fetch_feed, geojson_to_df
from .transform import build_preprocessor, split_data


def load_stream_data() -> pd.DataFrame:
    """Load all JSONL stream files from the stream directory."""

    frames = []

    config.STREAM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in sorted(
        config.STREAM_DIR.glob("*.jsonl")
    ):
        try:
            frame = pd.read_json(
                path,
                lines=True,
            )

            if not frame.empty:
                frames.append(frame)

        except ValueError as exc:
            print(
                f"Warning: could not read {path}: {exc}"
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def load_base_data() -> pd.DataFrame:
    """Fetch the live feed, falling back to local GeoJSON."""

    try:
        payload = fetch_feed("all_week")

        print("Loaded earthquake data from USGS.")

        return geojson_to_df(payload)

    except Exception as exc:
        print(
            f"Live feed failed: {exc}"
        )

        if not config.FALLBACK_PATH.exists():
            raise FileNotFoundError(
                "USGS feed failed and fallback file does not exist: "
                f"{config.FALLBACK_PATH}"
            )

        print(
            f"Using fallback file: "
            f"{config.FALLBACK_PATH}"
        )

        with config.FALLBACK_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        return geojson_to_df(payload)


def merge_stream_data(
    base_df: pd.DataFrame,
    stream_df: pd.DataFrame,
):
    """Merge stream data and report changed event versions."""

    if stream_df.empty:
        return base_df, 0

    combined = pd.concat(
        [base_df, stream_df],
        ignore_index=True,
    )

    changed_events = 0

    if (
        "id" in base_df.columns
        and "updated" in base_df.columns
        and "id" in stream_df.columns
        and "updated" in stream_df.columns
    ):
        base_updates = (
            base_df[["id", "updated"]]
            .dropna()
            .groupby("id")["updated"]
            .max()
        )

        stream_updates = (
            stream_df[["id", "updated"]]
            .dropna()
            .groupby("id")["updated"]
            .max()
        )

        common_ids = (
            set(base_updates.index)
            & set(stream_updates.index)
        )

        for event_id in common_ids:
            if (
                stream_updates[event_id]
                > base_updates[event_id]
            ):
                changed_events += 1

    return dedupe_latest(combined), changed_events


def main() -> None:
    """Run the complete preprocessing pipeline."""

    config.PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Load base data
    # ---------------------------------------------------------

    raw_df = load_base_data()

    raw_rows = len(raw_df)

    # ---------------------------------------------------------
    # 2. Load stream data
    # ---------------------------------------------------------

    stream_df = load_stream_data()

    stream_rows = len(stream_df)

    combined_df, changed_events = merge_stream_data(
        raw_df,
        stream_df,
    )

    # ---------------------------------------------------------
    # 3. Clean
    # ---------------------------------------------------------

    cleaned_df = clean(combined_df)

    # ---------------------------------------------------------
    # 4. Feature engineering
    # ---------------------------------------------------------

    featured_df = add_time_features(cleaned_df)

    featured_df = add_quality_features(
        featured_df
    )

    featured_df = add_location_features(
        featured_df
    )

    featured_df["region"] = group_rare(
        featured_df["region"],
        top_k=15,
    )

    featured_df = add_target(
        featured_df
    )

    featured_df = drop_leaky_columns(
        featured_df
    )

    # ---------------------------------------------------------
    # 5. Train/test split
    # ---------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(featured_df)

    # ---------------------------------------------------------
    # 6. Build and fit preprocessor
    # ---------------------------------------------------------

    preprocessor = build_preprocessor(
        scaler="robust"
    )

    X_train_transformed = (
        preprocessor.fit_transform(X_train)
    )

    X_test_transformed = (
        preprocessor.transform(X_test)
    )

    # ---------------------------------------------------------
    # 7. Convert transformed data to DataFrames
    # ---------------------------------------------------------

    try:
        feature_names = (
            preprocessor.get_feature_names_out()
        )

    except AttributeError:
        feature_names = [
            f"feature_{i}"
            for i in range(
                X_train_transformed.shape[1]
            )
        ]

    train_output = pd.DataFrame(
        X_train_transformed,
        columns=feature_names,
        index=X_train.index,
    )

    test_output = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index,
    )

    train_output[config.TARGET] = (
        y_train.values
    )

    test_output[config.TARGET] = (
        y_test.values
    )

    # ---------------------------------------------------------
    # 8. Save processed files
    # ---------------------------------------------------------

    train_path = (
        config.PROCESSED_DIR
        / "train.csv"
    )

    test_path = (
        config.PROCESSED_DIR
        / "test.csv"
    )

    preprocessor_path = (
        config.PROCESSED_DIR
        / "preprocessor.joblib"
    )

    train_output.to_csv(
        train_path,
        index=False,
    )

    test_output.to_csv(
        test_path,
        index=False,
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    # ---------------------------------------------------------
    # 9. Report
    # ---------------------------------------------------------

    print("\n========== PIPELINE REPORT ==========")

    print(
        f"Raw rows: {raw_rows}"
    )

    print(
        f"Rows after cleaning/features: "
        f"{len(featured_df)}"
    )

    print(
        f"Stream rows merged: {stream_rows}"
    )

    print(
        f"Events whose updated value changed: "
        f"{changed_events}"
    )

    print(
        f"X_train shape: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"X_test shape: "
        f"{X_test_transformed.shape}"
    )

    print(
        f"Positive-class rate in train: "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Positive-class rate in test: "
        f"{y_test.mean():.4f}"
    )

    print(
        f"\nSaved train data to: {train_path}"
    )

    print(
        f"Saved test data to: {test_path}"
    )

    print(
        f"Saved preprocessor to: "
        f"{preprocessor_path}"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()