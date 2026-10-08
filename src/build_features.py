"""Build the model-ready feature table.

    python -m src.build_features

Reads the cleaned flights + raw weather and writes data/processed/features.parquet.
"""
import time

import pandas as pd

from src.config import PREDICTION_POINT, PROCESSED_DIR, RAW_DIR, TARGET
from src.data.airports import load_raw_flights
from src.features import (
    add_airport_state_features,
    add_rotation_features,
    add_route_features,
    add_time_features,
    add_weather_features,
    build_rotation_table,
)

# Identifiers and post-arrival information - never model inputs.
NON_FEATURES = [
    "flight_date", "tail_number", "flight_number", "origin_state", "dest_state",
    "crs_dep_time", "crs_arr_time", "arr_delay", TARGET,
]

CATEGORICAL = ["carrier", "origin", "dest", "route"]


def build() -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_parquet(PROCESSED_DIR / "flights_clean.parquet")
    weather = pd.read_parquet(RAW_DIR / "weather_hourly.parquet")

    rotation = build_rotation_table(load_raw_flights())
    df = add_time_features(df)
    df = add_route_features(df)
    df = add_rotation_features(df, rotation)
    df = add_airport_state_features(df)
    df = add_weather_features(df, weather)

    weather_cols = [c for c in df.columns if c.startswith(("origin_", "dest_")) and df[c].isna().any()]
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())

    df["departed_late"] = (df["dep_delay"] >= 15).astype("int8")

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    print(f"features built: {df.shape} in {time.time() - t0:.0f}s")
    return df


DEPARTURE_FEATURES = ["dep_delay", "departed_late"]


def feature_columns(df: pd.DataFrame, prediction_point: str = PREDICTION_POINT) -> list[str]:
    cols = [c for c in df.columns if c not in NON_FEATURES and c not in DEPARTURE_FEATURES]
    if prediction_point == "departure":
        cols += DEPARTURE_FEATURES
    elif prediction_point != "pre_departure":
        raise ValueError(f"unknown prediction point: {prediction_point}")
    return cols


def main():
    df = build()
    out = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"[save] {out}  ({len(feature_columns(df))} feature columns)")


if __name__ == "__main__":
    main()
