"""Build the model-ready feature table.

    python -m src.build_features

Reads the cleaned flights + raw weather and writes data/processed/features.parquet.
"""
import time

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, TARGET
from src.data.airports import load_raw_flights
from src.features import (
    add_airport_state_features,
    add_rotation_features,
    add_route_features,
    add_time_features,
    add_weather_features,
    build_rotation_table,
)

# Columns that are only known after the flight (or are identifiers) - never model inputs.
NON_FEATURES = [
    "flight_date", "tail_number", "flight_number", "origin_state", "dest_state",
    "crs_dep_time", "crs_arr_time", "dep_delay", "arr_delay", TARGET,
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

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    print(f"features built: {df.shape} in {time.time() - t0:.0f}s")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def main():
    df = build()
    out = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"[save] {out}  ({len(feature_columns(df))} feature columns)")


if __name__ == "__main__":
    main()
