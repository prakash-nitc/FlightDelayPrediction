"""Load the feature table and produce time-based train / validation / test splits."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.build_features import CATEGORICAL, feature_columns
from src.config import (
    PREDICTION_POINT, PROCESSED_DIR, RANDOM_STATE, TARGET, TEST_MONTHS, TRAIN_MONTHS, VALID_MONTHS,
)


@dataclass
class Split:
    X: pd.DataFrame
    y: pd.Series

    def sample(self, n: int | None, random_state: int = RANDOM_STATE) -> "Split":
        """Stratified random subsample (keeps the class ratio)."""
        if n is None or n >= len(self.y):
            return self
        frac = n / len(self.y)
        idx = (self.y.groupby(self.y, group_keys=False)
                     .apply(lambda s: s.sample(frac=frac, random_state=random_state)).index)
        return Split(self.X.loc[idx], self.y.loc[idx])


def load_splits(prediction_point: str = PREDICTION_POINT) -> tuple[Split, Split, Split, list[str]]:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    features = feature_columns(df, prediction_point)

    # float64 -> float32 halves memory; plenty of precision for these features.
    f64 = df[features].select_dtypes("float64").columns
    df[f64] = df[f64].astype(np.float32)

    def make(months):
        part = df[df["month"].isin(months)]
        return Split(part[features], part[TARGET].astype(int))

    return make(TRAIN_MONTHS), make(VALID_MONTHS), make(TEST_MONTHS), features


def numeric_and_categorical(features: list[str]) -> tuple[list[str], list[str]]:
    cats = [c for c in features if c in CATEGORICAL]
    nums = [c for c in features if c not in CATEGORICAL]
    return nums, cats
