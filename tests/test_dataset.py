import numpy as np
import pandas as pd
import pytest

from src.build_features import DEPARTURE_FEATURES, NON_FEATURES, feature_columns
from src.dataset import Split, numeric_and_categorical

MODEL_INPUTS = ["carrier", "route", "distance", "prev_arr_delay"]


def _feature_table():
    return pd.DataFrame(columns=NON_FEATURES + DEPARTURE_FEATURES + MODEL_INPUTS)


def test_departure_point_includes_departure_delay():
    cols = feature_columns(_feature_table(), "departure")
    assert set(DEPARTURE_FEATURES) <= set(cols)
    assert not set(NON_FEATURES) & set(cols)


def test_pre_departure_point_drops_departure_delay():
    cols = feature_columns(_feature_table(), "pre_departure")
    assert cols == MODEL_INPUTS  # no identifiers, target or departure delay


def test_unknown_prediction_point_raises():
    with pytest.raises(ValueError):
        feature_columns(_feature_table(), "after_landing")


def test_numeric_and_categorical_split():
    nums, cats = numeric_and_categorical(["distance", "carrier", "route", "prev_arr_delay"])
    assert nums == ["distance", "prev_arr_delay"]
    assert cats == ["carrier", "route"]


def test_split_sample_keeps_class_ratio_and_is_reproducible():
    y = pd.Series([1] * 300 + [0] * 700)
    split = Split(pd.DataFrame({"x": np.arange(len(y))}), y)
    a, b = split.sample(100), split.sample(100)
    assert len(a.y) == 100
    assert a.y.mean() == pytest.approx(0.3)
    assert a.X.index.equals(a.y.index)
    assert a.X.index.equals(b.X.index)  # same seed -> same rows


def test_split_sample_none_returns_everything():
    split = Split(pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1]))
    assert split.sample(None) is split
    assert split.sample(10) is split
