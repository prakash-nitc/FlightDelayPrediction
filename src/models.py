"""Model zoo: every candidate is a full sklearn Pipeline (encoding + estimator).

Each entry also says how many training rows it gets. Tree ensembles and linear
models scale to hundreds of thousands of rows; an RBF-kernel SVM is O(n^2) in
memory/time, so it is trained on a smaller stratified sample.
"""
from dataclasses import dataclass, field

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    train_rows: int | None = None  # None = use all available training rows
    params: dict = field(default_factory=dict)


def _onehot_scaled(nums, cats, min_frequency=200):
    """Linear / kernel models: standardise numerics, one-hot categoricals."""
    return ColumnTransformer([
        ("num", StandardScaler(), nums),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency), cats),
    ])


def _ordinal(nums, cats):
    """Tree models: categories as integer codes, numerics untouched."""
    return ColumnTransformer([
        ("num", "passthrough", nums),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), cats),
    ])


def build_models(nums: list[str], cats: list[str]) -> list[ModelSpec]:
    # HistGradientBoosting supports native categoricals only up to 255 levels;
    # `route` (~2k levels) is passed as an ordinal code instead.
    cat_mask = [False] * len(nums) + [c != "route" for c in cats]

    return [
        ModelSpec("Logistic Regression", Pipeline([
            ("prep", _onehot_scaled(nums, cats)),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
        ]), train_rows=400_000),

        ModelSpec("Decision Tree", Pipeline([
            ("prep", _ordinal(nums, cats)),
            ("clf", DecisionTreeClassifier(max_depth=12, min_samples_leaf=100, random_state=RANDOM_STATE)),
        ]), train_rows=400_000),

        ModelSpec("Random Forest", Pipeline([
            ("prep", _ordinal(nums, cats)),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=20,
                                           max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
        ]), train_rows=400_000),

        ModelSpec("SVM (RBF)", Pipeline([
            ("prep", _onehot_scaled(nums, cats, min_frequency=50)),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000, random_state=RANDOM_STATE)),
        ]), train_rows=30_000),

        ModelSpec("Gradient Boosting", Pipeline([
            ("prep", _ordinal(nums, cats)),
            ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                                   categorical_features=cat_mask,
                                                   random_state=RANDOM_STATE)),
        ]), train_rows=400_000),

        ModelSpec("XGBoost", Pipeline([
            ("prep", _ordinal(nums, cats)),
            ("clf", XGBClassifier(n_estimators=500, learning_rate=0.08, max_depth=8, subsample=0.8,
                                  colsample_bytree=0.7, tree_method="hist", n_jobs=-1,
                                  eval_metric="logloss", random_state=RANDOM_STATE)),
        ]), train_rows=400_000),
    ]
