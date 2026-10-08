"""Hyperparameter tuning for the best model family (XGBoost) and final evaluation.

    python -m src.tune

1. RandomizedSearchCV with a *predefined* split: fit on a Jan-Apr sample,
   score ROC-AUC on a May sample. A time-ordered split avoids the optimistic
   bias of shuffled K-fold on temporally correlated flight data.
2. Refit the best configuration on all of Jan-Apr with early stopping on May.
3. Pick the decision threshold on May, then report once on June (test).
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PREDICTION_POINT, RANDOM_STATE, REPORTS_DIR
from src.dataset import load_splits
from src.evaluate import (
    best_threshold, compute_metrics, plot_confusion_matrix, plot_feature_importance,
    plot_threshold_tradeoff,
)

SEARCH_SPACE = {
    "max_depth": randint(4, 12),
    "learning_rate": loguniform(0.02, 0.3),
    "n_estimators": randint(200, 900),
    "min_child_weight": randint(1, 40),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.4, 0.6),
    "gamma": uniform(0, 5),
    "reg_lambda": loguniform(0.1, 20),
    "reg_alpha": loguniform(1e-3, 5),
    "max_cat_to_onehot": [1, 8],
}


def base_xgb(**params) -> XGBClassifier:
    return XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=-1,
                         eval_metric="auc", random_state=RANDOM_STATE, **params)


def search(train, valid, n_iter: int, train_rows: int, valid_rows: int):
    tr, va = train.sample(train_rows), valid.sample(valid_rows)
    X = pd.concat([tr.X, va.X])
    y = pd.concat([tr.y, va.y])
    fold = np.r_[np.full(len(tr.y), -1), np.zeros(len(va.y))]  # -1 = always train

    rs = RandomizedSearchCV(
        base_xgb(), SEARCH_SPACE, n_iter=n_iter, scoring="roc_auc",
        cv=PredefinedSplit(fold), refit=False, random_state=RANDOM_STATE, verbose=1, n_jobs=1,
    )
    t0 = time.time()
    rs.fit(X, y)
    print(f"search done in {(time.time() - t0) / 60:.1f} min - best val AUC {rs.best_score_:.4f}")

    cv = pd.DataFrame(rs.cv_results_)
    cols = [c for c in cv.columns if c.startswith("param_")] + ["mean_test_score", "mean_fit_time"]
    return rs.best_params_, cv[cols].sort_values("mean_test_score", ascending=False)


def main(prediction_point: str, n_iter: int):
    train, valid, test, features = load_splits(prediction_point)

    best, history = search(train, valid, n_iter=n_iter, train_rows=250_000, valid_rows=120_000)
    history.round(5).to_csv(REPORTS_DIR / f"tuning_results_{prediction_point}.csv", index=False)
    print("best params:", best)

    # Refit on all training months; let early stopping choose the number of trees.
    final_params = {**best, "n_estimators": 3000, "early_stopping_rounds": 100}
    model = base_xgb(**final_params)
    t0 = time.time()
    model.fit(train.X, train.y, eval_set=[(valid.X, valid.y)], verbose=False)
    print(f"final fit {time.time() - t0:.0f}s, best iteration {model.best_iteration}")

    p_valid = model.predict_proba(valid.X)[:, 1]
    p_test = model.predict_proba(test.X)[:, 1]
    thr = best_threshold(valid.y, p_valid)
    metrics = {
        "prediction_point": prediction_point,
        "model": "XGBoost (tuned)",
        "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
        "best_iteration": int(model.best_iteration),
        "train_rows": int(len(train.y)),
        "test_rows": int(len(test.y)),
        "validation": compute_metrics(valid.y, p_valid, thr),
        "test": compute_metrics(test.y, p_test, thr),
        "test_at_0.5": compute_metrics(test.y, p_test, 0.5),
    }
    with open(REPORTS_DIR / f"final_metrics_{prediction_point}.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(json.dumps(metrics["test"], indent=2, default=float))

    suffix = prediction_point
    plot_confusion_matrix(test.y, (p_test >= thr).astype(int),
                          f"Tuned XGBoost — June 2023 (threshold {thr:.2f})", name=f"confusion_matrix_{suffix}")
    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(features).fillna(0)
    plot_feature_importance(gain / gain.sum(), "Tuned XGBoost — feature importance (share of gain)",
                            name=f"feature_importance_{suffix}")
    plot_threshold_tradeoff(test.y, p_test, thr, name=f"threshold_tradeoff_{suffix}")

    joblib.dump({"model": model, "features": features, "threshold": thr},
                MODELS_DIR / f"xgboost_tuned_{suffix}.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-point", choices=["departure", "pre_departure"], default=PREDICTION_POINT)
    parser.add_argument("--n-iter", type=int, default=30)
    args = parser.parse_args()
    main(args.prediction_point, args.n_iter)
