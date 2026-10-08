"""Train and compare all candidate models.

    python -m src.train

For each model: fit on (a sample of) Jan-Apr, pick the F1-optimal threshold on
May, then report metrics on June - a month the model has never seen.
Writes reports/model_comparison_<prediction_point>.csv and comparison plots.
"""
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import PREDICTION_POINT, REPORTS_DIR
from src.dataset import load_splits, numeric_and_categorical
from src.evaluate import (
    best_threshold, compute_metrics, plot_model_comparison, plot_pr_curves, plot_roc_curves,
    predict_scores,
)
from src.models import build_models


def score(model, X, parallel: bool = False, chunks: int = 16) -> np.ndarray:
    """Kernel SVM scoring is slow; split it across cores."""
    if not parallel:
        return predict_scores(model, X)
    parts = np.array_split(np.arange(len(X)), chunks)
    out = Parallel(n_jobs=-1)(delayed(predict_scores)(model, X.iloc[p]) for p in parts)
    return np.concatenate(out)


def main(prediction_point: str = PREDICTION_POINT):
    train, valid, test, features = load_splits(prediction_point)
    nums, cats = numeric_and_categorical(features)
    print(f"prediction point: {prediction_point} | {len(features)} features | "
          f"train {len(train.y):,}  valid {len(valid.y):,}  test {len(test.y):,}")

    rows, test_scores = [], {}
    for spec in build_models(nums, cats):
        tr = train.sample(spec.train_rows)
        t0 = time.time()
        spec.pipeline.fit(tr.X, tr.y)
        fit_s = time.time() - t0

        is_svm = spec.name.startswith("SVM")
        thr = best_threshold(valid.y, score(spec.pipeline, valid.X, parallel=is_svm))
        s_test = score(spec.pipeline, test.X, parallel=is_svm)
        m = compute_metrics(test.y, s_test, thr)
        m.update(model=spec.name, train_rows=len(tr.y), fit_seconds=round(fit_s, 1))
        rows.append(m)
        test_scores[spec.name] = s_test
        print(f"{spec.name:<20} acc {m['accuracy']:.4f}  auc {m['roc_auc']:.4f}  "
              f"f1 {m['f1']:.4f}  f1w {m['f1_weighted']:.4f}  ({fit_s:.0f}s fit)")

    results = pd.DataFrame(rows).set_index("model")
    out = REPORTS_DIR / f"model_comparison_{prediction_point}.csv"
    results.round(4).to_csv(out)
    print(f"[save] {out}")

    plot_model_comparison(results, name=f"model_comparison_{prediction_point}")
    plot_roc_curves(test.y, test_scores, name=f"roc_curves_{prediction_point}")
    plot_pr_curves(test.y, test_scores, name=f"pr_curves_{prediction_point}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-point", choices=["departure", "pre_departure"], default=PREDICTION_POINT)
    main(parser.parse_args().prediction_point)
