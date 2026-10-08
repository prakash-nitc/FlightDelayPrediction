"""Evaluation metrics and plots shared by training and tuning."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from src import plot_style as ps
from src.config import FIGURES_DIR

ps.apply()


def predict_scores(model, X) -> np.ndarray:
    """Probability of delay if available, otherwise a monotonic decision score (SVM)."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:  # SVC without probability=True
            pass
    return model.decision_function(X)


def best_threshold(y_true, scores) -> float:
    """Threshold that maximises F1 for the delayed class (chosen on validation data)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def compute_metrics(y_true, scores, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "threshold": threshold,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_roc_curves(y_true, scores_by_model: dict, name="roc_curves"):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, (model, s) in enumerate(scores_by_model.items()):
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, color=ps.CATEGORICAL[i], lw=2,
                label=f"{model}  (AUC {roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], color=ps.TEXT_MUTED, lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — test set (June 2023)")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def plot_pr_curves(y_true, scores_by_model: dict, name="pr_curves"):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, (model, s) in enumerate(scores_by_model.items()):
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, color=ps.CATEGORICAL[i], lw=2,
                label=f"{model}  (AP {average_precision_score(y_true, s):.3f})")
    ax.axhline(np.mean(y_true), color=ps.TEXT_MUTED, lw=1, ls="--")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves — test set")
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, name="model_comparison"):
    metrics = ["accuracy", "roc_auc", "f1_macro", "f1"]
    labels = ["Accuracy", "ROC-AUC", "F1 (macro)", "F1 (delayed class)"]
    res = results.sort_values("roc_auc")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, m, label in zip(axes, metrics, labels):
        ax.barh(res.index, res[m], color=ps.BLUE, height=0.6)
        for i, v in enumerate(res[m]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9, color=ps.TEXT_MUTED)
        lo = max(0.0, res[m].min() - 0.1)
        ax.set_xlim(lo, 1.0)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Model comparison on the June 2023 test set", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, title, name="confusion_matrix"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm_norm, cmap=ps.SEQUENTIAL, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n{cm_norm[i, j]:.1%}", ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else ps.TEXT, fontsize=10)
    ax.set_xticks([0, 1], ["On time", "Delayed"])
    ax.set_yticks([0, 1], ["On time", "Delayed"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    ax.set_title(title, fontsize=11)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def plot_feature_importance(importance: pd.Series, title, name="feature_importance", top=20):
    imp = importance.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.3 * top + 1))
    ax.barh(imp.index, imp.values, color=ps.BLUE, height=0.65)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=11)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def plot_threshold_tradeoff(y_true, scores, chosen, name="threshold_tradeoff"):
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = [compute_metrics(y_true, scores, t) for t in thresholds]
    df = pd.DataFrame(rows, index=thresholds)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for col, label, color in [("precision", "Precision", ps.BLUE), ("recall", "Recall", ps.ORANGE),
                              ("f1", "F1", ps.AQUA), ("accuracy", "Accuracy", ps.VIOLET)]:
        ax.plot(df.index, df[col], label=label, color=color)
    ax.axvline(chosen, color=ps.TEXT_MUTED, lw=1, ls="--")
    ax.text(chosen, 0.02, f" chosen = {chosen:.2f}", color=ps.TEXT_MUTED)
    ax.set_xlabel("Decision threshold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title("Threshold trade-off (tuned XGBoost, test set)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)
