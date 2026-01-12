import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvalResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    threshold_used: float
    predicted_positives: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str,
) -> EvalResult:
    y_pred = (y_proba >= threshold).astype(int)
    return EvalResult(
        model=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        threshold_used=float(threshold),
        predicted_positives=int(y_pred.sum()),
    )


def save_plots(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    fig_dir: str,
    prefix: str,
) -> None:
    _ensure_dir(fig_dir)

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {prefix}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Not Fraud", "Fraud"])
    plt.yticks([0, 1], ["Not Fraud", "Fraud"])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, val, ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_confusion_matrix.png"), dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {prefix}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_roc_curve.png"), dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {prefix}")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_pr_curve.png"), dpi=200)
    plt.close()


def write_metrics_csv(rows: list[EvalResult], output_path: str) -> None:
    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(output_path, index=False)
