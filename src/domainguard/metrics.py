from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def source_youden_threshold(y_true, y_score) -> float:
    """Select an operating threshold using SOURCE-DOMAIN data only."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    j = tpr[finite] - fpr[finite]
    return float(thresholds[finite][np.argmax(j)])


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(y_score, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)


def binary_metrics(y_true, y_score, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": expected_calibration_error(y_true, y_score),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }
