from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def confidence_from_probability(probability):
    p = np.asarray(probability, dtype=float)
    return np.abs(p - 0.5) * 2.0


def coverage_performance_curve(y_true, y_score, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """Evaluate discrimination after rejecting the least-confident predictions."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    confidence = confidence_from_probability(p)
    order = np.argsort(-confidence)
    rows = []
    for coverage in coverages:
        k = max(1, int(np.ceil(len(y) * float(coverage))))
        idx = order[:k]
        if np.unique(y[idx]).size < 2:
            auroc = np.nan
        else:
            auroc = roc_auc_score(y[idx], p[idx])
        rows.append({
            "coverage": float(k / len(y)),
            "retained_n": int(k),
            "auroc": float(auroc) if np.isfinite(auroc) else np.nan,
        })
    return rows
