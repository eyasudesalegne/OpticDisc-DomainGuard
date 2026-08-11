from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


def percentile_bootstrap_ci(y_true, y_score, metric=roc_auc_score, groups=None, n_boot=1000, seed=42):
    """Percentile bootstrap CI; optionally resample grouped evaluation units."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    values = []
    if groups is None:
        units = np.arange(len(y))
        for _ in range(n_boot):
            idx = rng.choice(units, size=len(units), replace=True)
            if np.unique(y[idx]).size < 2:
                continue
            values.append(metric(y[idx], s[idx]))
    else:
        g = np.asarray(groups)
        units = np.unique(g)
        for _ in range(n_boot):
            sampled = rng.choice(units, size=len(units), replace=True)
            idx = np.concatenate([np.flatnonzero(g == u) for u in sampled])
            if np.unique(y[idx]).size < 2:
                continue
            values.append(metric(y[idx], s[idx]))
    if not values:
        return np.nan, np.nan
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))


def paired_bootstrap_auc_difference(y_true, score_a, score_b, groups=None, n_boot=1000, seed=42):
    y, a, b = map(np.asarray, (y_true, score_a, score_b))
    rng = np.random.default_rng(seed)
    diffs = []
    units = np.arange(len(y)) if groups is None else np.unique(np.asarray(groups))
    g = None if groups is None else np.asarray(groups)
    for _ in range(n_boot):
        sampled = rng.choice(units, size=len(units), replace=True)
        idx = sampled if g is None else np.concatenate([np.flatnonzero(g == u) for u in sampled])
        if np.unique(y[idx]).size < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    estimate = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = 2 * min(np.mean(np.asarray(diffs) <= 0), np.mean(np.asarray(diffs) >= 0))
    return {"difference": estimate, "ci_low": float(lo), "ci_high": float(hi), "p": float(min(1, p))}


def mcnemar_exact(y_true, pred_a, pred_b):
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = a == y, b == y
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(min(n01, n10), n=n, p=0.5).pvalue
    return {"a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "p": float(p)}


def holm_bonferroni(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted
