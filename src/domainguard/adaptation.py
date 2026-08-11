from __future__ import annotations

import numpy as np


def _matrix_sqrt_psd(matrix: np.ndarray, inverse: bool = False, eps: float = 1e-8) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eps)
    power = -0.5 if inverse else 0.5
    return (vectors * (values ** power)) @ vectors.T


def coral_align_source_to_target(X_source, X_target, eps: float = 1e-6):
    """Unsupervised CORAL alignment using target covariates, never target labels.

    Returns a transformed source representation and the unchanged target representation.
    This helper is for explicitly labelled harmonization experiments and must not be
    silently substituted for the manuscript's fixed source-only transfer condition.
    """
    xs = np.asarray(X_source, dtype=float)
    xt = np.asarray(X_target, dtype=float)
    ms, mt = xs.mean(axis=0), xt.mean(axis=0)
    xs0, xt0 = xs - ms, xt - mt
    cs = np.cov(xs0, rowvar=False) + eps * np.eye(xs.shape[1])
    ct = np.cov(xt0, rowvar=False) + eps * np.eye(xt.shape[1])
    aligned = xs0 @ _matrix_sqrt_psd(cs, inverse=True) @ _matrix_sqrt_psd(ct) + mt
    return aligned, xt.copy()
