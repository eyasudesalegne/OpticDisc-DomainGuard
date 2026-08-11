from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def dataset_identity_audit(X, domains, n_splits: int = 5, seed: int = 42) -> dict[str, float]:
    """Quantify how strongly a representation retains dataset-origin information."""
    X = np.asarray(X)
    domains = np.asarray(domains)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, domains, cv=cv, method="predict")
    return {
        "domain_accuracy": float(accuracy_score(domains, pred)),
        "domain_balanced_accuracy": float(balanced_accuracy_score(domains, pred)),
        "n": int(len(domains)),
        "n_domains": int(len(np.unique(domains))),
    }
