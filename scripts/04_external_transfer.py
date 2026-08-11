#!/usr/bin/env python
"""Strict source-trained external transfer on precomputed feature tables.

Expected feature CSV columns: domain,label,group_id,<numeric feature columns...>
The scaler, classifier, and operating threshold are fitted/selected on HYGD only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from domainguard.metrics import binary_metrics, source_youden_threshold

META = {"domain", "label", "group_id", "path", "image_id"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--source", default="HYGD")
    ap.add_argument("--targets", nargs="+", default=["ACRIMA", "G1020"])
    ap.add_argument("--out", default="results/metrics/fixed_external_transfer.json")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    feature_cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    source = df[df.domain == args.source].copy()
    if source.empty or not feature_cols:
        raise ValueError("Source rows or numeric feature columns are missing.")

    Xs, ys = source[feature_cols].to_numpy(), source.label.to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    model.fit(Xs, ys)
    source_prob = model.predict_proba(Xs)[:, 1]
    threshold = source_youden_threshold(ys, source_prob)

    report = {"source": args.source, "source_selected_threshold": threshold, "targets": {}}
    for target_name in args.targets:
        target = df[df.domain == target_name].copy()
        if target.empty:
            continue
        # Critical: transform/predict only. No target labels enter fitting or threshold selection.
        prob = model.predict_proba(target[feature_cols].to_numpy())[:, 1]
        report["targets"][target_name] = binary_metrics(target.label.to_numpy(), prob, threshold)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
