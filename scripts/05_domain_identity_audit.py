#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from domainguard.domain_audit import dataset_identity_audit

META = {"domain", "label", "group_id", "path", "image_id"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="results/metrics/domain_identity_audit.json")
    args = ap.parse_args()
    df = pd.read_csv(args.features)
    cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    report = dataset_identity_audit(df[cols].to_numpy(), df.domain.to_numpy())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
