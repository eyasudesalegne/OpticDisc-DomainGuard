#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from domainguard.risk_gating import coverage_performance_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="CSV containing label and probability columns")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--score-col", default="probability")
    ap.add_argument("--out", default="results/metrics/risk_gating.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.predictions)
    rows = coverage_performance_curve(df[args.label_col], df[args.score_col])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
