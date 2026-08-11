#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--risk",help="risk-gating CSV"); ap.add_argument("--fewshot",help="few-shot CSV"); ap.add_argument("--outdir",default="results/figures"); a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    if a.risk:
        df=pd.read_csv(a.risk); plt.figure(figsize=(6,4)); plt.plot(df.coverage,df.auroc,marker="o"); plt.xlabel("Accepted coverage"); plt.ylabel("Retained-case AUROC"); plt.title("Domain-risk gating"); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/"risk_gating.png",dpi=300); plt.close()
    if a.fewshot:
        df=pd.read_csv(a.fewshot); summary=df.groupby(["target","shots_per_class"],as_index=False).auroc.mean(); plt.figure(figsize=(6,4));
        for target,g in summary.groupby("target"): plt.plot(g.shots_per_class,g.auroc,marker="o",label=target)
        plt.xlabel("Labelled target support per class"); plt.ylabel("Target AUROC"); plt.title("Few-shot target adaptation"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/"fewshot_adaptation.png",dpi=300); plt.close()
if __name__=="__main__": main()
