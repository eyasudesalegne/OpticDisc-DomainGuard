#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from domainguard.metrics import binary_metrics, source_youden_threshold
from domainguard.statistics import percentile_bootstrap_ci

META={"domain","label","group_id","path","image_id"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--folds",type=int,default=10); ap.add_argument("--out",default="results/metrics/internal_hygd.json"); a=ap.parse_args()
    df=pd.read_csv(a.features); df=df[df.domain=="HYGD"].reset_index(drop=True)
    cols=[c for c in df if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    X,y,g=df[cols].to_numpy(),df.label.to_numpy(),df.group_id.to_numpy()
    oof=np.full(len(df),np.nan); cv=StratifiedGroupKFold(n_splits=a.folds,shuffle=True,random_state=42)
    for tr,te in cv.split(X,y,g):
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight="balanced")); model.fit(X[tr],y[tr]); oof[te]=model.predict_proba(X[te])[:,1]
    threshold=source_youden_threshold(y,oof); report=binary_metrics(y,oof,threshold); lo,hi=percentile_bootstrap_ci(y,oof,groups=g,n_boot=300); report["auroc_ci95"]=[lo,hi]; report["cv"]="10-fold StratifiedGroupKFold" if a.folds==10 else f"{a.folds}-fold StratifiedGroupKFold"
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); pd.DataFrame({"label":y,"probability":oof,"group_id":g}).to_csv(out.with_suffix(".predictions.csv"),index=False); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
