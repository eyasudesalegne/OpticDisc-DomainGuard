#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
META={"domain","label","group_id","path","image_id"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--target",required=True,choices=["ACRIMA","G1020"]); ap.add_argument("--shots-per-class",type=int,nargs="+",default=[5,10,20,50]); ap.add_argument("--repeats",type=int,default=20); ap.add_argument("--out",default="results/metrics/fewshot_adaptation.csv"); a=ap.parse_args()
    df=pd.read_csv(a.features); cols=[c for c in df if c not in META and pd.api.types.is_numeric_dtype(df[c])]; src=df[df.domain=="HYGD"]; tgt=df[df.domain==a.target].reset_index(drop=True); rng=np.random.default_rng(42); rows=[]
    for shots in a.shots_per_class:
      for rep in range(a.repeats):
        support=[]
        for label in [0,1]:
          pool=np.flatnonzero(tgt.label.to_numpy()==label)
          if len(pool)<=shots: continue
          support.extend(rng.choice(pool,size=shots,replace=False))
        support=np.asarray(support,dtype=int); eval_idx=np.setdiff1d(np.arange(len(tgt)),support)
        if len(support)!=2*shots or np.unique(tgt.iloc[eval_idx].label).size<2: continue
        train=pd.concat([src,tgt.iloc[support]],ignore_index=True); model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight="balanced")); model.fit(train[cols],train.label); p=model.predict_proba(tgt.iloc[eval_idx][cols])[:,1]
        rows.append({"target":a.target,"shots_per_class":shots,"repeat":rep,"eval_n":len(eval_idx),"auroc":roc_auc_score(tgt.iloc[eval_idx].label,p)})
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print(pd.DataFrame(rows).groupby(["target","shots_per_class"]).auroc.agg(["mean","std","count"]))
if __name__=="__main__": main()
