#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from domainguard.preprocessing import read_rgb, preprocess
from domainguard.features import build_frozen_backbone, extract_batches, image_to_tensor

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--backbone",default="resnet50",choices=["resnet50","efficientnet_b0","convnext_tiny","swin_tiny"]); ap.add_argument("--branch",default="raw_resize",choices=["raw_resize","fallback_disc_color_clahe"]); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--out",required=True); a=ap.parse_args()
    df=pd.read_csv(a.manifest); tensors=[]
    for p in df.path:
        rgb=preprocess(read_rgb(p),a.branch,224); tensors.append(image_to_tensor(rgb))
    model,device=build_frozen_backbone(a.backbone); z=extract_batches(model,tensors,device,a.batch_size)
    feat=pd.DataFrame(z,columns=[f"f{i:04d}" for i in range(z.shape[1])]); meta=[c for c in ["path","label","domain","group_id","image_id"] if c in df]; out=pd.concat([df[meta].reset_index(drop=True),feat],axis=1); Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False); print(f"saved {len(out)} rows x {z.shape[1]} features -> {a.out}")
if __name__=="__main__": main()
