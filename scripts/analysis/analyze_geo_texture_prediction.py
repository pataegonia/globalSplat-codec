#!/usr/bin/env python3
"""Fit train-only linear geometry-to-texture predictors and test residual rates."""
from __future__ import annotations
import argparse
import numpy as np
import torch
from codec_diagnostic_utils import common_args, extract, quantize_torch, seed_all, setup, to_device, write_json
from globalsplat.compression.arithmetic_codec import _compressai_cdf, _rans_encode

def qsymbols(x,bits):
    q=(1<<(bits-1))-1; scale=np.maximum(np.max(np.abs(x),axis=0)/q,np.finfo(np.float32).tiny); return np.clip(np.rint(x/scale),-q,q).astype(np.int64)+q
def entropy(s,a):
    c=np.bincount(s.ravel(),minlength=a); p=c[c>0]/c.sum(); return float(-(p*np.log2(p)).sum())
def payload(s,a):
    flat=s.ravel(); return len(_rans_encode(flat,_compressai_cdf(flat,a)))

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); common_args(p); p.add_argument("--calibration-scenes",type=int,default=200); p.add_argument("--test-scenes",type=int,default=200)
    p.add_argument("--token-subsample",type=int,default=4); p.add_argument("--ridge",type=float,nargs="+",default=[0.0,1e-3,1e-2,1e-1]); a=p.parse_args(); seed_all(a.seed)
    model,train,test,dev=setup(a); n=0; sx=np.zeros(512); sy=np.zeros(512); xx=np.zeros((512,512)); xy=np.zeros((512,512))
    for i,b in enumerate(train):
        if i>=a.calibration_scenes: break
        t,g=extract(model,to_device(b["inputs"],dev)); x=g[0,::a.token_subsample].cpu().numpy(); y=t[0,::a.token_subsample].cpu().numpy()
        n+=len(x); sx+=x.sum(0); sy+=y.sum(0); xx+=x.T@x; xy+=x.T@y
    mx=sx/n; my=sy/n; cxx=xx-n*np.outer(mx,mx); cxy=xy-n*np.outer(mx,my); scale=np.trace(cxx)/512
    weights={str(r):(np.linalg.lstsq(cxx,cxy,rcond=1e-8)[0] if r==0 else np.linalg.solve(cxx+r*scale*np.eye(512),cxy)) for r in a.ridge}; stats={k:{"geometry_variance":[],"geometry_quantization_mse":[],"raw_var":[],"residual_var_original_geo":[],"residual_var_quantized_geo":[],"raw_entropy":[],"residual_entropy_original_geo":[],"residual_entropy_quantized_geo":[],"raw_payload_bytes":[],"residual_payload_bytes_original_geo":[],"residual_payload_bytes_quantized_geo":[]} for k in weights}
    for i,b in enumerate(test):
        if i>=a.test_scenes: break
        t,g=extract(model,to_device(b["inputs"],dev)); y=t.cpu().numpy().reshape(-1,512); x=g.cpu().numpy().reshape(-1,512); xq=quantize_torch(g,9).cpu().numpy().reshape(-1,512)
        raw=qsymbols(y,7); raw_h=entropy(raw,127); raw_p=payload(raw,127)
        for k,w in weights.items():
            ro=y-((x-mx)@w+my); rq=y-((xq-mx)@w+my); so=qsymbols(ro,7); sq=qsymbols(rq,7); d=stats[k]
            d["geometry_variance"].append(float(np.var(x))); d["geometry_quantization_mse"].append(float(np.mean((x-xq)**2))); d["raw_var"].append(float(np.var(y))); d["residual_var_original_geo"].append(float(np.var(ro))); d["residual_var_quantized_geo"].append(float(np.var(rq))); d["raw_entropy"].append(raw_h)
            d["residual_entropy_original_geo"].append(entropy(so,127)); d["residual_entropy_quantized_geo"].append(entropy(sq,127)); d["raw_payload_bytes"].append(raw_p)
            d["residual_payload_bytes_original_geo"].append(payload(so,127)); d["residual_payload_bytes_quantized_geo"].append(payload(sq,127))
    summary={k:{m:float(np.mean(v)) for m,v in d.items()} for k,d in stats.items()}
    for d in summary.values():
        d["variance_reduction_quantized_geo_percent"]=100*(1-d["residual_var_quantized_geo"]/d["raw_var"]); d["actual_payload_reduction_quantized_geo_percent"]=100*(1-d["residual_payload_bytes_quantized_geo"]/d["raw_payload_bytes"])
    write_json(a.output,{"calibration_scenes":a.calibration_scenes,"test_scenes":a.test_scenes,"token_subsample":a.token_subsample,"predictors":summary})
if __name__=="__main__": main()
