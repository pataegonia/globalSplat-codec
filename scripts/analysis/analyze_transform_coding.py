#!/usr/bin/env python3
"""Fit train-only channel KLTs and estimate held-out transform-coding rates."""
from __future__ import annotations
import argparse, math
import numpy as np
import torch
from codec_diagnostic_utils import common_args, extract, seed_all, setup, to_device, write_json

class Moments:
    def __init__(self,c): self.n=0; self.s=np.zeros(c); self.xx=np.zeros((c,c))
    def add(self,x): x=x.detach().cpu().double().reshape(-1,x.shape[-1]).numpy(); self.n+=len(x); self.s+=x.sum(0); self.xx+=x.T@x
    def finish(self):
        mean=self.s/self.n; cov=(self.xx-np.outer(self.s,self.s)/self.n)/max(1,self.n-1); return mean,cov

def transforms(cov,blocks):
    c=len(cov); result={"identity":np.eye(c)}
    for b in blocks:
        w=np.zeros((c,c))
        for st in range(0,c,b):
            en=min(st+b,c); _,v=np.linalg.eigh(cov[st:en,st:en]); w[st:en,st:en]=v[:,::-1]
        result[f"block_{b}"]=w
    _,v=np.linalg.eigh(cov); result["full"]=v[:,::-1]; return result

def quant_stats(x,bits):
    qmax=(1<<(bits-1))-1; scale=np.maximum(np.max(np.abs(x),axis=0)/qmax,np.finfo(np.float32).tiny)
    q=np.clip(np.rint(x/scale),-qmax,qmax).astype(np.int32); mse=float(np.mean((x-q*scale)**2)); return q,mse

def hist(q,bits):
    a=(1<<bits)-1; c=q.shape[1]; codes=(q+((1<<(bits-1))-1)).T+np.arange(c)[:,None]*a
    return np.bincount(codes.ravel(),minlength=c*a).reshape(c,a)

def rate(counts,pmf): return float(-np.sum(counts*np.log2(pmf)))
def empirical_rate(counts):
    counts=counts.astype(float); total=counts.sum(1,keepdims=True); nz=counts>0
    return float(np.sum(counts[nz]*(np.log2(np.broadcast_to(total,counts.shape)[nz])-np.log2(counts[nz]))))

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); common_args(p); p.add_argument("--calibration-scenes",type=int,default=200); p.add_argument("--test-scenes",type=int,default=200)
    p.add_argument("--blocks",type=int,nargs="+",default=[16,32,64]); a=p.parse_args(); seed_all(a.seed); model,train,test,dev=setup(a)
    mom={s:Moments(512) for s in ("texture","geometry")}
    for i,b in enumerate(train):
        if i>=a.calibration_scenes: break
        t,g=extract(model,to_device(b["inputs"],dev)); mom["texture"].add(t); mom["geometry"].add(g)
    fitted={}; report={"calibration_scenes":a.calibration_scenes,"test_scenes":a.test_scenes,"streams":{}}
    for s in mom:
        mean,cov=mom[s].finish(); eig=np.linalg.eigvalsh(cov)[::-1].clip(min=0); fitted[s]=(mean,cov,transforms(cov,a.blocks))
        report["streams"][s]={"mean":mean.tolist(),"covariance":cov.tolist(),"correlation":(cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov))).clip(min=1e-20)).tolist(),
          "eigenvalues":eig.tolist(),"cumulative_explained_variance":(np.cumsum(eig)/eig.sum()).tolist(),"modes":{}}
    cal_counts={s:{n:None for n in fitted[s][2]} for s in fitted}; test_counts={s:{n:None for n in fitted[s][2]} for s in fitted}; mses={s:{n:[] for n in fitted[s][2]} for s in fitted}
    for split,loader,limit,target in (("cal",train,a.calibration_scenes,cal_counts),("test",test,a.test_scenes,test_counts)):
        for i,b in enumerate(loader):
            if i>=limit: break
            t,g=extract(model,to_device(b["inputs"],dev))
            for s,z,bits in (("texture",t,7),("geometry",g,9)):
                x=z.cpu().numpy().reshape(-1,512); mean,_,ts=fitted[s]
                for n,w in ts.items():
                    q,mse=quant_stats((x-mean)@w,bits); h=hist(q,bits); target[s][n]=h if target[s][n] is None else target[s][n]+h
                    if split=="test": mses[s][n].append(mse)
    for s in fitted:
        for n in fitted[s][2]:
            c=cal_counts[s][n].astype(float); pmf=(c+.5)/(c.sum(1,keepdims=True)+.5*c.shape[1]); e=test_counts[s][n]
            cross=rate(e,pmf); empirical=empirical_rate(e)
            report["streams"][s]["modes"][n]={"heldout_cross_entropy_bits":cross,"heldout_bits_per_value":cross/e.sum(),
              "heldout_empirical_entropy_bits":empirical,"heldout_empirical_bits_per_value":empirical/e.sum(),
              "mean_transform_quantization_mse":float(np.mean(mses[s][n]))}
    write_json(a.output,report)
if __name__=="__main__": main()
