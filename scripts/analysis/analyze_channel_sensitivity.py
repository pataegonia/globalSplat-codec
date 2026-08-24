#!/usr/bin/env python3
"""Rank latent channel groups by downstream rendering sensitivity."""
from __future__ import annotations
import argparse, statistics
import torch
from codec_diagnostic_utils import common_args, extract, quantize_torch, seed_all, setup, to_device, write_json
from globalsplat.model.rendering import render_static_batched
from globalsplat.model.types import Gaussians

def decode(model, pair):
    v=model.gaussian_decoder(pair)
    return Gaussians(means=v[0],rotations=v[1],scales=v[2],sh=v[3],opacities=v[4],reg=v[5])

@torch.no_grad()
def evaluate(model, loader, device, scenes, group_size, factor, metrics):
    rows={s:[] for s in ("texture","geometry")}; baseline=[]
    for i,batch in enumerate(loader):
        if i>=scenes: break
        batch=to_device(batch,device); tex,geo=extract(model,batch["inputs"])
        base=(quantize_torch(tex,7),quantize_torch(geo,9)); gt=batch["targets"]["images"][0].float().clamp(0,1)
        def score(pair):
            pred=render_static_batched(decode(model,pair),batch["targets"],render_depth=False)["img"].view_as(gt).clamp(0,1)
            return {n:float(fn(gt,pred).mean()) for n,fn in metrics.items()}
        bs=score(base); baseline.append(bs)
        for stream,x,bits in (("texture",tex,7),("geometry",geo,9)):
            for start in range(0,x.shape[-1],group_size):
                end=min(start+group_size,x.shape[-1]); changed=base[0 if stream=="texture" else 1].clone()
                changed[...,start:end]=quantize_torch(x[...,start:end],bits,factor)
                pair=(changed,base[1]) if stream=="texture" else (base[0],changed)
                sc=score(pair); mse=float((x-changed).square().mean())
                rows[stream].append({"scene":i,"start":start,"end":end,"psnr_drop":bs["psnr"]-sc["psnr"],
                    "ssim_drop":bs["ssim"]-sc["ssim"],"lpips_increase":sc["lpips"]-bs["lpips"],"latent_mse":mse})
    return baseline,rows

def summarize(rows):
    out=[]
    for key in sorted({(r["start"],r["end"]) for r in rows}):
        part=[r for r in rows if (r["start"],r["end"])==key]
        row={"start":key[0],"end":key[1],"scenes":len(part)}
        for m in ("psnr_drop","ssim_drop","lpips_increase","latent_mse"): row[f"mean_{m}"]=statistics.fmean(r[m] for r in part)
        out.append(row)
    # Keep the ranking interpretable: PSNR impact is the primary key while SSIM,
    # LPIPS, and latent MSE remain independent reported diagnostics.
    out.sort(key=lambda r:r["mean_psnr_drop"],reverse=True)
    for rank,row in enumerate(out,1): row["rank"]=rank
    return out

def main():
    p=argparse.ArgumentParser(); common_args(p); p.add_argument("--calibration-scenes",type=int,default=20); p.add_argument("--test-scenes",type=int,default=20)
    p.add_argument("--group-size",type=int,default=16); p.add_argument("--coarsen-factor",type=float,default=2.0); a=p.parse_args(); seed_all(a.seed)
    model,train,test,dev=setup(a)
    from src.evaluation.metrics import compute_psnr,compute_ssim,compute_lpips
    metrics={"psnr":compute_psnr,"ssim":compute_ssim,"lpips":compute_lpips}
    cb,cr=evaluate(model,train,dev,a.calibration_scenes,a.group_size,a.coarsen_factor,metrics); tb,tr=evaluate(model,test,dev,a.test_scenes,a.group_size,a.coarsen_factor,metrics)
    report={"method":"channel_group_coarsening","group_size":a.group_size,"coarsen_factor":a.coarsen_factor,
      "baseline_bits":{"texture":7,"geometry":9},"calibration_scenes":len(cb),"test_scenes":len(tb),
      "calibration_ranked":{s:summarize(cr[s]) for s in cr},"heldout_ranked":{s:summarize(tr[s]) for s in tr}}
    write_json(a.output,report)
if __name__=="__main__": main()
