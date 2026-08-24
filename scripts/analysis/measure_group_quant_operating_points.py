#!/usr/bin/env python3
"""Measure calibration-only RD curves for every latent group and multiplier."""
from __future__ import annotations
import argparse, hashlib, statistics
import numpy as np
import torch
from codec_diagnostic_utils import common_args, extract, seed_all, setup, to_device, write_json
from globalsplat.compression import decode_scene_latents_from_bytes, encode_scene_latents_to_bytes, load_grouped_cdf_profile
from globalsplat.model.rendering import render_static_batched
from globalsplat.model.types import Gaussians

def quant_profile(stream, group, multiplier, channels=512, group_size=16):
    tables={name:np.ones(channels,dtype=np.float32) for name in ("texture","geometry")}
    tables[stream][group*group_size:(group+1)*group_size]=float(multiplier)
    digest=hashlib.sha256(stream.encode()+np.asarray([group,multiplier],dtype="<f4").tobytes()).hexdigest()
    return {"profile_id":digest,"policy":f"{stream}_g{group}_m{multiplier:g}","ablation":"joint",
      "group_size":group_size,"groups":channels//group_size,"streams":{
        name:{"channel_multipliers":table} for name,table in tables.items()}}

def gaussians(model,pair):
    v=model.gaussian_decoder(pair)
    return Gaussians(means=v[0],rotations=v[1],scales=v[2],sh=v[3],opacities=v[4],reg=v[5])

def mean(rows,key): return statistics.fmean(float(row[key]) for row in rows)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); common_args(p)
    p.add_argument("--cdf-profile",required=True)
    p.add_argument("--calibration-scenes",type=int,default=50)
    p.add_argument("--group-size",type=int,default=16)
    p.add_argument("--multipliers",type=float,nargs="+",default=[1.0,1.25,1.5,1.75,2.0,2.5,3.0])
    a=p.parse_args(); seed_all(a.seed)
    if any(m<1.0 for m in a.multipliers): raise SystemExit("all multipliers must be >= 1.0")
    model,train,_,dev=setup(a)
    cdf=load_grouped_cdf_profile(a.cdf_profile,texture_groups=32,geometry_groups=32,alpha=.5,precision=16)
    from src.evaluation.metrics import compute_psnr,compute_ssim,compute_lpips
    metrics={"psnr":compute_psnr,"ssim":compute_ssim,"lpips":compute_lpips}
    groups=512//a.group_size; keys=[(s,g,m) for s in ("texture","geometry") for g in range(groups) for m in a.multipliers if m!=1.0]
    collected={key:[] for key in keys}; baselines=[]
    encode_args=dict(texture_bits=7,geometry_bits=9,backend="compressai",scale_mode="channel",clip_percentile=None,format_version=2,fixed_cdf_profile=cdf,verify_symbols=False)
    for scene,batch in enumerate(train):
        if scene>=a.calibration_scenes: break
        batch=to_device(batch,dev); tex,geo=extract(model,batch["inputs"]); gt=batch["targets"]["images"][0].float().clamp(0,1)
        data,base_rate=encode_scene_latents_to_bytes(tex,geo,**encode_args)
        bt,bg,base_decode=decode_scene_latents_from_bytes(data,device=dev,fixed_cdf_profile=cdf,return_stats=True,verify_symbols=False)
        base_pred=render_static_batched(gaussians(model,(bt,bg)),batch["targets"],render_depth=False)["img"].view_as(gt).clamp(0,1)
        base_score={name:float(fn(gt,base_pred).mean()) for name,fn in metrics.items()}
        base={**base_score,"bytes":base_rate["bytes"],"payload_bytes":base_rate["payload_bytes"],
          "texture_payload_bytes":base_rate["texture_payload_bytes"],"geometry_payload_bytes":base_rate["geometry_payload_bytes"],
          "texture_mse":float((tex-bt).square().mean()),"geometry_mse":float((geo-bg).square().mean()),
          "encode_seconds":base_rate["total_encode_seconds"],"decode_seconds":base_decode["total_decode_seconds"]}
        baselines.append(base)
        for stream,group,multiplier in keys:
            qp=quant_profile(stream,group,multiplier,group_size=a.group_size)
            data,rate=encode_scene_latents_to_bytes(tex,geo,group_quantization_profile=qp,**encode_args)
            th,gh,dec=decode_scene_latents_from_bytes(data,device=dev,fixed_cdf_profile=cdf,return_stats=True,verify_symbols=False)
            pred=render_static_batched(gaussians(model,(th,gh)),batch["targets"],render_depth=False)["img"].view_as(gt).clamp(0,1)
            score={name:float(fn(gt,pred).mean()) for name,fn in metrics.items()}
            collected[(stream,group,multiplier)].append({
              "payload_delta_bytes":rate["payload_bytes"]-base["payload_bytes"],
              "texture_payload_delta_bytes":rate["texture_payload_bytes"]-base["texture_payload_bytes"],
              "geometry_payload_delta_bytes":rate["geometry_payload_bytes"]-base["geometry_payload_bytes"],
              "psnr_drop":base["psnr"]-score["psnr"],"ssim_drop":base["ssim"]-score["ssim"],"lpips_increase":score["lpips"]-base["lpips"],
              "texture_mse":float((tex-th).square().mean()),"geometry_mse":float((geo-gh).square().mean()),
              "encode_seconds":rate["total_encode_seconds"],"decode_seconds":dec["total_decode_seconds"]})
        print(f"calibration scene {scene+1}/{a.calibration_scenes}",flush=True)
    if not baselines: raise RuntimeError("calibration loader produced no scenes")
    baseline={key:mean(baselines,key) for key in baselines[0]}
    rows=[]
    for stream in ("texture","geometry"):
      for group in range(groups):
        rows.append({"stream":stream,"group":group,"channel_start":group*a.group_size,"channel_end":(group+1)*a.group_size,"multiplier":1.0,
          "payload_delta_bytes":0.0,"texture_payload_delta_bytes":0.0,"geometry_payload_delta_bytes":0.0,
          "psnr_drop":0.0,"ssim_drop":0.0,"lpips_increase":0.0,
          "texture_mse":baseline["texture_mse"],"geometry_mse":baseline["geometry_mse"],
          "encode_seconds":baseline["encode_seconds"],"decode_seconds":baseline["decode_seconds"]})
        for multiplier in a.multipliers:
          if multiplier==1.0: continue
          values=collected[(stream,group,multiplier)]
          rows.append({"stream":stream,"group":group,"channel_start":group*a.group_size,"channel_end":(group+1)*a.group_size,"multiplier":multiplier,
            **{key:mean(values,key) for key in values[0]}})
    write_json(a.output,{"calibration_split":"train","calibration_scenes":len(baselines),"group_size":a.group_size,
      "groups_per_stream":groups,"multipliers":a.multipliers,"baseline":baseline,"operating_points":rows,
      "rate_note":"payload deltas are actual fixed-G32/T32 rANS bytes; container metadata is excluded from additive allocation",
      "distortion_note":"all quality deltas are rendered against each scene's G9/T7 baseline"})
if __name__=="__main__": main()
