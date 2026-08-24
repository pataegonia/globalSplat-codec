#!/usr/bin/env python3
"""Summarize group-variable quantization runs and identify joint Pareto points."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from globalsplat.compression import load_group_quantization_profile

RUN_LABELS = {
    "baseline", "conservative_geo", "conservative_tex", "conservative_joint",
    "moderate_geo", "moderate_tex", "moderate_joint", "aggressive_geo",
    "aggressive_tex", "aggressive_joint",
}

def infer_label(codec_path, input_root):
    relative = codec_path.relative_to(input_root)
    for part in relative.parts:
        if part in RUN_LABELS: return part
    return relative.parent.name

def dominates(a,b):
    weak=(a["bytes"]<=b["bytes"] and a["psnr"]>=b["psnr"] and a["ssim"]>=b["ssim"] and a["lpips"]<=b["lpips"])
    strict=(a["bytes"]<b["bytes"] or a["psnr"]>b["psnr"] or a["ssim"]>b["ssim"] or a["lpips"]<b["lpips"])
    return weak and strict

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,required=True); p.add_argument("--profile",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    rows=[]
    for codec_path in sorted(a.input_root.rglob("codec_summary.json")):
        score_path=codec_path.with_name("scores_all_avg.json")
        if not score_path.is_file(): continue
        c=json.loads(codec_path.read_text()); s=json.loads(score_path.read_text()); label=infer_label(codec_path,a.input_root)
        rows.append({"label":label,"source":str(codec_path),"policy":c.get("group_quantization_policy","baseline"),"ablation":c.get("group_quantization_ablation","none"),
          "bytes":c["mean_bytes"],"texture_payload_bytes":c["mean_texture_payload_bytes"],"geometry_payload_bytes":c["mean_geometry_payload_bytes"],
          "bits_per_value":c["mean_bits_per_value"],"psnr":s["psnr"],"ssim":s["ssim"],"lpips":s["lpips"],
          "texture_mse":c["mean_texture_mse"],"geometry_mse":c["mean_geometry_mse"],"encode_seconds":c.get("mean_total_encode_seconds",c["mean_rans_encode_seconds"]),"decode_seconds":c.get("mean_total_decode_seconds",c["mean_rans_decode_seconds"])})
    baseline=next((r for r in rows if r["label"]=="baseline"),None)
    if baseline is None: baseline=next((r for r in rows if r["policy"]=="baseline"),None)
    if baseline is None:
        discovered="\n".join(f"  {r['label']}: {r['source']} (policy={r['policy']})" for r in rows) or "  <none>"
        raise SystemExit(f"baseline result not found under {a.input_root}\ndiscovered complete runs:\n{discovered}\nCheck SLURM array task 0 and its log.")
    for r in rows:
        for key in ("bytes","texture_payload_bytes","geometry_payload_bytes","bits_per_value","psnr","ssim","lpips","texture_mse","geometry_mse","encode_seconds","decode_seconds"):
            r[f"delta_{key}"]=r[key]-baseline[key]
        r["rate_change_percent"]=100*(r["bytes"]/baseline["bytes"]-1)
    joint=[r for r in rows if r["label"]=="baseline" or r["ablation"]=="joint"]
    pareto=[r["label"] for r in joint if not any(dominates(q,r) for q in joint if q is not r)]
    profile=json.loads(a.profile.read_text())
    allocations={}
    for policy in profile["policies"]:
        loaded=load_group_quantization_profile(a.profile,policy=policy,ablation="joint")
        allocations[policy]={stream:{
          "tiers":value["tiers"],"group_multipliers":value["group_multipliers"].tolist(),
          "critical_groups":[i for i,tier in enumerate(value["tiers"]) if tier=="critical"],
          "high_groups":[i for i,tier in enumerate(value["tiers"]) if tier=="high"],
          "medium_groups":[i for i,tier in enumerate(value["tiers"]) if tier=="medium"],
          "low_groups":[i for i,tier in enumerate(value["tiers"]) if tier=="low"],
        } for stream,value in loaded["streams"].items()}
    report={"baseline":baseline,"runs":rows,"pareto_joint":pareto,"profile":profile,"expanded_allocations":allocations}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2))
    print("label\tbytes\trate%\tpsnr\tssim\tlpips\tpareto")
    for r in rows: print(f"{r['label']}\t{r['bytes']:.1f}\t{r['rate_change_percent']:+.3f}\t{r['psnr']:.5f}\t{r['ssim']:.6f}\t{r['lpips']:.6f}\t{r['label'] in pareto}")
if __name__=="__main__": main()
