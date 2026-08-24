#!/usr/bin/env python3
"""Solve calibration-derived discrete group quantization RD allocations."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

STREAMS=("geometry","texture")

def distortion(row,w):
    # Negative measured deltas are calibration noise, not a quality credit.
    return (w[0]*max(0.0,float(row["psnr_drop"]))
            +w[1]*max(0.0,float(row["ssim_drop"]))
            +w[2]*max(0.0,float(row["lpips_increase"])))

def solve(groups,target_bytes,bin_bytes,weights):
    target_bin=int(math.ceil(target_bytes/bin_bytes)); states={0:(0.0,())}
    for options in groups:
        nxt={}
        for saved_bin,(cost,choices) in states.items():
            for row in options:
                saving=max(0.0,-float(row["payload_delta_bytes"]))
                new_bin=min(target_bin,saved_bin+int(round(saving/bin_bytes)))
                candidate=(cost+distortion(row,weights),choices+(row,))
                if new_bin not in nxt or candidate[0]<nxt[new_bin][0]: nxt[new_bin]=candidate
        # Remove states whose cost is no better than a state with more savings.
        best=float("inf"); pruned={}
        for saved_bin in sorted(nxt,reverse=True):
            if nxt[saved_bin][0]<best:
                pruned[saved_bin]=nxt[saved_bin]; best=nxt[saved_bin][0]
        states=pruned
    reached=target_bin in states
    selected=states[target_bin][1] if reached else states[max(states)][1]
    return reached,selected

def main():
    p=argparse.ArgumentParser(); p.add_argument("--operating-points",required=True,type=Path); p.add_argument("--output-profile",required=True,type=Path)
    p.add_argument("--targets",type=float,nargs="+",default=[5,7.5,10,12.5,15]); p.add_argument("--rate-bin-bytes",type=float,default=256)
    p.add_argument("--psnr-weight",type=float,default=1.0); p.add_argument("--ssim-weight",type=float,default=10.0); p.add_argument("--lpips-weight",type=float,default=1.0); a=p.parse_args()
    data=json.loads(a.operating_points.read_text()); rows=data["operating_points"]; group_count=int(data["groups_per_stream"])
    if any(float(r["multiplier"])<1.0 for r in rows): raise SystemExit("operating points contain multiplier below 1.0")
    groups=[]
    for stream in STREAMS:
      for group in range(group_count):
        options=sorted([r for r in rows if r["stream"]==stream and int(r["group"])==group],key=lambda r:r["multiplier"])
        if not options or float(options[0]["multiplier"])!=1.0: raise SystemExit(f"missing 1.0 option for {stream} group {group}")
        groups.append(options)
    baseline_bytes=float(data["baseline"]["bytes"]); weights=(a.psnr_weight,a.ssim_weight,a.lpips_weight)
    policies={}; summaries={}
    for target in a.targets:
        # :g removes the insignificant .0 from integer-valued float arguments,
        # keeping profile names aligned with SLURM labels (target_5pct rather
        # than target_5p0pct; target_7p5pct remains unchanged).
        target_label=f"{target:g}".replace(".","p")
        name=f"target_{target_label}pct"; reached,selected=solve(groups,baseline_bytes*target/100,a.rate_bin_bytes,weights)
        tables={stream:[1.0]*group_count for stream in STREAMS}
        for row in selected: tables[row["stream"]][int(row["group"])]=float(row["multiplier"])
        policies[name]=tables
        summaries[name]={"target_rate_reduction_percent":target,"target_reached_in_additive_model":reached,
          "predicted_payload_saving_bytes":sum(max(0.0,-float(r["payload_delta_bytes"])) for r in selected),
          "predicted_psnr_drop":sum(float(r["psnr_drop"]) for r in selected),
          "predicted_ssim_drop":sum(float(r["ssim_drop"]) for r in selected),
          "predicted_lpips_increase":sum(float(r["lpips_increase"]) for r in selected),
          "geometry_groups_at_1x":sum(m==1 for m in tables["geometry"]),"texture_groups_at_1x":sum(m==1 for m in tables["texture"]),
          "geometry_max_multiplier":max(tables["geometry"]),"texture_max_multiplier":max(tables["texture"])}
    sensitivity={stream:[0.0]*group_count for stream in STREAMS}
    for row in rows:
        if float(row["multiplier"])==2.0: sensitivity[row["stream"]][int(row["group"])]=float(row["psnr_drop"])
    profile={"format":"globalsplat-optimized-group-quantization-v1","group_size":int(data["group_size"]),
      "source_operating_points":str(a.operating_points),"calibration_scenes":data["calibration_scenes"],
      "constraint":"all multipliers >= 1.0","objective_weights":{"psnr":weights[0],"ssim":weights[1],"lpips":weights[2]},
      "rate_bin_bytes":a.rate_bin_bytes,"sensitivity_psnr_drop":sensitivity,
      "policy_group_multipliers":policies,"allocation_summaries":summaries}
    a.output_profile.parent.mkdir(parents=True,exist_ok=True); a.output_profile.write_text(json.dumps(profile,indent=2))
    print(json.dumps(summaries,indent=2))
if __name__=="__main__": main()
