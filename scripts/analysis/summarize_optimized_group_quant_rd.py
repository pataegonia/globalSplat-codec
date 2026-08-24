#!/usr/bin/env python3
"""Create held-out Pareto table and plots for optimized group allocation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def dominates(a,b):
    return (a["bytes"]<=b["bytes"] and a["psnr"]>=b["psnr"] and a["ssim"]>=b["ssim"] and a["lpips"]<=b["lpips"] and
      (a["bytes"]<b["bytes"] or a["psnr"]>b["psnr"] or a["ssim"]>b["ssim"] or a["lpips"]<b["lpips"]))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-root",required=True,type=Path); p.add_argument("--optimized-profile",required=True,type=Path); p.add_argument("--output",required=True,type=Path); a=p.parse_args()
    rows=[]
    for cp in sorted(a.input_root.rglob("codec_summary.json")):
        sp=cp.with_name("scores_all_avg.json")
        if not sp.is_file(): continue
        c=json.loads(cp.read_text()); s=json.loads(sp.read_text()); label=cp.relative_to(a.input_root).parts[0]
        rows.append({"label":label,"bytes":c["mean_bytes"],"geometry_payload_bytes":c["mean_geometry_payload_bytes"],"texture_payload_bytes":c["mean_texture_payload_bytes"],
          "bits_per_value":c["mean_bits_per_value"],"psnr":s["psnr"],"ssim":s["ssim"],"lpips":s["lpips"],"geometry_mse":c["mean_geometry_mse"],"texture_mse":c["mean_texture_mse"],
          "encode_seconds":c["mean_total_encode_seconds"],"decode_seconds":c["mean_total_decode_seconds"]})
    base=next((r for r in rows if r["label"]=="baseline"),None)
    if base is None: raise SystemExit("baseline not found")
    for r in rows:
        r["rate_reduction_percent"]=100*(1-r["bytes"]/base["bytes"]); r["psnr_drop"]=base["psnr"]-r["psnr"]
        r["ssim_drop"]=base["ssim"]-r["ssim"]; r["lpips_increase"]=r["lpips"]-base["lpips"]
    pareto=[r["label"] for r in rows if not any(dominates(q,r) for q in rows if q is not r)]
    profile=json.loads(a.optimized_profile.read_text()); policies=profile["policy_group_multipliers"]
    allocation_analysis={}
    sensitivity=profile["sensitivity_psnr_drop"]
    try:
        from scipy.stats import spearmanr
        for name,tables in policies.items():
            allocation_analysis[name]={}
            for stream in ("geometry","texture"):
                values=tables[stream]
                correlation=(None if len(set(values))<2 or len(set(sensitivity[stream]))<2
                  else float(spearmanr(sensitivity[stream],values).statistic))
                allocation_analysis[name][stream]={
                  "groups_at_1x":[i for i,m in enumerate(values) if m==1.0],
                  "groups_at_max":[i for i,m in enumerate(values) if m==max(values)],
                  "max_multiplier":max(values),
                  "sensitivity_vs_multiplier_spearman":correlation,
                }
    except ImportError: allocation_analysis={"note":"scipy unavailable; rank correlation omitted"}
    favorable_over_7p58=[r["label"] for r in rows if r["rate_reduction_percent"]>7.58 and r["psnr_drop"]<.05]
    near_10=[r["label"] for r in rows if r["rate_reduction_percent"]>=9.5]
    report={"baseline":base,"runs":rows,"pareto":pareto,"optimized_profile":profile,"allocation_analysis":allocation_analysis,
      "more_than_7p58_percent_below_0p05db":favorable_over_7p58,"approximately_10_percent_or_more":near_10}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2))
    print("label\trate_reduction%\tpsnr_drop\tssim_drop\tlpips_increase\tpareto")
    for r in rows: print(f"{r['label']}\t{r['rate_reduction_percent']:.3f}\t{r['psnr_drop']:.5f}\t{r['ssim_drop']:.6f}\t{r['lpips_increase']:.6f}\t{r['label'] in pareto}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ordered=sorted(rows,key=lambda r:r["rate_reduction_percent"]); fig,ax=plt.subplots(1,2,figsize=(11,4.5))
        ax[0].plot([r["rate_reduction_percent"] for r in ordered],[r["psnr_drop"] for r in ordered],"o-"); ax[0].set(xlabel="Rate reduction (%)",ylabel="PSNR drop (dB)"); ax[0].grid(alpha=.3)
        ax[1].plot([r["rate_reduction_percent"] for r in ordered],[r["lpips_increase"] for r in ordered],"o-"); ax[1].set(xlabel="Rate reduction (%)",ylabel="LPIPS increase"); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(a.output.with_suffix(".png"),dpi=180); plt.close(fig)
    except ImportError: pass
if __name__=="__main__": main()
