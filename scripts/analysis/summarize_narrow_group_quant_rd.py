#!/usr/bin/env python3
"""Summarize the final 8.0--9.5% group-quantization held-out sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = ("target_8pct", "target_8p5pct", "target_9pct", "target_9p5pct")
REFERENCES = ("target_7p5pct", "moderate_joint", "target_10pct")


def dominates(a: dict, b: dict) -> bool:
    no_worse = (
        a["bytes"] <= b["bytes"]
        and a["psnr"] >= b["psnr"]
        and a["ssim"] >= b["ssim"]
        and a["lpips"] <= b["lpips"]
    )
    strict = (
        a["bytes"] < b["bytes"]
        or a["psnr"] > b["psnr"]
        or a["ssim"] > b["ssim"]
        or a["lpips"] < b["lpips"]
    )
    return no_worse and strict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reference-comparison", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-psnr-drop", type=float, default=0.05)
    args = parser.parse_args()

    previous = json.loads(args.reference_comparison.read_text())
    previous_by_label = {row["label"]: row for row in previous["runs"]}
    baseline = previous_by_label["baseline"]
    profile = json.loads(args.profile.read_text())

    new_runs = []
    for label in LABELS:
        root = args.input_root / label / "globalsplat_re10k_32k"
        codec = json.loads((root / "codec_summary.json").read_text())
        scores = json.loads((root / "scores_all_avg.json").read_text())
        if int(codec["scenes"]) != 5601:
            raise SystemExit(f"{label}: expected 5601 scenes, got {codec['scenes']}")
        row = {
            "label": label,
            "scenes": codec["scenes"],
            "bytes": codec["mean_bytes"],
            "geometry_payload_bytes": codec["mean_geometry_payload_bytes"],
            "texture_payload_bytes": codec["mean_texture_payload_bytes"],
            "bits_per_value": codec["mean_bits_per_value"],
            "psnr": scores["psnr"],
            "ssim": scores["ssim"],
            "lpips": scores["lpips"],
            "geometry_mse": codec["mean_geometry_mse"],
            "texture_mse": codec["mean_texture_mse"],
            "encode_seconds": codec["mean_total_encode_seconds"],
            "decode_seconds": codec["mean_total_decode_seconds"],
        }
        row.update(
            rate_reduction_percent=100.0 * (1.0 - row["bytes"] / baseline["bytes"]),
            psnr_drop=baseline["psnr"] - row["psnr"],
            ssim_drop=baseline["ssim"] - row["ssim"],
            lpips_increase=row["lpips"] - baseline["lpips"],
        )
        new_runs.append(row)

    pareto_pool = [dict(row) for row in previous["runs"]] + new_runs
    for row in new_runs:
        row["pareto"] = not any(
            dominates(other, row) for other in pareto_pool if other is not row
        )

    feasible = [row for row in new_runs if row["psnr_drop"] <= args.max_psnr_drop]
    if not feasible:
        raise SystemExit("no narrow-sweep point satisfies the PSNR constraint")
    selected = max(feasible, key=lambda row: row["rate_reduction_percent"])

    comparisons = {}
    for label in REFERENCES:
        reference = previous_by_label[label]
        comparisons[label] = {
            "rate_reduction_difference_percentage_points": selected[
                "rate_reduction_percent"
            ]
            - reference["rate_reduction_percent"],
            "psnr_drop_difference_db": selected["psnr_drop"]
            - reference["psnr_drop"],
            "ssim_drop_difference": selected["ssim_drop"]
            - reference["ssim_drop"],
            "lpips_increase_difference": selected["lpips_increase"]
            - reference["lpips_increase"],
            "selected_strictly_dominates": dominates(selected, reference),
            "reference_strictly_dominates": dominates(reference, selected),
        }

    report = {
        "experiment": "final narrow decoder-aware scalar/group-variable quantization sweep",
        "constraint": "maximum actual rate reduction subject to PSNR drop <= 0.05 dB",
        "baseline": baseline,
        "new_runs": new_runs,
        "references": {label: previous_by_label[label] for label in REFERENCES},
        "selected": selected,
        "selected_multiplier_table": profile["policy_group_multipliers"][
            selected["label"]
        ],
        "selected_vs_references": comparisons,
        "quantizer_status": "final; do not continue tuning",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("label\trate reduction (%)\tPSNR drop (dB)\tpareto")
    for row in new_runs:
        print(
            f"{row['label']}\t{row['rate_reduction_percent']:.6f}\t"
            f"{row['psnr_drop']:.6f}\t{row['pareto']}"
        )
    print(f"selected: {selected['label']}")


if __name__ == "__main__":
    main()
