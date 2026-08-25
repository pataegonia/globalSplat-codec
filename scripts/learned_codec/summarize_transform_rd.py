#!/usr/bin/env python3
"""Compare final learned-transform actual-rANS results with scalar baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-root", required=True, type=Path)
    parser.add_argument("--scalar-comparison", required=True, type=Path)
    parser.add_argument("--narrow-comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    scalar = json.loads(args.scalar_comparison.read_text())
    narrow = json.loads(args.narrow_comparison.read_text())
    scalar_by_label = {row["label"]: row for row in scalar["runs"]}
    baseline = scalar_by_label["baseline"]
    target_8p5 = next(
        row for row in narrow["new_runs"] if row["label"] == "target_8p5pct"
    )
    baselines = {
        "baseline": baseline,
        "target_7p5pct": scalar_by_label["target_7p5pct"],
        "target_8p5pct": target_8p5,
        "target_10pct": scalar_by_label["target_10pct"],
    }

    learned = []
    for codec_path in sorted(args.learned_root.glob("lambda_*/globalsplat_re10k_32k/codec_summary.json")):
        score_path = codec_path.with_name("scores_all_avg.json")
        if not score_path.is_file():
            raise SystemExit(f"missing scores beside {codec_path}")
        codec = json.loads(codec_path.read_text())
        scores = json.loads(score_path.read_text())
        label = codec_path.parents[1].name
        row = {
            "label": label,
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
            "trainable_codec_parameters": codec["trainable_codec_parameters"],
            "entropy_profile_bytes_raw": codec["entropy_profile_bytes_raw"],
            "additional_per_scene_side_information_bytes": codec[
                "additional_per_scene_side_information_bytes"
            ],
        }
        row.update(
            rate_reduction_percent=100.0 * (1.0 - row["bytes"] / baseline["bytes"]),
            psnr_drop=baseline["psnr"] - row["psnr"],
            ssim_drop=baseline["ssim"] - row["ssim"],
            lpips_increase=row["lpips"] - baseline["lpips"],
        )
        row["beats_primary_success_criterion"] = (
            row["psnr_drop"] <= 0.05
            and row["rate_reduction_percent"] > 8.38523931440307
        )
        row["vs_target_8p5pct"] = {
            "rate_reduction_difference_percentage_points": row[
                "rate_reduction_percent"
            ]
            - target_8p5["rate_reduction_percent"],
            "psnr_drop_difference_db": row["psnr_drop"] - target_8p5["psnr_drop"],
            "ssim_drop_difference": row["ssim_drop"] - target_8p5["ssim_drop"],
            "lpips_increase_difference": row["lpips_increase"]
            - target_8p5["lpips_increase"],
        }
        learned.append(row)
    if not learned:
        raise SystemExit(f"no learned results found under {args.learned_root}")

    report = {
        "research_question": "does the lightweight transform move the measured RD frontier?",
        "primary_success_criterion": "PSNR drop <= 0.05 dB and actual rate reduction > 8.385239%",
        "baselines": baselines,
        "learned_operating_points": learned,
        "successful_points": [
            row["label"] for row in learned if row["beats_primary_success_criterion"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
