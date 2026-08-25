#!/usr/bin/env python3
"""Calibrate a fixed K=32 entropy profile for transformed training latents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from globalsplat.compression import LearnedLatentTransformCodec
from globalsplat.compression.cdf_analysis import (
    agglomerative_entropy_mappings,
    grouped_counts,
    quantized_channel_histogram,
)
from scripts.analysis.codec_diagnostic_utils import extract, seed_all, setup, to_device


def scene_name(batch: dict, fallback: str) -> str:
    value = (batch.get("scene_info", {}) or {}).get("scene", fallback)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--transform-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-profile", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--experiment", default="re10k_32k")
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--calibration-scenes", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_workers != 0:
        raise SystemExit("num-workers must be 0 for train calibration")
    for output in (args.output_profile, args.output_report):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {output}")
    seed_all(args.seed)

    model, train_loader, _, device = setup(args)
    model.requires_grad_(False)
    codec = LearnedLatentTransformCodec().to(device).eval()
    saved = torch.load(
        args.transform_checkpoint, map_location=device, weights_only=False
    )
    codec.load_state_dict(saved["codec_state_dict"])

    totals: dict[str, np.ndarray] = {}
    scene_ids = []
    with torch.no_grad():
        for index, batch in enumerate(train_loader):
            batch_gpu = to_device(batch, device)
            texture, geometry = extract(model, batch_gpu["inputs"])
            texture, geometry = codec.analysis(texture, geometry)
            for name, latent, bits in (
                ("texture", texture, 7),
                ("geometry", geometry, 9),
            ):
                counts, _ = quantized_channel_histogram(
                    latent, bits, scale_mode="channel", clip_percentile=None
                )
                totals[name] = totals.get(name, np.zeros_like(counts)) + counts
            scene_ids.append(scene_name(batch, f"scene_{index:06d}"))
            if len(scene_ids) == 1 or len(scene_ids) % 50 == 0:
                print(f"calibrated {len(scene_ids)} scenes", flush=True)
            if len(scene_ids) >= args.calibration_scenes:
                break
    if len(scene_ids) != args.calibration_scenes:
        raise RuntimeError(
            f"expected {args.calibration_scenes} calibration scenes, got {len(scene_ids)}"
        )

    arrays = {}
    group_report = {}
    for name in ("texture", "geometry"):
        mapping = agglomerative_entropy_mappings(
            totals[name], target_groups=(32,)
        )[32]
        counts = grouped_counts(totals[name], mapping)
        arrays[f"{name}_mapping_k32"] = mapping.astype(np.uint16)
        arrays[f"{name}_calibration_counts_k32"] = counts.astype(np.int64)
        group_report[name] = {
            "channels": int(mapping.size),
            "groups": 32,
            "alphabet": int(counts.shape[1]),
            "values": int(counts.sum()),
        }

    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_profile, **arrays)
    report = {
        "format": "globalsplat-learned-transform-cdf-calibration-v1",
        "split": "train",
        "heldout_test_used": False,
        "calibration_scenes": len(scene_ids),
        "scene_ids": scene_ids,
        "transform_checkpoint": str(args.transform_checkpoint),
        "texture_bits": 7,
        "geometry_bits": 9,
        "scale_mode": "channel_max",
        "groups_per_stream": 32,
        "group_quantization_multipliers": "all 1.0",
        "streams": group_report,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report | {"scene_ids": f"{len(scene_ids)} ids"}, indent=2))


if __name__ == "__main__":
    main()
