#!/usr/bin/env python3
"""Evaluate a learned latent transform with actual fixed-profile rANS bytes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from globalsplat.compression import (
    LearnedLatentTransformCodec,
    decode_scene_latents_from_bytes,
    encode_scene_latents_to_bytes,
    load_grouped_cdf_profile,
)
from globalsplat.model.rendering import render_static_batched
from globalsplat.model.types import Gaussians
from scripts.analysis.codec_diagnostic_utils import extract, seed_all, setup, to_device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gaussians(model, texture, geometry) -> Gaussians:
    values = model.gaussian_decoder((texture, geometry))
    return Gaussians(
        means=values[0], rotations=values[1], scales=values[2], sh=values[3],
        opacities=values[4], reg=values[5]
    )


def scene_name(batch: dict, fallback: str) -> str:
    value = (batch.get("scene_info", {}) or {}).get("scene", fallback)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def mean(rows, key):
    return float(statistics.fmean(float(row[key]) for row in rows))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--transform-checkpoint", required=True, type=Path)
    parser.add_argument("--cdf-profile", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment", default="re10k_32k")
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--scenes", type=int, default=5601)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-symbols", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    seed_all(args.seed)

    model, _, test_loader, device = setup(args)
    model.requires_grad_(False)
    codec = LearnedLatentTransformCodec().to(device).eval()
    saved = torch.load(
        args.transform_checkpoint, map_location=device, weights_only=False
    )
    codec.load_state_dict(saved["codec_state_dict"])
    fixed_profile = load_grouped_cdf_profile(
        args.cdf_profile,
        texture_groups=32,
        geometry_groups=32,
        alpha=0.5,
        precision=16,
    )
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    rows = []
    for index, batch in enumerate(test_loader):
        batch_gpu = to_device(batch, device)
        texture, geometry = extract(model, batch_gpu["inputs"])

        synchronize(device)
        encode_start = time.perf_counter()
        texture_analysis, geometry_analysis = codec.analysis(texture, geometry)
        bitstream, rate = encode_scene_latents_to_bytes(
            texture_analysis,
            geometry_analysis,
            texture_bits=7,
            geometry_bits=9,
            backend="compressai",
            scale_mode="channel",
            clip_percentile=None,
            fixed_cdf_profile=fixed_profile,
            group_quantization_profile=None,
            verify_symbols=args.verify_symbols,
        )
        synchronize(device)
        encode_seconds = time.perf_counter() - encode_start

        synchronize(device)
        decode_start = time.perf_counter()
        texture_transformed, geometry_transformed, decode = decode_scene_latents_from_bytes(
            bitstream,
            device=device,
            fixed_cdf_profile=fixed_profile,
            return_stats=True,
            verify_symbols=args.verify_symbols,
        )
        texture_hat, geometry_hat = codec.synthesis(
            texture_transformed, geometry_transformed
        )
        synchronize(device)
        decode_seconds = time.perf_counter() - decode_start

        rendered = render_static_batched(
            gaussians(model, texture_hat, geometry_hat),
            batch_gpu["targets"],
            render_depth=False,
        )["img"].float().clamp(0, 1)
        ground_truth = batch_gpu["targets"]["images"].flatten(0, 1).float().clamp(0, 1)
        row = {
            "scene": scene_name(batch, f"scene_{index:06d}"),
            "bytes": rate["bytes"],
            "payload_bytes": rate["payload_bytes"],
            "geometry_payload_bytes": rate["geometry_payload_bytes"],
            "texture_payload_bytes": rate["texture_payload_bytes"],
            "bits_per_value": 8.0 * rate["bytes"] / (texture.numel() + geometry.numel()),
            "psnr": float(compute_psnr(ground_truth, rendered).mean()),
            "ssim": float(compute_ssim(ground_truth, rendered).mean()),
            "lpips": float(compute_lpips(ground_truth, rendered).mean()),
            "geometry_mse": float((geometry - geometry_hat).square().mean()),
            "texture_mse": float((texture - texture_hat).square().mean()),
            "total_encode_seconds": encode_seconds,
            "total_decode_seconds": decode_seconds,
        }
        rows.append(row)
        if len(rows) == 1 or len(rows) % 50 == 0:
            print(f"evaluated {len(rows)}/{args.scenes}", flush=True)
        if len(rows) >= args.scenes:
            break
    if len(rows) != args.scenes:
        raise RuntimeError(f"expected {args.scenes} scenes, got {len(rows)}")

    metric_keys = ("psnr", "ssim", "lpips")
    codec_keys = (
        "bytes", "payload_bytes", "geometry_payload_bytes", "texture_payload_bytes",
        "bits_per_value", "geometry_mse", "texture_mse", "total_encode_seconds",
        "total_decode_seconds",
    )
    scores = {key: mean(rows, key) for key in metric_keys}
    codec_summary = {
        "format": "globalsplat-learned-latent-transform-evaluation-v1",
        "scenes": len(rows),
        **{f"mean_{key}": mean(rows, key) for key in codec_keys},
        "trainable_codec_parameters": codec.trainable_parameter_count,
        "transform_parameters": codec.transform_parameter_count,
        "transform_bytes_raw_fp32": codec.transform_parameter_count * 4,
        "training_entropy_parameters": codec.entropy_parameter_count,
        "training_entropy_model_deployed": False,
        "entropy_profile_bytes_raw": fixed_profile["profile_bytes_raw"],
        "shared_transform_and_profile_not_charged_per_scene": True,
        "per_scene_scale_side_information_bytes": 4096,
        "additional_per_scene_side_information_bytes": 0,
        "texture_bits": 7,
        "geometry_bits": 9,
        "group_quantization_multipliers": "all 1.0",
        "fixed_groups_per_stream": 32,
        "actual_rans_bytes": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores_all_avg.json").write_text(json.dumps(scores, indent=2) + "\n")
    (args.output_dir / "codec_summary.json").write_text(json.dumps(codec_summary, indent=2) + "\n")
    (args.output_dir / "codec_per_scene.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps({"scores": scores, "codec": codec_summary}, indent=2))


if __name__ == "__main__":
    main()
