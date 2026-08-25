#!/usr/bin/env python3
"""A/B benchmark legacy file codec path against the in-memory path.

Both arms use the same G9/T7 scalar quantizer, fixed G32/T32 CDF tables and
CompressAI rANS. The legacy arm also recreates the full group-index vectors on
every encode/decode, matching the pre-optimization behavior.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from globalsplat.compression import (
    decode_scene_latents,
    decode_scene_latents_from_bytes,
    encode_scene_latents,
    encode_scene_latents_to_bytes,
    load_grouped_cdf_profile,
)
from globalsplat.model.rendering import render_static_batched
from globalsplat.model.types import Gaussians
from scripts.analysis.analyze_cdf_phase1 import (
    extract_latents,
    load_weights,
    scene_name,
    seed_everything,
    to_device,
)


def config_for(args):
    overrides = [
        f"+experiment={args.experiment}",
        f"dataset=re10k_eval_ctx{args.context}",
        "mode=test",
        "optimizer.batch_size=1",
        f"optimizer.num_workers={args.num_workers}",
        f"dataset.dataset_roots=['{args.dataset_root.as_posix()}']",
        "model.latent_codec_bits=0",
        "model.latent_codec_texture_bits=0",
        "model.latent_codec_geometry_bits=0",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return compose(config_name="main", overrides=overrides)


def clear_index_cache(profile: dict) -> None:
    for stream in ("texture", "geometry"):
        profile[stream]["index_cache"].clear()


def codec_kwargs(profile: dict) -> dict:
    return dict(
        texture_bits=7,
        geometry_bits=9,
        backend="compressai",
        scale_mode="channel",
        clip_percentile=None,
        format_version=2,
        fixed_cdf_profile=profile,
        verify_symbols=False,
    )


def run_legacy(texture, geometry, path: Path, profile: dict):
    clear_index_cache(profile)
    start = time.perf_counter()
    encoded = encode_scene_latents(texture, geometry, path, **codec_kwargs(profile))
    encode_s = time.perf_counter() - start
    clear_index_cache(profile)
    start = time.perf_counter()
    tex_hat, geo_hat, decoded = decode_scene_latents(
        path, device=texture.device, fixed_cdf_profile=profile,
        return_stats=True, verify_symbols=False,
    )
    decode_s = time.perf_counter() - start
    return path.read_bytes(), encoded, decoded, tex_hat, geo_hat, encode_s, decode_s


def run_memory(texture, geometry, profile: dict):
    start = time.perf_counter()
    data, encoded = encode_scene_latents_to_bytes(
        texture, geometry, **codec_kwargs(profile)
    )
    encode_s = time.perf_counter() - start
    start = time.perf_counter()
    tex_hat, geo_hat, decoded = decode_scene_latents_from_bytes(
        data, device=texture.device, fixed_cdf_profile=profile,
        return_stats=True, verify_symbols=False,
    )
    decode_s = time.perf_counter() - start
    return data, encoded, decoded, tex_hat, geo_hat, encode_s, decode_s


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def decode_gaussians(model, encoded_patch) -> Gaussians:
    means, rotations, scales, sh, opacities, reg = model.gaussian_decoder(encoded_patch)
    return Gaussians(
        means=means, scales=scales, rotations=rotations, sh=sh,
        opacities=opacities, reg=reg,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--profile-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment", default="re10k_32k")
    parser.add_argument("--context", type=int, choices=(12, 24, 36), default=12)
    parser.add_argument("--scene-limit", type=int, default=5601)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.profile_bundle = args.profile_bundle.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    for path, label in ((args.checkpoint, "checkpoint"), (args.profile_bundle, "profile")):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not (args.dataset_root / "test" / "index.json").is_file():
        raise SystemExit(f"missing dataset index: {args.dataset_root / 'test/index.json'}")

    # Separate objects prevent legacy cache clearing from perturbing memory mode.
    profile_args = dict(texture_groups=32, geometry_groups=32, alpha=0.5, precision=16)
    legacy_profile = load_grouped_cdf_profile(args.profile_bundle, **profile_args)
    memory_profile = load_grouped_cdf_profile(args.profile_bundle, **profile_args)

    seed_everything(args.seed)
    from globalsplat.main import build_datamodule, build_model

    cfg = config_for(args)
    device = torch.device("cuda")
    model = build_model(cfg.model).to(device).eval()
    model.set_stage(int(cfg.curriculum.final_stage), mix=1.0)
    load_weights(model, args.checkpoint)
    dm, _ = build_datamodule(cfg)
    dm.setup("test")
    seed_everything(args.seed)

    # Use exactly the same metric implementation as normal GlobalSplat eval.
    upstream = ROOT / "third_party" / "ZPressor" / "mvsplat"
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    rows = []
    with tempfile.TemporaryDirectory(prefix="globalsplat-codec-ab-") as tmp:
        path = Path(tmp) / "scene.gsac"
        for index, batch in enumerate(dm.test_dataloader()):
            inputs = to_device(batch["inputs"], device)
            targets = to_device(batch["targets"], device)
            texture, geometry = extract_latents(model, inputs)
            if index % 2 == 0:
                legacy = run_legacy(texture, geometry, path, legacy_profile)
                memory = run_memory(texture, geometry, memory_profile)
            else:
                memory = run_memory(texture, geometry, memory_profile)
                legacy = run_legacy(texture, geometry, path, legacy_profile)

            ldata, lstats, _, ltex, lgeo, lenc, ldec = legacy
            mdata, mstats, _, mtex, mgeo, menc, mdec = memory
            bitstream_exact = ldata == mdata
            texture_exact = torch.equal(ltex, mtex)
            geometry_exact = torch.equal(lgeo, mgeo)
            if not (bitstream_exact and texture_exact and geometry_exact):
                raise RuntimeError(f"A/B mismatch at scene {index}")

            legacy_gaussians = decode_gaussians(model, (ltex, lgeo))
            memory_gaussians = decode_gaussians(model, (mtex, mgeo))
            legacy_rgb = render_static_batched(
                legacy_gaussians, targets, render_depth=False
            )["img"]
            memory_rgb = render_static_batched(
                memory_gaussians, targets, render_depth=False
            )["img"]
            b, t, _, h, w = targets["images"].shape
            legacy_rgb = legacy_rgb.view(b, t, 3, h, w)[0].clamp(0, 1)
            memory_rgb = memory_rgb.view(b, t, 3, h, w)[0].clamp(0, 1)
            gt = targets["images"][0].float().clamp(0, 1)
            render_exact = torch.equal(legacy_rgb, memory_rgb)
            render_max_abs_diff = float((legacy_rgb - memory_rgb).abs().max().item())
            memory_psnr = float(compute_psnr(gt, memory_rgb).mean().item())
            memory_ssim = float(compute_ssim(gt, memory_rgb).mean().item())
            memory_lpips = float(compute_lpips(gt, memory_rgb).mean().item())
            if render_exact:
                legacy_psnr, legacy_ssim, legacy_lpips = (
                    memory_psnr, memory_ssim, memory_lpips
                )
            else:
                legacy_psnr = float(compute_psnr(gt, legacy_rgb).mean().item())
                legacy_ssim = float(compute_ssim(gt, legacy_rgb).mean().item())
                legacy_lpips = float(compute_lpips(gt, legacy_rgb).mean().item())
            row = {
                "index": index,
                "scene": scene_name(batch, f"scene_{index:06d}"),
                "legacy_encode_seconds": lenc,
                "legacy_decode_seconds": ldec,
                "legacy_total_seconds": lenc + ldec,
                "memory_encode_seconds": menc,
                "memory_decode_seconds": mdec,
                "memory_total_seconds": menc + mdec,
                "legacy_bytes": lstats["bytes"],
                "memory_bytes": mstats["bytes"],
                "legacy_payload_bytes": lstats["payload_bytes"],
                "memory_payload_bytes": mstats["payload_bytes"],
                "legacy_bits_per_value": lstats["bits_per_value"],
                "memory_bits_per_value": mstats["bits_per_value"],
                "bitstream_exact": bitstream_exact,
                "texture_decoded_exact": texture_exact,
                "geometry_decoded_exact": geometry_exact,
                "render_rgb_exact": render_exact,
                "render_rgb_max_abs_diff": render_max_abs_diff,
                "legacy_psnr": legacy_psnr,
                "memory_psnr": memory_psnr,
                "legacy_ssim": legacy_ssim,
                "memory_ssim": memory_ssim,
                "legacy_lpips": legacy_lpips,
                "memory_lpips": memory_lpips,
            }
            rows.append(row)
            done = index + 1
            if done == 1 or (args.progress_every and done % args.progress_every == 0):
                print(
                    f"{done} scenes | legacy={mean(rows, 'legacy_total_seconds')*1e3:.2f} ms "
                    f"memory={mean(rows, 'memory_total_seconds')*1e3:.2f} ms "
                    f"bytes={mean(rows, 'memory_bytes'):.1f}", flush=True,
                )
            if args.scene_limit > 0 and done >= args.scene_limit:
                break

    summary = {
        "scenes": len(rows),
        "codec": "G9/T7 fixed G32/T32 CDF16 CompressAI rANS",
        "legacy_mean_encode_ms": mean(rows, "legacy_encode_seconds") * 1e3,
        "legacy_mean_decode_ms": mean(rows, "legacy_decode_seconds") * 1e3,
        "legacy_mean_total_ms": mean(rows, "legacy_total_seconds") * 1e3,
        "memory_mean_encode_ms": mean(rows, "memory_encode_seconds") * 1e3,
        "memory_mean_decode_ms": mean(rows, "memory_decode_seconds") * 1e3,
        "memory_mean_total_ms": mean(rows, "memory_total_seconds") * 1e3,
        "total_speedup": mean(rows, "legacy_total_seconds") / mean(rows, "memory_total_seconds"),
        "mean_final_bytes": mean(rows, "memory_bytes"),
        "mean_payload_bytes": mean(rows, "memory_payload_bytes"),
        "mean_bits_per_value": mean(rows, "memory_bits_per_value"),
        "legacy_mean_psnr": mean(rows, "legacy_psnr"),
        "memory_mean_psnr": mean(rows, "memory_psnr"),
        "legacy_mean_ssim": mean(rows, "legacy_ssim"),
        "memory_mean_ssim": mean(rows, "memory_ssim"),
        "legacy_mean_lpips": mean(rows, "legacy_lpips"),
        "memory_mean_lpips": mean(rows, "memory_lpips"),
        "psnr_delta": mean(rows, "memory_psnr") - mean(rows, "legacy_psnr"),
        "ssim_delta": mean(rows, "memory_ssim") - mean(rows, "legacy_ssim"),
        "lpips_delta": mean(rows, "memory_lpips") - mean(rows, "legacy_lpips"),
        "all_bitstreams_exact": all(row["bitstream_exact"] for row in rows),
        "all_decoded_latents_exact": all(
            row["texture_decoded_exact"] and row["geometry_decoded_exact"] for row in rows
        ),
        "all_rendered_rgb_exact": all(row["render_rgb_exact"] for row in rows),
        "max_render_rgb_abs_diff": max(row["render_rgb_max_abs_diff"] for row in rows),
        "bitrate_unchanged": all(
            row["legacy_bytes"] == row["memory_bytes"]
            and row["legacy_payload_bytes"] == row["memory_payload_bytes"]
            and row["legacy_bits_per_value"] == row["memory_bits_per_value"]
            for row in rows
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "per_scene.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
