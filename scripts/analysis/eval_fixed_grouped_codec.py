#!/usr/bin/env python3
"""Evaluate actual CompressAI rANS payloads for fixed grouped CDF profiles.

This extracts scene latents only: it does not run the Gaussian decoder or
renderer because every mode decodes exactly the same G9/T7 integer symbols.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_cdf_phase1 import (
    extract_latents,
    load_weights,
    scene_name,
    seed_everything,
    to_device,
)
from globalsplat.compression import (
    decode_scene_latents,
    encode_scene_latents,
    load_grouped_cdf_profile,
)


MODES = {
    "adaptive_k1": None,
    "fixed_k1": (1, 1),
    "fixed_g32_t16": (16, 32),
    "fixed_g32_t32": (32, 32),
}


def compose_eval_config(args):
    from hydra import compose, initialize_config_dir

    overrides = [
        f"+experiment={args.experiment}",
        "mode=test",
        "optimizer.batch_size=1",
        f"dataset=re10k_eval_ctx{args.context}",
        f"dataset.dataset_roots=['{args.dataset_root.resolve().as_posix()}']",
        f"optimizer.num_workers={args.num_workers}",
        "model.latent_codec_bits=0",
        "model.latent_codec_texture_bits=0",
        "model.latent_codec_geometry_bits=0",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return compose(config_name="main", overrides=overrides)


def mean_numeric(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows], dtype=np.float64))


@torch.no_grad()
def evaluate(args, model, loader, profile: dict | None) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    numeric_keys = (
        "bytes",
        "payload_bytes",
        "texture_payload_bytes",
        "geometry_payload_bytes",
        "header_bytes",
        "json_header_bytes",
        "scale_side_info_bytes",
        "cdf_side_info_bytes",
        "other_metadata_bytes",
        "framing_bytes",
        "container_bytes",
        "bits_per_value",
        "rans_encode_seconds",
        "rans_decode_seconds",
        "texture_rans_encode_seconds",
        "geometry_rans_encode_seconds",
        "texture_rans_decode_seconds",
        "geometry_rans_decode_seconds",
        "group_dispatch_encode_seconds",
        "group_dispatch_decode_seconds",
        "cdf_build_seconds",
        "total_encode_seconds",
        "total_decode_seconds",
        "latent_extract_seconds",
    )
    with tempfile.TemporaryDirectory(prefix="globalsplat-fixed-cdf-") as tmp:
        bitstream = Path(tmp) / "scene.gsac"
        for index, batch in enumerate(loader):
            inputs = to_device(batch["inputs"], args.device)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            extract_start = time.perf_counter()
            texture, geometry = extract_latents(model, inputs)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            extract_seconds = time.perf_counter() - extract_start
            if texture.shape[0] != 1:
                raise RuntimeError("actual codec evaluation requires one scene per batch")

            encode_start = time.perf_counter()
            encoded = encode_scene_latents(
                texture,
                geometry,
                bitstream,
                texture_bits=args.texture_bits,
                geometry_bits=args.geometry_bits,
                backend="compressai",
                scale_mode="channel",
                clip_percentile=None,
                format_version=2,
                fixed_cdf_profile=profile,
            )
            total_encode_seconds = time.perf_counter() - encode_start

            decode_start = time.perf_counter()
            _, _, decoded = decode_scene_latents(
                bitstream,
                device="cpu",
                fixed_cdf_profile=profile,
                return_stats=True,
            )
            total_decode_seconds = time.perf_counter() - decode_start
            texture_exact = (
                encoded["texture_symbol_sha256"]
                == decoded["texture_symbol_sha256"]
            )
            geometry_exact = (
                encoded["geometry_symbol_sha256"]
                == decoded["geometry_symbol_sha256"]
            )
            if not texture_exact or not geometry_exact:
                raise RuntimeError(
                    f"quantized symbol mismatch at scene {index}: "
                    f"texture={texture_exact}, geometry={geometry_exact}"
                )

            row = {
                **encoded,
                **decoded,
                "scene": scene_name(batch, f"scene_{index:06d}"),
                "index": index,
                "texture_symbol_exact": texture_exact,
                "geometry_symbol_exact": geometry_exact,
                "symbol_exact_match": texture_exact and geometry_exact,
                "total_encode_seconds": total_encode_seconds,
                "total_decode_seconds": total_decode_seconds,
                "latent_extract_seconds": extract_seconds,
            }
            rows.append(row)
            processed = index + 1
            if args.progress_every > 0 and (
                processed == 1 or processed % args.progress_every == 0
            ):
                print(
                    f"[{args.mode}] {processed} scenes: "
                    f"mean total={mean_numeric(rows, 'bytes'):.1f} B, "
                    f"exact={sum(r['symbol_exact_match'] for r in rows)}/{processed}",
                    flush=True,
                )
            if args.scene_limit > 0 and processed >= args.scene_limit:
                break

    if not rows:
        raise RuntimeError("test dataloader produced no scenes")
    summary = {
        "mode": args.mode,
        "scenes": len(rows),
        "texture_bits": args.texture_bits,
        "geometry_bits": args.geometry_bits,
        "scale_mode": "channel",
        "format_version": 2,
        "cdf_mode": rows[0]["cdf_mode"],
        "texture_groups": rows[0]["texture_groups"],
        "geometry_groups": rows[0]["geometry_groups"],
        "shared_profile_bytes_raw": rows[0]["shared_profile_bytes_raw"],
        "shared_profile_is_not_charged_per_scene": True,
        "symbol_exact_scenes": int(sum(row["symbol_exact_match"] for row in rows)),
        "symbol_exact_match": all(row["symbol_exact_match"] for row in rows),
    }
    if profile is not None:
        summary["profile_id"] = profile["profile_id"]
        summary["profile_source"] = profile["source"]
    for key in numeric_keys:
        summary[f"mean_{key}"] = mean_numeric(rows, key)
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--profile-bundle", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment", default="re10k_32k")
    parser.add_argument("--context", type=int, choices=(12, 24, 36), default=12)
    parser.add_argument("--scene-limit", type=int, default=5601)
    parser.add_argument("--texture-bits", type=int, default=7)
    parser.add_argument("--geometry-bits", type=int, default=9)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cdf-precision", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not (args.dataset_root / "test" / "index.json").is_file():
        raise SystemExit(f"test dataset index not found under {args.dataset_root}")
    groups = MODES[args.mode]
    profile = None
    if groups is not None:
        if args.profile_bundle is None:
            raise SystemExit("fixed modes require --profile-bundle")
        texture_groups, geometry_groups = groups
        profile = load_grouped_cdf_profile(
            args.profile_bundle,
            texture_groups=texture_groups,
            geometry_groups=geometry_groups,
            alpha=args.alpha,
            precision=args.cdf_precision,
        )
        print(
            f"loaded fixed profile {profile['profile_id'][:16]}: "
            f"G{geometry_groups}/T{texture_groups}, "
            f"{profile['profile_bytes_raw']} shared raw bytes",
            flush=True,
        )

    seed_everything(args.seed)
    from globalsplat.main import build_datamodule, build_model

    cfg = compose_eval_config(args)
    device = torch.device(args.device)
    args.device = device
    model = build_model(cfg.model).to(device).eval()
    model.set_stage(int(cfg.curriculum.final_stage))
    load_weights(model, args.checkpoint)
    datamodule, _ = build_datamodule(cfg)
    datamodule.setup("test")
    seed_everything(args.seed)

    summary, rows = evaluate(args, model, datamodule.test_dataloader(), profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "per_scene.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote actual codec results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
