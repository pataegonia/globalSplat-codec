#!/usr/bin/env python3
"""Phase-1 CDF analysis for GlobalSplat G9/T7 scene latents.

No network weights are trained and no Gaussians are decoded or rendered.  The
script extracts the geometry/texture scene tokens, applies the production
per-scene channel quantizer, collects calibration/evaluation histograms, and
compares global versus per-channel zero-order entropy models.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from globalsplat.compression import encode_scene_latents
from globalsplat.compression.cdf_analysis import (
    cdf16_probabilities,
    cross_entropy_bits,
    empirical_entropy_bits,
    out_of_calibration_support,
    quantized_channel_histogram,
    smoothed_pmf,
)

BREAKDOWN_KEYS = (
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
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(to_device(value, device) for value in obj)
    return obj


def load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw)
    if any(key.startswith("model.") for key in state):
        state = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )


@torch.no_grad()
def extract_latents(model, inputs: dict[str, torch.Tensor]):
    content = model._tokenize(inputs["images"], inputs["intrinsic"], inputs["c2w"])
    state = model.scene_tokens.unsqueeze(0).expand(content.shape[0], -1, -1)
    _, (texture, geometry), _ = model.slot_encoder(content, state=state)
    return texture.float(), geometry.float()


def scene_name(batch: dict, fallback: str) -> str:
    value = (batch.get("scene_info", {}) or {}).get("scene", fallback)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    return {
        f"mean_{key}": float(sum(float(row[key]) for row in rows) / len(rows))
        for key in BREAKDOWN_KEYS
    } | {"scenes": len(rows)}


@torch.no_grad()
def collect_histograms(
    model,
    loader,
    *,
    device: torch.device,
    scene_limit: int,
    texture_bits: int,
    geometry_bits: int,
    progress_every: int,
    header_scenes: int = 0,
) -> dict:
    totals: dict[str, np.ndarray] = {}
    scene_adaptive_bits = {"texture": 0.0, "geometry": 0.0}
    ids: list[str] = []
    header_rows: list[dict[str, float]] = []

    with tempfile.TemporaryDirectory(prefix="globalsplat-cdf-phase1-") as tmp:
        bitstream = Path(tmp) / "scene.gsac"
        for index, batch in enumerate(loader):
            inputs = to_device(batch["inputs"], device)
            texture, geometry = extract_latents(model, inputs)
            if texture.shape[0] != 1 or geometry.shape[0] != 1:
                raise RuntimeError("CDF analysis requires one scene per batch")

            for name, latent, bits in (
                ("texture", texture, texture_bits),
                ("geometry", geometry, geometry_bits),
            ):
                counts, _ = quantized_channel_histogram(
                    latent, bits, scale_mode="channel", clip_percentile=None
                )
                if name not in totals:
                    totals[name] = np.zeros_like(counts, dtype=np.int64)
                if totals[name].shape != counts.shape:
                    raise RuntimeError(
                        f"{name} histogram shape changed: {totals[name].shape} -> {counts.shape}"
                    )
                totals[name] += counts
                scene_adaptive_bits[name] += empirical_entropy_bits(counts.sum(axis=0))

            current_id = scene_name(batch, f"scene_{index:06d}")
            ids.append(current_id)
            if len(header_rows) < header_scenes:
                stats = encode_scene_latents(
                    texture,
                    geometry,
                    bitstream,
                    texture_bits=texture_bits,
                    geometry_bits=geometry_bits,
                    backend="compressai",
                    scale_mode="channel",
                    clip_percentile=None,
                )
                row = {key: float(stats[key]) for key in BREAKDOWN_KEYS}
                row["scene"] = current_id
                header_rows.append(row)

            processed = index + 1
            if progress_every > 0 and (
                processed == 1 or processed % progress_every == 0
            ):
                print(f"processed {processed} scenes", flush=True)
            if scene_limit > 0 and processed >= scene_limit:
                break

    if not ids:
        raise RuntimeError("dataloader produced no scenes")
    for name, counts in totals.items():
        expected = len(ids) * int(model.scene_tokens.shape[0])
        per_channel = counts.sum(axis=1)
        if not np.all(per_channel == expected):
            raise RuntimeError(
                f"{name} histogram invariant failed: expected {expected} values/channel"
            )
    return {
        "counts": totals,
        "scene_adaptive_bits": scene_adaptive_bits,
        "scene_ids": ids,
        "header_rows": header_rows,
        "header_summary": _mean_rows(header_rows),
    }


def rate(bits: float, values: int, scenes: int) -> dict[str, float]:
    return {
        "total_bits": float(bits),
        "bits_per_value": float(bits / values),
        "bytes_per_scene": float(bits / (8 * scenes)),
    }


def analyze_stream(
    calibration: np.ndarray,
    evaluation: np.ndarray,
    *,
    scene_adaptive_bits: float,
    scenes: int,
    alpha: float,
    cdf_precision: int | None,
) -> dict:
    if calibration.shape != evaluation.shape:
        raise ValueError(
            f"calibration/evaluation shape mismatch: {calibration.shape} vs {evaluation.shape}"
        )
    values = int(evaluation.sum())
    calibration_global = calibration.sum(axis=0)
    evaluation_global = evaluation.sum(axis=0)
    fixed_global_pmf = smoothed_pmf(calibration_global, alpha)
    fixed_channel_pmf = smoothed_pmf(calibration, alpha)

    rates = {
        "scene_adaptive_global_ideal": rate(scene_adaptive_bits, values, scenes),
        "empirical_test_global": rate(
            empirical_entropy_bits(evaluation_global), values, scenes
        ),
        "fixed_global_ideal": rate(
            cross_entropy_bits(evaluation_global, fixed_global_pmf), values, scenes
        ),
        "empirical_test_per_channel": rate(
            empirical_entropy_bits(evaluation), values, scenes
        ),
        "fixed_per_channel_ideal": rate(
            cross_entropy_bits(evaluation, fixed_channel_pmf), values, scenes
        ),
    }
    cdf_error = None
    if cdf_precision is not None:
        try:
            rates["fixed_global_cdf_quantized"] = rate(
                cross_entropy_bits(
                    evaluation_global,
                    cdf16_probabilities(fixed_global_pmf, cdf_precision),
                ),
                values,
                scenes,
            )
            rates["fixed_per_channel_cdf_quantized"] = rate(
                cross_entropy_bits(
                    evaluation,
                    cdf16_probabilities(fixed_channel_pmf, cdf_precision),
                ),
                values,
                scenes,
            )
        except RuntimeError as exc:
            cdf_error = str(exc)

    fixed_global = rates["fixed_global_ideal"]["total_bits"]
    fixed_channel = rates["fixed_per_channel_ideal"]["total_bits"]
    empirical_global = rates["empirical_test_global"]["total_bits"]
    empirical_channel = rates["empirical_test_per_channel"]["total_bits"]
    return {
        "channels": int(calibration.shape[0]),
        "alphabet": int(calibration.shape[1]),
        "calibration_values": int(calibration.sum()),
        "evaluation_values": values,
        "smoothing_alpha": alpha,
        "out_of_calibration_support": out_of_calibration_support(
            calibration, evaluation
        ),
        "rates": rates,
        "fixed_per_channel_saving_vs_fixed_global": {
            "bits": float(fixed_global - fixed_channel),
            "percent": float(100.0 * (fixed_global - fixed_channel) / fixed_global),
        },
        "empirical_channel_information": {
            "bits": float(empirical_global - empirical_channel),
            "percent": float(
                100.0 * (empirical_global - empirical_channel) / empirical_global
            ),
        },
        "fixed_per_channel_mismatch_penalty_bits": float(
            fixed_channel - empirical_channel
        ),
        "cdf_quantization_error": cdf_error,
    }


def combine_stream_rates(streams: dict[str, dict], scenes: int) -> dict:
    common = set.intersection(
        *(set(report["rates"].keys()) for report in streams.values())
    )
    output = {}
    for name in sorted(common):
        bits = sum(report["rates"][name]["total_bits"] for report in streams.values())
        values = sum(report["evaluation_values"] for report in streams.values())
        output[name] = rate(bits, values, scenes)
    fixed_global = output["fixed_global_ideal"]["total_bits"]
    fixed_channel = output["fixed_per_channel_ideal"]["total_bits"]
    empirical_global = output["empirical_test_global"]["total_bits"]
    empirical_channel = output["empirical_test_per_channel"]["total_bits"]
    return {
        "rates": output,
        "fixed_per_channel_saving_vs_fixed_global": {
            "bytes_per_scene": float((fixed_global - fixed_channel) / (8 * scenes)),
            "percent": float(100.0 * (fixed_global - fixed_channel) / fixed_global),
        },
        "empirical_channel_information": {
            "bytes_per_scene": float(
                (empirical_global - empirical_channel) / (8 * scenes)
            ),
            "percent": float(
                100.0 * (empirical_global - empirical_channel) / empirical_global
            ),
        },
    }


def compose_configs(args):
    from hydra import compose, initialize_config_dir

    common = [
        f"+experiment={args.experiment}",
        "mode=test",
        "optimizer.batch_size=1",
        "model.latent_codec_bits=0",
        "model.latent_codec_texture_bits=0",
        "model.latent_codec_geometry_bits=0",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        calibration = compose(
            config_name="main",
            overrides=common
            + [
                "dataset=re10k",
                f"dataset.dataset_roots=['{args.dataset_root.resolve().as_posix()}']",
                f"dataset.num_context_views={args.context}",
                "dataset.num_target_views=3",
                "dataset.augment=false",
                f"optimizer.num_workers={args.calibration_num_workers}",
            ],
        )
        evaluation = compose(
            config_name="main",
            overrides=common
            + [
                f"dataset=re10k_eval_ctx{args.context}",
                f"dataset.dataset_roots=['{args.dataset_root.resolve().as_posix()}']",
                f"optimizer.num_workers={args.evaluation_num_workers}",
            ],
        )
    return calibration, evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment", default="re10k_32k")
    parser.add_argument("--context", type=int, choices=(12, 24, 36), default=12)
    parser.add_argument("--calibration-scenes", type=int, default=1000)
    parser.add_argument("--evaluation-scenes", type=int, default=5601)
    parser.add_argument("--header-scenes", type=int, default=100)
    parser.add_argument("--geometry-bits", type=int, default=9)
    parser.add_argument("--texture-bits", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cdf-precision", type=int, default=16)
    parser.add_argument("--skip-cdf-quantization", action="store_true")
    parser.add_argument("--calibration-num-workers", type=int, default=0)
    parser.add_argument("--evaluation-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    for split in ("train", "test"):
        index = dataset_root / split / "index.json"
        if not index.is_file():
            raise SystemExit(f"dataset index not found: {index}")
    if args.calibration_num_workers != 0:
        raise SystemExit(
            "calibration-num-workers must be 0: the upstream train IterableDataset "
            "does not shard chunks across workers"
        )
    if args.calibration_scenes <= 0 or args.evaluation_scenes <= 0:
        raise SystemExit("calibration-scenes and evaluation-scenes must be positive")

    args.checkpoint = checkpoint
    args.dataset_root = dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    from globalsplat.main import build_datamodule, build_model

    calibration_cfg, evaluation_cfg = compose_configs(args)
    device = torch.device(args.device)
    model = build_model(calibration_cfg.model).to(device).eval()
    model.set_stage(int(calibration_cfg.curriculum.final_stage))
    load_weights(model, checkpoint)

    calibration_dm, _ = build_datamodule(calibration_cfg)
    calibration_dm.setup("fit")
    print("collecting train calibration histograms", flush=True)
    seed_everything(args.seed)
    calibration = collect_histograms(
        model,
        calibration_dm.train_dataloader(),
        device=device,
        scene_limit=args.calibration_scenes,
        texture_bits=args.texture_bits,
        geometry_bits=args.geometry_bits,
        progress_every=args.progress_every,
    )

    evaluation_dm, _ = build_datamodule(evaluation_cfg)
    evaluation_dm.setup("test")
    print("collecting deterministic test histograms", flush=True)
    seed_everything(args.seed)
    evaluation = collect_histograms(
        model,
        evaluation_dm.test_dataloader(),
        device=device,
        scene_limit=args.evaluation_scenes,
        texture_bits=args.texture_bits,
        geometry_bits=args.geometry_bits,
        progress_every=args.progress_every,
        header_scenes=args.header_scenes,
    )

    np.savez_compressed(
        output_dir / "histograms.npz",
        calibration_texture_counts=calibration["counts"]["texture"],
        calibration_geometry_counts=calibration["counts"]["geometry"],
        evaluation_texture_counts=evaluation["counts"]["texture"],
        evaluation_geometry_counts=evaluation["counts"]["geometry"],
    )
    (output_dir / "scene_ids.json").write_text(
        json.dumps(
            {
                "calibration": calibration["scene_ids"],
                "evaluation": evaluation["scene_ids"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if evaluation["header_rows"]:
        (output_dir / "header_breakdown_per_scene.json").write_text(
            json.dumps(evaluation["header_rows"], indent=2), encoding="utf-8"
        )

    cdf_precision = None if args.skip_cdf_quantization else args.cdf_precision
    stream_reports = {}
    for name in ("texture", "geometry"):
        stream_reports[name] = analyze_stream(
            calibration["counts"][name],
            evaluation["counts"][name],
            scene_adaptive_bits=evaluation["scene_adaptive_bits"][name],
            scenes=len(evaluation["scene_ids"]),
            alpha=args.alpha,
            cdf_precision=cdf_precision,
        )

    report = {
        "config": {
            "checkpoint": str(checkpoint),
            "dataset_root": str(dataset_root),
            "experiment": args.experiment,
            "context_views": args.context,
            "target_views_for_normalization": 3,
            "geometry_bits": args.geometry_bits,
            "texture_bits": args.texture_bits,
            "scale_mode": "channel",
            "clip_percentile": None,
            "smoothing_alpha": args.alpha,
            "cdf_precision": cdf_precision,
            "seed": args.seed,
        },
        "calibration_scenes": len(calibration["scene_ids"]),
        "evaluation_scenes": len(evaluation["scene_ids"]),
        "header_breakdown": evaluation["header_summary"],
        "streams": stream_reports,
        "combined": combine_stream_rates(
            stream_reports, scenes=len(evaluation["scene_ids"])
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote Phase-1 CDF analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
