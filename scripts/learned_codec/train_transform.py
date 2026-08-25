#!/usr/bin/env python3
"""Train the first lightweight learned GlobalSplat latent transform.

Only the selected residual transforms and their training-time factorized
entropy models are optimized. The optional texture-only mode bypasses and
freezes the geometry transform and entropy estimator. The pretrained encoder,
Gaussian decoder, and renderer stay frozen, while decoder/renderer autograd
remains enabled so rendered RGB distortion reaches the codec transforms.
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from globalsplat.compression import (
    LearnedLatentTransformCodec,
    decode_scene_latents_from_bytes,
    encode_scene_latents_to_bytes,
    load_grouped_cdf_profile,
)
from globalsplat.model.optim import build_param_groups
from globalsplat.model.rendering import render_static_batched
from globalsplat.model.types import Gaussians
from scripts.analysis.codec_diagnostic_utils import extract, seed_all, setup, to_device


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def codec_from_checkpoint(saved: dict, device) -> LearnedLatentTransformCodec:
    checkpoint_args = saved.get("args", {})
    return LearnedLatentTransformCodec(
        texture_only=bool(checkpoint_args.get("texture_only", False))
    ).to(device).float()


def scene_name(batch: dict, fallback: str) -> str:
    value = (batch.get("scene_info", {}) or {}).get("scene", fallback)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


class MalformedExampleTracker:
    """Attach dataset skip messages to the active Stage A data phase."""

    _pattern = re.compile(r"^Skipped bad example ([^.]+)\. (.*)$")

    def __init__(self) -> None:
        self.phase = "setup"
        self.records: list[dict[str, str]] = []
        self.yielded = {"calibration": 0, "validation": 0, "train": 0}
        self._original_print = builtins.print

    def install(self) -> None:
        def tracked_print(*values, **kwargs):
            if values and isinstance(values[0], str):
                match = self._pattern.match(values[0])
                if match:
                    record = {
                        "scene_id": match.group(1),
                        "reason": match.group(2),
                        "phase": self.phase,
                    }
                    self.records.append(record)
                    self._original_print(
                        json.dumps({"malformed_example": record}), flush=True
                    )
                    return
            self._original_print(*values, **kwargs)

        builtins.print = tracked_print

    def restore(self) -> None:
        builtins.print = self._original_print

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def note_yield(self, phase: str) -> None:
        self.yielded[phase] += 1

    def report(self, maximum_fraction: float) -> dict:
        by_phase = {}
        systematic = False
        for phase in ("calibration", "validation", "train"):
            skipped = sum(record["phase"] == phase for record in self.records)
            yielded = self.yielded[phase]
            fraction = skipped / max(1, skipped + yielded)
            # A few known malformed RE10K examples are not a systematic loader
            # failure. Require at least three skips before applying the rate gate.
            phase_systematic = skipped >= 3 and fraction > maximum_fraction
            systematic = systematic or phase_systematic
            by_phase[phase] = {
                "yielded_examples": yielded,
                "skipped_examples": skipped,
                "skipped_fraction": fraction,
                "systematic": phase_systematic,
            }
        return {
            "records": self.records,
            "by_phase": by_phase,
            "maximum_allowed_fraction": maximum_fraction,
            "systematic_data_loader_problem": systematic,
            "pass": not systematic,
        }


def gaussians_from_latents(
    model,
    texture: torch.Tensor,
    geometry: torch.Tensor,
    *,
    mixed_precision: bool,
) -> Gaussians:
    with torch.autocast(
        device_type=texture.device.type,
        dtype=torch.bfloat16,
        enabled=mixed_precision and texture.device.type == "cuda",
    ):
        values = model.gaussian_decoder((texture, geometry))
    return Gaussians(
        means=values[0],
        rotations=values[1],
        scales=values[2],
        sh=values[3],
        opacities=values[4],
        reg=values[5],
    )


def rendered_distortion(
    model,
    texture: torch.Tensor,
    geometry: torch.Tensor,
    targets: dict,
    *,
    mixed_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gaussians = gaussians_from_latents(
        model, texture, geometry, mixed_precision=mixed_precision
    )
    rendered = render_static_batched(gaussians, targets, render_depth=False)["img"].float()
    ground_truth = targets["images"].flatten(0, 1).float()
    rendered = rendered.clamp(0.0, 1.0)
    ground_truth = ground_truth.clamp(0.0, 1.0)
    distortion = (255.0**2) * F.mse_loss(rendered, ground_truth)
    return distortion, ground_truth, rendered


def reserve_batches(
    loader,
    calibration_scenes: int,
    validation_scenes: int,
    malformed: MalformedExampleTracker,
) -> tuple[list, list, object]:
    iterator = iter(loader)
    calibration, validation = [], []
    total = calibration_scenes + validation_scenes
    while len(calibration) + len(validation) < total:
        phase = "calibration" if len(calibration) < calibration_scenes else "validation"
        malformed.set_phase(phase)
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("training loader ended before reserve subsets were built") from exc
        malformed.note_yield(phase)
        destination = calibration if len(calibration) < calibration_scenes else validation
        destination.append(batch)
    return calibration, validation, iterator


@torch.no_grad()
def initialize_entropy_models(codec, model, batches, device) -> dict:
    moments = {
        "texture": {"sum": None, "square": None, "count": 0},
        "geometry": {"sum": None, "square": None, "count": 0},
    }
    for batch in batches:
        inputs = to_device(batch["inputs"], device)
        texture, geometry = extract(model, inputs)
        texture_analysis, geometry_analysis = codec.analysis(texture, geometry)
        for name, latent, bits in (
            ("texture", texture_analysis, codec.texture_bits),
            ("geometry", geometry_analysis, codec.geometry_bits),
        ):
            symbols = codec.quantize(latent, bits, straight_through=False).symbols
            total = symbols.sum(dim=(0, 1)).double().cpu()
            square = symbols.square().sum(dim=(0, 1)).double().cpu()
            if moments[name]["sum"] is None:
                moments[name]["sum"] = total
                moments[name]["square"] = square
            else:
                moments[name]["sum"] += total
                moments[name]["square"] += square
            moments[name]["count"] += symbols.shape[0] * symbols.shape[1]

    report = {}
    for name in ("texture", "geometry"):
        count = moments[name]["count"]
        mean = moments[name]["sum"] / count
        variance = moments[name]["square"] / count - mean.square()
        standard_deviation = variance.clamp_min(0.25).sqrt().float().to(device)
        entropy = getattr(codec, f"{name}_entropy")
        entropy.initialize_from_moments(mean.float().to(device), standard_deviation)
        report[name] = {
            "symbol_mean_min": float(mean.min()),
            "symbol_mean_max": float(mean.max()),
            "symbol_std_min": float(standard_deviation.min()),
            "symbol_std_max": float(standard_deviation.max()),
            "symbol_values": int(count),
        }
    return report


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(left: list[float], right: list[float]) -> tuple[float, float]:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(average_tie_ranks(x), average_tie_ranks(y))[0, 1])
    return pearson, spearman


def quality_metrics(ground_truth: torch.Tensor, rendered: torch.Tensor) -> dict:
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    return {
        "psnr": float(compute_psnr(ground_truth, rendered).mean()),
        "ssim": float(compute_ssim(ground_truth, rendered).mean()),
        "lpips": float(compute_lpips(ground_truth, rendered).mean()),
    }


def encode_actual_rans(
    texture: torch.Tensor,
    geometry: torch.Tensor,
    fixed_profile: dict,
    *,
    verify_symbols: bool,
) -> tuple[bytes, dict]:
    return encode_scene_latents_to_bytes(
        texture,
        geometry,
        texture_bits=7,
        geometry_bits=9,
        backend="compressai",
        scale_mode="channel",
        clip_percentile=None,
        fixed_cdf_profile=fixed_profile,
        group_quantization_profile=None,
        verify_symbols=verify_symbols,
    )


@torch.no_grad()
def measure_identity_equivalence(
    codec,
    model,
    batches,
    device,
    fixed_profile,
    *,
    mixed_precision: bool,
    latent_tolerance: float,
    metric_tolerance: float,
) -> dict:
    """Compare production scalar coding with the identity learned path."""
    rows = []
    symbol_matches = 0
    symbol_checks = 0
    for batch in batches:
        batch_gpu = to_device(batch, device)
        texture, geometry = extract(model, batch_gpu["inputs"])

        production_data, production_stats = encode_actual_rans(
            texture, geometry, fixed_profile, verify_symbols=True
        )
        production_texture, production_geometry, production_decode = (
            decode_scene_latents_from_bytes(
                production_data,
                device=device,
                fixed_cdf_profile=fixed_profile,
                return_stats=True,
                verify_symbols=True,
            )
        )

        learned_output = codec(texture, geometry, straight_through=False)
        learned_data, learned_stats = encode_actual_rans(
            learned_output["texture_analysis"],
            learned_output["geometry_analysis"],
            fixed_profile,
            verify_symbols=True,
        )
        transformed_texture, transformed_geometry, learned_decode = (
            decode_scene_latents_from_bytes(
                learned_data,
                device=device,
                fixed_cdf_profile=fixed_profile,
                return_stats=True,
                verify_symbols=True,
            )
        )
        learned_texture, learned_geometry = codec.synthesis(
            transformed_texture, transformed_geometry
        )

        _, production_truth, production_rendered = rendered_distortion(
            model,
            production_texture,
            production_geometry,
            batch_gpu["targets"],
            mixed_precision=mixed_precision,
        )
        _, learned_truth, learned_rendered = rendered_distortion(
            model,
            learned_texture,
            learned_geometry,
            batch_gpu["targets"],
            mixed_precision=mixed_precision,
        )
        production_quality = quality_metrics(production_truth, production_rendered)
        learned_quality = quality_metrics(learned_truth, learned_rendered)

        production_quantized_texture = codec.quantize(
            texture, codec.texture_bits, straight_through=False
        )
        production_quantized_geometry = codec.quantize(
            geometry, codec.geometry_bits, straight_through=False
        )
        production_estimated_rate = float(
            (
                codec.texture_entropy.bits(production_quantized_texture.symbols).sum()
                + codec.geometry_entropy.bits(production_quantized_geometry.symbols).sum()
            )
            / float(texture.numel() + geometry.numel())
        )

        stream_matches = []
        for stats, decoded in (
            (production_stats, production_decode),
            (learned_stats, learned_decode),
        ):
            for stream in ("texture", "geometry"):
                exact = (
                    stats[f"{stream}_symbol_sha256"]
                    == decoded[f"{stream}_symbol_sha256"]
                )
                stream_matches.append(exact)
        symbol_matches += sum(stream_matches)
        symbol_checks += len(stream_matches)

        texture_difference = (production_texture - learned_texture).abs()
        geometry_difference = (production_geometry - learned_geometry).abs()
        difference_values = torch.cat(
            [texture_difference.reshape(-1), geometry_difference.reshape(-1)]
        )
        rows.append(
            {
                "scene_id": scene_name(batch, f"identity_{len(rows)}"),
                "production": {
                    **production_quality,
                    "estimated_rate_bits_per_scalar": production_estimated_rate,
                    "actual_total_bytes": int(production_stats["bytes"]),
                    "actual_payload_bytes": int(production_stats["payload_bytes"]),
                    "texture_mse": float((texture - production_texture).square().mean()),
                    "geometry_mse": float((geometry - production_geometry).square().mean()),
                },
                "learned_identity": {
                    **learned_quality,
                    "estimated_rate_bits_per_scalar": float(
                        learned_output["rate_bits_per_scalar"]
                    ),
                    "actual_total_bytes": int(learned_stats["bytes"]),
                    "actual_payload_bytes": int(learned_stats["payload_bytes"]),
                    "texture_mse": float((texture - learned_texture).square().mean()),
                    "geometry_mse": float((geometry - learned_geometry).square().mean()),
                },
                "reconstructed_latent_max_abs_difference": float(
                    difference_values.max()
                ),
                "reconstructed_latent_mean_abs_difference": float(
                    difference_values.mean()
                ),
                "bitstream_exact_match": production_data == learned_data,
                "all_symbol_checks_exact": all(stream_matches),
            }
        )

    metric_names = (
        "psnr",
        "ssim",
        "lpips",
        "estimated_rate_bits_per_scalar",
        "actual_total_bytes",
        "actual_payload_bytes",
        "texture_mse",
        "geometry_mse",
    )
    production_mean = {
        name: float(np.mean([row["production"][name] for row in rows]))
        for name in metric_names
    }
    learned_mean = {
        name: float(np.mean([row["learned_identity"][name] for row in rows]))
        for name in metric_names
    }
    metric_max_abs_difference = {
        name: float(
            max(
                abs(row["production"][name] - row["learned_identity"][name])
                for row in rows
            )
        )
        for name in ("psnr", "ssim", "lpips", "estimated_rate_bits_per_scalar")
    }
    max_latent_difference = max(
        row["reconstructed_latent_max_abs_difference"] for row in rows
    )
    mean_latent_difference = float(
        np.mean([row["reconstructed_latent_mean_abs_difference"] for row in rows])
    )
    exact_bitstreams = all(row["bitstream_exact_match"] for row in rows)
    symbol_exact_rate = symbol_matches / max(1, symbol_checks)
    passed = (
        max_latent_difference <= latent_tolerance
        and all(value <= metric_tolerance for value in metric_max_abs_difference.values())
        and exact_bitstreams
        and symbol_exact_rate == 1.0
    )
    return {
        "scenes": len(rows),
        "quantization_mode": "hard_round_clamp",
        "production_mean": production_mean,
        "learned_identity_mean": learned_mean,
        "metric_max_abs_difference": metric_max_abs_difference,
        "reconstructed_latent_max_abs_difference": max_latent_difference,
        "reconstructed_latent_mean_abs_difference": mean_latent_difference,
        "actual_bitstream_exact_match": exact_bitstreams,
        "symbol_exact_match_rate": symbol_exact_rate,
        "latent_tolerance": latent_tolerance,
        "metric_tolerance": metric_tolerance,
        "pass": passed,
        "per_scene": rows,
    }


@torch.no_grad()
def measure_initial_scales(
    codec,
    model,
    batches,
    device,
    fixed_profile,
    *,
    mixed_precision: bool,
    minimum_correlation: float,
) -> dict:
    estimated_rates, actual_rates, actual_bytes, distortions = [], [], [], []
    for batch in batches:
        batch_gpu = to_device(batch, device)
        texture, geometry = extract(model, batch_gpu["inputs"])
        output = codec(texture, geometry, straight_through=False)
        distortion, _, _ = rendered_distortion(
            model,
            output["texture_hat"],
            output["geometry_hat"],
            batch_gpu["targets"],
            mixed_precision=mixed_precision,
        )
        _, stats = encode_actual_rans(
            output["texture_analysis"],
            output["geometry_analysis"],
            fixed_profile,
            verify_symbols=False,
        )
        scalars = texture.numel() + geometry.numel()
        estimated_rates.append(float(output["rate_bits_per_scalar"]))
        actual_rates.append(8.0 * float(stats["payload_bytes"]) / scalars)
        actual_bytes.append(int(stats["payload_bytes"]))
        distortions.append(float(distortion))

    rate0 = float(np.mean(estimated_rates))
    distortion0 = float(np.mean(distortions))
    if not math.isfinite(rate0) or not math.isfinite(distortion0) or distortion0 <= 0:
        raise RuntimeError(f"invalid initial scales: R0={rate0}, D0={distortion0}")
    pearson, spearman = correlation(estimated_rates, actual_rates)
    correlation_pass = (
        math.isfinite(pearson)
        and math.isfinite(spearman)
        and pearson >= minimum_correlation
        and spearman >= minimum_correlation
    )
    lambda0 = rate0 / distortion0
    multipliers = (0.25, 0.5, 1.0, 2.0, 4.0)
    return {
        "R0_bits_per_scalar": rate0,
        "D0_255_squared_rgb_mse": distortion0,
        "lambda0": lambda0,
        "lambda_sweep": {f"{value:g}x": value * lambda0 for value in multipliers},
        "rate_correlation_scenes": len(batches),
        "minimum_required_correlation": minimum_correlation,
        "estimated_vs_actual_payload_pearson": pearson,
        "estimated_vs_actual_payload_spearman": spearman,
        "mean_estimated_bits_per_scalar": rate0,
        "mean_actual_rans_payload_bits_per_scalar": float(np.mean(actual_rates)),
        "mean_actual_rans_payload_bytes": float(np.mean(actual_bytes)),
        "estimated_rate_per_scene": estimated_rates,
        "actual_rans_payload_bits_per_scalar_per_scene": actual_rates,
        "actual_rans_payload_bytes_per_scene": actual_bytes,
        "distortion_per_scene": distortions,
        "pass": correlation_pass,
    }


def scheduler_factor(step: int, warmup_steps: int, schedule_steps: int, minimum_ratio: float) -> float:
    if step < warmup_steps:
        return max(1e-3, float(step + 1) / float(warmup_steps))
    progress = min(1.0, (step - warmup_steps) / max(1, schedule_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def checkpoint_payload(codec, optimizer, scheduler, step, args, scale_report, trainable) -> dict:
    return {
        "format": "globalsplat-learned-latent-transform-v1",
        "codec_state_dict": codec.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": int(step),
        "lambda_multiplier": float(args.lambda_multiplier),
        "lambda": float(scale_report["lambda0"] * args.lambda_multiplier),
        "initial_scale_report": scale_report,
        "trainable_parameters": trainable,
        "args": vars(args),
    }


def gradient_report(codec, model) -> dict:
    all_codec_tensors = list(codec.named_parameters())
    codec_tensors = [
        (name, parameter)
        for name, parameter in all_codec_tensors
        if parameter.requires_grad
    ]
    frozen_codec_tensors = [
        name
        for name, parameter in all_codec_tensors
        if not parameter.requires_grad
    ]
    with_grad = [
        (name, parameter) for name, parameter in codec_tensors if parameter.grad is not None
    ]
    without_grad = [
        name for name, parameter in codec_tensors if parameter.grad is None
    ]
    nonfinite = [
        name
        for name, parameter in with_grad
        if not torch.isfinite(parameter.grad).all()
    ]
    zero = [
        name
        for name, parameter in with_grad
        if not torch.count_nonzero(parameter.grad).item()
    ]
    pretrained_with_grad = [
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    ]
    module_nonzero = {}
    for name in (
        "g_a_geometry",
        "g_s_geometry",
        "g_a_texture",
        "g_s_texture",
        "geometry_entropy",
        "texture_entropy",
    ):
        module = getattr(codec, name)
        trainable_parameters = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        module_nonzero[name] = (
            None
            if not trainable_parameters
            else any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in trainable_parameters
            )
        )
    return {
        "trainable_codec_tensor_count": len(codec_tensors),
        "frozen_codec_tensor_count": len(frozen_codec_tensors),
        "frozen_codec_tensors": frozen_codec_tensors,
        "trainable_codec_tensors_with_gradients": len(with_grad),
        "trainable_codec_tensors_without_gradients": len(without_grad),
        "trainable_codec_tensors_with_nonfinite_gradients": len(nonfinite),
        "trainable_codec_tensors_with_zero_gradients": len(zero),
        "tensors_without_gradients": without_grad,
        "tensors_with_nonfinite_gradients": nonfinite,
        "tensors_with_zero_gradients": zero,
        "module_has_finite_nonzero_gradient": module_nonzero,
        "pretrained_tensors_with_non_none_gradients": len(pretrained_with_grad),
        "pretrained_tensors_with_gradients": pretrained_with_grad,
        "all_codec_tensors_have_finite_nonzero_gradients": (
            len(with_grad) == len(codec_tensors) and not nonfinite and not zero
        ),
        "all_codec_modules_have_finite_nonzero_gradient": all(
            value for value in module_nonzero.values() if value is not None
        ),
        "zero_pretrained_gradients": not pretrained_with_grad,
    }


def global_gradient_norm(parameters) -> float:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2))


@torch.no_grad()
def validate(
    codec, model, batches, device, lambda_value: float, *, mixed_precision: bool
) -> dict:
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    rows = []
    for batch in batches:
        batch_gpu = to_device(batch, device)
        texture, geometry = extract(model, batch_gpu["inputs"])
        output = codec(texture, geometry, straight_through=False)
        distortion, ground_truth, rendered = rendered_distortion(
            model,
            output["texture_hat"],
            output["geometry_hat"],
            batch_gpu["targets"],
            mixed_precision=mixed_precision,
        )
        rows.append(
            {
                "rate": float(output["rate_bits_per_scalar"]),
                "distortion": float(distortion),
                "lambda_distortion": float(lambda_value * distortion),
                "loss": float(output["rate_bits_per_scalar"] + lambda_value * distortion),
                "psnr": float(compute_psnr(ground_truth, rendered).mean()),
                "ssim": float(compute_ssim(ground_truth, rendered).mean()),
                "lpips": float(compute_lpips(ground_truth, rendered).mean()),
                "texture_mse": float((texture - output["texture_hat"]).square().mean()),
                "geometry_mse": float((geometry - output["geometry_hat"]).square().mean()),
            }
        )
    return {
        f"mean_{key}": float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    } | {
        "scenes": len(rows),
        "quantization_mode": "hard_round_clamp",
        "straight_through_used": False,
    }


@torch.no_grad()
def checkpoint_roundtrip_report(
    codec,
    model,
    batch,
    checkpoint_path: Path,
    device,
    *,
    mixed_precision: bool,
    tolerance: float,
) -> dict:
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored = codec_from_checkpoint(saved, device)
    restored.load_state_dict(saved["codec_state_dict"])
    restored.eval()

    batch_gpu = to_device(batch, device)
    texture, geometry = extract(model, batch_gpu["inputs"])
    original = codec(texture, geometry, straight_through=False)
    reloaded = restored(texture, geometry, straight_through=False)
    _, ground_truth, original_rendered = rendered_distortion(
        model,
        original["texture_hat"],
        original["geometry_hat"],
        batch_gpu["targets"],
        mixed_precision=mixed_precision,
    )
    _, _, reloaded_rendered = rendered_distortion(
        model,
        reloaded["texture_hat"],
        reloaded["geometry_hat"],
        batch_gpu["targets"],
        mixed_precision=mixed_precision,
    )
    original_psnr = quality_metrics(ground_truth, original_rendered)["psnr"]
    reloaded_psnr = quality_metrics(ground_truth, reloaded_rendered)["psnr"]
    texture_difference = (original["texture_hat"] - reloaded["texture_hat"]).abs()
    geometry_difference = (original["geometry_hat"] - reloaded["geometry_hat"]).abs()
    latent_difference = torch.cat(
        [texture_difference.reshape(-1), geometry_difference.reshape(-1)]
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "scene_id": scene_name(batch, "checkpoint_roundtrip"),
        "quantization_mode": "hard_round_clamp",
        "estimated_rate_before_reload": float(original["rate_bits_per_scalar"]),
        "estimated_rate_after_reload": float(reloaded["rate_bits_per_scalar"]),
        "estimated_rate_abs_difference": abs(
            float(original["rate_bits_per_scalar"])
            - float(reloaded["rate_bits_per_scalar"])
        ),
        "psnr_before_reload": original_psnr,
        "psnr_after_reload": reloaded_psnr,
        "psnr_abs_difference": abs(original_psnr - reloaded_psnr),
        "reconstructed_latent_max_abs_difference": float(latent_difference.max()),
        "reconstructed_latent_mean_abs_difference": float(latent_difference.mean()),
        "tolerance": tolerance,
    }
    report["pass"] = all(
        report[key] <= tolerance
        for key in (
            "estimated_rate_abs_difference",
            "psnr_abs_difference",
            "reconstructed_latent_max_abs_difference",
        )
    )
    return report


@torch.no_grad()
def checkpoint_rate_diagnostics(
    checkpoint_path: Path,
    model,
    batches,
    device,
    fixed_profile,
) -> dict:
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    codec = codec_from_checkpoint(saved, device)
    codec.load_state_dict(saved["codec_state_dict"])
    codec.eval()
    accumulators = {
        stream: {
            "analysis_sum": 0.0,
            "analysis_square_sum": 0.0,
            "analysis_count": 0,
            "analysis_max_abs": 0.0,
            "scale_sum": 0.0,
            "scale_count": 0,
            "scale_min": float("inf"),
            "scale_max": 0.0,
            "zero_symbols": 0,
            "saturated_symbols": 0,
            "symbol_count": 0,
            "estimated_bits": 0.0,
            "actual_payload_bytes": 0,
        }
        for stream in ("texture", "geometry")
    }
    timing_totals = {
        "total_encode_seconds": 0.0,
        "total_decode_seconds": 0.0,
        "texture_rans_encode_seconds": 0.0,
        "geometry_rans_encode_seconds": 0.0,
        "texture_rans_decode_seconds": 0.0,
        "geometry_rans_decode_seconds": 0.0,
    }
    scene_ids = []
    for index, batch in enumerate(batches):
        scene_ids.append(scene_name(batch, f"rate_diagnostic_{index}"))
        batch_gpu = to_device(batch, device)
        texture, geometry = extract(model, batch_gpu["inputs"])
        output = codec(texture, geometry, straight_through=False)
        data, stats = encode_actual_rans(
            output["texture_analysis"],
            output["geometry_analysis"],
            fixed_profile,
            verify_symbols=False,
        )
        _, _, decode_stats = decode_scene_latents_from_bytes(
            data,
            device="cpu",
            fixed_cdf_profile=fixed_profile,
            return_stats=True,
            verify_symbols=False,
        )
        timing_totals["total_encode_seconds"] += stats["total_encode_seconds"]
        timing_totals["total_decode_seconds"] += decode_stats[
            "total_decode_seconds"
        ]
        for stream in ("texture", "geometry"):
            timing_totals[f"{stream}_rans_encode_seconds"] += stats[
                f"{stream}_rans_encode_seconds"
            ]
            timing_totals[f"{stream}_rans_decode_seconds"] += decode_stats[
                f"{stream}_rans_decode_seconds"
            ]
        for stream, bits in (
            ("texture", codec.texture_bits),
            ("geometry", codec.geometry_bits),
        ):
            accumulator = accumulators[stream]
            analysis = output[f"{stream}_analysis"].float()
            quantized = output[f"{stream}_quantized"]
            symbols = quantized.symbols
            scales = quantized.scale.float()
            qmax = (1 << (bits - 1)) - 1
            accumulator["analysis_sum"] += float(analysis.sum())
            accumulator["analysis_square_sum"] += float(analysis.square().sum())
            accumulator["analysis_count"] += analysis.numel()
            accumulator["analysis_max_abs"] = max(
                accumulator["analysis_max_abs"], float(analysis.abs().max())
            )
            accumulator["scale_sum"] += float(scales.sum())
            accumulator["scale_count"] += scales.numel()
            accumulator["scale_min"] = min(
                accumulator["scale_min"], float(scales.min())
            )
            accumulator["scale_max"] = max(
                accumulator["scale_max"], float(scales.max())
            )
            accumulator["zero_symbols"] += int((symbols == 0).sum())
            accumulator["saturated_symbols"] += int(
                (symbols.abs() == qmax).sum()
            )
            accumulator["symbol_count"] += symbols.numel()
            accumulator["estimated_bits"] += float(output[f"{stream}_bits"])
            accumulator["actual_payload_bytes"] += int(
                stats[f"{stream}_payload_bytes"]
            )

    streams = {}
    for stream, accumulator in accumulators.items():
        count = accumulator["analysis_count"]
        mean = accumulator["analysis_sum"] / count
        variance = max(
            0.0, accumulator["analysis_square_sum"] / count - mean * mean
        )
        symbol_count = accumulator["symbol_count"]
        streams[stream] = {
            "transformed_mean": mean,
            "transformed_std": math.sqrt(variance),
            "transformed_max_abs": accumulator["analysis_max_abs"],
            "quantization_scale_min": accumulator["scale_min"],
            "quantization_scale_mean": (
                accumulator["scale_sum"] / accumulator["scale_count"]
            ),
            "quantization_scale_max": accumulator["scale_max"],
            "zero_symbol_fraction": accumulator["zero_symbols"] / symbol_count,
            "saturation_fraction": accumulator["saturated_symbols"] / symbol_count,
            "estimated_bits_per_scalar": accumulator["estimated_bits"] / symbol_count,
            "actual_rans_payload_bits_per_scalar": (
                8.0 * accumulator["actual_payload_bytes"] / symbol_count
            ),
            "mean_actual_rans_payload_bytes_per_scene": (
                accumulator["actual_payload_bytes"] / len(batches)
            ),
            "scalars": symbol_count,
        }
    total_scalars = sum(value["scalars"] for value in streams.values())
    total_estimated_bits = sum(
        accumulators[stream]["estimated_bits"] for stream in accumulators
    )
    total_actual_bytes = sum(
        accumulators[stream]["actual_payload_bytes"] for stream in accumulators
    )
    return {
        "checkpoint": str(checkpoint_path),
        "step": int(saved["step"]),
        "codec_mode": (
            "texture_only"
            if bool(saved.get("args", {}).get("texture_only", False))
            else "joint_geometry_texture"
        ),
        "trainable_parameters": saved.get("trainable_parameters"),
        "subset": "fixed_validation",
        "scenes": len(batches),
        "scene_ids": scene_ids,
        "quantization_mode": "hard_round_clamp",
        "estimated_bits_per_scalar": total_estimated_bits / total_scalars,
        "actual_rans_payload_bits_per_scalar": 8.0 * total_actual_bytes / total_scalars,
        "mean_actual_rans_payload_bytes_per_scene": total_actual_bytes / len(batches),
        "mean_timing_seconds_per_scene": {
            name: value / len(batches) for name, value in timing_totals.items()
        },
        "streams": streams,
    }


@torch.no_grad()
def checkpoint_validation(
    checkpoint_path: Path,
    model,
    batches,
    device,
    lambda_value: float,
    *,
    mixed_precision: bool,
) -> dict:
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    codec = codec_from_checkpoint(saved, device)
    codec.load_state_dict(saved["codec_state_dict"])
    codec.eval()
    result = validate(
        codec,
        model,
        batches,
        device,
        lambda_value,
        mixed_precision=mixed_precision,
    )
    result["step"] = int(saved["step"])
    result["gradient_source"] = "selection_reference_checkpoint"
    return result


def print_gate_table(gates: dict[str, bool]) -> None:
    print("SMOKE GATE", flush=True)
    for name, passed in gates.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)


def run_gradient_probe(
    codec,
    model,
    batch,
    device,
    lambda_value: float,
    *,
    mixed_precision: bool,
) -> dict:
    codec.zero_grad(set_to_none=True)
    batch_gpu = to_device(batch, device)
    with torch.no_grad():
        texture, geometry = extract(model, batch_gpu["inputs"])
    output = codec(texture, geometry, straight_through=True)
    distortion, _, _ = rendered_distortion(
        model,
        output["texture_hat"],
        output["geometry_hat"],
        batch_gpu["targets"],
        mixed_precision=mixed_precision,
    )
    rate = output["rate_bits_per_scalar"].float()
    loss = rate + lambda_value * distortion.float()
    loss.backward()
    report = gradient_report(codec, model)
    gradient_norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), float("inf"))
    report.update(
        {
            "scene_id": scene_name(batch, "gradient_probe"),
            "rate_bits_per_scalar": float(rate.detach()),
            "distortion_255_squared_rgb_mse": float(distortion.detach()),
            "lambda_distortion": float(lambda_value * distortion.detach()),
            "total_rd_loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
        }
    )
    codec.zero_grad(set_to_none=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cdf-profile", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment", default="re10k_32k")
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-scenes", type=int, default=16)
    parser.add_argument("--validation-scenes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--schedule-steps", type=int, default=40000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--early-stopping-steps", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--final-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--lambda-multiplier", type=float, default=1.0)
    parser.add_argument("--texture-only", action="store_true")
    parser.add_argument("--lambda-config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fixed-resume-learning-rate", type=float)
    parser.add_argument("--selection-reference-checkpoint", type=Path)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-rate-correlation", type=float, default=0.2)
    parser.add_argument("--identity-latent-tolerance", type=float, default=1e-6)
    parser.add_argument("--identity-metric-tolerance", type=float, default=1e-6)
    parser.add_argument("--checkpoint-tolerance", type=float, default=1e-6)
    parser.add_argument("--maximum-malformed-fraction", type=float, default=0.05)
    parser.add_argument("--maximum-smoke-psnr-drop", type=float, default=0.05)
    args = parser.parse_args()

    if args.num_workers != 0:
        raise SystemExit("num-workers must be 0: the upstream train IterableDataset is not worker-sharded")
    if args.max_steps <= 0 or args.schedule_steps < args.max_steps:
        raise SystemExit("require 0 < max-steps <= schedule-steps")
    if args.final_learning_rate <= 0 or args.final_learning_rate > args.learning_rate:
        raise SystemExit("final learning rate must be in (0, learning rate]")
    if args.fixed_resume_learning_rate is not None:
        if args.resume is None:
            raise SystemExit("fixed-resume-learning-rate requires --resume")
        if args.fixed_resume_learning_rate <= 0:
            raise SystemExit("fixed-resume-learning-rate must be positive")
    if not 0.0 <= args.minimum_rate_correlation <= 1.0:
        raise SystemExit("minimum-rate-correlation must be in [0, 1]")
    if not 0.0 <= args.maximum_malformed_fraction <= 1.0:
        raise SystemExit("maximum-malformed-fraction must be in [0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    model, loader, _, device = setup(args)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("failed to freeze pretrained GlobalSplat")

    codec = LearnedLatentTransformCodec(texture_only=args.texture_only).to(device).float()
    trainable = [name for name, parameter in codec.named_parameters() if parameter.requires_grad]
    trainable_report = {
        "mode": "texture_only" if args.texture_only else "joint_geometry_texture",
        "names": trainable,
        "transform_parameters": codec.transform_parameter_count,
        "entropy_parameters": codec.entropy_parameter_count,
        "total_trainable_parameters": codec.trainable_parameter_count,
        "total_codec_parameters": sum(
            parameter.numel() for parameter in codec.parameters()
        ),
        "pretrained_trainable_parameters": 0,
        "geometry_path_is_production_identity": bool(args.texture_only),
    }
    print(json.dumps(trainable_report, indent=2), flush=True)
    write_json(args.output_dir / "trainable_parameters.json", trainable_report)

    malformed = MalformedExampleTracker()
    malformed.install()
    calibration_batches, validation_batches, training_iterator = reserve_batches(
        loader, args.calibration_scenes, args.validation_scenes, malformed
    )
    reserved_ids = {
        scene_name(batch, f"reserved_{index}")
        for index, batch in enumerate(calibration_batches + validation_batches)
    }
    if len(reserved_ids) != args.calibration_scenes + args.validation_scenes:
        raise RuntimeError("calibration and validation scene IDs must be unique and disjoint")
    write_json(
        args.output_dir / "data_split.json",
        {
            "calibration_scene_ids": [
                scene_name(batch, f"calibration_{index}")
                for index, batch in enumerate(calibration_batches)
            ],
            "validation_scene_ids": [
                scene_name(batch, f"validation_{index}")
                for index, batch in enumerate(validation_batches)
            ],
            "heldout_test_used_for_training_or_selection": False,
        },
    )

    entropy_initialization = initialize_entropy_models(
        codec, model, calibration_batches, device
    )
    fixed_profile = load_grouped_cdf_profile(
        args.cdf_profile, texture_groups=32, geometry_groups=32, alpha=0.5, precision=16
    )
    identity_report = measure_identity_equivalence(
        codec,
        model,
        validation_batches,
        device,
        fixed_profile,
        mixed_precision=args.mixed_precision,
        latent_tolerance=args.identity_latent_tolerance,
        metric_tolerance=args.identity_metric_tolerance,
    )
    write_json(args.output_dir / "identity_equivalence.json", identity_report)
    print(
        json.dumps(
            {
                "identity_equivalence": {
                    key: value
                    for key, value in identity_report.items()
                    if key != "per_scene"
                }
            },
            indent=2,
        ),
        flush=True,
    )
    if not identity_report["pass"]:
        print_gate_table({"Identity equivalence": False})
        raise RuntimeError(
            "identity-initialized learned codec is not production-equivalent; "
            "training was not started"
        )

    scale_report = measure_initial_scales(
        codec,
        model,
        calibration_batches,
        device,
        fixed_profile,
        mixed_precision=args.mixed_precision,
        minimum_correlation=args.minimum_rate_correlation,
    )
    scale_report["entropy_initialization"] = entropy_initialization
    if args.lambda_config:
        frozen_scale = json.loads(args.lambda_config.read_text())
        scale_report["measured_lambda0_this_job"] = scale_report["lambda0"]
        scale_report["lambda0"] = float(frozen_scale["lambda0"])
        scale_report["lambda_sweep"] = frozen_scale["lambda_sweep"]
        scale_report["frozen_lambda_source"] = str(args.lambda_config)
    write_json(args.output_dir / "initial_scale.json", scale_report)
    print(
        json.dumps(
            {
                "R0_bits_per_scalar": scale_report["R0_bits_per_scalar"],
                "D0_255_squared_rgb_mse": scale_report["D0_255_squared_rgb_mse"],
                "lambda0": scale_report["lambda0"],
                "planned_lambda_sweep": scale_report["lambda_sweep"],
                "mean_estimated_bits_per_scalar": scale_report[
                    "mean_estimated_bits_per_scalar"
                ],
                "mean_actual_rans_payload_bits_per_scalar": scale_report[
                    "mean_actual_rans_payload_bits_per_scalar"
                ],
                "mean_actual_rans_payload_bytes": scale_report[
                    "mean_actual_rans_payload_bytes"
                ],
                "estimated_vs_actual_payload_pearson": scale_report[
                    "estimated_vs_actual_payload_pearson"
                ],
                "estimated_vs_actual_payload_spearman": scale_report[
                    "estimated_vs_actual_payload_spearman"
                ],
                "minimum_required_correlation": scale_report[
                    "minimum_required_correlation"
                ],
                "pass": scale_report["pass"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not scale_report["pass"]:
        print_gate_table(
            {
                "Identity equivalence": True,
                "Rate-vs-rANS correlation": False,
            }
        )
        raise RuntimeError(
            "rate estimator correlation is weak or negative; training was not started"
        )

    lambda_value = float(scale_report["lambda0"] * args.lambda_multiplier)
    optimizer = torch.optim.AdamW(
        build_param_groups(codec, args.weight_decay), lr=args.learning_rate
    )
    minimum_ratio = args.final_learning_rate / args.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_factor(
            step, args.warmup_steps, args.schedule_steps, minimum_ratio
        ),
    )
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        saved_texture_only = bool(saved.get("args", {}).get("texture_only", False))
        if saved_texture_only != args.texture_only:
            raise RuntimeError("resume checkpoint texture-only mode does not match")
        codec.load_state_dict(saved["codec_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_step = int(saved["step"])
        if not math.isclose(float(saved["lambda"]), lambda_value, rel_tol=1e-8):
            raise RuntimeError("resume checkpoint lambda does not match requested lambda")
        if args.fixed_resume_learning_rate is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.fixed_resume_learning_rate

    def next_training_batch():
        nonlocal training_iterator
        while True:
            malformed.set_phase("train")
            try:
                candidate = next(training_iterator)
            except StopIteration:
                training_iterator = iter(loader)
                malformed.set_phase("train")
                candidate = next(training_iterator)
            malformed.note_yield("train")
            if scene_name(candidate, "unknown") not in reserved_ids:
                return candidate

    gradient_probe_batch = next_training_batch()
    initial_gradient_report = run_gradient_probe(
        codec,
        model,
        gradient_probe_batch,
        device,
        lambda_value,
        mixed_precision=args.mixed_precision,
    )
    initial_gradient_pass = (
        initial_gradient_report["trainable_codec_tensors_without_gradients"] == 0
        and initial_gradient_report["trainable_codec_tensors_with_nonfinite_gradients"] == 0
        and initial_gradient_report["all_codec_modules_have_finite_nonzero_gradient"]
        and initial_gradient_report["zero_pretrained_gradients"]
    )
    initial_gradient_report["pass"] = initial_gradient_pass
    initial_gradient_report["note"] = (
        "Resume-state preoptimization audit."
        if args.resume
        else (
            "At exact identity, zero-initialized final residual layers can make some "
            "upstream tensor gradients numerically zero. A second audit after optimizer "
            "step 1 requires finite nonzero gradients for every trainable codec tensor."
        )
    )
    write_json(
        args.output_dir / f"gradient_audit_step_{start_step:06d}.json",
        initial_gradient_report,
    )
    print(json.dumps({"initial_gradient_audit": initial_gradient_report}, indent=2), flush=True)
    if not initial_gradient_pass:
        print_gate_table(
            {
                "Identity equivalence": True,
                "Rate-vs-rANS correlation": True,
                "Finite codec gradients": False,
                "Zero pretrained gradients": initial_gradient_report[
                    "zero_pretrained_gradients"
                ],
            }
        )
        raise RuntimeError("initial autograd probe failed; training was not started")

    history = []
    initial_validation = validate(
        codec,
        model,
        validation_batches,
        device,
        lambda_value,
        mixed_precision=args.mixed_precision,
    )
    initial_validation["step"] = start_step
    initial_validation["gradient_norm"] = initial_gradient_report["gradient_norm"]
    initial_validation["pre_clip_gradient_norm"] = initial_gradient_report[
        "gradient_norm"
    ]
    initial_validation["post_clip_gradient_norm"] = min(
        initial_gradient_report["gradient_norm"], args.gradient_clip
    )
    initial_validation["gradient_clipping_active"] = (
        initial_gradient_report["gradient_norm"] > args.gradient_clip
    )
    initial_validation["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    initial_validation["gradient_source"] = (
        "resume_preoptimization_probe"
        if args.resume
        else "identity_preoptimization_probe"
    )
    write_json(
        args.output_dir / f"validation_step_{start_step:06d}.json",
        initial_validation,
    )
    print(json.dumps({"initial_validation": initial_validation}, indent=2), flush=True)
    best_validation = initial_validation["mean_loss"]
    best_step = start_step
    initial_checkpoint = checkpoint_payload(
        codec,
        optimizer,
        scheduler,
        start_step,
        args,
        scale_report,
        trainable_report,
    )
    initial_checkpoint_path = args.output_dir / f"step_{start_step:06d}.pt"
    torch.save(initial_checkpoint, initial_checkpoint_path)
    torch.save(initial_checkpoint, args.output_dir / "best.pt")
    validation_checkpoint_paths = {start_step: initial_checkpoint_path}
    validation_results = {start_step: initial_validation}
    if args.selection_reference_checkpoint:
        reference_validation = checkpoint_validation(
            args.selection_reference_checkpoint,
            model,
            validation_batches,
            device,
            lambda_value,
            mixed_precision=args.mixed_precision,
        )
        reference_step = reference_validation["step"]
        if reference_step == start_step:
            raise RuntimeError(
                "selection reference checkpoint must precede the resume checkpoint"
            )
        validation_checkpoint_paths[reference_step] = (
            args.selection_reference_checkpoint
        )
        validation_results[reference_step] = reference_validation
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    optimizer_step = start_step
    post_update_gradient_report = None
    final_validation = initial_validation

    while optimizer_step < args.max_steps:
        batch = to_device(next_training_batch(), device)
        with torch.no_grad():
            texture, geometry = extract(model, batch["inputs"])
        output = codec(texture, geometry, straight_through=True)
        distortion, _, _ = rendered_distortion(
            model,
            output["texture_hat"],
            output["geometry_hat"],
            batch["targets"],
            mixed_precision=args.mixed_precision,
        )
        rate = output["rate_bits_per_scalar"].float()
        loss = rate + lambda_value * distortion.float()
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss: rate={float(rate)}, distortion={float(distortion)}"
            )
        (loss / args.accumulate_grad_batches).backward()
        accumulation += 1

        if accumulation < args.accumulate_grad_batches:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), args.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm: {gradient_norm}")
        post_clip_gradient_norm = global_gradient_norm(codec.parameters())
        clipping_active = float(gradient_norm) > args.gradient_clip
        optimizer.step()
        if args.fixed_resume_learning_rate is None:
            scheduler.step()
        else:
            for parameter_group in optimizer.param_groups:
                if not math.isclose(
                    parameter_group["lr"],
                    args.fixed_resume_learning_rate,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise RuntimeError("optimizer learning rate drifted from fixed value")
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0
        optimizer_step += 1

        if optimizer_step == start_step + 1:
            post_update_gradient_report = run_gradient_probe(
                codec,
                model,
                gradient_probe_batch,
                device,
                lambda_value,
                mixed_precision=args.mixed_precision,
            )
            post_update_gradient_report["pass"] = (
                post_update_gradient_report[
                    "all_codec_tensors_have_finite_nonzero_gradients"
                ]
                and post_update_gradient_report["zero_pretrained_gradients"]
            )
            write_json(
                args.output_dir
                / f"gradient_audit_after_step_{optimizer_step:06d}.json",
                post_update_gradient_report,
            )
            print(
                json.dumps(
                    {"post_update_gradient_audit": post_update_gradient_report},
                    indent=2,
                ),
                flush=True,
            )
            if not post_update_gradient_report["pass"]:
                print_gate_table(
                    {
                        "Identity equivalence": True,
                        "Rate-vs-rANS correlation": True,
                        "Finite codec gradients": False,
                        "Zero pretrained gradients": post_update_gradient_report[
                            "zero_pretrained_gradients"
                        ],
                    }
                )
                raise RuntimeError("post-update autograd audit failed")

        row = {
            "step": optimizer_step,
            "loss": float(loss.detach()),
            "rate_bits_per_scalar": float(rate.detach()),
            "distortion_255_squared_rgb_mse": float(distortion.detach()),
            "lambda_distortion": float(lambda_value * distortion.detach()),
            "pre_clip_gradient_norm": float(gradient_norm),
            "post_clip_gradient_norm": post_clip_gradient_norm,
            "gradient_norm": float(gradient_norm),
            "gradient_clip_max_norm": float(args.gradient_clip),
            "gradient_clipping_active": clipping_active,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if optimizer_step == 1 or optimizer_step % 10 == 0:
            print(json.dumps(row), flush=True)

        should_validate = optimizer_step % args.validate_every == 0 or optimizer_step == args.max_steps
        if should_validate:
            validation = validate(
                codec,
                model,
                validation_batches,
                device,
                lambda_value,
                mixed_precision=args.mixed_precision,
            )
            validation["step"] = optimizer_step
            validation["pre_clip_gradient_norm"] = float(gradient_norm)
            validation["post_clip_gradient_norm"] = post_clip_gradient_norm
            validation["gradient_clipping_active"] = clipping_active
            validation["gradient_norm"] = float(gradient_norm)
            validation["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            validation["gradient_source"] = "accumulated_training_step"
            final_validation = validation
            print(json.dumps({"validation": validation}, indent=2), flush=True)
            write_json(args.output_dir / f"validation_step_{optimizer_step:06d}.json", validation)
            validation_checkpoint_path = (
                args.output_dir / f"step_{optimizer_step:06d}.pt"
            )
            validation_checkpoint = checkpoint_payload(
                codec,
                optimizer,
                scheduler,
                optimizer_step,
                args,
                scale_report,
                trainable_report,
            )
            torch.save(validation_checkpoint, validation_checkpoint_path)
            validation_checkpoint_paths[optimizer_step] = validation_checkpoint_path
            validation_results[optimizer_step] = validation
            if validation["mean_loss"] < best_validation:
                best_validation = validation["mean_loss"]
                best_step = optimizer_step
                torch.save(validation_checkpoint, args.output_dir / "best.pt")
            if optimizer_step - best_step >= args.early_stopping_steps:
                print(f"early stopping at step {optimizer_step}; best step {best_step}", flush=True)
                torch.save(
                    checkpoint_payload(
                        codec,
                        optimizer,
                        scheduler,
                        optimizer_step,
                        args,
                        scale_report,
                        trainable_report,
                    ),
                    args.output_dir / "last.pt",
                )
                break

        if optimizer_step % args.save_every == 0 or optimizer_step == args.max_steps:
            payload = checkpoint_payload(
                codec, optimizer, scheduler, optimizer_step, args, scale_report, trainable_report
            )
            torch.save(payload, args.output_dir / "last.pt")

    smoke_checkpoint = args.output_dir / "smoke_checkpoint.pt"
    torch.save(
        checkpoint_payload(
            codec,
            optimizer,
            scheduler,
            optimizer_step,
            args,
            scale_report,
            trainable_report,
        ),
        smoke_checkpoint,
    )
    roundtrip_report = checkpoint_roundtrip_report(
        codec,
        model,
        validation_batches[0],
        smoke_checkpoint,
        device,
        mixed_precision=args.mixed_precision,
        tolerance=args.checkpoint_tolerance,
    )
    write_json(args.output_dir / "checkpoint_roundtrip.json", roundtrip_report)
    print(json.dumps({"checkpoint_roundtrip": roundtrip_report}, indent=2), flush=True)

    validation_checkpoint_diagnostics = {}
    for step, checkpoint_path in sorted(validation_checkpoint_paths.items()):
        diagnostic = checkpoint_rate_diagnostics(
            checkpoint_path,
            model,
            validation_batches,
            device,
            fixed_profile,
        )
        diagnostic["validation"] = validation_results[step]
        validation_checkpoint_diagnostics[str(step)] = diagnostic

    identity_step = (
        min(validation_results)
        if args.selection_reference_checkpoint
        else start_step
    )
    identity_diagnostic = validation_checkpoint_diagnostics[str(identity_step)]
    identity_validation = validation_results[identity_step]
    identity_actual_bytes = identity_diagnostic[
        "mean_actual_rans_payload_bytes_per_scene"
    ]

    def checkpoint_selection(step: int) -> dict:
        diagnostic = validation_checkpoint_diagnostics[str(step)]
        validation = validation_results[step]
        actual_bytes = diagnostic["mean_actual_rans_payload_bytes_per_scene"]
        bytes_saved = identity_actual_bytes - actual_bytes
        return {
            "step": step,
            "validation": validation,
            "actual_rans_payload_bits_per_scalar": diagnostic[
                "actual_rans_payload_bits_per_scalar"
            ],
            "mean_actual_rans_payload_bytes_per_scene": actual_bytes,
            "geometry_actual_rans_payload_bits_per_scalar": diagnostic["streams"]
            ["geometry"]["actual_rans_payload_bits_per_scalar"],
            "texture_actual_rans_payload_bits_per_scalar": diagnostic["streams"]
            ["texture"]["actual_rans_payload_bits_per_scalar"],
            "geometry_mean_actual_rans_payload_bytes_per_scene": diagnostic[
                "streams"
            ]["geometry"]["mean_actual_rans_payload_bytes_per_scene"],
            "texture_mean_actual_rans_payload_bytes_per_scene": diagnostic[
                "streams"
            ]["texture"]["mean_actual_rans_payload_bytes_per_scene"],
            "actual_rans_payload_bytes_saved_per_scene": bytes_saved,
            "actual_rans_rate_reduction_fraction": bytes_saved
            / identity_actual_bytes,
            "actual_rans_rate_reduction_percent": 100.0
            * bytes_saved
            / identity_actual_bytes,
            "psnr_degradation_db": identity_validation["mean_psnr"]
            - validation["mean_psnr"],
            "ssim_change": validation["mean_ssim"]
            - identity_validation["mean_ssim"],
            "lpips_change": validation["mean_lpips"]
            - identity_validation["mean_lpips"],
            "validation_rd_loss_change": validation["mean_loss"]
            - identity_validation["mean_loss"],
        }

    minimum_validation_rd_step = min(
        validation_results,
        key=lambda step: validation_results[step]["mean_loss"],
    )
    quality_constrained_steps_0p05 = [
        step
        for step, validation in validation_results.items()
        if identity_validation["mean_psnr"] - validation["mean_psnr"]
        <= 0.05
    ]
    quality_constrained_steps_0p10 = [
        step
        for step, validation in validation_results.items()
        if identity_validation["mean_psnr"] - validation["mean_psnr"]
        <= 0.10
    ]
    maximum_actual_rate_saving_step_0p05 = min(
        quality_constrained_steps_0p05,
        key=lambda step: validation_checkpoint_diagnostics[str(step)][
            "mean_actual_rans_payload_bytes_per_scene"
        ],
    )
    maximum_actual_rate_saving_step_0p10 = min(
        quality_constrained_steps_0p10,
        key=lambda step: validation_checkpoint_diagnostics[str(step)][
            "mean_actual_rans_payload_bytes_per_scene"
        ],
    )
    checkpoint_diagnostics = {
        "identity_reference": checkpoint_selection(identity_step),
        "validation_best": validation_checkpoint_diagnostics[
            str(minimum_validation_rd_step)
        ],
        f"final_step_{optimizer_step}": validation_checkpoint_diagnostics[
            str(optimizer_step)
        ],
        "all_validation_steps": validation_checkpoint_diagnostics,
        "selections": {
            "minimum_validation_rd_loss": checkpoint_selection(
                minimum_validation_rd_step
            ),
            "maximum_actual_rate_saving_psnr_loss_le_0p05_db": checkpoint_selection(
                maximum_actual_rate_saving_step_0p05
            ),
            "maximum_actual_rate_saving_psnr_loss_le_0p10_db": checkpoint_selection(
                maximum_actual_rate_saving_step_0p10
            ),
        },
        "selection_constraints": {
            "reference_step": identity_step,
            "psnr_degradation_le_0p05_db": {
                "maximum_psnr_degradation_db": 0.05,
                "eligible_steps": sorted(quality_constrained_steps_0p05),
            },
            "psnr_degradation_le_0p10_db": {
                "maximum_psnr_degradation_db": 0.10,
                "eligible_steps": sorted(quality_constrained_steps_0p10),
            },
        },
    }
    write_json(
        args.output_dir / "checkpoint_rate_diagnostics.json",
        checkpoint_diagnostics,
    )
    compact_rate_trajectory = {
        step: {
            "validation": diagnostic["validation"],
            "estimated_bits_per_scalar": diagnostic["estimated_bits_per_scalar"],
            "actual_rans_payload_bits_per_scalar": diagnostic[
                "actual_rans_payload_bits_per_scalar"
            ],
            "mean_actual_rans_payload_bytes_per_scene": diagnostic[
                "mean_actual_rans_payload_bytes_per_scene"
            ],
            "mean_timing_seconds_per_scene": diagnostic[
                "mean_timing_seconds_per_scene"
            ],
            "geometry": {
                "actual_rans_payload_bits_per_scalar": diagnostic["streams"]
                ["geometry"]["actual_rans_payload_bits_per_scalar"],
                "mean_actual_rans_payload_bytes_per_scene": diagnostic["streams"]
                ["geometry"]["mean_actual_rans_payload_bytes_per_scene"],
            },
            "texture": {
                "actual_rans_payload_bits_per_scalar": diagnostic["streams"]
                ["texture"]["actual_rans_payload_bits_per_scalar"],
                "mean_actual_rans_payload_bytes_per_scene": diagnostic["streams"]
                ["texture"]["mean_actual_rans_payload_bytes_per_scene"],
            },
        }
        for step, diagnostic in validation_checkpoint_diagnostics.items()
    }
    print(
        json.dumps(
            {
                "checkpoint_rate_diagnostics_summary": {
                    "all_validation_steps": compact_rate_trajectory,
                    "selections": checkpoint_diagnostics["selections"],
                    "selection_constraints": checkpoint_diagnostics[
                        "selection_constraints"
                    ],
                }
            },
            indent=2,
        ),
        flush=True,
    )

    malformed_report = malformed.report(args.maximum_malformed_fraction)
    write_json(args.output_dir / "malformed_examples.json", malformed_report)
    print(json.dumps({"malformed_data": malformed_report}, indent=2), flush=True)

    psnr_drop = initial_validation["mean_psnr"] - final_validation["mean_psnr"]
    rate_change = final_validation["mean_rate"] - initial_validation["mean_rate"]
    loss_change = final_validation["mean_loss"] - initial_validation["mean_loss"]
    unfavorable_lower_rate_quality = (
        rate_change < 0.0 and psnr_drop > args.maximum_smoke_psnr_drop
    )
    optimization_sanity = (
        optimizer_step == args.max_steps
        and math.isfinite(final_validation["mean_loss"])
        and final_validation["mean_loss"] <= initial_validation["mean_loss"]
        and psnr_drop <= args.maximum_smoke_psnr_drop
    )
    optimization_report = {
        "initial_step": initial_validation["step"],
        "final_step": final_validation["step"],
        "initial": initial_validation,
        "final": final_validation,
        "rate_change_bits_per_scalar": rate_change,
        "psnr_drop_db": psnr_drop,
        "ssim_change": final_validation["mean_ssim"] - initial_validation["mean_ssim"],
        "lpips_change": final_validation["mean_lpips"] - initial_validation["mean_lpips"],
        "rd_loss_change": loss_change,
        "maximum_allowed_psnr_drop_db": args.maximum_smoke_psnr_drop,
        "lower_rate_with_substantially_worse_quality": unfavorable_lower_rate_quality,
        "pass": optimization_sanity,
    }

    finite_codec_gradients = bool(
        post_update_gradient_report
        and post_update_gradient_report["pass"]
    )
    zero_pretrained_gradients = bool(
        initial_gradient_report["zero_pretrained_gradients"]
        and post_update_gradient_report
        and post_update_gradient_report["zero_pretrained_gradients"]
    )
    hard_validation = (
        initial_validation["quantization_mode"] == "hard_round_clamp"
        and final_validation["quantization_mode"] == "hard_round_clamp"
        and not initial_validation["straight_through_used"]
        and not final_validation["straight_through_used"]
    )
    gates = {
        "Identity equivalence": bool(identity_report["pass"]),
        "Rate-vs-rANS correlation": bool(scale_report["pass"]),
        "Finite codec gradients": finite_codec_gradients,
        "Zero pretrained gradients": zero_pretrained_gradients,
        "Hard validation quantization": hard_validation,
        "Checkpoint round-trip": bool(roundtrip_report["pass"]),
        "Malformed-data handling": bool(malformed_report["pass"]),
        "Optimization sanity": optimization_sanity,
    }
    implementation_gate_names = tuple(gates)[:-1]
    all_implementation_gates_pass = all(gates[name] for name in implementation_gate_names)
    recommendation = "Do not submit Stage B."
    if all_implementation_gates_pass and unfavorable_lower_rate_quality:
        recommendation += (
            " Run one additional 300-step smoke at 2*lambda0; consider 4*lambda0 "
            "only if 2*lambda0 remains too rate-heavy. Do not launch either automatically."
        )
    elif all(gates.values()):
        recommendation += " Review the complete smoke report before manually approving Stage B."
    elif not all_implementation_gates_pass:
        recommendation += " Diagnose failed implementation/instrumentation gates before more training."

    summary = {
        "completed_step": optimizer_step,
        "best_validation_step": best_step,
        "best_validation_loss": best_validation,
        "lambda": lambda_value,
        "lambda_multiplier": args.lambda_multiplier,
        "R0_bits_per_scalar": scale_report["R0_bits_per_scalar"],
        "D0_255_squared_rgb_mse": scale_report["D0_255_squared_rgb_mse"],
        "lambda0": scale_report["lambda0"],
        "lambda_sweep": scale_report["lambda_sweep"],
        "history": history,
        "identity_equivalence": identity_report,
        "rate_estimator_validation": scale_report,
        "initial_gradient_audit": initial_gradient_report,
        "post_update_gradient_audit": post_update_gradient_report,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "checkpoint_roundtrip": roundtrip_report,
        "checkpoint_rate_diagnostics": checkpoint_diagnostics,
        "malformed_data": malformed_report,
        "optimization_sanity": optimization_report,
        "smoke_gates": gates,
        "all_implementation_gates_pass": all_implementation_gates_pass,
        "all_smoke_gates_pass": all(gates.values()),
        "stage_b_submitted": False,
        "recommendation": recommendation,
    }
    write_json(args.output_dir / "training_summary.json", summary)
    write_json(
        args.output_dir / "smoke_gate.json",
        {
            "gates": gates,
            "all_implementation_gates_pass": all_implementation_gates_pass,
            "all_smoke_gates_pass": all(gates.values()),
            "optimization": optimization_report,
            "recommendation": recommendation,
        },
    )
    print(json.dumps({"optimization_sanity": optimization_report}, indent=2), flush=True)
    print_gate_table(gates)
    print(recommendation, flush=True)
    malformed.restore()


if __name__ == "__main__":
    main()
