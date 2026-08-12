#!/usr/bin/env python3
"""Profile GlobalSplat stages on one deterministic RE10K example.

Reports tokenize / slot encoder / Gaussian decoder / gsplat render latency,
CUDA peak memory, tensor input/output sizes, and MAC estimates.  Transformer
MACs are computed analytically; render MACs are undefined for gsplat's
data-dependent rasterizer and are therefore reported as null.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from hydra import compose, initialize_config_dir

from globalsplat.main import build_datamodule, build_model
from globalsplat.model.rendering import render_static_batched


ROOT = Path(__file__).resolve().parents[2]


def tensor_bytes(obj: Any, seen: set[tuple] | None = None) -> int:
    """Count unique tensor storage bytes recursively (aliases count once)."""
    if seen is None:
        seen = set()
    if torch.is_tensor(obj):
        key = (obj.device.type, obj.device.index, obj.untyped_storage().data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(tensor_bytes(v, seen) for v in obj)
    if is_dataclass(obj):
        return sum(tensor_bytes(getattr(obj, f.name), seen) for f in fields(obj))
    return 0


def to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


def load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw)
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("perceptual_loss")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def cuda_latency_ms(fn: Callable[[], Any], warmup: int, repeats: int) -> tuple[Any, list[float]]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return out, values


def cuda_peak(fn: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    mib = 1024.0 ** 2
    stats = {
        "baseline_allocated_mib": base_alloc / mib,
        "baseline_reserved_mib": base_reserved / mib,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / mib,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / mib,
        "incremental_peak_allocated_mib":
            (torch.cuda.max_memory_allocated() - base_alloc) / mib,
    }
    return out, stats


def profiler_macs(fn: Callable[[], Any]) -> float:
    """Profiler-visible MAC lower bound; fused/custom kernels may report zero FLOPs."""
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        fn()
        torch.cuda.synchronize()
    flops = sum(float(getattr(event, "flops", 0) or 0) for event in prof.key_averages())
    return flops / 2.0


def theoretical_macs(model, context_views: int, height: int, width: int) -> dict[str, float | None]:
    ph, pw = model.patch_size
    patches = (height // ph) * (width // pw)
    memory = context_views * patches + model.slot_encoder.mem_reg_amount
    slots = (
        model.slot_encoder.num_aux_queries
        + model.scene_tokens.shape[0]
        + 1
        + model.slot_encoder.slot_reg_amount
    )
    d = model.dim_latents
    dt = model.dim_rgb_feat + model.dim_rays
    heads = model.slot_encoder.heads
    del heads  # head splitting does not change dense-attention MAC count.
    ratio = 4
    rounds = model.slot_encoder.rounds
    self_blocks = model.slot_encoder.slot_calib_layers_per_round

    # Per input frame: RGB/ray patch projections plus PatchViewPE MLP/projection.
    tokenize_per_frame = patches * ((ph * pw * 3) * model.dim_rgb_feat)
    tokenize_per_frame += patches * ((ph * pw * 6) * model.dim_rays)
    tokenize_per_frame += 4 * 64 + 64 * 64 + (64 + 27) * model.dim_rays

    # One stream round: Q/K/V/out, QK^T + AV, cross-attn MLP, then self blocks.
    cross = slots * d * d + 2 * memory * dt * d + slots * d * d
    cross += 2 * slots * memory * d
    cross += 2 * slots * d * (ratio * d)
    self_one = 4 * slots * d * d + 2 * slots * slots * d
    self_one += 2 * slots * d * (ratio * d)
    stream_round = cross + self_blocks * self_one
    pair_adapter = 4 * slots * d * d
    encoder = 2 * slots * d * d + rounds * (2 * stream_round + pair_adapter)

    # Decoder readout linears: 16*(3 + 10 + SH(48) + gate(1)) = 992 outputs/token.
    decoder_outputs = model.gaussian_decoder.M_max * (
        3 + 10 + model.gaussian_decoder.color_dim + 1
    )
    decoder = model.scene_tokens.shape[0] * d * decoder_outputs
    return {
        "tokenize_total": float(context_views * tokenize_per_frame),
        "tokenize_per_input_frame": float(tokenize_per_frame),
        "encoder_total": float(encoder),
        "encoder_per_input_frame_amortized": float(encoder / context_views),
        "gaussian_decoder_total": float(decoder),
        "gaussian_decoder_per_input_frame_amortized": float(decoder / context_views),
        "render": None,
    }


def summarize_stage(
    name: str,
    fn: Callable[[], Any],
    stage_input: Any,
    warmup: int,
    repeats: int,
    macs_theoretical: float | None,
) -> tuple[Any, dict[str, Any]]:
    output, latencies = cuda_latency_ms(fn, warmup, repeats)
    del output
    output, memory = cuda_peak(fn)
    visible_macs = profiler_macs(fn)
    result = {
        "stage": name,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "repeats": repeats,
        },
        "cuda_memory": memory,
        "input_mib": tensor_bytes(stage_input) / (1024.0 ** 2),
        "output_mib": tensor_bytes(output) / (1024.0 ** 2),
        "theoretical_macs": macs_theoretical,
        "profiler_visible_macs_lower_bound": visible_macs,
    }
    return output, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default="/data3/local_datasets/re10k")
    ap.add_argument("--experiment", default="re10k_32k")
    ap.add_argument("--context", type=int, choices=(12, 24, 36), default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    overrides = [
        f"+experiment={args.experiment}",
        f"dataset=re10k_eval_ctx{args.context}",
        "mode=test",
        "optimizer.batch_size=1",
        f"optimizer.num_workers={args.num_workers}",
        f"dataset.dataset_roots=['{args.dataset_root}']",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg = compose(config_name="main", overrides=overrides)

    device = torch.device("cuda")
    model = build_model(cfg.model).to(device).eval()
    model.set_stage(int(cfg.curriculum.final_stage), mix=1.0)
    load_weights(model, checkpoint)

    dm, _ = build_datamodule(cfg)
    dm.setup("test")
    batch = next(iter(dm.test_dataloader()))
    batch = to_device(batch, device)
    inputs, targets = batch["inputs"], batch["targets"]
    b, views, _, height, width = inputs["images"].shape
    if b != 1:
        raise RuntimeError(f"expected batch size 1, got {b}")

    theory = theoretical_macs(model, views, height, width)
    results = []
    with torch.inference_mode():
        content, row = summarize_stage(
            "tokenize", lambda: model._tokenize(
                inputs["images"], inputs["intrinsic"], inputs["c2w"]
            ), inputs, args.warmup, args.repeats, theory["tokenize_total"]
        )
        results.append(row)

        state = model.scene_tokens.unsqueeze(0).expand(b, -1, -1)
        encoded_all, row = summarize_stage(
            "dual_stream_encoder",
            lambda: model.slot_encoder(content, state=state),
            (content, state), args.warmup, args.repeats, theory["encoder_total"],
        )
        results.append(row)
        encoded = encoded_all[1]

        decoded, row = summarize_stage(
            "gaussian_decoder", lambda: model.gaussian_decoder(encoded),
            encoded, args.warmup, args.repeats, theory["gaussian_decoder_total"],
        )
        results.append(row)
        from globalsplat.model.types import Gaussians
        gaussians = Gaussians(
            means=decoded[0], rotations=decoded[1], scales=decoded[2],
            sh=decoded[3], opacities=decoded[4], reg=decoded[5],
        )

        rendered, row = summarize_stage(
            "gsplat_render_3_targets",
            lambda: render_static_batched(gaussians, targets, render_depth=False),
            (gaussians, targets), args.warmup, args.repeats, None,
        )
        results.append(row)

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "checkpoint": str(checkpoint),
        "context_views": views,
        "target_views": int(targets["images"].shape[1]),
        "image_shape": [height, width],
        "gaussians": gaussians.num_gaussians,
        "model_parameter_bytes": tensor_bytes(dict(model.named_parameters())),
        "theoretical_macs": theory,
        "mac_note": (
            "1 MAC = multiply-accumulate = 2 FLOPs. Theoretical transformer/readout "
            "MACs exclude normalization, activations, reductions and ray math. Profiler "
            "MACs are a lower bound because fused SDPA and gsplat CUDA kernels may not "
            "publish FLOP metadata. Render MAC is data-dependent and intentionally null."
        ),
        "stages": results,
    }
    output = Path(args.output or f"outputs/profile_re10k_ctx{views}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
