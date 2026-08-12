#!/usr/bin/env python3
"""Aggregate per-stage GlobalSplat profiling statistics over RE10K scenes."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
from hydra import compose, initialize_config_dir

from globalsplat.main import build_datamodule, build_model
from globalsplat.model.rendering import render_static_batched
from globalsplat.model.types import Gaussians
from profile_re10k_pipeline import load_weights, tensor_bytes, theoretical_macs, to_device


ROOT = Path(__file__).resolve().parents[2]
MIB = 1024.0 ** 2


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


@torch.inference_mode()
def measure(fn: Callable[[], Any], stage_input: Any) -> tuple[Any, dict[str, float]]:
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return output, {
        "latency_ms": float(start.elapsed_time(end)),
        "baseline_allocated_mib": base_alloc / MIB,
        "baseline_reserved_mib": base_reserved / MIB,
        "peak_allocated_mib": peak_alloc / MIB,
        "peak_reserved_mib": peak_reserved / MIB,
        "incremental_peak_allocated_mib": (peak_alloc - base_alloc) / MIB,
        "input_mib": tensor_bytes(stage_input) / MIB,
        "output_mib": tensor_bytes(output) / MIB,
    }


@torch.inference_mode()
def run_scene(model, batch: dict[str, Any]) -> dict[str, dict[str, float]]:
    inputs, targets = batch["inputs"], batch["targets"]
    b, views = inputs["images"].shape[:2]

    content, tokenize = measure(
        lambda: model._tokenize(inputs["images"], inputs["intrinsic"], inputs["c2w"]),
        inputs,
    )
    state = model.scene_tokens.unsqueeze(0).expand(b, -1, -1)
    encoded_all, encoder = measure(
        lambda: model.slot_encoder(content, state=state), (content, state)
    )
    encoded = encoded_all[1]
    decoded, decoder = measure(lambda: model.gaussian_decoder(encoded), encoded)
    gaussians = Gaussians(
        means=decoded[0], rotations=decoded[1], scales=decoded[2],
        sh=decoded[3], opacities=decoded[4], reg=decoded[5],
    )
    rendered, render = measure(
        lambda: render_static_batched(gaussians, targets, render_depth=False),
        (gaussians, targets),
    )

    target_views = int(targets["images"].shape[1])
    tokenize["latency_ms_per_input_frame"] = tokenize["latency_ms"] / views
    tokenize["input_mib_per_input_frame"] = tokenize["input_mib"] / views
    tokenize["output_mib_per_input_frame"] = tokenize["output_mib"] / views
    for row in (encoder, decoder):
        row["latency_ms_per_input_frame_amortized"] = row["latency_ms"] / views
        row["output_mib_per_input_frame_amortized"] = row["output_mib"] / views
    render["latency_ms_per_target_frame"] = render["latency_ms"] / target_views
    render["output_mib_per_target_frame"] = render["output_mib"] / target_views
    del rendered
    return {
        "tokenize": tokenize,
        "dual_stream_encoder": encoder,
        "gaussian_decoder": decoder,
        "gsplat_render_3_targets": render,
    }


def aggregate(records: list[dict[str, dict[str, float]]]) -> dict[str, Any]:
    result = {}
    for stage in records[0]:
        metric_names = records[0][stage].keys()
        result[stage] = {
            metric: stats([record[stage][metric] for record in records])
            for metric in metric_names
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default="/data3/local_datasets/re10k")
    ap.add_argument("--experiment", default="re10k_32k")
    ap.add_argument("--context", type=int, choices=(12, 24, 36), default=12)
    ap.add_argument("--num-scenes", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    overrides = [
        f"+experiment={args.experiment}", f"dataset=re10k_eval_ctx{args.context}",
        "mode=test", "optimizer.batch_size=1",
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
    iterator = iter(dm.test_dataloader())

    first = to_device(next(iterator), device)
    for _ in range(args.warmup):
        run_scene(model, first)
    torch.cuda.synchronize()

    records = [run_scene(model, first)]
    while len(records) < args.num_scenes:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch = to_device(batch, device)
        records.append(run_scene(model, batch))
        if len(records) % 25 == 0:
            print(f"profiled {len(records)}/{args.num_scenes} scenes", flush=True)
        del batch

    views = int(first["inputs"]["images"].shape[1])
    height, width = map(int, first["inputs"]["images"].shape[-2:])
    target_views = int(first["targets"]["images"].shape[1])
    theory = theoretical_macs(model, views, height, width)
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "checkpoint": str(checkpoint),
        "sample_count": len(records),
        "context_views": views,
        "target_views": target_views,
        "image_shape": [height, width],
        "gaussians": model.scene_tokens.shape[0] * (1 << model.gaussian_decoder.stage),
        "theoretical_macs": theory,
        "mac_note": (
            "Transformer/readout MACs are shape-derived and constant across scenes; "
            "per-input-frame values are amortized totals. gsplat render MAC is null "
            "because rasterization work is visibility/tile dependent."
        ),
        "statistics_note": "std is sample standard deviation (N-1). Latency excludes data loading and CPU-to-GPU transfer.",
        "stages": aggregate(records),
    }
    output = Path(args.output or f"outputs/profile/re10k_ctx{views}_n{len(records)}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
