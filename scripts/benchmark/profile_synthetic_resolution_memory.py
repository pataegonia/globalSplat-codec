#!/usr/bin/env python3
"""Sweep synthetic image resolutions and report inference CUDA peak memory.

Each resolution runs in a fresh subprocess so a CUDA OOM cannot poison later
measurements.  No dataset is loaded.  The worker profiles the logical inference
stages separately and then profiles the complete model + renderer path.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir

from globalsplat.main import build_model
from globalsplat.model.rendering import render_static_batched
from globalsplat.model.types import Gaussians

DEFAULT_RESOLUTIONS = (256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048)


def parse_resolution(value: str | int) -> tuple[int, int]:
    text = str(value).lower().replace(" ", "")
    if "x" in text:
        h_text, w_text = text.split("x", 1)
        height, width = int(h_text), int(w_text)
    else:
        height = width = int(text)
    if height <= 0 or width <= 0:
        raise ValueError(f"resolution must be positive, got {value!r}")
    return height, width


def resolution_label(shape: tuple[int, int]) -> str:
    height, width = shape
    return str(height) if height == width else f"{height}x{width}"


def tensor_storage_bytes(obj: Any, seen: set[tuple[Any, ...]] | None = None) -> int:
    """Count unique tensor storage, so expanded target placeholders stay tiny."""
    if seen is None:
        seen = set()
    if torch.is_tensor(obj):
        storage = obj.untyped_storage()
        key = (obj.device.type, obj.device.index, storage.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return int(storage.nbytes())
    if isinstance(obj, dict):
        return sum(tensor_storage_bytes(value, seen) for value in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(tensor_storage_bytes(value, seen) for value in obj)
    if is_dataclass(obj):
        return sum(tensor_storage_bytes(getattr(obj, field.name), seen) for field in fields(obj))
    return 0


def load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw)
    if any(key.startswith("model.") for key in state):
        state = {
            key[len("model."):]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def autocast_context(precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def process_rss_bytes() -> int:
    """Best-effort current RSS without requiring psutil."""
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        pass
    try:
        statm = Path(f"/proc/{os.getpid()}/statm").read_text().split()
        return int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, IndexError, ValueError, AttributeError):
        return 0


class RssSampler:
    """Sample process RSS while NumPy/CompressAI work runs on the CPU."""

    def __init__(self, interval_seconds: float = 0.002):
        self.interval_seconds = interval_seconds
        self.before = process_rss_bytes()
        self.peak = self.before
        self.after = self.before
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak = max(self.peak, process_rss_bytes())

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.after = process_rss_bytes()
        self.peak = max(self.peak, self.after)


def make_camera_poses(
    batch_size: int,
    views: int,
    device: torch.device,
    *,
    phase: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create normalized, forward-facing cameras around the world origin."""
    angles = torch.linspace(-0.35, 0.35, views, device=device) + phase
    poses = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch_size, views, 1, 1)
    poses[:, :, 0, 3] = 0.35 * torch.sin(angles)
    poses[:, :, 1, 3] = 0.05 * torch.cos(angles)
    # Rotations are identity, so the inverse is an exact translation negation.
    # Avoiding torch.linalg.inv also keeps cuBLAS initialization out of the
    # synthetic-input allocation measurement.
    world_to_camera = poses.clone()
    world_to_camera[:, :, :3, 3].neg_()
    return poses, world_to_camera


def make_intrinsics(
    batch_size: int,
    views: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    intrinsics = torch.zeros(batch_size, views, 3, 3, device=device)
    focal = 0.8 * max(height, width)
    intrinsics[..., 0, 0] = focal
    intrinsics[..., 1, 1] = focal
    intrinsics[..., 0, 2] = width / 2.0
    intrinsics[..., 1, 2] = height / 2.0
    intrinsics[..., 2, 2] = 1.0
    return intrinsics


def make_synthetic_batch(
    *,
    batch_size: int,
    context_views: int,
    target_views: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    images = torch.rand(batch_size, context_views, 3, height, width, device=device)
    context_c2w, context_w2c = make_camera_poses(batch_size, context_views, device)
    target_c2w, target_w2c = make_camera_poses(
        batch_size, target_views, device, phase=0.025
    )
    inputs = {
        "images": images,
        "intrinsic": make_intrinsics(
            batch_size, context_views, height, width, device
        ),
        "c2w": context_c2w,
        "extrinsic": context_w2c,
    }

    # The renderer only uses target_images.shape.  An expanded one-element view
    # models production inference without allocating unavailable ground truth.
    target_image_shape = (batch_size, target_views, 3, height, width)
    target_images = torch.empty(1, device=device).expand(target_image_shape)
    targets = {
        "images": target_images,
        "intrinsic": make_intrinsics(
            batch_size, target_views, height, width, device
        ),
        "c2w": target_c2w,
        "extrinsic": target_w2c,
    }
    return inputs, targets


def cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def profile_cuda_stage(
    name: str,
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
) -> tuple[Any | None, dict[str, Any]]:
    """Profile one stage, returning a partial OOM/error row when it fails."""
    phase = "warmup"
    baseline_allocated = baseline_reserved = 0
    rss_sampler: RssSampler | None = None
    try:
        for _ in range(warmup):
            temporary = function()
            torch.cuda.synchronize(device)
            del temporary

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)

        phase = "measured_call"
        rss_sampler = RssSampler()
        rss_sampler.start()
        started = time.perf_counter()
        output = function()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        rss_sampler.stop()
        memory = cuda_memory_snapshot(device)
        row = {
            "stage": name,
            "status": "success",
            "seconds": seconds,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "allocated_after_bytes": memory["allocated_bytes"],
            "reserved_after_bytes": memory["reserved_bytes"],
            "peak_allocated_bytes": memory["peak_allocated_bytes"],
            "incremental_peak_allocated_bytes": max(
                0, memory["peak_allocated_bytes"] - baseline_allocated
            ),
            "peak_reserved_bytes": memory["peak_reserved_bytes"],
            "incremental_peak_reserved_bytes": max(
                0, memory["peak_reserved_bytes"] - baseline_reserved
            ),
            "process_rss_before_bytes": rss_sampler.before,
            "process_rss_after_bytes": rss_sampler.after,
            "process_peak_rss_bytes": rss_sampler.peak,
            "incremental_process_peak_rss_bytes": max(
                0, rss_sampler.peak - rss_sampler.before
            ),
            "output_storage_bytes": tensor_storage_bytes(output),
        }
        return output, row
    except Exception as exc:
        if rss_sampler is not None and rss_sampler._thread.is_alive():
            rss_sampler.stop()
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        try:
            memory = cuda_memory_snapshot(device)
        except Exception:
            memory = {
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
            }
        row = {
            "stage": name,
            "status": "oom" if is_oom else "error",
            "failed_phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": memory["peak_allocated_bytes"],
            "peak_reserved_bytes": memory["peak_reserved_bytes"],
            "incremental_peak_allocated_bytes": max(
                0, memory["peak_allocated_bytes"] - baseline_allocated
            ),
            "incremental_peak_reserved_bytes": max(
                0, memory["peak_reserved_bytes"] - baseline_reserved
            ),
            "process_rss_before_bytes": rss_sampler.before if rss_sampler else process_rss_bytes(),
            "process_rss_after_bytes": rss_sampler.after if rss_sampler else process_rss_bytes(),
            "process_peak_rss_bytes": rss_sampler.peak if rss_sampler else process_rss_bytes(),
            "incremental_process_peak_rss_bytes": (
                max(0, rss_sampler.peak - rss_sampler.before) if rss_sampler else 0
            ),
        }
        return None, row


def codec_roundtrip(model, encoded_patch):
    """Run the same in-memory latent codec path used by GlobalSplat.forward."""
    from globalsplat.compression import (
        decode_scene_latents_from_bytes,
        encode_scene_latents_to_bytes,
    )

    texture, geometry = encoded_patch
    if texture.shape[0] != 1:
        raise ValueError("latent codec currently expects batch size 1")
    bitstream, stats = encode_scene_latents_to_bytes(
        texture,
        geometry,
        bits=model.latent_codec_bits,
        texture_bits=model.latent_codec_texture_bits or model.latent_codec_bits,
        geometry_bits=model.latent_codec_geometry_bits or model.latent_codec_bits,
        backend=model.latent_codec_backend,
        scale_mode=model.latent_codec_scale_mode,
        clip_percentile=model.latent_codec_clip_percentile,
        fixed_cdf_profile=model.latent_codec_fixed_profile,
        verify_symbols=model.latent_codec_verify_symbols,
    )
    texture_hat, geometry_hat, decode_stats = decode_scene_latents_from_bytes(
        bitstream,
        device=texture.device,
        fixed_cdf_profile=model.latent_codec_fixed_profile,
        return_stats=True,
        verify_symbols=model.latent_codec_verify_symbols,
    )
    stats.update(decode_stats)
    restored = (
        texture_hat.to(texture.dtype),
        geometry_hat.to(geometry.dtype),
    )
    return restored, stats


def scalar_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stats.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def profile_resolution(
    args: argparse.Namespace,
    shape: tuple[int, int],
    progress_output: Path | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    height, width = shape

    overrides = [f"+experiment={args.experiment}"]
    if args.codec:
        overrides.append(f"+codec={args.codec}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg = compose(config_name="main", overrides=overrides)

    model = build_model(cfg.model).to(device).eval()
    model.set_stage(int(cfg.curriculum.final_stage), mix=1.0)
    if args.checkpoint:
        load_weights(model, Path(args.checkpoint).expanduser().resolve())

    ph, pw = model.patch_size
    if height % ph or width % pw:
        raise ValueError(
            f"resolution {height}x{width} must be divisible by patch size {ph}x{pw}"
        )

    report: dict[str, Any] = {
        "status": "running",
        "resolution": [height, width],
        "batch_size": args.batch_size,
        "context_views": args.context_views,
        "target_views": args.target_views,
        "precision": args.precision,
        "experiment": args.experiment,
        "codec_preset": args.codec,
        "codec_enabled": bool(
            model.latent_codec_bits or model.latent_codec_texture_bits
        ),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None,
        "device": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "model_parameter_storage_bytes": tensor_storage_bytes(dict(model.named_parameters())),
        "model_cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "patch_size": [ph, pw],
        "patches_per_view": (height // ph) * (width // pw),
        "stages": [],
    }

    def append_stage(row: dict[str, Any]) -> None:
        report["stages"].append(row)
        if progress_output is not None:
            progress_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if progress_output is not None:
        progress_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with torch.inference_mode():
        batch, row = profile_cuda_stage(
            "synthetic_inputs",
            lambda: make_synthetic_batch(
                batch_size=args.batch_size,
                context_views=args.context_views,
                target_views=args.target_views,
                height=height,
                width=width,
                device=device,
            ),
            device=device,
            warmup=0,
        )
        append_stage(row)
        if batch is None:
            report["status"] = row["status"]
            return report
        inputs, targets = batch
        report["synthetic_input_storage_bytes"] = tensor_storage_bytes(batch)

        def measured(function: Callable[[], Any]) -> Any:
            with autocast_context(args.precision):
                return function()

        content, row = profile_cuda_stage(
            "view_tokenizer",
            lambda: measured(
                lambda: model._tokenize(
                    inputs["images"], inputs["intrinsic"], inputs["c2w"]
                )
            ),
            device=device,
            warmup=args.warmup,
        )
        append_stage(row)
        if content is None:
            report["status"] = row["status"]
            return report

        state = model.scene_tokens.unsqueeze(0).expand(args.batch_size, -1, -1)
        encoded_all, row = profile_cuda_stage(
            "dual_stream_slot_encoder",
            lambda: measured(lambda: model.slot_encoder(content, state=state)),
            device=device,
            warmup=args.warmup,
        )
        append_stage(row)
        if encoded_all is None:
            report["status"] = row["status"]
            return report
        encoded = encoded_all[1]

        codec_enabled = bool(
            model.latent_codec_bits or model.latent_codec_texture_bits
        )
        if codec_enabled:
            codec_output, row = profile_cuda_stage(
                "latent_codec_roundtrip",
                lambda: codec_roundtrip(model, encoded),
                device=device,
                warmup=args.warmup,
            )
            append_stage(row)
            if codec_output is None:
                report["status"] = row["status"]
                return report
            encoded, codec_stats = codec_output
            report["codec_stats"] = scalar_stats(codec_stats)

        decoded, row = profile_cuda_stage(
            "gaussian_decoder",
            lambda: measured(lambda: model.gaussian_decoder(encoded)),
            device=device,
            warmup=args.warmup,
        )
        append_stage(row)
        if decoded is None:
            report["status"] = row["status"]
            return report
        gaussians = Gaussians(
            means=decoded[0],
            rotations=decoded[1],
            scales=decoded[2],
            sh=decoded[3],
            opacities=decoded[4],
            reg=decoded[5],
        )
        report["num_gaussians"] = gaussians.num_gaussians

        # A complete network-only row is useful even on hosts where gsplat is
        # not installed, and checks the peak produced by normal Python tensor
        # lifetimes rather than the manually retained stage intermediates.
        del content, state, encoded_all, encoded, decoded
        gc.collect()
        torch.cuda.empty_cache()

        predicted, row = profile_cuda_stage(
            "model_end_to_end",
            lambda: measured(lambda: model(inputs)),
            device=device,
            warmup=args.warmup,
        )
        append_stage(row)
        if predicted is None:
            report["status"] = row["status"]
            return report
        del predicted
        gc.collect()
        torch.cuda.empty_cache()

        rendered, row = profile_cuda_stage(
            "gsplat_renderer",
            lambda: measured(
                lambda: render_static_batched(
                    gaussians, targets, render_depth=args.render_depth
                )
            ),
            device=device,
            warmup=args.warmup,
        )
        append_stage(row)
        if rendered is None:
            report["status"] = row["status"]
            return report
        del rendered, gaussians
        gc.collect()
        torch.cuda.empty_cache()

        def end_to_end():
            predicted = model(inputs)
            return render_static_batched(
                predicted, targets, render_depth=args.render_depth
            )

        rendered, row = profile_cuda_stage(
            "model_and_renderer_end_to_end",
            lambda: measured(end_to_end),
            device=device,
            warmup=args.warmup,
        )
        append_stage(row)
        if rendered is None:
            report["status"] = row["status"]
            return report
        del rendered

    report["status"] = "success"
    return report


def bytes_to_gib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / (1024.0 ** 3)


def tail_text(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "resolution",
        "resolution_status",
        "stage",
        "stage_status",
        "seconds",
        "baseline_allocated_gib",
        "peak_allocated_gib",
        "incremental_peak_allocated_gib",
        "peak_reserved_gib",
        "incremental_peak_reserved_gib",
        "process_rss_before_gib",
        "process_peak_rss_gib",
        "incremental_process_peak_rss_gib",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in report["results"]:
            label = resolution_label(tuple(result["resolution"]))
            for stage in result.get("stages", []):
                writer.writerow({
                    "resolution": label,
                    "resolution_status": result["status"],
                    "stage": stage["stage"],
                    "stage_status": stage["status"],
                    "seconds": stage.get("seconds"),
                    "baseline_allocated_gib": bytes_to_gib(stage.get("baseline_allocated_bytes")),
                    "peak_allocated_gib": bytes_to_gib(stage.get("peak_allocated_bytes")),
                    "incremental_peak_allocated_gib": bytes_to_gib(
                        stage.get("incremental_peak_allocated_bytes")
                    ),
                    "peak_reserved_gib": bytes_to_gib(stage.get("peak_reserved_bytes")),
                    "incremental_peak_reserved_gib": bytes_to_gib(
                        stage.get("incremental_peak_reserved_bytes")
                    ),
                    "process_rss_before_gib": bytes_to_gib(
                        stage.get("process_rss_before_bytes")
                    ),
                    "process_peak_rss_gib": bytes_to_gib(
                        stage.get("process_peak_rss_bytes")
                    ),
                    "incremental_process_peak_rss_gib": bytes_to_gib(
                        stage.get("incremental_process_peak_rss_bytes")
                    ),
                    "error": stage.get("error"),
                })


def print_result_summary(result: dict[str, Any]) -> None:
    label = resolution_label(tuple(result["resolution"]))
    print(f"[{label}] {result['status']}")
    for stage in result.get("stages", []):
        peak = bytes_to_gib(stage.get("peak_allocated_bytes"))
        delta = bytes_to_gib(stage.get("incremental_peak_allocated_bytes"))
        rss_delta = bytes_to_gib(stage.get("incremental_process_peak_rss_bytes"))
        peak_text = "-" if peak is None else f"{peak:.3f} GiB"
        delta_text = "-" if delta is None else f"+{delta:.3f} GiB"
        rss_text = "-" if rss_delta is None else f"+{rss_delta:.3f} GiB"
        print(
            f"  {stage['stage']:<30} {stage['status']:<7} "
            f"cuda_peak={peak_text} cuda_delta={delta_text} rss_delta={rss_text}"
        )


def worker_command(args: argparse.Namespace, shape: tuple[int, int], output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-resolution",
        resolution_label(shape),
        "--_worker-output",
        str(output),
        "--experiment",
        args.experiment,
        "--context-views",
        str(args.context_views),
        "--target-views",
        str(args.target_views),
        "--batch-size",
        str(args.batch_size),
        "--precision",
        args.precision,
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.codec:
        command.extend(["--codec", args.codec])
    if args.render_depth:
        command.append("--render-depth")
    return command


def run_parent(args: argparse.Namespace) -> None:
    shapes = [parse_resolution(value) for value in args.resolutions]
    if args.checkpoint and not Path(args.checkpoint).expanduser().is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with tempfile.TemporaryDirectory(prefix="synthetic_mem_", dir=output.parent) as temp_dir:
        for index, shape in enumerate(shapes):
            label = resolution_label(shape)
            worker_output = Path(temp_dir) / f"{index:03d}_{label}.json"
            print(f"profiling {label} in an isolated process...", flush=True)
            try:
                completed = subprocess.run(
                    worker_command(args, shape, worker_output),
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=args.resolution_timeout or None,
                )
            except subprocess.TimeoutExpired as exc:
                if worker_output.is_file():
                    result = json.loads(worker_output.read_text(encoding="utf-8"))
                else:
                    result = {"resolution": list(shape), "stages": []}
                result["status"] = "timeout"
                result["timeout_seconds"] = args.resolution_timeout
                result["stdout"] = tail_text(exc.stdout)
                result["stderr"] = tail_text(exc.stderr)
                results.append(result)
                print_result_summary(result)
                continue
            if worker_output.is_file():
                result = json.loads(worker_output.read_text(encoding="utf-8"))
            else:
                result = {
                    "status": "worker_crash",
                    "resolution": list(shape),
                    "stages": [],
                    "returncode": completed.returncode,
                    "stdout": tail_text(completed.stdout),
                    "stderr": tail_text(completed.stderr),
                }
            results.append(result)
            print_result_summary(result)

    report = {
        "measurement": (
            "steady-state inference CUDA allocator peak"
            if args.warmup
            else "first-call inference CUDA allocator peak"
        ),
        "warmup_calls_per_stage": args.warmup,
        "memory_units": "bytes",
        "peak_allocated_definition": "absolute PyTorch allocator peak during the stage",
        "incremental_peak_definition": "peak allocated minus live allocated memory at stage entry",
        "process_peak_definition": "peak process RSS sampled every 2 ms during the stage",
        "target_images": "one-element expanded placeholder; no ground-truth image allocation",
        "worker_isolation": "one fresh subprocess per resolution",
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_output = output.with_suffix(".csv")
    write_csv(report, csv_output)
    print(f"saved JSON: {output}")
    print(f"saved CSV:  {csv_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile dataset-free GlobalSplat inference peak memory by resolution."
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=[str(value) for value in DEFAULT_RESOLUTIONS],
        help="Square side lengths or HxW values (default: 256 ... 2048).",
    )
    parser.add_argument("--experiment", default="re10k_32k")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--codec",
        default=None,
        help="Optional Hydra codec preset, e.g. final_g9_t7_g32_t32.",
    )
    parser.add_argument("--context-views", type=int, default=12)
    parser.add_argument("--target-views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up calls per stage before the measured call (default: 1).",
    )
    parser.add_argument(
        "--resolution-timeout",
        type=float,
        default=0,
        help="Seconds allowed per resolution worker; 0 disables the timeout.",
    )
    parser.add_argument("--render-depth", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/synthetic_resolution_memory.json",
    )
    parser.add_argument("--_worker-resolution", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.context_views <= 0 or args.target_views <= 0:
        raise SystemExit("batch size and view counts must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.resolution_timeout < 0:
        raise SystemExit("--resolution-timeout must be >= 0")

    if args._worker_resolution is None:
        run_parent(args)
        return

    if not args._worker_output:
        raise SystemExit("internal worker requires --_worker-output")
    shape = parse_resolution(args._worker_resolution)
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        progress_output = Path(args._worker_output)
        progress_output.parent.mkdir(parents=True, exist_ok=True)
        result = profile_resolution(args, shape, progress_output=progress_output)
    except Exception as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        result = {
            "status": "oom" if is_oom else "error",
            "resolution": list(shape),
            "stages": [],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    worker_output = Path(args._worker_output)
    worker_output.parent.mkdir(parents=True, exist_ok=True)
    worker_output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
