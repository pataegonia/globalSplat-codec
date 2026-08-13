"""Synchronized wall-time and CUDA peak-memory profiling helpers."""
from __future__ import annotations

import time
import os
import sys
from collections.abc import Callable
from typing import Any

import torch


def _process_rss_bytes() -> int:
    """Best-effort current resident-set size without a hard psutil dependency."""
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        return 0


def _process_peak_rss_bytes() -> int:
    """Best-effort cumulative process high-water RSS (Linux/macOS)."""
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return 0


def profile_call(
    function: Callable[[], Any],
    *,
    device: torch.device | str | None,
) -> tuple[Any, dict[str, float | int]]:
    """Run one module and report synchronized time plus CUDA memory peaks.

    ``peak_*`` values are absolute process allocator peaks during the call.
    ``peak_*_delta`` values subtract memory already live at module entry and
    therefore expose the module's additional transient allocator pressure.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        reserved_before = int(torch.cuda.memory_reserved(device))
    else:
        allocated_before = reserved_before = 0
    process_rss_before = _process_rss_bytes()

    start = time.perf_counter()
    output = function()
    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    if use_cuda:
        allocated_after = int(torch.cuda.memory_allocated(device))
        reserved_after = int(torch.cuda.memory_reserved(device))
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        allocated_after = reserved_after = peak_allocated = peak_reserved = 0
    process_rss_after = _process_rss_bytes()

    return output, {
        "seconds": float(elapsed),
        "cuda_allocated_before_bytes": allocated_before,
        "cuda_allocated_after_bytes": allocated_after,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_allocated_delta_bytes": max(0, peak_allocated - allocated_before),
        "cuda_reserved_before_bytes": reserved_before,
        "cuda_reserved_after_bytes": reserved_after,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_reserved_delta_bytes": max(0, peak_reserved - reserved_before),
        "process_rss_before_bytes": process_rss_before,
        "process_rss_after_bytes": process_rss_after,
        "process_rss_change_bytes": process_rss_after - process_rss_before,
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_module_profiles(
    rows: list[dict[str, dict[str, float | int]]],
    *,
    warmup_steps: int = 0,
) -> dict:
    """Aggregate per-scene module profiles after dropping warm-up scenes."""
    selected = rows[max(0, int(warmup_steps)) :]
    if not selected:
        return {
            "scenes_total": len(rows),
            "warmup_scenes_skipped": min(len(rows), max(0, int(warmup_steps))),
            "scenes_profiled": 0,
            "modules": {},
        }
    module_names = sorted(set().union(*(row.keys() for row in selected)))
    modules = {}
    for name in module_names:
        samples = [row[name] for row in selected if name in row]
        seconds = [float(sample["seconds"]) for sample in samples]
        modules[name] = {
            "calls": len(samples),
            "mean_seconds": sum(seconds) / len(seconds),
            "p50_seconds": _percentile(seconds, 0.50),
            "p95_seconds": _percentile(seconds, 0.95),
            "min_seconds": min(seconds),
            "max_seconds": max(seconds),
        }
        for key in (
            "cuda_peak_allocated_bytes",
            "cuda_peak_allocated_delta_bytes",
            "cuda_peak_reserved_bytes",
            "cuda_peak_reserved_delta_bytes",
            "process_rss_after_bytes",
            "process_peak_rss_bytes",
        ):
            values = [int(sample[key]) for sample in samples]
            modules[name][f"mean_{key}"] = sum(values) / len(values)
            modules[name][f"max_{key}"] = max(values)
    total_seconds = [
        sum(float(module["seconds"]) for module in row.values()) for row in selected
    ]
    return {
        "scenes_total": len(rows),
        "warmup_scenes_skipped": len(rows) - len(selected),
        "scenes_profiled": len(selected),
        "timing_is_cuda_synchronized": True,
        "memory_units": "bytes",
        "absolute_peak_definition": "allocator peak while the module executes",
        "peak_delta_definition": "absolute peak minus live memory at module entry",
        "process_peak_rss_definition": (
            "cumulative process resident-set high-water mark observed after the call"
        ),
        "mean_sum_of_profiled_module_seconds": sum(total_seconds) / len(total_seconds),
        "modules": modules,
    }
