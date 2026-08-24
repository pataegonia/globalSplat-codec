#!/usr/bin/env python3
"""Test whether codec-input latents contain conventional local redundancy.

Only the two tensors passed to ``encode_scene_latents_to_bytes`` are analysed:
texture and geometry, normally shaped [1, 2048, 512].  No Gaussian decoding or
rendering is performed.

The experiment deliberately separates two questions:

* correlation: does the native token/channel ordering contain dependence?
* predictability: does a conventional 1-D/2-D residual lower the entropy of
  symbols produced by the production scalar quantizer?

Artificial rasterisations are controls, not claims of spatial topology.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from globalsplat.compression.arithmetic_codec import _quantize
from scripts.analysis.codec_diagnostic_utils import common_args, extract, seed_all, setup, to_device


def _pearson_columns(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson r for corresponding columns, observations on axis zero."""
    a = a.double() - a.double().mean(0, keepdim=True)
    b = b.double() - b.double().mean(0, keepdim=True)
    denominator = a.square().sum(0).sqrt() * b.square().sum(0).sqrt()
    return ((a * b).sum(0) / denominator.clamp_min(1e-30)).float()


def _describe(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    finite = values[torch.isfinite(values)]
    if not finite.numel():
        return {key: float("nan") for key in ("mean", "mean_abs", "median_abs", "p90_abs", "max_abs")}
    absolute = finite.abs()
    return {
        "mean": float(finite.mean()),
        "mean_abs": float(absolute.mean()),
        "median_abs": float(absolute.median()),
        "p90_abs": float(torch.quantile(absolute, 0.90)),
        "max_abs": float(absolute.max()),
    }


def _corr_matrix(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    x = x.double()
    y = x if y is None else y.double()
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    x = x / x.square().sum(0).sqrt().clamp_min(1e-30)
    y = y / y.square().sum(0).sqrt().clamp_min(1e-30)
    return (x.T @ y).float()


def _off_diagonal_report(matrix: torch.Tensor) -> dict[str, float]:
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    values = matrix[mask]
    result = _describe(values)
    result.update({
        "fraction_abs_gt_0.1": float((values.abs() > 0.1).float().mean()),
        "fraction_abs_gt_0.3": float((values.abs() > 0.3).float().mean()),
        "off_diagonal_energy_ratio": float(
            values.square().sum() / matrix.square().sum().clamp_min(1e-30)
        ),
    })
    return result


def _entropy(symbols: np.ndarray) -> float:
    _, counts = np.unique(symbols, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _quantized_symbols(x: torch.Tensor, bits: int, scale_mode: str) -> torch.Tensor:
    symbols, _ = _quantize(x, bits, scale_mode, None)
    return torch.from_numpy(symbols.reshape(tuple(x.shape))).to(torch.int64)


def _predictors(x: torch.Tensor, rows: int, cols: int, permutation: torch.Tensor) -> dict[str, torch.Tensor]:
    # x contains production-quantized integer symbols.  Differencing symbols,
    # rather than independently requantizing float residuals, preserves exact
    # reconstruction and gives a fair predictive-coding comparison.
    grid = x.reshape(x.shape[0], rows, cols, x.shape[-1])
    shuffled = x[:, permutation]
    return {
        "raw": x,
        "token_delta_lag1": x[:, 1:] - x[:, :-1],
        "shuffle_delta_lag1": shuffled[:, 1:] - shuffled[:, :-1],
        "raster_horizontal": grid[:, :, 1:] - grid[:, :, :-1],
        "raster_vertical": grid[:, 1:] - grid[:, :-1],
        "raster_paeth_linear": grid[:, 1:, 1:] - (
            grid[:, 1:, :-1] + grid[:, :-1, 1:] - grid[:, :-1, :-1]
        ),
    }


def _shape_arg(value: str) -> tuple[int, int]:
    try:
        rows, cols = (int(part) for part in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("reshape must look like 32x64") from exc
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("reshape dimensions must be positive")
    return rows, cols


def _automatic_reshapes(token_count: int) -> tuple[tuple[int, int], ...]:
    """Choose a near-square raster and a second, wider factorisation."""
    rows = math.isqrt(token_count)
    while token_count % rows:
        rows -= 1
    primary = (rows, token_count // rows)
    wider_rows = max(1, rows // 2)
    while token_count % wider_rows:
        wider_rows -= 1
    secondary = (wider_rows, token_count // wider_rows)
    return (primary,) if secondary == primary else (primary, secondary)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("--scenes", type=int, default=50)
    parser.add_argument("--texture-bits", type=int, default=7)
    parser.add_argument("--geometry-bits", type=int, default=9)
    parser.add_argument("--scale-mode", choices=("global", "channel"), default="channel")
    parser.add_argument("--lags", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32, 64, 128, 256))
    parser.add_argument(
        "--reshapes", type=_shape_arg, nargs="+", default=None,
        help="Artificial rasters such as 64x64. Default: infer two factor pairs from L.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.scenes <= 0:
        raise SystemExit("--scenes must be positive")
    if not 2 <= args.texture_bits <= 12 or not 2 <= args.geometry_bits <= 12:
        raise SystemExit("bit depths must be in [2, 12]")
    seed_all(args.seed)
    model, _, loader, device = setup(args)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    streams: dict[str, list[torch.Tensor]] = {"texture": [], "geometry": []}
    for index, batch in enumerate(loader):
        texture, geometry = extract(model, to_device(batch["inputs"], device))
        if texture.shape[0] != 1 or geometry.shape[0] != 1:
            raise RuntimeError("analysis requires one scene per batch")
        streams["texture"].append(texture.cpu())
        streams["geometry"].append(geometry.cpu())
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(f"extracted {index + 1}/{args.scenes} scenes", flush=True)
        if index + 1 >= args.scenes:
            break
    if not streams["texture"]:
        raise RuntimeError("test dataloader produced no scenes")

    arrays = {name: torch.cat(values, dim=0) for name, values in streams.items()}
    shape = arrays["texture"].shape
    if arrays["geometry"].shape != shape:
        raise RuntimeError("texture/geometry shape mismatch")
    _, token_count, channels = shape
    if args.reshapes is None:
        args.reshapes = _automatic_reshapes(token_count)
        print(
            "auto reshapes: " + ", ".join(f"{r}x{c}" for r, c in args.reshapes),
            flush=True,
        )
    for rows, cols in args.reshapes:
        if rows * cols != token_count:
            raise SystemExit(f"reshape {rows}x{cols} does not contain {token_count} tokens")
    lags = sorted(set(args.lags))
    if not lags or lags[0] <= 0 or lags[-1] >= token_count:
        raise SystemExit(f"lags must be in [1, {token_count - 1}]")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "setting": {
            "checkpoint": str(args.checkpoint.resolve()), "experiment": args.experiment,
            "context": args.context, "scenes": int(shape[0]), "seed": args.seed,
            "shape_per_scene": [1, token_count, channels], "lags": lags,
            "reshapes": [f"{r}x{c}" for r, c in args.reshapes],
            "scale_mode": args.scale_mode,
            "texture_bits": args.texture_bits, "geometry_bits": args.geometry_bits,
            "interpretation": "artificial reshapes test raster predictability, not spatial topology",
        },
        "streams": {},
    }
    matrices: dict[str, np.ndarray] = {}

    for name, x3 in arrays.items():
        x = x3.reshape(-1, channels)
        corr = _corr_matrix(x)
        matrices[f"{name}_channel_correlation"] = corr.numpy()
        stream_report: dict = {
            "values_per_scene": token_count * channels,
            "channel_correlation": _off_diagonal_report(corr),
            "token_lag_correlation": {},
            "raster_correlation": {},
            "quantized_entropy_bits_per_value": {},
        }
        for lag in lags:
            values = _pearson_columns(x3[:, :-lag].reshape(-1, channels), x3[:, lag:].reshape(-1, channels))
            stream_report["token_lag_correlation"][str(lag)] = _describe(values)

        permutation = torch.randperm(token_count, generator=generator)
        shuffled_r = _pearson_columns(
            x3[:, permutation[:-1]].reshape(-1, channels),
            x3[:, permutation[1:]].reshape(-1, channels),
        )
        stream_report["shuffle_lag1_correlation"] = _describe(shuffled_r)

        bits = args.texture_bits if name == "texture" else args.geometry_bits
        all_symbols = _quantized_symbols(x3, bits, args.scale_mode)
        raw_entropy = _entropy(all_symbols.numpy())
        entropy_rows: dict[str, list[float]] = {}
        for rows, cols in args.reshapes:
            grid = x3.reshape(shape[0], rows, cols, channels)
            stream_report["raster_correlation"][f"{rows}x{cols}"] = {
                "horizontal": _describe(_pearson_columns(grid[:, :, :-1].reshape(-1, channels), grid[:, :, 1:].reshape(-1, channels))),
                "vertical": _describe(_pearson_columns(grid[:, :-1].reshape(-1, channels), grid[:, 1:].reshape(-1, channels))),
            }
            # Production quantization is scene-adaptive, so average per-scene
            # entropy rather than deriving one scale from the entire dataset.
            for scene in x3:
                scene_perm = torch.randperm(token_count, generator=generator)
                symbols = _quantized_symbols(scene.unsqueeze(0), bits, args.scale_mode)
                for predictor, residual in _predictors(symbols, rows, cols, scene_perm).items():
                    key = predictor if predictor in ("raw", "token_delta_lag1", "shuffle_delta_lag1") else f"{rows}x{cols}.{predictor}"
                    entropy_rows.setdefault(key, []).append(_entropy(residual.numpy().reshape(-1)))
        raw_mean = float(np.mean(entropy_rows["raw"]))
        for key, values in entropy_rows.items():
            # raw/1-D rows repeat once per requested reshape; that does not alter
            # their mean, but retaining a single implementation avoids drift.
            mean = float(np.mean(values))
            stream_report["quantized_entropy_bits_per_value"][key] = {
                "mean": mean, "std": float(np.std(values)),
                "gain_vs_raw": float(1.0 - mean / raw_mean) if key != "raw" else 0.0,
            }
        stream_report["dataset_global_raw_entropy"] = raw_entropy
        report["streams"][name] = stream_report

    cross = _corr_matrix(arrays["texture"].reshape(-1, channels), arrays["geometry"].reshape(-1, channels))
    matrices["texture_geometry_cross_correlation"] = cross.numpy()
    report["texture_geometry_cross_correlation"] = _describe(cross)
    report["texture_geometry_cross_correlation"]["fraction_abs_gt_0.1"] = float((cross.abs() > 0.1).float().mean())

    np.savez_compressed(output / "correlation_matrices.npz", **matrices)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    command = (
        "python scripts/analysis/analyze_codec_decorrelation.py "
        f"--checkpoint {args.checkpoint} --dataset-root {args.dataset_root} --output {args.output}"
    )
    (output / "command.txt").write_text(command + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
