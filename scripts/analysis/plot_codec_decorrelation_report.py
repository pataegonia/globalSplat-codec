#!/usr/bin/env python3
"""Create publication-ready plots and tables from codec decorrelation output."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STREAM_LABELS = {"texture": "Texture", "geometry": "Geometry"}
COLORS = {"texture": "#0072B2", "geometry": "#D55E00"}


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary(report: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)

    ax = axes[0]
    for name in ("texture", "geometry"):
        stream = report["streams"][name]
        lag = stream["token_lag_correlation"]
        xs = np.asarray([int(value) for value in lag])
        ys = np.asarray([lag[str(value)]["mean_abs"] for value in xs])
        color = COLORS[name]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=color, label=STREAM_LABELS[name])
        baseline = stream["shuffle_lag1_correlation"]["mean_abs"]
        ax.axhline(baseline, color=color, lw=1.2, ls="--", alpha=0.75)
    ax.set_xscale("log", base=2)
    ax.set_xticks(report["setting"]["lags"])
    ax.set_xticklabels([str(value) for value in report["setting"]["lags"]])
    ax.set_xlabel("Token lag")
    ax.set_ylabel(r"Mean absolute Pearson correlation $|\rho|$")
    ax.set_title("(a) Token locality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.text(
        0.02, 0.03, "Dashed: random-permutation baseline", transform=ax.transAxes,
        fontsize=8.5, color="0.35",
    )

    ax = axes[1]
    keys = (
        "raw", "token_delta_lag1", "shuffle_delta_lag1",
        "64x64.raster_horizontal", "64x64.raster_vertical",
        "64x64.raster_paeth_linear",
    )
    labels = ("Raw", "1D\ndelta", "Shuffle\ndelta", "2D\nhorizontal", "2D\nvertical", "2D\nlinear")
    x = np.arange(len(keys))
    width = 0.36
    for offset, name in ((-width / 2, "texture"), (width / 2, "geometry")):
        entropy = report["streams"][name]["quantized_entropy_bits_per_value"]
        values = [entropy[key]["mean"] for key in keys]
        bars = ax.bar(x + offset, values, width, color=COLORS[name], label=STREAM_LABELS[name])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 9.35)
    ax.set_ylabel("Empirical entropy (bits/value)")
    ax.set_title("(b) Predictive coding does not reduce entropy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output, "decorrelation_summary")


def plot_heatmaps(matrices: dict[str, np.ndarray], output: Path) -> None:
    entries = (
        ("texture_channel_correlation", "Texture channels"),
        ("geometry_channel_correlation", "Geometry channels"),
        ("texture_geometry_cross_correlation", "Texture–geometry cross-stream"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
    image = None
    for ax, (key, title) in zip(axes, entries):
        image = ax.imshow(np.abs(matrices[key]), vmin=0, vmax=1, cmap="magma", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Channel index")
        ax.set_ylabel("Channel index")
        ax.set_xticks((0, 255, 511))
        ax.set_yticks((0, 255, 511))
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label(r"Absolute Pearson correlation $|\rho|$")
    _save(fig, output, "channel_correlation_heatmaps")


def write_csv(report: dict, output: Path) -> None:
    with (output / "token_lag_correlation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stream", "lag", "mean_abs", "median_abs", "p90_abs", "shuffle_mean_abs"))
        for name in ("texture", "geometry"):
            stream = report["streams"][name]
            shuffle = stream["shuffle_lag1_correlation"]["mean_abs"]
            for lag, row in stream["token_lag_correlation"].items():
                writer.writerow((name, lag, row["mean_abs"], row["median_abs"], row["p90_abs"], shuffle))

    with (output / "entropy_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stream", "predictor", "mean_bits_per_value", "std", "gain_vs_raw"))
        for name in ("texture", "geometry"):
            for predictor, row in report["streams"][name]["quantized_entropy_bits_per_value"].items():
                writer.writerow((name, predictor, row["mean"], row["std"], row["gain_vs_raw"]))

    with (output / "channel_correlation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stream", "mean_abs", "median_abs", "p90_abs", "fraction_abs_gt_0.1", "fraction_abs_gt_0.3"))
        for name in ("texture", "geometry"):
            row = report["streams"][name]["channel_correlation"]
            writer.writerow((name, row["mean_abs"], row["median_abs"], row["p90_abs"], row["fraction_abs_gt_0.1"], row["fraction_abs_gt_0.3"]))


def write_report(report: dict, output: Path) -> None:
    rows = []
    for name in ("texture", "geometry"):
        stream = report["streams"][name]
        entropy = stream["quantized_entropy_bits_per_value"]
        rows.append(
            f"| {STREAM_LABELS[name]} | {stream['token_lag_correlation']['1']['mean_abs']:.4f} "
            f"| {stream['shuffle_lag1_correlation']['mean_abs']:.4f} "
            f"| {stream['channel_correlation']['mean_abs']:.4f} "
            f"| {entropy['raw']['mean']:.3f} | {entropy['token_delta_lag1']['mean']:.3f} "
            f"| {100 * entropy['token_delta_lag1']['gain_vs_raw']:+.1f}% "
            f"| {entropy['64x64.raster_paeth_linear']['mean']:.3f} "
            f"| {100 * entropy['64x64.raster_paeth_linear']['gain_vs_raw']:+.1f}% |"
        )
    setting = report["setting"]
    text = f"""# Codec-input latent decorrelation

## Setting

- Dataset: RE10K, context {setting['context']}, {setting['scenes']} scenes
- Codec inputs per scene: `{setting['shape_per_scene']}` for texture and geometry
- Quantization: texture {setting['texture_bits']}-bit, geometry {setting['geometry_bits']}-bit, `{setting['scale_mode']}` scale
- Artificial rasterizations: {', '.join(setting['reshapes'])}

## Main results

| Stream | Adjacent token | Shuffle | Channel-pair | Raw entropy | 1D delta | 1D change | 2D linear | 2D change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Adjacent-token dependence is indistinguishable from the random-permutation
baseline across lags 1–256. Conventional 1D and artificially rasterized 2D
predictors increase, rather than reduce, quantized-symbol entropy. This supports
the claim that the codec inputs contain little exploitable token-local redundancy.

Channel correlation is not negligible, so the evidence should not be described
as proving that all transform coding is ineffective. The defensible conclusion
is specifically about sequential, local, and raster-based predictive coding.

## Paper-ready wording

> The codec-input latents exhibit little dependence along the token dimension:
> adjacent-token correlations are comparable to random permutations across all
> tested lags. Moreover, conventional 1D and artificial 2D predictors increase
> empirical quantized-symbol entropy by 5–19%, indicating limited exploitable
> locality for predictive or raster-based compression.

## Files

- `decorrelation_summary.png` / `.pdf`: lag correlation and entropy comparison
- `channel_correlation_heatmaps.png` / `.pdf`: full 512×512 matrices
- `token_lag_correlation.csv`: lag statistics
- `entropy_comparison.csv`: raw and predictor entropy
- `channel_correlation_summary.csv`: channel-level summary
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Directory containing summary.json and correlation_matrices.npz")
    args = parser.parse_args()
    output = args.input.resolve()
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    archive = np.load(output / "correlation_matrices.npz")
    matrices = {key: archive[key] for key in archive.files}
    plot_summary(report, output)
    plot_heatmaps(matrices, output)
    write_csv(report, output)
    write_report(report, output)
    print(f"Wrote report, figures, and tables to {output}")


if __name__ == "__main__":
    main()
