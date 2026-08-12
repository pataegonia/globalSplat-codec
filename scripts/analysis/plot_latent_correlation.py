#!/usr/bin/env python3
"""Plot distributions produced by analyze_scene_latent_correlation.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def values(raw: dict[str, torch.Tensor], key: str) -> np.ndarray:
    return raw[key].detach().float().cpu().numpy().reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="*_raw.pt correlation file")
    ap.add_argument("--output", required=True, type=Path, help="Output PNG")
    ap.add_argument("--title", default="Scene latent correlation")
    ap.add_argument("--bins", type=int, default=100)
    args = ap.parse_args()

    raw = torch.load(args.input, map_location="cpu", weights_only=True)
    bins = np.linspace(-1.0, 1.0, args.bins + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    comparisons = [
        ("geo", "Geometry token correlation"),
        ("tex", "Texture token correlation"),
    ]
    for ax, (stream, title) in zip(axes[0], comparisons):
        adjacent = values(raw, f"{stream}.adjacent_token_pearson")
        random = values(raw, f"{stream}.random_token_pearson")
        ax.hist(random, bins=bins, density=True, alpha=0.50, label=f"random |ρ| mean={np.abs(random).mean():.3f}")
        ax.hist(adjacent, bins=bins, density=True, histtype="step", linewidth=1.8,
                label=f"adjacent |ρ| mean={np.abs(adjacent).mean():.3f}")
        ax.axvline(0.0, color="0.35", linewidth=0.8)
        ax.set(title=title, xlabel="Pearson correlation ρ", ylabel="Density", xlim=(-1, 1))
        ax.legend(frameon=False)
        ax.grid(alpha=0.18)

    ax = axes[1, 0]
    geo_ch = values(raw, "geo.adjacent_channel_pearson")
    tex_ch = values(raw, "tex.adjacent_channel_pearson")
    ax.hist(geo_ch, bins=bins, density=True, alpha=0.50,
            label=f"geometry |ρ| mean={np.abs(geo_ch).mean():.3f}")
    ax.hist(tex_ch, bins=bins, density=True, histtype="step", linewidth=1.8,
            label=f"texture |ρ| mean={np.abs(tex_ch).mean():.3f}")
    ax.axvline(0.0, color="0.35", linewidth=0.8)
    ax.set(title="Adjacent channel correlation", xlabel="Pearson correlation ρ",
           ylabel="Density", xlim=(-1, 1))
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)

    ax = axes[1, 1]
    cross = values(raw, "geo_tex.same_token_pearson")
    ax.hist(cross, bins=bins, density=True, alpha=0.70,
            label=f"geo–texture |ρ| mean={np.abs(cross).mean():.3f}")
    ax.axvline(0.0, color="0.35", linewidth=0.8)
    ax.set(title="Same-token geometry–texture correlation", xlabel="Pearson correlation ρ",
           ylabel="Density", xlim=(-1, 1))
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)

    fig.suptitle(args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
