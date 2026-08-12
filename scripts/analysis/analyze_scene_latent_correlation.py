#!/usr/bin/env python3
"""Measure local and global correlation in GlobalSplat scene latents.

The script extracts the geometry/texture outputs immediately after the slot
encoder.  It reports adjacent-channel, adjacent-token, random-token, and
geometry-vs-texture correlations over one or more dataset batches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw)
    if any(k.startswith("model.") for k in state):
        state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


@torch.no_grad()
def extract_latents(model, inputs: dict[str, torch.Tensor]):
    content = model._tokenize(inputs["images"], inputs["intrinsic"], inputs["c2w"])
    state = model.scene_tokens.unsqueeze(0).expand(content.shape[0], -1, -1)
    _, (tex, geo), _ = model.slot_encoder(content, state=state)
    return tex.float(), geo.float()


def pearson_last_dim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    numerator = (a * b).sum(dim=-1)
    denominator = a.square().sum(dim=-1).sqrt() * b.square().sum(dim=-1).sqrt()
    return numerator / denominator.clamp_min(1e-12)


def summary(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().flatten().cpu()
    return {
        "mean": x.mean().item(),
        "mean_abs": x.abs().mean().item(),
        "std": x.std(unbiased=False).item(),
        "min": x.min().item(),
        "max": x.max().item(),
    }


def stream_metrics(x: torch.Tensor, generator: torch.Generator) -> dict[str, torch.Tensor]:
    # Channel c vs c+1, with (batch, token) observations. This is Pearson
    # correlation in the conventional statistical sense for each channel pair.
    flat = x.reshape(-1, x.shape[-1])
    a, b = flat[:, :-1], flat[:, 1:]
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    adjacent_channel = (a * b).sum(0) / (
        a.square().sum(0).sqrt() * b.square().sum(0).sqrt()
    ).clamp_min(1e-12)

    adjacent_token_cosine = F.cosine_similarity(x[:, :-1], x[:, 1:], dim=-1)
    adjacent_token_pearson = pearson_last_dim(x[:, :-1], x[:, 1:])

    # A same-sized random-pair baseline tells us whether index adjacency carries
    # more dependence than arbitrary token pairing.
    token_count = x.shape[1]
    perm = torch.randperm(token_count, generator=generator, device="cpu").to(x.device)
    random_token_cosine = F.cosine_similarity(x, x[:, perm], dim=-1)
    random_token_pearson = pearson_last_dim(x, x[:, perm])
    return {
        "adjacent_channel_pearson": adjacent_channel,
        "adjacent_token_cosine": adjacent_token_cosine,
        "adjacent_token_pearson": adjacent_token_pearson,
        "random_token_cosine": random_token_cosine,
        "random_token_pearson": random_token_pearson,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--experiment", default="re10k_32k")
    ap.add_argument("--context", type=int, choices=(12, 24, 36), default=12)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, default=Path("latent_correlation.json"))
    ap.add_argument("--raw-output", type=Path, default=None,
                    help="Optional .pt file containing every per-pair correlation value")
    args = ap.parse_args()

    # Keep module import / --help usable in lightweight environments; these are
    # required only when actually loading the configured model and dataset.
    from hydra import compose, initialize_config_dir
    from globalsplat.main import build_datamodule, build_model

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "test" / "index.json").is_file():
        raise SystemExit(f"dataset index not found: {dataset_root / 'test' / 'index.json'}")
    device = torch.device(args.device)

    dataset = f"re10k_eval_ctx{args.context}"

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg = compose(
            config_name="main",
            overrides=[
                f"+experiment={args.experiment}",
                f"dataset={dataset}",
                "mode=test",
                f"dataset.dataset_roots=['{dataset_root.as_posix()}']",
                "optimizer.batch_size=1",
                f"optimizer.num_workers={args.num_workers}",
            ],
        )

    model = build_model(cfg.model).to(device).eval()
    model.set_stage(int(cfg.curriculum.final_stage))
    load_weights(model, checkpoint)
    dm, _ = build_datamodule(cfg)
    dm.setup("test")

    collected: dict[str, list[torch.Tensor]] = {}
    generator = torch.Generator().manual_seed(int(cfg.seed))
    processed = 0
    for batch in dm.test_dataloader():
        inputs = to_device(batch["inputs"], device)
        tex, geo = extract_latents(model, inputs)
        for stream_name, latent in (("geo", geo), ("tex", tex)):
            for metric_name, values in stream_metrics(latent, generator).items():
                collected.setdefault(f"{stream_name}.{metric_name}", []).append(values.cpu())
        collected.setdefault("geo_tex.same_token_cosine", []).append(
            F.cosine_similarity(geo, tex, dim=-1).cpu()
        )
        collected.setdefault("geo_tex.same_token_pearson", []).append(
            pearson_last_dim(geo, tex).cpu()
        )
        processed += 1
        if processed >= args.batches:
            break

    if not processed:
        raise RuntimeError("test dataloader produced no batches")
    raw = {name: torch.cat([v.flatten() for v in values]) for name, values in collected.items()}
    report = {
        "checkpoint": str(checkpoint),
        "experiment": args.experiment,
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "batches": processed,
        "metrics": {name: summary(values) for name, values in raw.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw, args.raw_output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
