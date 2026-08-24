"""Shared, read-only helpers for latent codec diagnostics."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value): return value.to(device, non_blocking=True)
    if isinstance(value, dict): return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(to_device(v, device) for v in value)
    return value


def load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw)
    if any(k.startswith("model.") for k in state):
        state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("perceptual_loss")]
    if missing or unexpected: raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def configs(args):
    from hydra import compose, initialize_config_dir
    common = [f"+experiment={args.experiment}", "mode=test", "optimizer.batch_size=1",
              "model.latent_codec_bits=0", "model.latent_codec_texture_bits=0",
              "model.latent_codec_geometry_bits=0"]
    root = args.dataset_root.resolve().as_posix()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        train = compose(config_name="main", overrides=common + ["dataset=re10k",
            f"dataset.dataset_roots=['{root}']", f"dataset.num_context_views={args.context}",
            "dataset.num_target_views=3", "dataset.augment=false", "optimizer.num_workers=0"])
        test = compose(config_name="main", overrides=common + [f"dataset=re10k_eval_ctx{args.context}",
            f"dataset.dataset_roots=['{root}']", f"optimizer.num_workers={args.num_workers}"])
    return train, test


def setup(args):
    from globalsplat.main import build_datamodule, build_model
    train_cfg, test_cfg = configs(args); device = torch.device(args.device)
    model = build_model(train_cfg.model).to(device).eval(); model.set_stage(int(train_cfg.curriculum.final_stage))
    load_weights(model, args.checkpoint.resolve())
    train_dm, _ = build_datamodule(train_cfg); train_dm.setup("fit")
    test_dm, _ = build_datamodule(test_cfg); test_dm.setup("test")
    return model, train_dm.train_dataloader(), test_dm.test_dataloader(), device


@torch.no_grad()
def extract(model, inputs):
    content = model._tokenize(inputs["images"], inputs["intrinsic"], inputs["c2w"])
    state = model.scene_tokens.unsqueeze(0).expand(content.shape[0], -1, -1)
    _, (texture, geometry), _ = model.slot_encoder(content, state=state)
    return texture.float(), geometry.float()


def quantize_torch(x: torch.Tensor, bits: int, factor: float = 1.0) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().amax(dim=tuple(range(x.ndim - 1)), keepdim=True).clamp_min(torch.finfo(torch.float32).tiny) / qmax
    scale = scale * factor
    return (x / scale).round().clamp(-qmax, qmax) * scale


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def common_args(parser) -> None:
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--experiment", default="re10k_32k")
    parser.add_argument("--context", type=int, choices=(12,24,36), default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
