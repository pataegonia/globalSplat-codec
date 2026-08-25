"""Lightweight learned transforms for the frozen GlobalSplat latent boundary.

The production representation remains 512-dimensional.  Analysis and synthesis
are separate identity-initialized token-wise residual MLPs for geometry and
texture.  Training uses the production per-scene/per-channel max quantizer with
a straight-through rounding estimator and a small factorized logistic density
model.  The density model is training-only; final bitstreams use a freshly
calibrated fixed K=32 rANS profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizedLatent:
    reconstructed: torch.Tensor
    symbols: torch.Tensor
    scale: torch.Tensor


class TokenResidualMLP(nn.Module):
    """``x + MLP(LayerNorm(x))`` with an initially zero residual branch."""

    def __init__(self, channels: int = 512, hidden: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.fc2(F.gelu(self.fc1(self.norm(value))))


class FactorizedLogisticEntropyModel(nn.Module):
    """Per-channel logistic probability masses for differentiable rate.

    This intentionally has no hyperprior, context model, token interaction, or
    cross-stream dependency.  Its parameters are discarded for deployment
    after a fixed K=32 categorical profile is calibrated on transformed
    training symbols.
    """

    def __init__(self, channels: int = 512, minimum_scale: float = 1e-3) -> None:
        super().__init__()
        self.channels = int(channels)
        self.minimum_scale = float(minimum_scale)
        self.location = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
        # softplus(2.0) ~= 2.13: safe until calibration moments are measured.
        self.log_scale = nn.Parameter(torch.full((channels,), 2.0, dtype=torch.float32))

    @torch.no_grad()
    def initialize_from_moments(
        self, location: torch.Tensor, standard_deviation: torch.Tensor
    ) -> None:
        if location.shape != (self.channels,) or standard_deviation.shape != (
            self.channels,
        ):
            raise ValueError("entropy moments must have one value per channel")
        # A logistic distribution has std = scale*pi/sqrt(3).  The stable
        # inverse-softplus expression avoids overflow for wide 9-bit symbols.
        scale = (
            standard_deviation.float() * (math.sqrt(3.0) / math.pi)
        ).clamp_min(self.minimum_scale)
        inverse_softplus = scale + torch.log(-torch.expm1(-scale))
        self.location.copy_(location.float())
        self.log_scale.copy_(inverse_softplus)

    def bits(self, symbols: torch.Tensor) -> torch.Tensor:
        if symbols.shape[-1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {symbols.shape[-1]}"
            )
        value = symbols.float()
        location = self.location.view(*([1] * (value.ndim - 1)), -1)
        scale = (
            F.softplus(self.log_scale).clamp_min(self.minimum_scale)
        ).view(*([1] * (value.ndim - 1)), -1)
        upper = torch.sigmoid((value + 0.5 - location) / scale)
        lower = torch.sigmoid((value - 0.5 - location) / scale)
        probability = (upper - lower).clamp_min(1e-9)
        return -torch.log2(probability)


class LearnedLatentTransformCodec(nn.Module):
    """Separate analysis/synthesis transforms and rate models per stream."""

    def __init__(
        self,
        channels: int = 512,
        hidden: int = 256,
        texture_bits: int = 7,
        geometry_bits: int = 9,
        texture_only: bool = False,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.texture_bits = int(texture_bits)
        self.geometry_bits = int(geometry_bits)
        self.texture_only = bool(texture_only)
        if self.texture_bits != 7 or self.geometry_bits != 9:
            raise ValueError("the first learned-transform experiment is fixed to G9/T7")

        self.g_a_geometry = TokenResidualMLP(channels, hidden)
        self.g_s_geometry = TokenResidualMLP(channels, hidden)
        self.g_a_texture = TokenResidualMLP(channels, hidden)
        self.g_s_texture = TokenResidualMLP(channels, hidden)
        self.geometry_entropy = FactorizedLogisticEntropyModel(channels)
        self.texture_entropy = FactorizedLogisticEntropyModel(channels)
        if self.texture_only:
            self.g_a_geometry.requires_grad_(False)
            self.g_s_geometry.requires_grad_(False)
            self.geometry_entropy.requires_grad_(False)

    @staticmethod
    def quantize(
        latent: torch.Tensor, bits: int, *, straight_through: bool
    ) -> QuantizedLatent:
        """Production max-scale quantization, independently for every scene."""
        if latent.ndim != 3:
            raise ValueError(f"expected [B,N,C] latent, got {tuple(latent.shape)}")
        value = latent.float()
        qmax = (1 << (int(bits) - 1)) - 1
        # The production evaluator has B=1 and reduces B,N.  Reducing N here
        # independently per scene is identical for B=1 and correct for training
        # batches larger than one.
        scale = value.abs().amax(dim=1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        ) / float(qmax)
        normalized = value / scale
        hard_symbols = normalized.round().clamp(-qmax, qmax)
        symbols = (
            normalized + (hard_symbols - normalized).detach()
            if straight_through
            else hard_symbols
        )
        return QuantizedLatent(
            reconstructed=symbols * scale,
            symbols=symbols,
            scale=scale,
        )

    def analysis(
        self, texture: torch.Tensor, geometry: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        geometry_analysis = (
            geometry.float()
            if self.texture_only
            else self.g_a_geometry(geometry.float())
        )
        return self.g_a_texture(texture.float()), geometry_analysis

    def synthesis(
        self, texture: torch.Tensor, geometry: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        geometry_hat = (
            geometry.float()
            if self.texture_only
            else self.g_s_geometry(geometry.float())
        )
        return self.g_s_texture(texture.float()), geometry_hat

    def forward(
        self,
        texture: torch.Tensor,
        geometry: torch.Tensor,
        *,
        straight_through: bool = True,
    ) -> dict[str, torch.Tensor | QuantizedLatent]:
        texture_analysis, geometry_analysis = self.analysis(texture, geometry)
        texture_quantized = self.quantize(
            texture_analysis, self.texture_bits, straight_through=straight_through
        )
        geometry_quantized = self.quantize(
            geometry_analysis, self.geometry_bits, straight_through=straight_through
        )
        texture_hat, geometry_hat = self.synthesis(
            texture_quantized.reconstructed, geometry_quantized.reconstructed
        )
        texture_bits = self.texture_entropy.bits(texture_quantized.symbols).sum()
        geometry_bits = self.geometry_entropy.bits(geometry_quantized.symbols).sum()
        scalar_count = texture.numel() + geometry.numel()
        rate = (texture_bits + geometry_bits) / float(scalar_count)
        return {
            "texture_analysis": texture_analysis,
            "geometry_analysis": geometry_analysis,
            "texture_quantized": texture_quantized,
            "geometry_quantized": geometry_quantized,
            "texture_hat": texture_hat,
            "geometry_hat": geometry_hat,
            "texture_bits": texture_bits,
            "geometry_bits": geometry_bits,
            "rate_bits_per_scalar": rate,
        }

    @property
    def transform_parameter_count(self) -> int:
        modules = (
            self.g_a_geometry,
            self.g_s_geometry,
            self.g_a_texture,
            self.g_s_texture,
        )
        return sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    @property
    def entropy_parameter_count(self) -> int:
        modules = (self.geometry_entropy, self.texture_entropy)
        return sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def inverse_psnr(psnr: float) -> float:
    """Return 255^2-scaled RGB MSE corresponding to a PSNR value."""
    return math.pow(10.0, -float(psnr) / 10.0) * 255.0**2
