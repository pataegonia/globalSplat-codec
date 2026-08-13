"""Zero-order entropy analysis for quantized scene-latent channels.

This module deliberately stops before channel grouping or learned entropy
models.  It provides the inexpensive Phase-1 measurements used to decide
whether channel-conditioned CDFs are worth implementing:

* scene-adaptive global entropy,
* train-fixed global cross-entropy,
* train-fixed per-channel (K=C) cross-entropy, and
* empirical held-out per-channel entropy.
"""
from __future__ import annotations

import heapq

import numpy as np
import torch

from .arithmetic_codec import _quantize


def channel_histogram(symbols: np.ndarray, alphabet: int) -> np.ndarray:
    """Return ``[C, alphabet]`` counts for symbols shaped ``[..., C]``."""
    symbols = np.asarray(symbols)
    if symbols.ndim < 2:
        raise ValueError(f"symbols must have at least two dimensions, got {symbols.shape}")
    if alphabet < 2:
        raise ValueError(f"alphabet must be at least 2, got {alphabet}")
    if symbols.size == 0:
        raise ValueError("symbols must not be empty")
    minimum = int(symbols.min())
    maximum = int(symbols.max())
    if minimum < 0 or maximum >= alphabet:
        raise ValueError(
            f"symbols outside [0, {alphabet}): min={minimum}, max={maximum}"
        )

    channels = symbols.shape[-1]
    flat = symbols.reshape(-1, channels)
    codes = flat.T.astype(np.int64, copy=False)
    codes = codes + np.arange(channels, dtype=np.int64)[:, None] * alphabet
    return np.bincount(
        codes.reshape(-1), minlength=channels * alphabet
    ).reshape(channels, alphabet)


def quantized_channel_histogram(
    latent: torch.Tensor,
    bits: int,
    *,
    scale_mode: str = "channel",
    clip_percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize one scene exactly like the codec and return counts plus scales.

    Per-channel codec scales are scene-adaptive.  Requiring a batch dimension of
    one prevents an analysis run from accidentally sharing scales across scenes.
    """
    if latent.ndim < 3 or latent.shape[0] != 1:
        raise ValueError(
            "latent must contain exactly one scene with shape [1, ..., C], "
            f"got {tuple(latent.shape)}"
        )
    flat_symbols, scales = _quantize(latent, bits, scale_mode, clip_percentile)
    symbols = flat_symbols.reshape(tuple(latent.shape))
    alphabet = (1 << int(bits)) - 1
    return channel_histogram(symbols, alphabet), scales


def empirical_entropy_bits(counts: np.ndarray) -> float:
    """Ideal zero-order code length, summing independent rows when present."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim == 0 or np.any(counts < 0):
        raise ValueError("counts must be a non-negative vector or matrix")
    if counts.ndim == 1:
        counts = counts[None, :]
    totals = counts.sum(axis=-1, keepdims=True)
    positive = counts > 0
    safe_totals = np.broadcast_to(totals, counts.shape)[positive]
    return float(
        np.sum(counts[positive] * (np.log2(safe_totals) - np.log2(counts[positive])))
    )


def smoothed_pmf(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Normalize counts after additive smoothing along the last dimension."""
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim == 0 or np.any(counts < 0):
        raise ValueError("counts must be a non-negative vector or matrix")
    smoothed = counts + float(alpha)
    return smoothed / smoothed.sum(axis=-1, keepdims=True)


def cross_entropy_bits(counts: np.ndarray, pmf: np.ndarray) -> float:
    """Code length of ``counts`` under a matching fixed PMF tensor."""
    counts = np.asarray(counts, dtype=np.float64)
    pmf = np.asarray(pmf, dtype=np.float64)
    if counts.shape != pmf.shape:
        raise ValueError(f"counts/pmf shape mismatch: {counts.shape} vs {pmf.shape}")
    if np.any(pmf <= 0) or not np.all(np.isfinite(pmf)):
        raise ValueError("pmf must be finite and strictly positive")
    return float(-np.sum(counts * np.log2(pmf)))


def cdf16_probabilities(pmf: np.ndarray, precision: int = 16) -> np.ndarray:
    """Quantize PMFs with CompressAI and return their actual rANS probabilities."""
    try:
        from compressai.entropy_models.entropy_models import pmf_to_quantized_cdf
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("CompressAI is required for CDF-quantized rates") from exc

    pmf = np.asarray(pmf, dtype=np.float64)
    original_shape = pmf.shape
    rows = pmf.reshape(-1, original_shape[-1])
    quantized = np.empty_like(rows)
    for index, row in enumerate(rows):
        cdf = (
            pmf_to_quantized_cdf(torch.from_numpy(row).float(), precision=precision)
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        frequencies = np.diff(cdf).astype(np.float64)
        if frequencies.shape[0] != row.shape[0] or np.any(frequencies <= 0):
            raise RuntimeError("CompressAI produced an invalid fixed CDF")
        quantized[index] = frequencies / float(cdf[-1])
    return quantized.reshape(original_shape)


def out_of_calibration_support(
    calibration_counts: np.ndarray,
    evaluation_counts: np.ndarray,
) -> dict[str, int]:
    """Report held-out symbols whose calibration count was zero."""
    calibration_counts = np.asarray(calibration_counts)
    evaluation_counts = np.asarray(evaluation_counts)
    if calibration_counts.shape != evaluation_counts.shape:
        raise ValueError("calibration/evaluation histogram shape mismatch")
    mask = (calibration_counts == 0) & (evaluation_counts > 0)
    return {
        "bins": int(mask.sum()),
        "values": int(evaluation_counts[mask].sum()),
    }


def grouped_counts(channel_counts: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Pool ``[C,A]`` channel histograms according to a fixed ``[C]`` mapping."""
    channel_counts = np.asarray(channel_counts)
    mapping = np.asarray(mapping, dtype=np.int64)
    if channel_counts.ndim != 2:
        raise ValueError(f"channel_counts must be [C,A], got {channel_counts.shape}")
    if mapping.shape != (channel_counts.shape[0],):
        raise ValueError(
            f"mapping must have shape {(channel_counts.shape[0],)}, got {mapping.shape}"
        )
    if mapping.size == 0 or int(mapping.min()) != 0:
        raise ValueError("mapping must contain contiguous group ids starting at zero")
    groups = int(mapping.max()) + 1
    if not np.array_equal(np.unique(mapping), np.arange(groups)):
        raise ValueError("mapping group ids must be contiguous")
    output = np.zeros((groups, channel_counts.shape[1]), dtype=channel_counts.dtype)
    np.add.at(output, mapping, channel_counts)
    return output


def agglomerative_entropy_mappings(
    channel_counts: np.ndarray,
    target_groups: tuple[int, ...] = (1, 4, 8, 16, 32, 512),
) -> dict[int, np.ndarray]:
    """Build a nested channel-group hierarchy minimizing entropy merge cost.

    Starting with one distribution per channel, the pair with the smallest
    increase in ideal calibration code length is merged repeatedly.  One run
    therefore yields consistent mappings for every requested K.
    """
    counts = np.asarray(channel_counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[0] == 0 or np.any(counts < 0):
        raise ValueError("channel_counts must be a non-empty non-negative [C,A] matrix")
    channels = counts.shape[0]
    targets = tuple(sorted(set(int(k) for k in target_groups), reverse=True))
    if not targets or targets[0] > channels or targets[-1] < 1:
        raise ValueError(f"target groups must lie in [1, {channels}]")

    group_counts_by_id = {index: counts[index].copy() for index in range(channels)}
    members = {index: [index] for index in range(channels)}
    entropies = {
        index: empirical_entropy_bits(group_counts_by_id[index])
        for index in range(channels)
    }
    active = set(range(channels))
    heap: list[tuple[float, int, int]] = []

    def merge_delta(left: int, right: int) -> float:
        merged = group_counts_by_id[left] + group_counts_by_id[right]
        return empirical_entropy_bits(merged) - entropies[left] - entropies[right]

    for left in range(channels):
        for right in range(left + 1, channels):
            heapq.heappush(heap, (merge_delta(left, right), left, right))

    snapshots: dict[int, np.ndarray] = {}

    def snapshot() -> None:
        group_order = sorted(active, key=lambda group_id: min(members[group_id]))
        mapping = np.empty(channels, dtype=np.int32)
        for output_id, group_id in enumerate(group_order):
            mapping[members[group_id]] = output_id
        snapshots[len(active)] = mapping

    if channels in targets:
        snapshot()
    next_id = channels
    minimum_target = min(targets)
    while len(active) > minimum_target:
        while heap:
            _, left, right = heapq.heappop(heap)
            if left in active and right in active:
                break
        else:  # pragma: no cover - indicates an internal heap invariant failure
            raise RuntimeError("agglomerative merge heap was exhausted")

        merged_counts = group_counts_by_id[left] + group_counts_by_id[right]
        group_counts_by_id[next_id] = merged_counts
        members[next_id] = members[left] + members[right]
        entropies[next_id] = empirical_entropy_bits(merged_counts)
        active.remove(left)
        active.remove(right)
        active.add(next_id)
        for other in active:
            if other == next_id:
                continue
            pair = (other, next_id) if other < next_id else (next_id, other)
            heapq.heappush(heap, (merge_delta(*pair), *pair))
        if len(active) in targets:
            snapshot()
        next_id += 1

    missing = set(targets) - set(snapshots)
    if missing:
        raise RuntimeError(f"failed to capture group mappings for K={sorted(missing)}")
    return {k: snapshots[k] for k in targets}


def evaluate_fixed_group_mapping(
    calibration_counts: np.ndarray,
    evaluation_counts: np.ndarray,
    mapping: np.ndarray,
    *,
    alpha: float = 0.5,
    cdf_precision: int | None = 16,
) -> dict[str, float | int | None]:
    """Evaluate a train-fixed channel grouping on held-out histograms."""
    calibration_grouped = grouped_counts(calibration_counts, mapping)
    evaluation_grouped = grouped_counts(evaluation_counts, mapping)
    pmf = smoothed_pmf(calibration_grouped, alpha)
    empirical_bits = empirical_entropy_bits(evaluation_grouped)
    fixed_ideal_bits = cross_entropy_bits(evaluation_grouped, pmf)
    fixed_cdf_bits = None
    if cdf_precision is not None:
        fixed_cdf_bits = cross_entropy_bits(
            evaluation_grouped, cdf16_probabilities(pmf, cdf_precision)
        )
    return {
        "groups": int(calibration_grouped.shape[0]),
        "empirical_bits": empirical_bits,
        "fixed_ideal_bits": fixed_ideal_bits,
        "fixed_cdf_quantized_bits": fixed_cdf_bits,
        "mismatch_penalty_bits": fixed_ideal_bits - empirical_bits,
    }
