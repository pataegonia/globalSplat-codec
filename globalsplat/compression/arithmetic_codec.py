"""Transform-free scalar quantization + static arithmetic coding.

Geometry and texture latents are coded independently.  Their only layout
operation is row-major flattening; the original shapes are stored in the
bitstream and restored by the decoder.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch


MAGIC = b"GSAC0001"
STATE_BITS = 32
FULL = 1 << STATE_BITS
HALF = FULL >> 1
QUARTER = HALF >> 1
THREE_QUARTERS = QUARTER * 3
MAX_TOTAL = 1 << 15


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.current = 0
        self.used = 0

    def write(self, bit: int) -> None:
        self.current = (self.current << 1) | int(bit)
        self.used += 1
        if self.used == 8:
            self.data.append(self.current)
            self.current = self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.data.append(self.current << (8 - self.used))
        return bytes(self.data)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte = 0
        self.bit = 0

    def read(self) -> int:
        if self.byte >= len(self.data):
            return 0
        value = (self.data[self.byte] >> (7 - self.bit)) & 1
        self.bit += 1
        if self.bit == 8:
            self.byte += 1
            self.bit = 0
        return value


def _cdf_from_symbols(symbols: np.ndarray, alphabet: int) -> np.ndarray:
    counts = np.bincount(symbols, minlength=alphabet).astype(np.int64)
    nonzero = counts > 0
    if counts.sum() > MAX_TOTAL:
        scaled = np.zeros_like(counts)
        scaled[nonzero] = np.maximum(
            1, np.floor(counts[nonzero] * (MAX_TOTAL / counts.sum())).astype(np.int64)
        )
        # Rounding can exceed MAX_TOTAL by at most the alphabet size.
        while scaled.sum() > MAX_TOTAL:
            idx = int(np.argmax(scaled))
            if scaled[idx] > 1:
                scaled[idx] -= 1
            else:
                break
        counts = scaled
    return np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))


def _arithmetic_encode(symbols: np.ndarray, cdf: np.ndarray) -> bytes:
    writer = _BitWriter()
    low, high, pending = 0, FULL - 1, 0
    total = int(cdf[-1])

    def emit(bit: int) -> None:
        nonlocal pending
        writer.write(bit)
        for _ in range(pending):
            writer.write(1 - bit)
        pending = 0

    for symbol in symbols:
        symbol = int(symbol)
        width = high - low + 1
        high = low + (width * int(cdf[symbol + 1]) // total) - 1
        low = low + (width * int(cdf[symbol]) // total)
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                low -= HALF
                high -= HALF
                emit(1)
            elif low >= QUARTER and high < THREE_QUARTERS:
                low -= QUARTER
                high -= QUARTER
                pending += 1
            else:
                break
            low = low << 1
            high = (high << 1) + 1
    pending += 1
    emit(0 if low < QUARTER else 1)
    return writer.finish()


def _arithmetic_decode(payload: bytes, cdf: np.ndarray, count: int) -> np.ndarray:
    reader = _BitReader(payload)
    low, high, code = 0, FULL - 1, 0
    for _ in range(STATE_BITS):
        code = (code << 1) | reader.read()
    total = int(cdf[-1])
    out = np.empty(count, dtype=np.int64)
    for i in range(count):
        width = high - low + 1
        value = ((code - low + 1) * total - 1) // width
        symbol = int(np.searchsorted(cdf, value, side="right") - 1)
        out[i] = symbol
        high = low + (width * int(cdf[symbol + 1]) // total) - 1
        low = low + (width * int(cdf[symbol]) // total)
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                code -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                low -= QUARTER
                high -= QUARTER
                code -= QUARTER
            else:
                break
            low = low << 1
            high = (high << 1) + 1
            code = (code << 1) | reader.read()
    return out


def _quantize(
    x: torch.Tensor,
    bits: int,
    scale_mode: str,
    clip_percentile: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    qmax = (1 << (bits - 1)) - 1
    array = x.detach().float().cpu().numpy()
    if scale_mode == "global":
        reduce_axes = None
    elif scale_mode == "channel":
        reduce_axes = tuple(range(array.ndim - 1))
    else:
        raise ValueError("scale_mode must be 'global' or 'channel'")
    abs_array = np.abs(array)
    if clip_percentile is None:
        threshold = np.max(abs_array, axis=reduce_axes)
    else:
        if not 0.0 < float(clip_percentile) <= 100.0:
            raise ValueError("clip_percentile must be in (0, 100]")
        threshold = np.percentile(abs_array, float(clip_percentile), axis=reduce_axes)
    scale = np.maximum(
        np.asarray(threshold, dtype=np.float32) / qmax,
        np.finfo(np.float32).tiny,
    )
    signed = np.clip(np.rint(array / scale), -qmax, qmax).astype(np.int64)
    return (signed + qmax).reshape(-1), scale


def _compressai_cdf(symbols: np.ndarray, alphabet: int) -> np.ndarray:
    try:
        from compressai.entropy_models.entropy_models import pmf_to_quantized_cdf
    except ImportError as exc:
        raise RuntimeError(
            "CompressAI backend requested but compressai is not installed. "
            "Install it with: pip install compressai"
        ) from exc
    counts = np.bincount(symbols, minlength=alphabet).astype(np.float64)
    pmf = torch.from_numpy(counts / counts.sum()).float()
    return pmf_to_quantized_cdf(pmf, precision=16).cpu().numpy().astype(np.int64)


def _rans_encode(symbols: np.ndarray, cdf: np.ndarray) -> bytes:
    from compressai.ans import RansEncoder

    symbol_list = symbols.astype(np.int32, copy=False).tolist()
    indexes = [0] * len(symbol_list)
    return RansEncoder().encode_with_indexes(
        symbol_list,
        indexes,
        [cdf.astype(np.int32, copy=False).tolist()],
        [int(cdf.size)],
        [0],
    )


def _rans_decode(payload: bytes, cdf: np.ndarray, count: int) -> np.ndarray:
    from compressai.ans import RansDecoder

    decoded = RansDecoder().decode_with_indexes(
        payload,
        [0] * count,
        [cdf.astype(np.int32, copy=False).tolist()],
        [int(cdf.size)],
        [0],
    )
    return np.asarray(decoded, dtype=np.int64)


def _encode_stream(
    x: torch.Tensor,
    bits: int,
    backend: str,
    scale_mode: str,
    clip_percentile: float | None,
) -> tuple[dict, bytes]:
    symbols, scale = _quantize(x, bits, scale_mode, clip_percentile)
    alphabet = (1 << bits) - 1
    if backend == "compressai":
        cdf = _compressai_cdf(symbols, alphabet)
        payload = _rans_encode(symbols, cdf)
    elif backend == "python":
        cdf = _cdf_from_symbols(symbols, alphabet)
        payload = _arithmetic_encode(symbols, cdf)
    else:
        raise ValueError(f"unknown codec backend: {backend!r}")
    meta = {
        "shape": list(x.shape),
        "scale": np.atleast_1d(scale).tolist(),
        "scale_mode": scale_mode,
        "clip_percentile": clip_percentile,
        "bits": bits,
        "count": int(symbols.size),
        "counts": np.diff(cdf).astype(np.int64).tolist(),
        "payload_bytes": len(payload),
        "backend": backend,
    }
    return meta, payload


def encode_scene_latents(
    texture: torch.Tensor,
    geometry: torch.Tensor,
    path: str | Path,
    *,
    bits: int = 8,
    texture_bits: int | None = None,
    geometry_bits: int | None = None,
    backend: str = "compressai",
    scale_mode: str = "global",
    clip_percentile: float | None = None,
) -> dict:
    """Encode `[L,C]` or `[1,L,C]` texture/geometry tensors to one file."""
    texture_bits = int(bits if texture_bits is None else texture_bits)
    geometry_bits = int(bits if geometry_bits is None else geometry_bits)
    if not 2 <= texture_bits <= 12 or not 2 <= geometry_bits <= 12:
        raise ValueError("texture_bits and geometry_bits must be in [2, 12]")
    if texture.shape != geometry.shape:
        raise ValueError(f"texture/geometry shape mismatch: {texture.shape} vs {geometry.shape}")
    tex_meta, tex_payload = _encode_stream(
        texture, texture_bits, backend, scale_mode, clip_percentile
    )
    geo_meta, geo_payload = _encode_stream(
        geometry, geometry_bits, backend, scale_mode, clip_percentile
    )
    header = json.dumps({"texture": tex_meta, "geometry": geo_meta}, separators=(",", ":")).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        f.write(tex_payload)
        f.write(geo_payload)
    return {
        "bytes": path.stat().st_size,
        "texture_payload_bytes": len(tex_payload),
        "geometry_payload_bytes": len(geo_payload),
        "num_values": texture.numel() + geometry.numel(),
        "bits_per_value": 8.0 * path.stat().st_size / (texture.numel() + geometry.numel()),
        "backend": backend,
        "texture_bits": texture_bits,
        "geometry_bits": geometry_bits,
        "scale_mode": scale_mode,
        "clip_percentile": clip_percentile,
    }


def decode_scene_latents(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a file and return `(texture, geometry)` float32 tensors."""
    data = Path(path).read_bytes()
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError("not a GlobalSplat arithmetic-coded latent file")
    offset = len(MAGIC)
    header_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    header = json.loads(data[offset : offset + header_len])
    offset += header_len
    outputs = []
    for name in ("texture", "geometry"):
        meta = header[name]
        size = int(meta["payload_bytes"])
        payload = data[offset : offset + size]
        offset += size
        counts = np.asarray(meta["counts"], dtype=np.int64)
        cdf = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        backend = meta.get("backend", "python")
        if backend == "compressai":
            symbols = _rans_decode(payload, cdf, int(meta["count"]))
        elif backend == "python":
            symbols = _arithmetic_decode(payload, cdf, int(meta["count"]))
        else:
            raise ValueError(f"unknown codec backend in bitstream: {backend!r}")
        qmax = (1 << (int(meta["bits"]) - 1)) - 1
        signed = (symbols - qmax).astype(np.float32).reshape(meta["shape"])
        scales = np.asarray(meta["scale"], dtype=np.float32)
        if meta.get("scale_mode", "global") == "channel":
            scales = scales.reshape(*([1] * (signed.ndim - 1)), signed.shape[-1])
            array = signed * scales
        else:
            array = signed * float(scales.reshape(-1)[0])
        outputs.append(torch.from_numpy(array.copy()).to(device))
    return outputs[0], outputs[1]
