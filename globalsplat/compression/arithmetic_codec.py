"""Transform-free scalar quantization + static arithmetic coding.

Geometry and texture latents are coded independently.  Their only layout
operation is row-major flattening; the original shapes are stored in the
bitstream and restored by the decoder.
"""
from __future__ import annotations

import json
import hashlib
import struct
import time
from pathlib import Path

import numpy as np
import torch


MAGIC_V1 = b"GSAC0001"
MAGIC_V2 = b"GSAC0002"
# New encodes use V2 (binary little-endian FP32 scales).  The decoder accepts
# both versions so existing experiment bitstreams remain readable.
MAGIC = MAGIC_V2
STATE_BITS = 32
FULL = 1 << STATE_BITS
HALF = FULL >> 1
QUARTER = HALF >> 1
THREE_QUARTERS = QUARTER * 3
MAX_TOTAL = 1 << 15


def _serialize_header(header: dict) -> bytes:
    """Serialize bitstream metadata in the canonical on-disk form."""
    return json.dumps(header, separators=(",", ":")).encode()


def _without_header_fields(header: dict, *fields: str) -> dict:
    """Return a shallow copy of each stream metadata dict without ``fields``."""
    omitted = set(fields)
    return {
        stream: {key: value for key, value in meta.items() if key not in omitted}
        for stream, meta in header.items()
    }


def _header_byte_breakdown(
    header_meta: dict,
    header: bytes,
    *,
    external_scale_bytes: int = 0,
) -> dict[str, int]:
    """Attribute the current JSON header to CDF, scale, and other metadata.

    V1 stores adaptive CDF ``counts`` and quantization ``scale`` arrays in one
    compact JSON object. V2 keeps counts in JSON but stores scales as raw FP32.
    Counterfactual serialization attributes every JSON byte (including field
    names and separators) exactly once.

    These values are returned as encoder statistics only; they are *not* added
    to the bitstream header, so existing files and decoders remain compatible.
    """
    no_cdf_meta = _without_header_fields(header_meta, "counts")
    no_cdf = _serialize_header(no_cdf_meta)
    json_header_bytes = len(header)
    cdf_side_info_bytes = json_header_bytes - len(no_cdf)
    has_json_scales = any("scale" in meta for meta in header_meta.values())
    if has_json_scales:
        no_side_info = _serialize_header(
            _without_header_fields(header_meta, "counts", "scale")
        )
        scale_side_info_bytes = len(no_cdf) - len(no_side_info)
        other_metadata_bytes = len(no_side_info)
    else:
        scale_side_info_bytes = int(external_scale_bytes)
        other_metadata_bytes = len(no_cdf)
    framing_bytes = len(MAGIC) + struct.calcsize("<I")
    return {
        "json_header_bytes": json_header_bytes,
        "framing_bytes": framing_bytes,
        "cdf_side_info_bytes": cdf_side_info_bytes,
        "scale_side_info_bytes": scale_side_info_bytes,
        "other_metadata_bytes": other_metadata_bytes,
        "header_bytes": framing_bytes + json_header_bytes + int(external_scale_bytes),
        "container_bytes": framing_bytes + other_metadata_bytes,
    }


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


def _quantized_cdfs_from_counts(
    counts: np.ndarray,
    *,
    alpha: float = 0.5,
    precision: int = 16,
) -> np.ndarray:
    """Convert grouped calibration counts to positive CompressAI CDF rows."""
    try:
        from compressai.entropy_models.entropy_models import pmf_to_quantized_cdf
    except ImportError as exc:
        raise RuntimeError(
            "Fixed grouped CDFs require CompressAI. Install it with: pip install compressai"
        ) from exc
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or np.any(counts < 0):
        raise ValueError("grouped counts must be a non-negative [K,A] matrix")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    pmfs = (counts + float(alpha)) / (
        counts.sum(axis=1, keepdims=True) + float(alpha) * counts.shape[1]
    )
    rows = []
    for pmf in pmfs:
        cdf = pmf_to_quantized_cdf(
            torch.from_numpy(pmf).float(), precision=int(precision)
        )
        row = cdf.cpu().numpy().astype(np.int32)
        if row.size != counts.shape[1] + 1 or np.any(np.diff(row) <= 0):
            raise RuntimeError("CompressAI produced an invalid fixed CDF")
        rows.append(row)
    return np.stack(rows)


def load_grouped_cdf_profile(
    path: str | Path,
    *,
    texture_groups: int,
    geometry_groups: int,
    alpha: float = 0.5,
    precision: int = 16,
) -> dict:
    """Load one deployment profile from the Phase-2 grouped histogram bundle.

    The returned profile contains only fixed channel mappings and integer CDF16
    tables. It is shared by all scenes and is never embedded in a scene file.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"grouped CDF profile bundle not found: {path}")
    selected: dict[str, dict] = {}
    digest = hashlib.sha256()
    with np.load(path) as bundle:
        for stream, groups in (
            ("texture", int(texture_groups)),
            ("geometry", int(geometry_groups)),
        ):
            mapping_key = f"{stream}_mapping_k{groups}"
            counts_key = f"{stream}_calibration_counts_k{groups}"
            if mapping_key not in bundle or counts_key not in bundle:
                raise KeyError(
                    f"profile bundle lacks {stream} K={groups}; "
                    f"missing {mapping_key!r} or {counts_key!r}"
                )
            mapping = np.asarray(bundle[mapping_key], dtype=np.int32)
            counts = np.asarray(bundle[counts_key], dtype=np.int64)
            if mapping.ndim != 1 or counts.ndim != 2:
                raise ValueError(f"invalid {stream} profile array shapes")
            if counts.shape[0] != groups or not np.array_equal(
                np.unique(mapping), np.arange(groups)
            ):
                raise ValueError(f"invalid {stream} K={groups} mapping")
            cdfs = _quantized_cdfs_from_counts(
                counts, alpha=alpha, precision=precision
            )
            digest.update(stream.encode())
            digest.update(np.asarray([groups, precision], dtype="<i4").tobytes())
            digest.update(mapping.astype("<i4", copy=False).tobytes())
            digest.update(cdfs.astype("<i4", copy=False).tobytes())
            selected[stream] = {
                "mapping": mapping,
                "cdfs": cdfs,
                "groups": groups,
                "channels": int(mapping.size),
                "alphabet": int(counts.shape[1]),
            }
    selected["profile_id"] = digest.hexdigest()
    selected["precision"] = int(precision)
    selected["alpha"] = float(alpha)
    selected["source"] = str(path)
    selected["profile_bytes_raw"] = int(
        sum(
            value["mapping"].astype(np.uint16).nbytes + value["cdfs"].nbytes
            for key, value in selected.items()
            if key in ("texture", "geometry")
        )
    )
    return selected


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


def _group_indexes(count: int, mapping: np.ndarray) -> np.ndarray:
    """Return rANS distribution indexes matching row-major ``[..., C]`` symbols."""
    mapping = np.asarray(mapping, dtype=np.int32)
    if mapping.ndim != 1 or count % mapping.size:
        raise ValueError(
            f"symbol count {count} is incompatible with {mapping.size} channels"
        )
    return np.tile(mapping, count // mapping.size)


def _rans_encode_grouped(
    symbols: np.ndarray, indexes: np.ndarray, cdfs: np.ndarray
) -> bytes:
    from compressai.ans import RansEncoder

    cdf_lists = cdfs.astype(np.int32, copy=False).tolist()
    return RansEncoder().encode_with_indexes(
        symbols.astype(np.int32, copy=False).tolist(),
        indexes.astype(np.int32, copy=False).tolist(),
        cdf_lists,
        [int(cdfs.shape[1])] * int(cdfs.shape[0]),
        [0] * int(cdfs.shape[0]),
    )


def _rans_decode_grouped(
    payload: bytes, indexes: np.ndarray, cdfs: np.ndarray
) -> np.ndarray:
    from compressai.ans import RansDecoder

    decoded = RansDecoder().decode_with_indexes(
        payload,
        indexes.astype(np.int32, copy=False).tolist(),
        cdfs.astype(np.int32, copy=False).tolist(),
        [int(cdfs.shape[1])] * int(cdfs.shape[0]),
        [0] * int(cdfs.shape[0]),
    )
    return np.asarray(decoded, dtype=np.int64)


def _symbol_sha256(symbols: np.ndarray) -> str:
    return hashlib.sha256(
        symbols.astype("<i4", copy=False).tobytes(order="C")
    ).hexdigest()


def _encode_stream(
    x: torch.Tensor,
    bits: int,
    backend: str,
    scale_mode: str,
    clip_percentile: float | None,
    fixed_profile: dict | None = None,
) -> tuple[dict, bytes, np.ndarray, dict]:
    symbols, scale = _quantize(x, bits, scale_mode, clip_percentile)
    alphabet = (1 << bits) - 1
    timing = {"group_dispatch_seconds": 0.0, "cdf_build_seconds": 0.0}
    if fixed_profile is not None:
        if backend != "compressai":
            raise ValueError("fixed grouped CDF profiles require backend='compressai'")
        if int(fixed_profile["alphabet"]) != alphabet:
            raise ValueError(
                f"profile alphabet {fixed_profile['alphabet']} does not match {bits}-bit "
                f"quantizer alphabet {alphabet}"
            )
        if int(fixed_profile["channels"]) != int(x.shape[-1]):
            raise ValueError("profile channel count does not match latent shape")
        dispatch_start = time.perf_counter()
        indexes = _group_indexes(symbols.size, fixed_profile["mapping"])
        timing["group_dispatch_seconds"] = time.perf_counter() - dispatch_start
        rans_start = time.perf_counter()
        payload = _rans_encode_grouped(symbols, indexes, fixed_profile["cdfs"])
        timing["rans_seconds"] = time.perf_counter() - rans_start
        cdf = None
    elif backend == "compressai":
        cdf_start = time.perf_counter()
        cdf = _compressai_cdf(symbols, alphabet)
        timing["cdf_build_seconds"] = time.perf_counter() - cdf_start
        rans_start = time.perf_counter()
        payload = _rans_encode(symbols, cdf)
        timing["rans_seconds"] = time.perf_counter() - rans_start
    elif backend == "python":
        cdf_start = time.perf_counter()
        cdf = _cdf_from_symbols(symbols, alphabet)
        timing["cdf_build_seconds"] = time.perf_counter() - cdf_start
        rans_start = time.perf_counter()
        payload = _arithmetic_encode(symbols, cdf)
        timing["rans_seconds"] = time.perf_counter() - rans_start
    else:
        raise ValueError(f"unknown codec backend: {backend!r}")
    meta = {
        "shape": list(x.shape),
        "scale": np.atleast_1d(scale).tolist(),
        "scale_mode": scale_mode,
        "clip_percentile": clip_percentile,
        "bits": bits,
        "count": int(symbols.size),
    }
    if fixed_profile is None:
        meta["counts"] = np.diff(cdf).astype(np.int64).tolist()
    meta["payload_bytes"] = len(payload)
    meta["backend"] = backend
    if fixed_profile is not None:
        meta["cdf_mode"] = "fixed_grouped"
        meta["groups"] = int(fixed_profile["groups"])
    timing["symbol_sha256"] = _symbol_sha256(symbols)
    return meta, payload, np.atleast_1d(scale).astype("<f4", copy=False), timing


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
    format_version: int = 2,
    fixed_cdf_profile: dict | None = None,
) -> dict:
    """Encode `[L,C]` or `[1,L,C]` texture/geometry tensors to one file.

    ``format_version=2`` stores per-scene scales as raw little-endian FP32;
    version 1 is retained only for compatibility experiments.
    """
    texture_bits = int(bits if texture_bits is None else texture_bits)
    geometry_bits = int(bits if geometry_bits is None else geometry_bits)
    if not 2 <= texture_bits <= 12 or not 2 <= geometry_bits <= 12:
        raise ValueError("texture_bits and geometry_bits must be in [2, 12]")
    if texture.shape != geometry.shape:
        raise ValueError(f"texture/geometry shape mismatch: {texture.shape} vs {geometry.shape}")
    if format_version not in (1, 2):
        raise ValueError("format_version must be 1 or 2")
    if fixed_cdf_profile is not None and format_version != 2:
        raise ValueError("fixed grouped CDF profiles require format_version=2")
    tex_profile = None if fixed_cdf_profile is None else fixed_cdf_profile["texture"]
    geo_profile = None if fixed_cdf_profile is None else fixed_cdf_profile["geometry"]
    tex_meta, tex_payload, tex_scale, tex_timing = _encode_stream(
        texture, texture_bits, backend, scale_mode, clip_percentile, tex_profile
    )
    geo_meta, geo_payload, geo_scale, geo_timing = _encode_stream(
        geometry, geometry_bits, backend, scale_mode, clip_percentile, geo_profile
    )
    magic = MAGIC_V1 if format_version == 1 else MAGIC_V2
    scale_blobs: list[bytes] = []
    if format_version == 2:
        for meta, scale in ((tex_meta, tex_scale), (geo_meta, geo_scale)):
            # Replace the verbose JSON number list with exact binary FP32.
            meta.pop("scale")
            blob = scale.tobytes(order="C")
            meta["scale_dtype"] = "<f4"
            meta["scale_count"] = int(scale.size)
            meta["scale_bytes"] = len(blob)
            scale_blobs.append(blob)
    header_meta = {"texture": tex_meta, "geometry": geo_meta}
    if fixed_cdf_profile is not None:
        header_meta["fixed_cdf"] = {
            "profile_id": fixed_cdf_profile["profile_id"],
            "precision": int(fixed_cdf_profile["precision"]),
            "texture_groups": int(tex_profile["groups"]),
            "geometry_groups": int(geo_profile["groups"]),
        }
    header = _serialize_header(header_meta)
    binary_scale_bytes = sum(len(blob) for blob in scale_blobs)
    header_stats = _header_byte_breakdown(
        header_meta, header, external_scale_bytes=binary_scale_bytes
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for blob in scale_blobs:
            f.write(blob)
        f.write(tex_payload)
        f.write(geo_payload)
    payload_bytes = len(tex_payload) + len(geo_payload)
    stats = {
        "bytes": path.stat().st_size,
        "payload_bytes": payload_bytes,
        "texture_payload_bytes": len(tex_payload),
        "geometry_payload_bytes": len(geo_payload),
        "num_values": texture.numel() + geometry.numel(),
        "bits_per_value": 8.0 * path.stat().st_size / (texture.numel() + geometry.numel()),
        "backend": backend,
        "texture_bits": texture_bits,
        "geometry_bits": geometry_bits,
        "scale_mode": scale_mode,
        "clip_percentile": clip_percentile,
        "format_version": format_version,
        "cdf_mode": "adaptive_scene_k1" if fixed_cdf_profile is None else "fixed_grouped",
        "texture_groups": 1 if fixed_cdf_profile is None else int(tex_profile["groups"]),
        "geometry_groups": 1 if fixed_cdf_profile is None else int(geo_profile["groups"]),
        "shared_profile_bytes_raw": (
            0 if fixed_cdf_profile is None else int(fixed_cdf_profile["profile_bytes_raw"])
        ),
        "texture_rans_encode_seconds": float(tex_timing["rans_seconds"]),
        "geometry_rans_encode_seconds": float(geo_timing["rans_seconds"]),
        "rans_encode_seconds": float(
            tex_timing["rans_seconds"] + geo_timing["rans_seconds"]
        ),
        "group_dispatch_encode_seconds": float(
            tex_timing["group_dispatch_seconds"] + geo_timing["group_dispatch_seconds"]
        ),
        "cdf_build_seconds": float(
            tex_timing["cdf_build_seconds"] + geo_timing["cdf_build_seconds"]
        ),
        "texture_symbol_sha256": tex_timing["symbol_sha256"],
        "geometry_symbol_sha256": geo_timing["symbol_sha256"],
    }
    stats.update(header_stats)
    if stats["bytes"] != stats["payload_bytes"] + stats["header_bytes"]:
        raise RuntimeError("internal codec byte accounting mismatch")
    return stats


def decode_scene_latents(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    fixed_cdf_profile: dict | None = None,
    return_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
    """Decode a file and return `(texture, geometry)` float32 tensors."""
    data = Path(path).read_bytes()
    magic = data[: len(MAGIC)]
    if magic not in (MAGIC_V1, MAGIC_V2):
        raise ValueError("not a GlobalSplat arithmetic-coded latent file")
    offset = len(MAGIC)
    header_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    header = json.loads(data[offset : offset + header_len])
    offset += header_len
    fixed_descriptor = header.get("fixed_cdf")
    if fixed_descriptor is not None:
        if fixed_cdf_profile is None:
            raise ValueError("this bitstream requires a shared fixed CDF profile")
        if fixed_cdf_profile["profile_id"] != fixed_descriptor["profile_id"]:
            raise ValueError("fixed CDF profile id does not match the bitstream")
    scales_by_stream: dict[str, np.ndarray] = {}
    if magic == MAGIC_V2:
        for name in ("texture", "geometry"):
            meta = header[name]
            if meta.get("scale_dtype") != "<f4":
                raise ValueError(f"unsupported V2 scale dtype: {meta.get('scale_dtype')!r}")
            scale_bytes = int(meta["scale_bytes"])
            scale_count = int(meta["scale_count"])
            if scale_bytes != scale_count * np.dtype("<f4").itemsize:
                raise ValueError("invalid V2 binary scale length")
            end = offset + scale_bytes
            if end > len(data):
                raise ValueError("truncated V2 binary scale data")
            scales_by_stream[name] = np.frombuffer(
                data[offset:end], dtype="<f4", count=scale_count
            ).astype(np.float32, copy=True)
            offset = end
    outputs = []
    decode_stats: dict[str, float | str] = {
        "rans_decode_seconds": 0.0,
        "group_dispatch_decode_seconds": 0.0,
    }
    for name in ("texture", "geometry"):
        meta = header[name]
        size = int(meta["payload_bytes"])
        payload = data[offset : offset + size]
        offset += size
        backend = meta.get("backend", "python")
        if meta.get("cdf_mode") == "fixed_grouped":
            stream_profile = fixed_cdf_profile[name]
            if int(meta["groups"]) != int(stream_profile["groups"]):
                raise ValueError(f"{name} group count does not match fixed profile")
            dispatch_start = time.perf_counter()
            indexes = _group_indexes(int(meta["count"]), stream_profile["mapping"])
            decode_stats["group_dispatch_decode_seconds"] += (
                time.perf_counter() - dispatch_start
            )
            rans_start = time.perf_counter()
            symbols = _rans_decode_grouped(payload, indexes, stream_profile["cdfs"])
            elapsed = time.perf_counter() - rans_start
        else:
            counts = np.asarray(meta["counts"], dtype=np.int64)
            cdf = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
            rans_start = time.perf_counter()
            if backend == "compressai":
                symbols = _rans_decode(payload, cdf, int(meta["count"]))
            elif backend == "python":
                symbols = _arithmetic_decode(payload, cdf, int(meta["count"]))
            else:
                raise ValueError(f"unknown codec backend in bitstream: {backend!r}")
            elapsed = time.perf_counter() - rans_start
        decode_stats[f"{name}_rans_decode_seconds"] = float(elapsed)
        decode_stats["rans_decode_seconds"] += float(elapsed)
        decode_stats[f"{name}_symbol_sha256"] = _symbol_sha256(symbols)
        qmax = (1 << (int(meta["bits"]) - 1)) - 1
        signed = (symbols - qmax).astype(np.float32).reshape(meta["shape"])
        scales = (
            scales_by_stream[name]
            if magic == MAGIC_V2
            else np.asarray(meta["scale"], dtype=np.float32)
        )
        if meta.get("scale_mode", "global") == "channel":
            scales = scales.reshape(*([1] * (signed.ndim - 1)), signed.shape[-1])
            array = signed * scales
        else:
            array = signed * float(scales.reshape(-1)[0])
        outputs.append(torch.from_numpy(array.copy()).to(device))
    if offset != len(data):
        raise ValueError(f"unexpected trailing bitstream bytes: {len(data) - offset}")
    if return_stats:
        return outputs[0], outputs[1], decode_stats
    return outputs[0], outputs[1]
