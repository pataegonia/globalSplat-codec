import json
import struct

import numpy as np
import pytest
import torch

from globalsplat.compression import (
    decode_scene_latents,
    encode_scene_latents,
    load_grouped_cdf_profile,
    ste_quantize_dequantize,
)
from globalsplat.compression.arithmetic_codec import _quantize


def test_arithmetic_codec_roundtrip(tmp_path):
    generator = torch.Generator().manual_seed(7)
    tex = torch.randn(1, 31, 8, generator=generator) * 0.7
    geo = torch.randn(1, 31, 8, generator=generator) * 1.3 + 0.2
    path = tmp_path / "latents.gsac"

    stats = encode_scene_latents(tex, geo, path, bits=8, backend="python")
    tex_hat, geo_hat = decode_scene_latents(path)

    assert tex_hat.shape == tex.shape
    assert geo_hat.shape == geo.shape
    assert (tex - tex_hat).abs().max() <= tex.abs().max() / 127 + 1e-6
    assert (geo - geo_hat).abs().max() <= geo.abs().max() / 127 + 1e-6
    assert stats["bytes"] == path.stat().st_size


def test_per_channel_asymmetric_codec_roundtrip(tmp_path):
    generator = torch.Generator().manual_seed(8)
    tex = torch.randn(1, 31, 8, generator=generator) * torch.linspace(0.1, 2.0, 8)
    geo = torch.randn(1, 31, 8, generator=generator) * torch.linspace(2.0, 0.1, 8)
    path = tmp_path / "latents_channel.gsac"

    stats = encode_scene_latents(
        tex,
        geo,
        path,
        texture_bits=6,
        geometry_bits=10,
        scale_mode="channel",
        backend="python",
    )
    tex_hat, geo_hat = decode_scene_latents(path)

    assert tex_hat.shape == tex.shape
    assert geo_hat.shape == geo.shape
    assert stats["texture_bits"] == 6
    assert stats["geometry_bits"] == 10
    assert stats["scale_mode"] == "channel"
    assert stats["format_version"] == 2
    assert path.read_bytes().startswith(b"GSAC0002")


def _compact_header(meta):
    return json.dumps(meta, separators=(",", ":")).encode()


def _drop_fields(meta, *fields):
    omitted = set(fields)
    return {
        stream: {key: value for key, value in values.items() if key not in omitted}
        for stream, values in meta.items()
    }


def test_codec_reports_byte_exact_header_breakdown(tmp_path):
    tex = torch.tensor([[[-1.0, -0.25], [0.0, 0.5], [1.0, 0.75]]])
    geo = torch.tensor([[[-2.0, 0.125], [-0.5, 0.0], [1.5, 0.25]]])
    path = tmp_path / "breakdown.gsac"

    stats = encode_scene_latents(
        tex,
        geo,
        path,
        texture_bits=4,
        geometry_bits=5,
        scale_mode="channel",
        backend="python",
        format_version=1,
    )
    data = path.read_bytes()
    header_len = struct.unpack_from("<I", data, 8)[0]
    header = json.loads(data[12 : 12 + header_len])
    no_cdf = _compact_header(_drop_fields(header, "counts"))
    no_side_info = _compact_header(_drop_fields(header, "counts", "scale"))

    assert stats["json_header_bytes"] == header_len
    assert stats["framing_bytes"] == 12
    assert stats["cdf_side_info_bytes"] == header_len - len(no_cdf)
    assert stats["scale_side_info_bytes"] == len(no_cdf) - len(no_side_info)
    assert stats["other_metadata_bytes"] == len(no_side_info)
    assert stats["container_bytes"] == 12 + len(no_side_info)
    assert stats["header_bytes"] == 12 + header_len
    assert stats["payload_bytes"] == (
        stats["texture_payload_bytes"] + stats["geometry_payload_bytes"]
    )
    assert stats["bytes"] == stats["payload_bytes"] + stats["header_bytes"]
    assert stats["bytes"] == (
        stats["payload_bytes"]
        + stats["scale_side_info_bytes"]
        + stats["cdf_side_info_bytes"]
        + stats["container_bytes"]
    )
    assert "header_bytes" not in header["texture"]
    assert "header_bytes" not in header["geometry"]


def test_v2_binary_scales_are_exact_and_v1_remains_readable(tmp_path):
    generator = torch.Generator().manual_seed(10)
    tex = torch.randn(1, 31, 8, generator=generator) * torch.linspace(0.1, 2.0, 8)
    geo = torch.randn(1, 31, 8, generator=generator) * torch.linspace(2.0, 0.1, 8)
    v1_path = tmp_path / "v1.gsac"
    v2_path = tmp_path / "v2.gsac"

    v1 = encode_scene_latents(
        tex, geo, v1_path, texture_bits=7, geometry_bits=9,
        scale_mode="channel", backend="python", format_version=1,
    )
    v2 = encode_scene_latents(
        tex, geo, v2_path, texture_bits=7, geometry_bits=9,
        scale_mode="channel", backend="python", format_version=2,
    )
    tex_v1, geo_v1 = decode_scene_latents(v1_path)
    tex_v2, geo_v2 = decode_scene_latents(v2_path)

    assert v1_path.read_bytes().startswith(b"GSAC0001")
    assert v2_path.read_bytes().startswith(b"GSAC0002")
    assert torch.equal(tex_v1, tex_v2)
    assert torch.equal(geo_v1, geo_v2)
    assert v2["scale_side_info_bytes"] == 2 * tex.shape[-1] * 4
    assert v2["scale_side_info_bytes"] < v1["scale_side_info_bytes"]
    assert v2["bytes"] < v1["bytes"]

    v2_header_len = struct.unpack_from("<I", v2_path.read_bytes(), 8)[0]
    v2_header = json.loads(v2_path.read_bytes()[12 : 12 + v2_header_len])
    for stream in ("texture", "geometry"):
        assert "scale" not in v2_header[stream]
        assert v2_header[stream]["scale_dtype"] == "<f4"
        assert v2_header[stream]["scale_count"] == tex.shape[-1]
        assert v2_header[stream]["scale_bytes"] == tex.shape[-1] * 4


def test_channel_scales_have_more_side_info_than_global_scales(tmp_path):
    generator = torch.Generator().manual_seed(9)
    tex = torch.randn(1, 15, 8, generator=generator)
    geo = torch.randn(1, 15, 8, generator=generator)
    channel = encode_scene_latents(
        tex,
        geo,
        tmp_path / "channel.gsac",
        bits=8,
        scale_mode="channel",
        backend="python",
    )
    global_ = encode_scene_latents(
        tex,
        geo,
        tmp_path / "global.gsac",
        bits=8,
        scale_mode="global",
        backend="python",
    )

    assert channel["scale_side_info_bytes"] > global_["scale_side_info_bytes"]


def test_fixed_grouped_compressai_roundtrip_and_zero_cdf_side_info(tmp_path):
    pytest.importorskip("compressai.ans")
    bundle = tmp_path / "profiles.npz"
    np.savez_compressed(
        bundle,
        texture_mapping_k2=np.asarray([0, 1, 1], dtype=np.uint16),
        texture_calibration_counts_k2=np.asarray(
            [[20, 3, 2, 1, 2, 3, 20], [3, 5, 9, 15, 9, 5, 3]], dtype=np.int64
        ),
        geometry_mapping_k2=np.asarray([0, 0, 1], dtype=np.uint16),
        geometry_calibration_counts_k2=np.asarray(
            [
                [12, 6, 4, 3, 2, 2, 1, 1, 1, 1, 2, 2, 3, 4, 6],
                [1, 1, 1, 2, 3, 5, 8, 13, 8, 5, 3, 2, 1, 1, 1],
            ],
            dtype=np.int64,
        ),
    )
    profile = load_grouped_cdf_profile(
        bundle, texture_groups=2, geometry_groups=2
    )
    tex = torch.tensor(
        [[[-1.0, -0.5, 0.0], [0.25, 0.5, 1.0], [0.75, -0.25, 0.5]]]
    )
    geo = 1.5 * tex + 0.1
    path = tmp_path / "fixed.gsac"

    encoded = encode_scene_latents(
        tex,
        geo,
        path,
        texture_bits=3,
        geometry_bits=4,
        backend="compressai",
        scale_mode="channel",
        fixed_cdf_profile=profile,
    )
    tex_hat, geo_hat, decoded = decode_scene_latents(
        path, fixed_cdf_profile=profile, return_stats=True
    )

    assert tex_hat.shape == tex.shape
    assert geo_hat.shape == geo.shape
    assert encoded["cdf_mode"] == "fixed_grouped"
    assert encoded["texture_groups"] == 2
    assert encoded["geometry_groups"] == 2
    assert encoded["cdf_side_info_bytes"] == 0
    assert encoded["shared_profile_bytes_raw"] == profile["profile_bytes_raw"]
    assert encoded["texture_symbol_sha256"] == decoded["texture_symbol_sha256"]
    assert encoded["geometry_symbol_sha256"] == decoded["geometry_symbol_sha256"]

    data = path.read_bytes()
    header_len = struct.unpack_from("<I", data, 8)[0]
    header = json.loads(data[12 : 12 + header_len])
    assert "counts" not in header["texture"]
    assert "counts" not in header["geometry"]
    assert header["fixed_cdf"]["profile_id"] == profile["profile_id"]
    assert encoded["bytes"] == encoded["payload_bytes"] + encoded["header_bytes"]


def test_ste_qdq_matches_production_quantizer_and_passes_gradient():
    generator = torch.Generator().manual_seed(19)
    latent = torch.randn(1, 17, 6, generator=generator, requires_grad=True)
    bits = 7

    actual = ste_quantize_dequantize(latent, bits, scale_mode="channel")
    symbols, scales = _quantize(latent, bits, "channel", None)
    qmax = (1 << (bits - 1)) - 1
    expected = torch.from_numpy(
        ((symbols.reshape(latent.shape) - qmax) * scales.reshape(1, 1, -1)).astype(
            np.float32
        )
    )

    assert torch.equal(actual.detach(), expected)
    actual.sum().backward()
    assert latent.grad is not None
    assert torch.all(torch.isfinite(latent.grad))
