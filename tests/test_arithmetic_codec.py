import json
import struct

import numpy as np
import pytest
import torch

from globalsplat.compression import (
    decode_scene_latents,
    decode_scene_latents_from_bytes,
    encode_scene_latents,
    encode_scene_latents_to_bytes,
    load_group_quantization_profile,
    load_grouped_cdf_profile,
)


def test_memory_api_matches_file_api_and_can_skip_hashes(tmp_path):
    generator = torch.Generator().manual_seed(11)
    tex = torch.randn(1, 17, 8, generator=generator)
    geo = torch.randn(1, 17, 8, generator=generator)
    path = tmp_path / "memory-equivalent.gsac"

    file_stats = encode_scene_latents(tex, geo, path, bits=8, backend="python")
    data, memory_stats = encode_scene_latents_to_bytes(
        tex, geo, bits=8, backend="python", verify_symbols=False
    )
    tex_hat, geo_hat, decode_stats = decode_scene_latents_from_bytes(
        data, return_stats=True, verify_symbols=False
    )

    assert data == path.read_bytes()
    assert memory_stats["bytes"] == file_stats["bytes"] == len(data)
    assert "texture_symbol_sha256" not in memory_stats
    assert "texture_symbol_sha256" not in decode_stats
    assert tex_hat.shape == tex.shape
    assert geo_hat.shape == geo.shape


def test_fixed_group_index_vector_is_cached(tmp_path):
    pytest.importorskip("compressai.ans")
    bundle = tmp_path / "cache-profile.npz"
    counts = np.ones((2, 7), dtype=np.int64)
    np.savez_compressed(
        bundle,
        texture_mapping_k2=np.asarray([0, 1], dtype=np.uint16),
        texture_calibration_counts_k2=counts,
        geometry_mapping_k2=np.asarray([1, 0], dtype=np.uint16),
        geometry_calibration_counts_k2=counts,
    )
    profile = load_grouped_cdf_profile(bundle, texture_groups=2, geometry_groups=2)
    tex = torch.randn(1, 5, 2)
    geo = torch.randn(1, 5, 2)

    data, _ = encode_scene_latents_to_bytes(
        tex, geo, texture_bits=3, geometry_bits=3, backend="compressai",
        scale_mode="channel", fixed_cdf_profile=profile,
    )
    texture_indexes = profile["texture"]["index_cache"][tex.numel()]
    decode_scene_latents_from_bytes(data, fixed_cdf_profile=profile)

    assert profile["texture"]["index_cache"][tex.numel()] is texture_indexes


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


def test_group_variable_quantization_is_opt_in_and_roundtrips(tmp_path):
    profile_path = tmp_path / "group-quant.json"
    profile_path.write_text(json.dumps({
        "group_size": 2,
        "sensitivity_psnr_drop": {
            "texture": [0.1, 0.001], "geometry": [0.2, 0.002],
        },
        "policies": {
            "moderate": {"critical": 1.0, "high": 1.25, "medium": 1.5, "low": 2.0},
        },
    }))
    profile = load_group_quantization_profile(
        profile_path, policy="moderate", ablation="joint", channels=4
    )
    tex = torch.tensor([[[-1.0, -0.5, 0.25, 0.1], [0.8, 0.4, -0.2, -0.05]]])
    geo = 2.0 * tex

    baseline, _ = encode_scene_latents_to_bytes(
        tex, geo, texture_bits=4, geometry_bits=5, scale_mode="channel",
        backend="python",
    )
    baseline_again, baseline_stats = encode_scene_latents_to_bytes(
        tex, geo, texture_bits=4, geometry_bits=5, scale_mode="channel",
        backend="python", group_quantization_profile=None,
    )
    varied, stats = encode_scene_latents_to_bytes(
        tex, geo, texture_bits=4, geometry_bits=5, scale_mode="channel",
        backend="python", group_quantization_profile=profile,
    )
    tex_hat, geo_hat = decode_scene_latents_from_bytes(varied)

    assert baseline == baseline_again
    assert baseline_stats["group_quantization_policy"] == "baseline"
    assert stats["group_quantization_policy"] == "moderate"
    assert stats["group_quantization_ablation"] == "joint"
    assert tex_hat.shape == tex.shape and geo_hat.shape == geo.shape
    assert not torch.equal(tex_hat, decode_scene_latents_from_bytes(baseline)[0])


def test_group_quantization_ablation_leaves_other_stream_at_one(tmp_path):
    profile_path = tmp_path / "group-quant.json"
    profile_path.write_text(json.dumps({
        "group_size": 2,
        "sensitivity_psnr_drop": {
            "texture": [0.1, 0.001], "geometry": [0.2, 0.002],
        },
        "policies": {
            "aggressive": {"critical": 0.75, "high": 1.0, "medium": 1.75, "low": 2.5},
        },
    }))
    profile = load_group_quantization_profile(
        profile_path, policy="aggressive", ablation="geometry_only", channels=4
    )
    assert np.all(profile["streams"]["texture"]["channel_multipliers"] == 1)
    assert np.any(profile["streams"]["geometry"]["channel_multipliers"] != 1)


def test_optimized_explicit_group_tables_and_no_clipping_constraint(tmp_path):
    profile_path = tmp_path / "optimized.json"
    payload = {
        "group_size": 2,
        "policy_group_multipliers": {
            "target_5pct": {
                "texture": [1.0, 2.0], "geometry": [1.25, 1.0],
            }
        },
    }
    profile_path.write_text(json.dumps(payload))
    profile = load_group_quantization_profile(
        profile_path, policy="target_5pct", channels=4
    )
    assert profile["streams"]["texture"]["group_multipliers"].tolist() == [1.0, 2.0]
    assert profile["streams"]["geometry"]["group_multipliers"].tolist() == [1.25, 1.0]

    payload["policy_group_multipliers"]["target_5pct"]["geometry"][0] = 0.75
    profile_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=">= 1.0"):
        load_group_quantization_profile(
            profile_path, policy="target_5pct", channels=4
        )
