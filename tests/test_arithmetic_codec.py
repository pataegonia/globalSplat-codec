import torch

from globalsplat.compression import decode_scene_latents, encode_scene_latents


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
