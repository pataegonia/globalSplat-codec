import numpy as np
import torch

from globalsplat.compression import LearnedLatentTransformCodec
from globalsplat.compression.arithmetic_codec import _quantize


def test_residual_transforms_start_as_exact_identity():
    codec = LearnedLatentTransformCodec(channels=8, hidden=4)
    texture = torch.randn(2, 7, 8)
    geometry = torch.randn(2, 7, 8)

    texture_analysis, geometry_analysis = codec.analysis(texture, geometry)
    texture_hat, geometry_hat = codec.synthesis(texture, geometry)

    torch.testing.assert_close(texture_analysis, texture)
    torch.testing.assert_close(geometry_analysis, geometry)
    torch.testing.assert_close(texture_hat, texture)
    torch.testing.assert_close(geometry_hat, geometry)


def test_hard_quantizer_matches_production_channel_max_quantizer():
    latent = torch.randn(1, 19, 8, generator=torch.Generator().manual_seed(4))
    result = LearnedLatentTransformCodec.quantize(
        latent, 7, straight_through=False
    )
    symbols, scales = _quantize(latent, 7, "channel", None)
    signed = torch.from_numpy(symbols.reshape(latent.shape) - 63)

    torch.testing.assert_close(result.symbols.cpu(), signed.float())
    np.testing.assert_allclose(
        result.scale.squeeze(0).squeeze(0).numpy(), scales, rtol=1e-6, atol=0
    )


def test_rate_is_finite_and_all_four_transforms_receive_gradients():
    codec = LearnedLatentTransformCodec(channels=8, hidden=4)
    texture = torch.randn(1, 11, 8)
    geometry = torch.randn(1, 11, 8)
    output = codec(texture, geometry, straight_through=True)
    loss = (
        output["rate_bits_per_scalar"]
        + output["texture_hat"].square().mean()
        + output["geometry_hat"].square().mean()
    )
    loss.backward()

    assert torch.isfinite(loss)
    for name in ("g_a_geometry", "g_s_geometry", "g_a_texture", "g_s_texture"):
        module = getattr(codec, name)
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        ), name


def test_all_codec_tensors_have_nonzero_gradients_after_first_update():
    codec = LearnedLatentTransformCodec(channels=8, hidden=4)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-4)
    texture = torch.randn(1, 11, 8)
    geometry = torch.randn(1, 11, 8)

    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = codec(texture, geometry, straight_through=True)
        loss = (
            output["rate_bits_per_scalar"]
            + output["texture_hat"].square().mean()
            + output["geometry_hat"].square().mean()
        )
        loss.backward()
        if update == 1:
            for name, parameter in codec.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                assert torch.count_nonzero(parameter.grad), name
        optimizer.step()


def test_codec_state_roundtrip_preserves_hard_outputs_and_rate():
    codec = LearnedLatentTransformCodec(channels=8, hidden=4)
    texture = torch.randn(1, 11, 8)
    geometry = torch.randn(1, 11, 8)
    original = codec(texture, geometry, straight_through=False)

    restored = LearnedLatentTransformCodec(channels=8, hidden=4)
    restored.load_state_dict(codec.state_dict())
    reloaded = restored(texture, geometry, straight_through=False)

    torch.testing.assert_close(
        original["rate_bits_per_scalar"], reloaded["rate_bits_per_scalar"]
    )
    torch.testing.assert_close(original["texture_hat"], reloaded["texture_hat"])
    torch.testing.assert_close(original["geometry_hat"], reloaded["geometry_hat"])


def test_recommended_model_parameter_count():
    codec = LearnedLatentTransformCodec()
    assert codec.transform_parameter_count == 1_055_744
    assert codec.entropy_parameter_count == 2_048
    assert codec.trainable_parameter_count == 1_057_792


def test_texture_only_mode_uses_production_identity_geometry_path():
    codec = LearnedLatentTransformCodec(channels=8, hidden=4, texture_only=True)
    texture = torch.randn(1, 11, 8)
    geometry = torch.randn(1, 11, 8)
    with torch.no_grad():
        for module in (codec.g_a_geometry, codec.g_s_geometry):
            for parameter in module.parameters():
                parameter.fill_(3.0)

    output = codec(texture, geometry, straight_through=False)
    production_geometry = codec.quantize(
        geometry, codec.geometry_bits, straight_through=False
    ).reconstructed

    torch.testing.assert_close(output["geometry_analysis"], geometry)
    torch.testing.assert_close(output["geometry_hat"], production_geometry)
    assert not any(
        parameter.requires_grad
        for module in (codec.g_a_geometry, codec.g_s_geometry, codec.geometry_entropy)
        for parameter in module.parameters()
    )


def test_texture_only_mode_trains_only_texture_modules():
    codec = LearnedLatentTransformCodec(channels=8, hidden=4, texture_only=True)
    texture = torch.randn(1, 11, 8)
    geometry = torch.randn(1, 11, 8)
    output = codec(texture, geometry, straight_through=True)
    loss = output["rate_bits_per_scalar"] + output["texture_hat"].square().mean()
    loss.backward()

    for name, parameter in codec.named_parameters():
        if name.startswith(("g_a_texture.", "g_s_texture.", "texture_entropy.")):
            assert parameter.grad is not None, name
        else:
            assert not parameter.requires_grad, name

    full = LearnedLatentTransformCodec()
    texture_only = LearnedLatentTransformCodec(texture_only=True)
    assert texture_only.transform_parameter_count == 527_872
    assert texture_only.entropy_parameter_count == 1_024
    assert texture_only.trainable_parameter_count == 528_896
    assert texture_only.trainable_parameter_count < full.trainable_parameter_count
