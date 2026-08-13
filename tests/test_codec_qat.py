from globalsplat.model.globalsplat import GlobalSplat


def test_decoder_only_qat_freezes_everything_except_gaussian_decoder():
    model = GlobalSplat(
        static_only=True,
        latent_rep_token_amount=2,
        M_max=2,
        latent_codec_texture_bits=7,
        latent_codec_geometry_bits=9,
        latent_codec_scale_mode="channel",
        latent_codec_qat=True,
        latent_codec_qat_scope="decoder",
    )

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("gaussian_decoder.") for name in trainable)

    model.train()
    assert not model.slot_encoder.training
    assert model.gaussian_decoder.training
