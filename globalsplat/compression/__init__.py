"""Lightweight codecs for GlobalSplat scene latents."""

from .arithmetic_codec import decode_scene_latents, encode_scene_latents

__all__ = ["encode_scene_latents", "decode_scene_latents"]
