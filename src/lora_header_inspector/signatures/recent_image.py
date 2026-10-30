"""Narrow image-model signatures; see docs/RECENT_IMAGE_SIGNATURES.md.

Require independent module anchors and adapter input widths. A sparse export
that omits these anchors remains unresolved rather than borrowing a filename.
"""
from __future__ import annotations

from ..models import SignatureMatch
from .registry import register


def _match(rule, family, architecture, message):
    return [SignatureMatch(rule_id=rule, family=family, architecture=architecture,
                           confidence="high", priority=110, message=message)]


@register("flux3-mode-streams", order=120)
def detect_flux3(ctx):
    image = r"(?:^|[._])content_mode_blocks[._]image[._]\d+[._](?:q_proj|attn_out)(?:[._]|$)"
    text = r"(?:^|[._])txt_mode_blocks[._]\d+[._](?:q_proj|attn_out)(?:[._]|$)"
    if ctx.dims(image) == {3072} and ctx.dims(text) == {3072}:
        return _match("flux3.mode-streams", "Flux3DiT", "Flux3",
                      "FLUX.3 Image Klein image/text mode streams with 3072-wide adapter inputs")
    return []


@register("moss-image-projections", order=121)
def detect_moss_image(ctx):
    gate = r"(?:^|[._])layers[._]\d+[._]proj[._]ffn_up_gate(?:[._]|$)"
    modulation = r"(?:^|[._])shared_block_modulation(?:[._]|$)"
    if ctx.dims(gate) == {3072} and ctx.dims(modulation) == {3072}:
        return _match("moss-image.shared-modulation", "MOSSImageDiT", "MOSSImage",
                      "MOSS-Image fused FFN gate and shared block modulation with 3072-wide inputs")
    return []


@register("fibo2-perceiver", order=122)
def detect_fibo2(ctx):
    perceiver = r"(?:^|[._])perceiver[._]layers[._]\d+[._]cross_attention[._]to_kv(?:[._]|$)"
    injection = r"(?:^|[._])layers[._]\d+[._]cross_attention[._]to_kv(?:[._]|$)"
    # Perceiver alone also matches the suffix of the injection pattern, so
    # explicitly exclude it when checking the independent main-layer anchor.
    main_keys = {key: info for key, info in ctx.header.items()
                 if "perceiver" not in key.lower()}
    main = type(ctx)(main_keys)
    if ctx.dims(perceiver) == {7680} and main.dims(injection) == {7680}:
        return _match("fibo2.perceiver-injection", "FIBO2DiT", "FIBO2",
                      "FIBO 2.0 Perceiver and main-layer cross-attention with 7680-wide context inputs")
    return []
