"""Flux-like dual/single stream model signatures."""

from __future__ import annotations

from typing import Iterable

from ..models import SignatureMatch
from ..normalization import DetectionContext
from .registry import register


@register("chroma-distilled-guidance", order=100)
def detect_chroma(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    if not ctx.has(r"(?:^|[._])distilled_guidance_layer(?:[._]|$)"):
        return []
    return [
        SignatureMatch(
            rule_id="chroma.distilled-guidance-layer",
            family="DualSingleStreamDiT",
            architecture="Chroma",
            confidence="high",
            priority=100,
            message="Chroma distilled_guidance_layer approximator naming",
        )
    ]


@register("hunyuan-token-refiner", order=110)
def detect_hunyuan(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    if not ctx.has(
        r"context_embedder.*token_refiner|context_embedder.*refiner_blocks|"
        r"token_refiner[._]refiner_blocks"
    ):
        return []
    return [
        SignatureMatch(
            rule_id="hunyuan.token-refiner",
            family="DualSingleStreamDiT",
            architecture="Hunyuan",
            confidence="high",
            priority=100,
            message="Hunyuan token-refiner module naming",
        )
    ]


@register("dual-single-stream-family", order=500)
def detect_dual_single_stream(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    explicit_dual = ctx.indices(
        r"double_blocks[._](\d+)",
        r"lora_transformer_double_blocks_(\d+)",
    )
    generic_transformer = ctx.indices(
        r"(?:^|[._])transformer_blocks[._](\d+)",
        r"lora_transformer_transformer_blocks_(\d+)",
    )
    single = ctx.indices(
        r"(?:single_blocks|single_transformer_blocks)[._](\d+)",
        r"lora_transformer_single_transformer_blocks_(\d+)",
    )
    if not explicit_dual and not single:
        return []

    dual = explicit_dual or (generic_transformer if single else set())
    dual_desc = f"{len(dual)} observed, max={max(dual)}" if dual else "none observed"
    single_desc = f"{len(single)} observed, max={max(single)}" if single else "none observed"
    details = [f"dual/single-stream layout; dual {dual_desc}; single {single_desc}"]

    if ctx.has(
        r"single_(?:transformer_)?blocks?[._]\d+.*(?:attn[._])?to_qkv_mlp_proj"
    ):
        return [
            SignatureMatch(
                rule_id="flux2.fused-single-qkv-mlp",
                family="DualSingleStreamDiT",
                architecture="Flux2",
                confidence="high",
                priority=100,
                message="Flux2 fused single-block QKV/MLP projection",
                details=tuple(details),
            )
        ]

    context_dims = ctx.dims(r"(?:^|[._])(?:context_embedder|txt_in)(?:[._]|$)")
    if 3584 in context_dims:
        return [
            SignatureMatch(
                rule_id="longcat.context-width-3584",
                family="DualSingleStreamDiT",
                architecture="LongCat",
                confidence="high",
                priority=95,
                message="3584-wide text-context projection in a dual/single-stream layout",
                details=tuple(details),
            )
        ]
    if 4096 in context_dims:
        details.append("4096-wide text-context projection is shared by Flux1 and Chroma")

    details.append("exact architecture unresolved; observed block coverage is not model depth")
    return [
        SignatureMatch(
            rule_id="flux-family.shared-dual-single",
            family="DualSingleStreamDiT",
            confidence="medium",
            priority=45,
            ambiguous=True,
            message=details[0],
            details=tuple(details[1:]),
        )
    ]
