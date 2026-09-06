"""Anima, MiniMax H3, Cosmos-style, and Wan structural signatures."""

from __future__ import annotations

from typing import Iterable

from ..models import SignatureMatch
from ..normalization import DetectionContext
from .registry import register


@register("anima-llm-adapter", order=20)
def detect_anima_adapter(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    if not ctx.has(
        r"(?:^|[._])llm_adapter[._]blocks[._]\d+[._]"
        r"(?:self_attn|cross_attn)[._](?:q_proj|k_proj|v_proj|o_proj)(?:[._]|$)"
    ):
        return []
    return [SignatureMatch(
        rule_id="anima.llm-adapter",
        family="CosmosPredict2DiT",
        architecture="Anima",
        confidence="high",
        priority=100,
        message="Anima-specific llm_adapter transformer path",
    )]


@register("minimax-h3", order=30)
def detect_minimax_h3(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    qkv = ctx.has(r"(?:^|[._])blocks[._]\d+[._]attn[._]qkv_proj(?:[._]|$)")
    media = ctx.has(r"(?:^|[._])video_patch_proj(?:[._]|$)") and ctx.has(
        r"(?:^|[._])audio_patch_proj(?:[._]|$)"
    )
    adaln = ctx.has(r"(?:^|[._])blocks[._]\d+[._]adaln_proj(?:[._]linear)?(?:[._]|$)")
    mlp = ctx.has(r"(?:^|[._])blocks[._]\d+[._]mlp[._](?:fc1|fc2)(?:[._]|$)")
    if not qkv and not media:
        return []
    qkv_dims = ctx.dims(r"(?:^|[._])blocks[._]\d+[._]attn[._]qkv_proj")
    fc2_dims = ctx.dims(r"(?:^|[._])blocks[._]\d+[._]mlp[._]fc2")
    details = ()
    if qkv_dims or fc2_dims:
        details = (
            "observed H3 widths="
            f"qkv:{','.join(map(str, sorted(qkv_dims))) or '-'};"
            f"fc2:{','.join(map(str, sorted(fc2_dims))) or '-'}",
        )
    if media or 5376 in qkv_dims or 14336 in fc2_dims or (qkv and adaln and mlp):
        return [SignatureMatch(
            rule_id="minimax-h3.omni-dit",
            family="MiniMaxH3OmniDiT",
            architecture="MiniMaxH3",
            confidence="high",
            priority=100,
            message="MiniMax H3 fused-QKV/AdaLN/audio-video projection layout",
            details=details,
        )]
    return [SignatureMatch(
        rule_id="minimax-h3.partial",
        family="MiniMaxH3OmniDiT",
        confidence="medium",
        priority=50,
        ambiguous=True,
        message="partial MiniMax H3 omni-DiT signature",
        details=details,
    )]


@register("cosmos-predict2-family", order=200)
def detect_cosmos_predict2(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    native = (
        ctx.has(r"(?:^|[._])blocks[._]\d+[._]self_attn[._](?:q_proj|k_proj|v_proj|output_proj)(?:[._]|$)")
        and ctx.has(r"(?:^|[._])blocks[._]\d+[._]cross_attn[._](?:q_proj|k_proj|v_proj|output_proj)(?:[._]|$)")
        and ctx.has(r"(?:^|[._])blocks[._]\d+[._]mlp[._](?:layer1|layer2)(?:[._]|$)")
    )
    if native:
        return [SignatureMatch(
            rule_id="cosmos-predict2.native-layout",
            family="CosmosPredict2DiT",
            confidence="medium",
            priority=55,
            ambiguous=True,
            message=(
                "Cosmos Predict2-style blocks/self_attn/cross_attn transformer; "
                "Anima requires metadata or llm_adapter evidence"
            ),
        )]
    diffusers = (
        ctx.has(r"(?:^|[._])transformer_blocks[._]\d+[._]attn1(?:[._]|$)")
        and ctx.has(r"(?:^|[._])transformer_blocks[._]\d+[._]attn2(?:[._]|$)")
        and ctx.has(r"(?:^|[._])transformer_blocks[._]\d+[._](?:ff|norm[123])(?:[._]|$)")
    )
    if diffusers:
        return [SignatureMatch(
            rule_id="dual-attention-transformer.shared-layout",
            family="DualAttentionTransformerDiT",
            confidence="medium",
            priority=45,
            ambiguous=True,
            message="generic transformer_blocks attn1/attn2/ff layout; exact video architecture requires metadata",
        )]
    return []


@register("wan-family", order=300)
def detect_wan(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    self_attn = ctx.has(r"(?:^|[._])blocks[._]\d+[._]self_attn(?:[._]|$)")
    cross_attn = ctx.has(r"(?:^|[._])blocks[._]\d+[._]cross_attn(?:[._]|$)")
    ffn = ctx.has(r"(?:^|[._])blocks[._]\d+[._]ffn(?:[._]|$)")
    if not (self_attn and cross_attn and ffn) or ctx.has(r"transformer_blocks"):
        return []
    dims = ctx.dims(r"blocks[._]\d+[._](?:self_attn|cross_attn)[._](?:k|to_k)")
    names = {1536: "1.3B", 3072: "5B", 5120: "14B"}
    known = [names[value] for value in sorted(dims) if value in names]
    detail = "Wan-style self-attention/cross-attention/FFN block layout"
    if known:
        detail += "; hidden-size hint=" + ",".join(known)
    return [SignatureMatch(
        rule_id="wan.block-layout",
        family="WanLikeVideoDiT",
        architecture="Wan",
        confidence="medium",
        priority=75,
        ambiguous=True,
        message=detail,
    )]
