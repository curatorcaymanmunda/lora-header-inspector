"""Single-stream, joint-attention, and modern image-DiT signatures."""

from __future__ import annotations

from typing import Iterable

from ..models import SignatureMatch
from ..normalization import DetectionContext
from .registry import register


@register("auraflow", order=40)
def detect_auraflow(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    diffusers = (
        ctx.has(
            r"(?:^|[._])joint_transformer_blocks[._]\d+[._]attn[._]"
            r"(?:add_[qkv]_proj|to_add_out)(?:[._]|$)"
        )
        and ctx.has(
            r"(?:^|[._])single_transformer_blocks[._]\d+[._]attn[._]"
            r"(?:to_[qkv]|to_out)(?:[._]|$)"
        )
    )
    original = ctx.has(
        r"(?:^|[._])double_layers[._]\d+[._]attn[._]w1[qkv](?:[._]|$)"
    ) or ctx.has(
        r"(?:^|[._])single_layers[._]\d+[._]attn[._]w1[qkv](?:[._]|$)"
    )
    if not diffusers and not original:
        return []
    return [
        SignatureMatch(
            rule_id="auraflow.joint-single" if diffusers else "auraflow.double-single-native",
            family="AuraFlowDiT",
            architecture="AuraFlow",
            confidence="high",
            priority=100,
            message=(
                "AuraFlow joint/single transformer layout"
                if diffusers
                else "AuraFlow double_layers/single_layers w1q layout"
            ),
        )
    ]


@register("sd3-native-joint-blocks", order=50)
def detect_sd3_native(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    if not ctx.has(r"(?:^|[._])joint_blocks[._]\d+[._](?:x_block|context_block)(?:[._]|$)"):
        return []
    return [
        SignatureMatch(
            rule_id="sd3.joint-blocks-native",
            family="JointAttentionDiT",
            architecture="SD3",
            confidence="high",
            priority=100,
            message="SD3 MMDiT joint_blocks x_block/context_block naming",
        )
    ]


@register("zimage-ideogram4", order=60)
def detect_zimage_ideogram4(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    zimage_refiner = ctx.has(
        r"(?:^|[._])(?:noise_refiner|context_refiner)[._]\d+[._]"
        r"(?:attention|feed_forward)(?:[._]|$)"
    )
    ideogram_modules = ctx.has(r"(?:^|[._])(?:llm_cond_proj|embed_image_indicator)(?:[._]|$)")
    separate_attention = ctx.has(
        r"(?:^|[._])layers[._]\d+[._]attention[._](?:to_[qkv]|to_out)(?:[._]|$)"
    )
    fused_attention = ctx.has(
        r"(?:^|[._])layers[._]\d+[._]attention[._](?:qkv|o)(?:[._]|$)"
    )
    swiglu = ctx.has(r"(?:^|[._])layers[._]\d+[._]feed_forward[._]w[123](?:[._]|$)")

    if zimage_refiner:
        return [
            SignatureMatch(
                rule_id="zimage.refiner-path",
                family="ZImageDiT",
                architecture="ZImage",
                confidence="high",
                priority=100,
                message="Z-Image noise/context refiner module naming",
            )
        ]
    if ideogram_modules:
        return [
            SignatureMatch(
                rule_id="ideogram4.llm-indicator-modules",
                family="Ideogram4DiT",
                architecture="Ideogram4",
                confidence="high",
                priority=100,
                message="Ideogram 4 LLM-condition/image-indicator module naming",
            )
        ]
    if not (separate_attention or fused_attention or swiglu):
        return []

    attention_dims = ctx.dims(
        r"(?:^|[._])layers[._]\d+[._]attention[._](?:to_[qkv]|to_out|qkv|o)"
    )
    w2_dims = ctx.dims(r"(?:^|[._])layers[._]\d+[._]feed_forward[._]w2")
    naming = "fused-qkv" if fused_attention else "separate-qkv"
    detail = f"layers.N attention ({naming}) + w1/w2/w3 single-stream layout"

    if 3840 in attention_dims or 10240 in w2_dims:
        return [
            SignatureMatch(
                rule_id="zimage.width-signature",
                family="ZImageDiT",
                architecture="ZImage",
                confidence="high",
                priority=95,
                message="Z-Image width signature (hidden=3840 or feed-forward=10240)",
                details=(detail,),
            )
        ]
    if 4608 in attention_dims or 12288 in w2_dims:
        return [
            SignatureMatch(
                rule_id="ideogram4.width-signature",
                family="Ideogram4DiT",
                architecture="Ideogram4",
                confidence="high",
                priority=95,
                message="Ideogram 4 width signature (hidden=4608 or feed-forward=12288)",
                details=(detail,),
            )
        ]

    widths = (
        "unrecognised single-stream widths="
        f"attention:{','.join(map(str, sorted(attention_dims))) or '-'};"
        f"w2:{','.join(map(str, sorted(w2_dims))) or '-'}"
        if attention_dims or w2_dims
        else "Z-Image/Ideogram-like layout without decisive adapter dimensions"
    )
    return [
        SignatureMatch(
            rule_id="nextdit.shared-single-stream",
            family="SingleStreamSwiGLUDiT",
            confidence="medium",
            priority=50,
            ambiguous=True,
            message=detail,
            details=(widths,),
        )
    ]


@register("ernie-image", order=70)
def detect_ernie_image(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    attention = ctx.has(
        r"(?:^|[._])layers[._]\d+[._]self_attention[._](?:to_[qkv]|to_out)(?:[._]|$)"
    )
    mlp = ctx.has(
        r"(?:^|[._])layers[._]\d+[._]mlp[._](?:gate_proj|up_proj|linear_fc2)(?:[._]|$)"
    )
    if not attention and not mlp:
        return []
    attention_dims = ctx.dims(
        r"(?:^|[._])layers[._]\d+[._]self_attention[._](?:to_[qkv]|to_out)"
    )
    fc2_dims = ctx.dims(r"(?:^|[._])layers[._]\d+[._]mlp[._]linear_fc2")
    details = []
    if attention_dims or fc2_dims:
        details.append(
            "observed ERNIE widths="
            f"attention:{','.join(map(str, sorted(attention_dims))) or '-'};"
            f"fc2:{','.join(map(str, sorted(fc2_dims))) or '-'}"
        )
    if 3072 in attention_dims or 8192 in fc2_dims or (attention and mlp):
        return [
            SignatureMatch(
                rule_id="ernie-image.layer-signature",
                family="ErnieImageDiT",
                architecture="ErnieImage",
                confidence="high",
                priority=95,
                message="ERNIE-Image self_attention + gated MLP signature",
                details=tuple(details),
            )
        ]
    return [
        SignatureMatch(
            rule_id="ernie-image.partial",
            family="ErnieImageDiT",
            confidence="medium",
            priority=50,
            ambiguous=True,
            message="partial ERNIE-Image layer signature",
            details=tuple(details),
        )
    ]


@register("krea2", order=80)
def detect_krea2(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    text_fusion = ctx.has(
        r"(?:^|[._])(?:text_fusion|txtfusion)[._](?:layerwise_blocks|refiner_blocks)[._]\d+"
    )
    attention_diffusers = ctx.has(
        r"(?:^|[._])transformer_blocks[._]\d+[._]attn[._]to_gate(?:[._]|$)"
    )
    attention_comfy = ctx.has(
        r"(?:^|[._])blocks[._]\d+[._]attn[._](?:wq|wk|wv|gate|wo)(?:[._]|$)"
    )
    swiglu_diffusers = ctx.has(
        r"(?:^|[._])transformer_blocks[._]\d+[._]ff[._](?:gate|up|down)(?:[._]|$)"
    )
    swiglu_comfy = ctx.has(
        r"(?:^|[._])blocks[._]\d+[._]mlp[._](?:gate|up|down)(?:[._]|$)"
    )
    attention = attention_diffusers or attention_comfy
    swiglu = swiglu_diffusers or swiglu_comfy
    if not text_fusion and not (attention and swiglu):
        return []

    attention_dims = ctx.dims(
        r"(?:^|[._])(?:transformer_blocks[._]\d+[._]attn[._]"
        r"(?:to_[qkv]|to_gate|to_out)|blocks[._]\d+[._]attn[._](?:wq|wk|wv|gate|wo))"
    )
    ff_down_dims = ctx.dims(
        r"(?:^|[._])(?:transformer_blocks[._]\d+[._]ff[._]down|"
        r"blocks[._]\d+[._]mlp[._]down)"
    )
    details = []
    if attention_comfy or swiglu_comfy:
        details.append("Krea 2 ComfyUI blocks/attn.wq/mlp naming")
    if attention_dims or ff_down_dims:
        details.append(
            "observed Krea 2 widths="
            f"attention:{','.join(map(str, sorted(attention_dims))) or '-'};"
            f"ff-down:{','.join(map(str, sorted(ff_down_dims))) or '-'}"
        )
    if text_fusion or 6144 in attention_dims or 16384 in ff_down_dims:
        return [
            SignatureMatch(
                rule_id="krea2.gqa-text-fusion-swiglu",
                family="Krea2DiT",
                architecture="Krea2",
                confidence="high",
                priority=100,
                message="Krea 2 grouped-query attention/text-fusion/SwiGLU signature",
                details=tuple(details),
            )
        ]
    return [
        SignatureMatch(
            rule_id="krea2.partial",
            family="Krea2DiT",
            confidence="medium",
            priority=50,
            ambiguous=True,
            message="partial Krea 2 transformer signature",
            details=tuple(details),
        )
    ]


@register("generic-transformer-blocks", order=950)
def detect_generic_transformer(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    indices = ctx.indices(
        r"(?:^|[._])transformer_blocks[._](\d+)",
        r"lora_transformer_transformer_blocks_(\d+)",
    )
    if not indices:
        return []

    if ctx.has(r"(?:^|[._])img_mod(?:[._]|$)") and ctx.has(r"(?:^|[._])txt_mod(?:[._]|$)"):
        return [
            SignatureMatch(
                rule_id="qwen-image.img-txt-mod",
                family="JointAttentionDiT",
                architecture="QwenImage",
                confidence="high",
                priority=100,
                message="paired img_mod and txt_mod modules",
            )
        ]

    if ctx.has(r"norm1_context|add_[qkv]_proj"):
        joint_dims = ctx.dims(
            r"(?:^|[._])transformer_blocks[._]\d+[._]attn2?[._]"
            r"(?:to_[qkv]|add_[qkv]_proj|to_out|to_add_out)"
        )
        known_sd3 = sorted(joint_dims.intersection({1536, 2432}))
        if len(known_sd3) == 1 and not joint_dims - {1536, 2432}:
            return [
                SignatureMatch(
                    rule_id="sd3.diffusers-width",
                    family="JointAttentionDiT",
                    architecture="SD3",
                    confidence="high",
                    priority=95,
                    message=f"joint-attention layout with SD3-family hidden width={known_sd3[0]}",
                )
            ]
        return [
            SignatureMatch(
                rule_id="joint-attention.shared-layout",
                family="JointAttentionDiT",
                confidence="medium",
                priority=45,
                ambiguous=True,
                message="joint-attention transformer layout; exact architecture unresolved",
            )
        ]

    if ctx.has(r"transformer_blocks[._]\d+.*attn1") and ctx.has(
        r"transformer_blocks[._]\d+.*attn2"
    ):
        return [
            SignatureMatch(
                rule_id="video-transformer.shared-attn1-attn2",
                family="VideoTransformerDiT",
                confidence="medium",
                priority=35,
                ambiguous=True,
                message="attn1/attn2 video-transformer layout; exact architecture unresolved",
            )
        ]

    return [
        SignatureMatch(
            rule_id="transformer.generic-blocks",
            family="TransformerDiT",
            confidence="low",
            priority=10,
            ambiguous=True,
            message="generic transformer_blocks layout",
        )
    ]
