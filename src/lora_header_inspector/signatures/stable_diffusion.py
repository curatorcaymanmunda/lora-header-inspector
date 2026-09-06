"""Stable Diffusion UNet and text-encoder signatures."""

from __future__ import annotations

from typing import Iterable

from ..models import SignatureMatch
from ..normalization import DetectionContext
from .registry import register


@register("stable-diffusion-unet", order=10)
def detect_stable_diffusion_unet(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    unet_layout = ctx.has(
        r"(?:^|[._])(?:down_blocks|input_blocks|up_blocks|output_blocks|"
        r"mid_block|middle_block)(?:[._]|$)"
    )
    if not unet_layout:
        return []

    dims = ctx.dims(r"attn2[._]to_k")
    known = sorted(dims.intersection({768, 1024, 2048}))
    details = []
    if len(known) == 1:
        architecture = {768: "SD15", 1024: "SD2", 2048: "SDXL"}[known[0]]
        return [
            SignatureMatch(
                rule_id=f"sd.unet.cross-attention.{known[0]}",
                family="SDUNet",
                architecture=architecture,
                confidence="high",
                priority=100,
                message=f"cross-attention context width={known[0]}",
            )
        ]
    if len(known) > 1:
        details.append(
            "multiple incompatible cross-attention widths="
            + ",".join(str(value) for value in known)
        )
        return [
            SignatureMatch(
                rule_id="sd.unet.cross-attention-conflict",
                family="SDUNet",
                confidence="high",
                priority=95,
                ambiguous=True,
                message=details[0],
            )
        ]
    if dims:
        details.append(
            "unrecognised cross-attention width(s)="
            + ",".join(str(value) for value in sorted(dims))
        )

    if ctx.has(r"(?:^|[._])lora_te2_|(?:^|[._])text_encoder_2(?:[._]|$)"):
        return [
            SignatureMatch(
                rule_id="sd.unet.second-text-encoder",
                family="SDUNet",
                architecture="SDXL",
                confidence="high",
                priority=90,
                message="second text encoder present with UNet layout",
                details=tuple(details),
            )
        ]

    naming = []
    if ctx.has(r"^lora_unet_input_blocks_"):
        naming.append("LDM-style input_blocks")
    if ctx.has(r"^lora_unet_down_blocks_|^unet[._]down_blocks"):
        naming.append("diffusers-style down_blocks")
    message = "UNet layout without a decisive cross-attention shape"
    if naming:
        message += " (" + ", ".join(naming) + ")"
    return [
        SignatureMatch(
            rule_id="sd.unet.shared-layout",
            family="SDUNet",
            confidence="medium" if naming else "low",
            priority=40,
            ambiguous=True,
            message=message,
            details=tuple(details),
        )
    ]


@register("stable-diffusion-text-encoder-only", order=900)
def detect_text_encoder_only(ctx: DetectionContext) -> Iterable[SignatureMatch]:
    if not ctx.has(r"^lora_te|^text_encoder(?:[._]|$)"):
        return []
    if ctx.has(r"(?:down_blocks|input_blocks|up_blocks|output_blocks|transformer_blocks)"):
        return []
    if ctx.has(r"^lora_te2_|^text_encoder_2(?:[._]|$)"):
        return [
            SignatureMatch(
                rule_id="sd.text-encoder-only.two-encoders",
                family="TextEncoderOnly",
                architecture="SDXL",
                confidence="medium",
                priority=35,
                ambiguous=True,
                message="two text encoders, no denoiser tensors",
            )
        ]
    return [
        SignatureMatch(
            rule_id="sd.text-encoder-only.single-encoder",
            family="TextEncoderOnly",
            confidence="low",
            priority=20,
            ambiguous=True,
            message="single text encoder, no denoiser tensors",
        )
    ]
