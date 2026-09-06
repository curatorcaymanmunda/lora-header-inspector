#!/usr/bin/env python3
"""Inspect LoRA safetensors headers and produce a conservative sort index.

The script reads only the safetensors header. It never imports torch or loads tensor
payloads. Classification keeps four ideas separate:

* structural_family: the broad tensor layout that is physically present
* arch: the exact base architecture, only when evidence supports one
* ecosystem / model_variant / techniques: descriptive tags that may coexist
* status: ok, ambiguous, conflict, or error

Nothing is moved by this script. When inspect_sort_lora.py is importable, a dry-run
sort plan is generated automatically.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

TOOL_VERSION = "0.5.0"
CSV_NAME = "lora_inspect_index.csv"
MAX_HEADER_BYTES = 256 * 1024 * 1024

CONF_RANK = {"low": 0, "medium": 1, "high": 2}

FIELDNAMES = [
    "input_path",
    "path",
    "source_was_symlink",
    "file_size",
    "mtime_ns",
    "header_sha256",
    "header_bytes",
    "arch",
    "structural_family",
    "model_version",
    "task_variant",
    "noise_stage",
    "confidence",
    "status",
    "ecosystem",
    "model_variant",
    "unresolved_hint",
    "techniques",
    "lineage",              # compatibility/display field
    "sort_subfolder",       # default secondary folder, hand-editable
    "format",
    "structure_arch",
    "metadata_arch",
    "name_arch",
    "evidence",
    "total_tensors",
    "error",
    "move",                 # filled or hand-edited later
    "destination",          # filled or hand-edited later
]


@dataclass(frozen=True)
class HeaderData:
    tensors: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    header_sha256: str
    header_bytes: int


@dataclass
class StructuralResult:
    family: str = "Unknown"
    arch: Optional[str] = None
    confidence: str = "low"
    ambiguous: bool = False
    evidence: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArchSignal:
    arch: str
    confidence: str
    source: str
    detail: str
    priority: int


@dataclass
class HintTags:
    ecosystems: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)
    unresolved_hints: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    model_version: str = ""
    task_variant: str = ""
    noise_stage: str = ""
    evidence: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Safetensors header reader
# ---------------------------------------------------------------------------

def read_header(path: Path, max_header_bytes: int = MAX_HEADER_BYTES) -> HeaderData:
    """Read and validate the JSON header without reading tensor payloads."""
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError("file too short to contain a safetensors header")

        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len < 2:
            raise ValueError("invalid safetensors header length")
        if header_len > max_header_bytes:
            raise ValueError(
                f"header length {header_len:,} exceeds safety limit {max_header_bytes:,}"
            )

        raw_header = fh.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError("truncated safetensors header")

    try:
        decoded = raw_header.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("safetensors header is not valid UTF-8") from exc

    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors JSON header: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("safetensors header root is not a JSON object")

    metadata_raw = parsed.pop("__metadata__", {}) or {}
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

    tensors: Dict[str, Dict[str, Any]] = {}
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, dict):
            tensors[key] = value

    return HeaderData(
        tensors=tensors,
        metadata=metadata,
        header_sha256=hashlib.sha256(raw_header).hexdigest(),
        header_bytes=header_len,
    )


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _absolute_without_resolving(path: Path) -> Path:
    """Return an absolute path while preserving the final symlink component."""
    return Path(os.path.abspath(os.fspath(path)))


def _unique(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _join_values(items: Iterable[str]) -> str:
    return ";".join(_unique(items))


def _camel_split(text: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)


def _hint_text(text: str) -> str:
    text = _camel_split(text)
    text = text.replace("\\", " ").replace("/", " ")
    return text.lower()


def _bounded(pattern: str) -> str:
    """Wrap a hint regex so it cannot start or end inside an alphanumeric word."""
    return r"(?<![a-z0-9])(?:" + pattern + r")(?![a-z0-9])"


def _matches_hint(text: str, pattern: str) -> bool:
    return re.search(_bounded(pattern), text, re.IGNORECASE) is not None


def _has(keys: Sequence[str], pattern: str) -> bool:
    regex = re.compile(pattern, re.IGNORECASE)
    return any(regex.search(key) for key in keys)


def _observed_indices(keys: Sequence[str], patterns: Sequence[str]) -> Set[int]:
    result: Set[int] = set()
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for key in keys:
        for regex in compiled:
            match = regex.search(key)
            if match:
                result.add(int(match.group(1)))
    return result


def _adapter_input_dims(
    header: Mapping[str, Mapping[str, Any]], key_pattern: str
) -> Set[int]:
    """Collect input widths from 2-D LoRA/LoHa/LoKr down-projection tensors."""
    key_regex = re.compile(key_pattern, re.IGNORECASE)
    down_regex = re.compile(
        r"(?:lora_down|lora_A(?:[._]default)?|hada_w1_b|lokr_w1_b|dora_down)",
        re.IGNORECASE,
    )
    dims: Set[int] = set()
    for key, info in header.items():
        if not key_regex.search(key) or not down_regex.search(key):
            continue
        shape = info.get("shape") if isinstance(info, Mapping) else None
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        try:
            dims.add(int(shape[-1]))
        except (TypeError, ValueError):
            continue
    return dims


def detect_format(keys: Sequence[str]) -> str:
    lower = [key.lower() for key in keys]
    if any("hada_" in key or "lokr_" in key for key in lower):
        return "lycoris"
    if any(
        key.startswith(("lora_unet_", "lora_te", "lora_transformer_"))
        for key in lower
    ):
        return "kohya"
    if any(
        key.startswith(("unet.", "text_encoder", "transformer."))
        for key in lower
    ):
        return "diffusers"
    if any(key.startswith("diffusion_model.") for key in lower):
        return "comfy"
    return "unknown"


# ---------------------------------------------------------------------------
# Structural classification
# ---------------------------------------------------------------------------

def arch_from_structure(header: Mapping[str, Mapping[str, Any]]) -> StructuralResult:
    """Classify tensor layout conservatively.

    A sparse LoRA shows only the modules it targets. Observed maximum block indices
    are therefore evidence about coverage, not reliable model-depth measurements.
    """
    keys = list(header)
    result = StructuralResult()

    # Stable Diffusion UNet family.
    unet_layout = _has(
        keys,
        r"(?:^|[._])(?:down_blocks|input_blocks|up_blocks|output_blocks|"
        r"mid_block|middle_block)(?:[._]|$)",
    )
    if unet_layout:
        result.family = "SDUNet"
        dims = _adapter_input_dims(header, r"attn2[._]to_k")
        known = sorted(dims.intersection({768, 1024, 2048}))
        if len(known) == 1:
            arch = {768: "SD15", 1024: "SD2", 2048: "SDXL"}[known[0]]
            result.arch = arch
            result.confidence = "high"
            result.evidence.append(f"cross-attention context width={known[0]}")
            return result
        if len(known) > 1:
            result.evidence.append(
                "multiple incompatible cross-attention widths="
                + ",".join(str(value) for value in known)
            )
            return result
        if dims:
            result.evidence.append(
                "unrecognised cross-attention width(s)="
                + ",".join(str(value) for value in sorted(dims))
            )

        if _has(keys, r"(?:^|[._])lora_te2_|(?:^|[._])text_encoder_2(?:[._]|$)"):
            result.arch = "SDXL"
            result.confidence = "high"
            result.evidence.append("second text encoder present with UNet layout")
            return result

        naming = []
        if _has(keys, r"^lora_unet_input_blocks_"):
            naming.append("LDM-style input_blocks")
        if _has(keys, r"^lora_unet_down_blocks_|^unet[._]down_blocks"):
            naming.append("diffusers-style down_blocks")
        result.confidence = "medium" if naming else "low"
        result.evidence.append(
            "UNet layout without a decisive cross-attention shape"
            + (f" ({', '.join(naming)})" if naming else "")
        )
        return result

    # Anima adds a model-specific LLM adapter around its Cosmos Predict2-style
    # denoiser. Most Anima LoRAs intentionally avoid this module, but when it is
    # present it is an exact signature rather than a family resemblance.
    if _has(
        keys,
        r"(?:^|[._])llm_adapter[._]blocks[._]\d+[._]"
        r"(?:self_attn|cross_attn)[._](?:q_proj|k_proj|v_proj|o_proj)(?:[._]|$)",
    ):
        result.family = "CosmosPredict2DiT"
        result.arch = "Anima"
        result.confidence = "high"
        result.evidence.append("Anima-specific llm_adapter transformer path")
        return result

    # MiniMax H3 is an omni-DiT with fused QKV, multimodal AdaLN, separate
    # audio/video patch projections, and a 5,376-wide main stream.
    minimax_qkv = _has(
        keys,
        r"(?:^|[._])blocks[._]\d+[._]attn[._]qkv_proj(?:[._]|$)",
    )
    minimax_media = _has(keys, r"(?:^|[._])video_patch_proj(?:[._]|$)") and _has(
        keys, r"(?:^|[._])audio_patch_proj(?:[._]|$)"
    )
    minimax_adaln = _has(
        keys,
        r"(?:^|[._])blocks[._]\d+[._]adaln_proj(?:[._]linear)?(?:[._]|$)",
    )
    minimax_mlp = _has(
        keys,
        r"(?:^|[._])blocks[._]\d+[._]mlp[._](?:fc1|fc2)(?:[._]|$)",
    )
    if minimax_media or minimax_qkv:
        result.family = "MiniMaxH3OmniDiT"
        qkv_dims = _adapter_input_dims(
            header,
            r"(?:^|[._])blocks[._]\d+[._]attn[._]qkv_proj",
        )
        fc2_dims = _adapter_input_dims(
            header,
            r"(?:^|[._])blocks[._]\d+[._]mlp[._]fc2",
        )
        exact_width = 5376 in qkv_dims or 14336 in fc2_dims
        exact_modules = minimax_media or (minimax_qkv and minimax_adaln and minimax_mlp)
        if exact_width or exact_modules:
            result.arch = "MiniMaxH3"
            result.confidence = "high"
            result.evidence.append(
                "MiniMax H3 fused-QKV/AdaLN/audio-video projection layout"
            )
            if qkv_dims or fc2_dims:
                result.evidence.append(
                    "observed H3 widths="
                    f"qkv:{','.join(map(str, sorted(qkv_dims))) or '-'};"
                    f"fc2:{','.join(map(str, sorted(fc2_dims))) or '-'}"
                )
            return result
        result.confidence = "medium"
        result.ambiguous = True
        result.evidence.append("partial MiniMax H3 omni-DiT signature")
        return result

    # AuraFlow uses joint_transformer_blocks plus single_transformer_blocks in
    # Diffusers, and double_layers/single_layers with w1q-style projections in
    # the original/Comfy naming. Pony V7 is an AuraFlow fine-tune.
    aura_diffusers = _has(
        keys,
        r"(?:^|[._])joint_transformer_blocks[._]\d+[._]attn[._]"
        r"(?:add_[qkv]_proj|to_add_out)(?:[._]|$)",
    ) and _has(
        keys,
        r"(?:^|[._])single_transformer_blocks[._]\d+[._]attn[._]"
        r"(?:to_[qkv]|to_out)(?:[._]|$)",
    )
    aura_original = _has(
        keys,
        r"(?:^|[._])double_layers[._]\d+[._]attn[._]w1[qkv](?:[._]|$)",
    ) or _has(
        keys,
        r"(?:^|[._])single_layers[._]\d+[._]attn[._]w1[qkv](?:[._]|$)",
    )
    if aura_diffusers or aura_original:
        result.family = "AuraFlowDiT"
        result.arch = "AuraFlow"
        result.confidence = "high"
        result.evidence.append(
            "AuraFlow joint/single transformer layout"
            if aura_diffusers
            else "AuraFlow double_layers/single_layers w1q layout"
        )
        return result

    # SD3-family MMDiT in the original/ComfyUI/kohya spelling keeps paired
    # image and context sub-blocks inside joint_blocks.N. No other supported
    # architecture uses x_block/context_block naming.
    if _has(
        keys,
        r"(?:^|[._])joint_blocks[._]\d+[._](?:x_block|context_block)(?:[._]|$)",
    ):
        result.family = "JointAttentionDiT"
        result.arch = "SD3"
        result.confidence = "high"
        result.evidence.append(
            "SD3 MMDiT joint_blocks x_block/context_block naming"
        )
        return result

    # Z-Image and Ideogram 4 share a single-stream layers.N transformer with
    # separate Q/K/V projections and a w1/w2/w3 SwiGLU feed-forward block. The
    # hidden widths and a few model-specific modules separate them.
    zimage_refiner = _has(
        keys,
        r"(?:^|[._])(?:noise_refiner|context_refiner)[._]\d+[._]"
        r"(?:attention|feed_forward)(?:[._]|$)",
    )
    ideogram_modules = _has(
        keys,
        r"(?:^|[._])(?:llm_cond_proj|embed_image_indicator)(?:[._]|$)",
    )
    shared_separate_attention = _has(
        keys,
        r"(?:^|[._])layers[._]\d+[._]attention[._]"
        r"(?:to_[qkv]|to_out)(?:[._]|$)",
    )
    # ComfyUI's native NextDiT/Lumina naming keeps QKV fused as
    # ``attention.qkv`` and calls the output projection ``attention.o``.
    shared_fused_attention = _has(
        keys,
        r"(?:^|[._])layers[._]\d+[._]attention[._](?:qkv|o)(?:[._]|$)",
    )
    shared_single_stream = shared_separate_attention or shared_fused_attention
    shared_swiglu = _has(
        keys,
        r"(?:^|[._])layers[._]\d+[._]feed_forward[._]w[123](?:[._]|$)",
    )

    if zimage_refiner:
        result.family = "ZImageDiT"
        result.arch = "ZImage"
        result.confidence = "high"
        result.evidence.append("Z-Image noise/context refiner module naming")
        return result

    if ideogram_modules:
        result.family = "Ideogram4DiT"
        result.arch = "Ideogram4"
        result.confidence = "high"
        result.evidence.append("Ideogram 4 LLM-condition/image-indicator module naming")
        return result

    if shared_single_stream or shared_swiglu:
        attention_dims = _adapter_input_dims(
            header,
            r"(?:^|[._])layers[._]\d+[._]attention[._]"
            r"(?:to_[qkv]|to_out|qkv|o)",
        )
        w2_dims = _adapter_input_dims(
            header,
            r"(?:^|[._])layers[._]\d+[._]feed_forward[._]w2",
        )
        result.family = "SingleStreamSwiGLUDiT"
        naming = "fused-qkv" if shared_fused_attention else "separate-qkv"
        result.evidence.append(
            f"layers.N attention ({naming}) + w1/w2/w3 single-stream layout"
        )

        if 3840 in attention_dims or (
            10240 in w2_dims and (shared_single_stream or shared_swiglu)
        ):
            result.family = "ZImageDiT"
            result.arch = "ZImage"
            result.confidence = "high"
            result.evidence.append(
                "Z-Image width signature (hidden=3840 or feed-forward=10240)"
            )
            return result

        if 4608 in attention_dims or (
            12288 in w2_dims and (shared_single_stream or shared_swiglu)
        ):
            result.family = "Ideogram4DiT"
            result.arch = "Ideogram4"
            result.confidence = "high"
            result.evidence.append(
                "Ideogram 4 width signature (hidden=4608 or feed-forward=12288)"
            )
            return result

        result.confidence = "medium"
        result.ambiguous = True
        if attention_dims or w2_dims:
            result.evidence.append(
                "unrecognised single-stream widths="
                f"attention:{','.join(map(str, sorted(attention_dims))) or '-'};"
                f"w2:{','.join(map(str, sorted(w2_dims))) or '-'}"
            )
        else:
            result.evidence.append(
                "Z-Image/Ideogram-like layout without decisive adapter dimensions"
            )
        return result

    # ERNIE-Image uses layers.N.self_attention and a distinctive
    # gate_proj/up_proj/linear_fc2 feed-forward block. Base and Turbo share it.
    ernie_attention = _has(
        keys,
        r"(?:^|[._])layers[._]\d+[._]self_attention[._]"
        r"(?:to_[qkv]|to_out)(?:[._]|$)",
    )
    ernie_mlp = _has(
        keys,
        r"(?:^|[._])layers[._]\d+[._]mlp[._]"
        r"(?:gate_proj|up_proj|linear_fc2)(?:[._]|$)",
    )
    if ernie_attention or ernie_mlp:
        result.family = "ErnieImageDiT"
        attention_dims = _adapter_input_dims(
            header,
            r"(?:^|[._])layers[._]\d+[._]self_attention[._](?:to_[qkv]|to_out)",
        )
        fc2_dims = _adapter_input_dims(
            header,
            r"(?:^|[._])layers[._]\d+[._]mlp[._]linear_fc2",
        )
        if 3072 in attention_dims or 8192 in fc2_dims or (
            ernie_attention and ernie_mlp
        ):
            result.arch = "ErnieImage"
            result.confidence = "high"
            result.evidence.append(
                "ERNIE-Image self_attention + gated MLP signature"
            )
            if attention_dims or fc2_dims:
                result.evidence.append(
                    "observed ERNIE widths="
                    f"attention:{','.join(map(str, sorted(attention_dims))) or '-'};"
                    f"fc2:{','.join(map(str, sorted(fc2_dims))) or '-'}"
                )
            return result
        result.confidence = "medium"
        result.ambiguous = True
        result.evidence.append("partial ERNIE-Image layer signature")
        return result

    # The main Anima denoiser is physically a Cosmos Predict2-style transformer.
    # These keys identify the family, but not whether the weights were trained on
    # Anima rather than another Cosmos Predict2 derivative.
    cosmos_diffusers = _has(
        keys,
        r"(?:^|[._])transformer_blocks[._]\d+[._]attn1(?:[._]|$)",
    ) and _has(
        keys,
        r"(?:^|[._])transformer_blocks[._]\d+[._]attn2(?:[._]|$)",
    ) and _has(
        keys,
        r"(?:^|[._])transformer_blocks[._]\d+[._](?:ff|norm[123])(?:[._]|$)",
    )
    cosmos_original = _has(
        keys,
        r"(?:^|[._])blocks[._]\d+[._]self_attn[._]"
        r"(?:q_proj|k_proj|v_proj|output_proj)(?:[._]|$)",
    ) and _has(
        keys,
        r"(?:^|[._])blocks[._]\d+[._]cross_attn[._]"
        r"(?:q_proj|k_proj|v_proj|output_proj)(?:[._]|$)",
    ) and _has(
        keys,
        r"(?:^|[._])blocks[._]\d+[._]mlp[._](?:layer1|layer2)(?:[._]|$)",
    )
    if cosmos_original:
        result.family = "CosmosPredict2DiT"
        result.confidence = "medium"
        result.ambiguous = True
        result.evidence.append(
            "Cosmos Predict2-style blocks/self_attn/cross_attn transformer; "
            "Anima requires metadata or llm_adapter evidence"
        )
        return result

    if cosmos_diffusers:
        # Diffusers' transformer_blocks/attn1/attn2/ff spelling is shared by
        # several video DiTs. It supports an Anima metadata claim, but is not an
        # independent Cosmos/Anima fingerprint.
        result.family = "DualAttentionTransformerDiT"
        result.confidence = "medium"
        result.ambiguous = True
        result.evidence.append(
            "generic transformer_blocks attn1/attn2/ff layout; "
            "exact video architecture requires metadata"
        )
        return result

    # Krea 2 uses grouped-query self-attention with a per-token sigmoid gate,
    # SwiGLU gate/up/down projections, and a model-specific text-fusion stage.
    # The official backbone is 6,144 wide with a 16,384-wide feed-forward path.
    krea_text_fusion = _has(
        keys,
        r"(?:^|[._])(?:text_fusion|txtfusion)[._]"
        r"(?:layerwise_blocks|refiner_blocks)[._]\d+",
    )
    krea_attention_diffusers = _has(
        keys,
        r"(?:^|[._])transformer_blocks[._]\d+[._]attn[._]to_gate(?:[._]|$)",
    )
    krea_attention_comfy = _has(
        keys,
        r"(?:^|[._])blocks[._]\d+[._]attn[._]"
        r"(?:wq|wk|wv|gate|wo)(?:[._]|$)",
    )
    krea_swiglu_diffusers = _has(
        keys,
        r"(?:^|[._])transformer_blocks[._]\d+[._]ff[._](?:gate|up|down)(?:[._]|$)",
    )
    krea_swiglu_comfy = _has(
        keys,
        r"(?:^|[._])blocks[._]\d+[._]mlp[._](?:gate|up|down)(?:[._]|$)",
    )
    krea_attention = krea_attention_diffusers or krea_attention_comfy
    krea_swiglu = krea_swiglu_diffusers or krea_swiglu_comfy
    if krea_text_fusion or (krea_attention and krea_swiglu):
        result.family = "Krea2DiT"
        attention_dims = _adapter_input_dims(
            header,
            r"(?:^|[._])(?:transformer_blocks[._]\d+[._]attn[._]"
            r"(?:to_[qkv]|to_gate|to_out)|"
            r"blocks[._]\d+[._]attn[._](?:wq|wk|wv|gate|wo))",
        )
        ff_down_dims = _adapter_input_dims(
            header,
            r"(?:^|[._])(?:transformer_blocks[._]\d+[._]ff[._]down|"
            r"blocks[._]\d+[._]mlp[._]down)",
        )
        if krea_text_fusion or 6144 in attention_dims or 16384 in ff_down_dims:
            result.arch = "Krea2"
            result.confidence = "high"
            result.evidence.append(
                "Krea 2 grouped-query attention/text-fusion/SwiGLU signature"
            )
            if krea_attention_comfy or krea_swiglu_comfy:
                result.evidence.append("Krea 2 ComfyUI blocks/attn.wq/mlp naming")
            if attention_dims or ff_down_dims:
                result.evidence.append(
                    "observed Krea 2 widths="
                    f"attention:{','.join(map(str, sorted(attention_dims))) or '-'};"
                    f"ff-down:{','.join(map(str, sorted(ff_down_dims))) or '-'}"
                )
            return result
        result.confidence = "medium"
        result.ambiguous = True
        result.evidence.append("partial Krea 2 transformer signature")
        return result

    # Chroma replaces Flux's per-block modulation with a distilled guidance
    # approximator. The module keeps the same name in Diffusers and ComfyUI and
    # is unique to Chroma, so it is exact even when a sparse LoRA omits the
    # main dual/single-stream blocks.
    if _has(keys, r"(?:^|[._])distilled_guidance_layer(?:[._]|$)"):
        result.family = "DualSingleStreamDiT"
        result.arch = "Chroma"
        result.confidence = "high"
        result.evidence.append(
            "Chroma distilled_guidance_layer approximator naming"
        )
        return result

    # HunyuanVideo's individual token-refiner path can be targeted even when a
    # sparse LoRA omits the main dual/single-stream blocks.
    if _has(
        keys,
        r"context_embedder.*token_refiner|"
        r"context_embedder.*refiner_blocks|"
        r"token_refiner[._]refiner_blocks",
    ):
        result.family = "DualSingleStreamDiT"
        result.arch = "Hunyuan"
        result.confidence = "high"
        result.evidence.append("Hunyuan token-refiner module naming")
        return result

    # Dual/single-stream DiT family. Flux, Flux2, Chroma, LongCat, Hunyuan,
    # and several other models share portions of this layout.
    explicit_dual_indices = _observed_indices(
        keys,
        (
            r"double_blocks[._](\d+)",
            r"lora_transformer_double_blocks_(\d+)",
        ),
    )
    generic_transformer_indices = _observed_indices(
        keys,
        (
            r"(?:^|[._])transformer_blocks[._](\d+)",
            r"lora_transformer_transformer_blocks_(\d+)",
        ),
    )
    single_indices = _observed_indices(
        keys,
        (
            r"(?:single_blocks|single_transformer_blocks)[._](\d+)",
            r"lora_transformer_single_transformer_blocks_(\d+)",
        ),
    )
    # Generic transformer_blocks alone is not enough: Qwen-Image, SD3, LTX,
    # and other DiTs use that spelling too. It is treated as the dual stream
    # only when an explicit single stream is also observed.
    dual_indices = explicit_dual_indices or (
        generic_transformer_indices if single_indices else set()
    )
    if explicit_dual_indices or single_indices:
        result.family = "DualSingleStreamDiT"
        dual_desc = (
            f"{len(dual_indices)} observed, max={max(dual_indices)}"
            if dual_indices
            else "none observed"
        )
        single_desc = (
            f"{len(single_indices)} observed, max={max(single_indices)}"
            if single_indices
            else "none observed"
        )
        result.evidence.append(
            f"dual/single-stream layout; dual {dual_desc}; single {single_desc}"
        )

        if _has(
            keys,
            r"single_(?:transformer_)?blocks?[._]\d+.*"
            r"(?:attn[._])?to_qkv_mlp_proj",
        ):
            result.arch = "Flux2"
            result.confidence = "high"
            result.evidence.append("Flux2 fused single-block QKV/MLP projection")
            return result

        if _has(
            keys,
            r"context_embedder.*token_refiner|"
            r"context_embedder.*refiner_blocks|"
            r"token_refiner[._]refiner_blocks",
        ):
            result.arch = "Hunyuan"
            result.confidence = "high"
            result.evidence.append("Hunyuan token-refiner module naming")
            return result

        # LongCat-Image is otherwise key-identical to Flux1 (19 double + 38
        # single blocks, hidden width 3072, same attention naming). Its
        # Qwen2.5-VL context stream makes the text-input projection 3584 wide,
        # against the 4096-wide T5 stream shared by Flux1 and Chroma.
        context_dims = _adapter_input_dims(
            header,
            r"(?:^|[._])(?:context_embedder|txt_in)(?:[._]|$)",
        )
        if 3584 in context_dims:
            result.arch = "LongCat"
            result.confidence = "high"
            result.evidence.append(
                "3584-wide text-context projection (Qwen2.5-VL) in a "
                "dual/single-stream layout"
            )
            return result
        if 4096 in context_dims:
            result.evidence.append(
                "4096-wide text-context projection is shared by Flux1 and Chroma"
            )

        result.confidence = "medium"
        result.evidence.append(
            "exact architecture unresolved; block coverage is not model depth"
        )
        return result

    # Wan and Wan-derived video DiTs commonly expose blocks.N.self_attn,
    # blocks.N.cross_attn, and blocks.N.ffn.
    wan_self = _has(keys, r"(?:^|[._])blocks[._]\d+[._]self_attn(?:[._]|$)")
    wan_cross = _has(keys, r"(?:^|[._])blocks[._]\d+[._]cross_attn(?:[._]|$)")
    wan_ffn = _has(keys, r"(?:^|[._])blocks[._]\d+[._]ffn(?:[._]|$)")
    if wan_self and wan_cross and wan_ffn and not _has(keys, r"transformer_blocks"):
        result.family = "WanLikeVideoDiT"
        dims = _adapter_input_dims(
            header,
            r"blocks[._]\d+[._](?:self_attn|cross_attn)[._](?:k|to_k)",
        )
        size_names = {1536: "1.3B", 3072: "5B", 5120: "14B"}
        known_sizes = [size_names[value] for value in sorted(dims) if value in size_names]
        result.arch = "Wan"
        result.confidence = "medium"
        result.ambiguous = True
        result.evidence.append(
            "Wan-style self-attention/cross-attention/FFN block layout"
            + (f"; hidden-size hint={','.join(known_sizes)}" if known_sizes else "")
        )
        return result

    transformer_blocks = generic_transformer_indices
    if transformer_blocks:
        if _has(keys, r"(?:^|[._])img_mod(?:[._]|$)") and _has(
            keys, r"(?:^|[._])txt_mod(?:[._]|$)"
        ):
            result.family = "JointAttentionDiT"
            result.arch = "QwenImage"
            result.confidence = "high"
            result.evidence.append("paired img_mod and txt_mod modules")
            return result

        if _has(keys, r"norm1_context|add_[qkv]_proj"):
            result.family = "JointAttentionDiT"
            # SD3 medium and SD3.5 medium run 24x64=1536 wide; SD3.5 large runs
            # 38x64=2432. Flux, Qwen-Image, and HunyuanVideo joint blocks are
            # all 3072, so these widths are decisive inside this layout. The
            # attn2? alternative covers SD3.5 dual-attention layers without
            # matching attn1/attn2 video-transformer naming.
            joint_dims = _adapter_input_dims(
                header,
                r"(?:^|[._])transformer_blocks[._]\d+[._]attn2?[._]"
                r"(?:to_[qkv]|add_[qkv]_proj|to_out|to_add_out)",
            )
            known_sd3 = sorted(joint_dims.intersection({1536, 2432}))
            if len(known_sd3) == 1 and not joint_dims - {1536, 2432}:
                result.arch = "SD3"
                result.confidence = "high"
                result.evidence.append(
                    "joint-attention layout with SD3-family hidden width="
                    f"{known_sd3[0]}"
                )
                return result
            result.confidence = "medium"
            result.ambiguous = True
            result.evidence.append(
                "joint-attention transformer layout; exact architecture unresolved"
            )
            return result

        if _has(keys, r"transformer_blocks[._]\d+.*attn1") and _has(
            keys, r"transformer_blocks[._]\d+.*attn2"
        ):
            result.family = "VideoTransformerDiT"
            result.confidence = "medium"
            result.ambiguous = True
            result.evidence.append(
                "attn1/attn2 video-transformer layout; exact architecture unresolved"
            )
            return result

        result.family = "TransformerDiT"
        result.confidence = "low"
        result.evidence.append("generic transformer_blocks layout")
        return result

    text_encoder_only = _has(keys, r"^lora_te|^text_encoder(?:[._]|$)")
    if text_encoder_only:
        result.family = "TextEncoderOnly"
        if _has(keys, r"^lora_te2_|^text_encoder_2(?:[._]|$)"):
            result.arch = "SDXL"
            result.confidence = "medium"
            result.ambiguous = True
            result.evidence.append("two text encoders, no denoiser tensors")
        else:
            result.confidence = "low"
            result.evidence.append("single text encoder, no denoiser tensors")
        return result

    result.evidence.append("no recognised structural signature")
    return result


# ---------------------------------------------------------------------------
# Metadata and filename hints
# ---------------------------------------------------------------------------

# Specific patterns must precede broad patterns. The third element is a scope:
#
# * SCOPE_ANY      - identity metadata fields, free-text metadata, and filenames
# * SCOPE_METADATA - trusted identity metadata fields only
#
# Bare tokens that double as common words or personal names (chroma, wan,
# anima, klein, ...) never fire from a filename or a free-text title. That
# keeps them from manufacturing a conflict against real structural or metadata
# evidence, which would block a move the structure alone had fully earned.
SCOPE_ANY = "any"
SCOPE_METADATA = "metadata"

_ARCH_PATTERNS: List[Tuple[str, str, str]] = [
    # Pony is an ecosystem spanning incompatible generations. V6 circulated in
    # explicit XL and 1.5 forms; V7 moved to AuraFlow. Bare "Pony" or bare V6 is
    # intentionally not an architecture signal.
    (r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*7", "AuraFlow", SCOPE_ANY),
    (r"aura[ ._-]*flow|auraflow(?:transformer2dmodel)?", "AuraFlow", SCOPE_ANY),
    (r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*6[ ._-]*(?:xl|sdxl)", "SDXL", SCOPE_ANY),
    (r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*6[ ._-]*(?:1[ ._-]*5|sd15)", "SD15", SCOPE_ANY),
    (r"pony(?:[ ._-]*diffusion)?[ ._-]*xl|pd[ ._-]*xl", "SDXL", SCOPE_ANY),

    (r"mini[ ._-]*max[ ._-]*h3|minimaxh3(?:model|ditmodel)?", "MiniMaxH3", SCOPE_ANY),
    (r"ideogram[ ._-]*(?:v[ ._-]*)?4|ideogram4transformer2dmodel", "Ideogram4", SCOPE_ANY),
    (r"z[ ._-]*image|zimagetransformer2dmodel", "ZImage", SCOPE_ANY),
    (r"ernie[ ._-]*image|ernieimagetransformer2dmodel", "ErnieImage", SCOPE_ANY),
    # Anima with an explicit variant or version is precise anywhere. The bare
    # word is a Latin/Jungian term and stays an identity-metadata signal.
    (
        r"anima(?!gine)[ ._-]*(?:base|turbo|aesthetic)|"
        r"anima(?!gine)[ ._-]*(?:v|version)[ ._-]*[0-9][0-9 ._-]*",
        "Anima",
        SCOPE_ANY,
    ),
    (r"anima(?!gine)", "Anima", SCOPE_METADATA),

    (r"flux[ ._-]*2|flux2", "Flux2", SCOPE_ANY),
    # Klein is a Flux2 release, but the bare word is a common surname. It only
    # classifies with flux context; a lone "klein" is recorded as evidence.
    (r"flux[ ._-]*(?:2[ ._-]*)?klein", "Flux2", SCOPE_ANY),
    (r"chroma", "Chroma", SCOPE_METADATA),
    (r"long[ ._-]*cat[ ._-]*image|longcatimagetransformer2dmodel", "LongCat", SCOPE_ANY),
    (r"long[ ._-]*cat", "LongCat", SCOPE_METADATA),
    (r"krea[ ._-]*2", "Krea2", SCOPE_ANY),
    (r"flux[ ._-]*1|flux1|flux[ ._-]*(?:dev|schnell|kontext)|flux[ ._-]*transformer2[ ._-]*d[ ._-]*model", "Flux1", SCOPE_ANY),
    (r"illustrious(?:[ ._-]*xl)?|il[ ._-]*xl|illust[ ._-]*xl", "SDXL", SCOPE_ANY),
    (r"stable[ ._-]*diffusion[ ._-]*xl|sd[ ._-]*xl|sdxl", "SDXL", SCOPE_ANY),
    (r"stable[ ._-]*diffusion[ ._-]*(?:v)?3(?:[ ._-]*5)?|sd3(?:[ ._-]*5)?", "SD3", SCOPE_ANY),
    (r"stable[ ._-]*diffusion[ ._-]*(?:v)?2|sd[ ._-]*v?2|sd2", "SD2", SCOPE_ANY),
    (r"stable[ ._-]*diffusion[ ._-]*(?:v)?1(?:[ ._-]*5)?|sd[ ._-]*1[ ._-]*5|sd15", "SD15", SCOPE_ANY),
    (r"hunyuan[ ._-]*video|hunyuanvideo|hyvideo", "Hunyuan", SCOPE_ANY),
    # Bernini-R adapts the Wan 2.2 renderer; its LoRAs remain Wan-shaped.
    (r"bernini(?:[ ._-]*r)?", "Wan", SCOPE_ANY),
    # Wan with a version or task token is precise anywhere. The bare syllable
    # ("Obi-Wan") stays an identity-metadata signal.
    (
        r"wan[ ._-]*2(?:[ ._-]*[12])?|"
        r"wan[ ._-]*(?:video|vace|fun|animate|i2v|t2v|ti2v|s2v|flf2v)",
        "Wan",
        SCOPE_ANY,
    ),
    (r"wan", "Wan", SCOPE_METADATA),
    (r"ltx(?:[ ._-]*(?:video|2))?", "LTX", SCOPE_ANY),
    (r"qwen(?:[ ._-]*image)?", "QwenImage", SCOPE_ANY),
]

# Bare tokens that are too imprecise to classify from a filename. They are
# surfaced as evidence so a reviewer still sees them, but they never create an
# architecture signal from a filename and therefore cannot manufacture a
# conflict. The middle element is the architecture the token would have
# suggested; when the filename already classified as that architecture through
# a precise pattern, no note is added.
_DEMOTED_NAME_TOKENS: List[Tuple[str, str, str]] = [
    ("Chroma", "Chroma", r"chroma"),
    ("Wan", "Wan", r"wan"),
    ("Anima", "Anima", r"anima(?!gine)"),
    ("LongCat", "LongCat", r"long[ ._-]*cat"),
    ("Klein", "Flux2", r"klein"),
    ("Schnell", "Flux1", r"schnell"),
    ("Kontext", "Flux1", r"kontext"),
]

_ECOSYSTEM_PATTERNS: List[Tuple[str, str]] = [
    ("Pony", r"pony(?!tail)(?:[ ._-]*diffusion)?|pd[ ._-]*xl"),
    ("Illustrious", r"illustrious(?:[ ._-]*xl)?|il[ ._-]*xl|ilxl|illust"),
    ("NoobAI", r"noob(?:[ ._-]*ai)?"),
    ("Animagine", r"animagine"),
]

# These are release/model variants, not stand-alone architectures. Patterns are
# scoped to the model name so a generic file called portrait_turbo remains a
# training-technique hint rather than becoming Z-Image Turbo by telepathy.
_VARIANT_PATTERNS: List[Tuple[str, str]] = [
    ("Bernini-R", r"bernini(?:[ ._-]*r)?"),
    ("Omni-Base", r"z[ ._-]*image[ ._-]*omni[ ._-]*base"),
    ("Base", r"(?:z[ ._-]*image|anima)[ ._-]*(?:base|foundation|undistilled)"),
    ("Turbo", r"(?:z[ ._-]*image|anima|ernie[ ._-]*image)[ ._-]*turbo"),
    ("Aesthetic", r"anima[ ._-]*aesthetic"),
    ("Aes", r"ernie[ ._-]*image[ ._-]*aes"),
    ("Edit", r"z[ ._-]*image[ ._-]*edit"),
    # Kontext, Schnell, and Klein are Flux releases, but the bare words are
    # ordinary German vocabulary and surnames. They only become variant tags
    # (and thereby architecture implications) with flux context present.
    ("Kontext", r"flux[ ._-]*(?:1[ ._-]*)?(?:dev[ ._-]*)?kontext"),
    ("Schnell", r"flux[ ._-]*(?:1[ ._-]*)?schnell"),
    ("Klein", r"flux[ ._-]*(?:2[ ._-]*)?klein"),
]

_UNRESOLVED_HINT_PATTERNS: List[Tuple[str, str]] = [
    # "Bernie" is not silently rewritten to either ByteDance Bernini or
    # Baidu ERNIE-Image. Both exist and are architecturally unrelated.
    ("Bernie (ambiguous: Bernini or ERNIE-Image)", r"bernie"),
    # MiniMax publishes several unrelated media models. H3 must be explicit before
    # this tool claims the MiniMaxH3 LoRA construction.
    ("MiniMax (family unspecified)", r"mini[ ._-]*max(?![ ._-]*h3)|minimax(?![ ._-]*h3)"),
]

_TECHNIQUE_PATTERNS: List[Tuple[str, str]] = [
    ("LCM", r"lcm"),
    ("Turbo", r"turbo"),
    ("Lightning", r"lightning"),
    ("Hyper", r"hyper(?:[ ._-]*sd)?"),
    ("DMD", r"dmd2?"),
]

# Illustrious, NoobAI, and Animagine remain SDXL-compatible ecosystems. Pony is
# version-dependent and is handled separately below.
_ECOSYSTEM_IMPLIES = {
    "Illustrious": "SDXL",
    "NoobAI": "SDXL",
    "Animagine": "SDXL",
}

_PONY_VERSION_IMPLIES = {"7": "AuraFlow"}

_VARIANT_IMPLIES = {
    "Bernini-R": "Wan",
    "Kontext": "Flux1",
    "Schnell": "Flux1",
    "Klein": "Flux2",
}

# An ecosystem may span more than one architecture generation. This map is a
# compatibility constraint, not an architecture claim. Generic Pony is allowed
# on SD1.5, SDXL, or AuraFlow, but never implies one without explicit generation,
# variant, metadata, or tensor evidence.
_ECOSYSTEM_ALLOWED_ARCHES = {
    "Pony": {"SD15", "SDXL", "AuraFlow"},
    "Illustrious": {"SDXL"},
    "NoobAI": {"SDXL"},
    "Animagine": {"SDXL"},
}

_ARCH_FAMILY = {
    "SD15": "SDUNet",
    "SD2": "SDUNet",
    "SDXL": "SDUNet",
    "AuraFlow": "AuraFlowDiT",
    "MiniMaxH3": "MiniMaxH3OmniDiT",
    "Krea2": "Krea2DiT",
    "Anima": "CosmosPredict2DiT",
    "ErnieImage": "ErnieImageDiT",
    "ZImage": "ZImageDiT",
    "Ideogram4": "Ideogram4DiT",
    "Flux1": "DualSingleStreamDiT",
    "Flux2": "DualSingleStreamDiT",
    "Chroma": "DualSingleStreamDiT",
    "LongCat": "DualSingleStreamDiT",
    "Hunyuan": "DualSingleStreamDiT",
    "Wan": "WanLikeVideoDiT",
    "LTX": "VideoTransformerDiT",
    "SD3": "JointAttentionDiT",
    "QwenImage": "JointAttentionDiT",
}

# The boolean marks identity fields: values that name the base model itself.
# modelspec.title is free-text prose, so bare metadata-scoped tokens (chroma,
# wan, anima, ...) do not fire from it; it is read with filename precision.
_METADATA_FIELDS: List[Tuple[str, str, bool]] = [
    ("modelspec.architecture", "high", True),
    ("architecture", "high", True),
    ("model_type", "high", True),
    ("ss_base_model_version", "high", True),
    ("ss_base_model", "high", True),
    ("base_model", "high", True),
    ("base_model_name", "high", True),
    ("base_model_name_or_path", "high", True),
    ("pretrained_model_name_or_path", "high", True),
    ("ss_sd_model_name", "medium", True),
    ("modelspec.title", "medium", False),
]


def _first_arch(text: str, source: str) -> Optional[str]:
    """Match the first architecture pattern permitted for ``source``.

    ``source`` is ``"metadata"`` for trusted identity fields and ``"filename"``
    for filenames and free-text metadata such as titles.
    """
    text = _hint_text(text)
    for pattern, arch, scope in _ARCH_PATTERNS:
        if scope == SCOPE_METADATA and source != "metadata":
            continue
        if _matches_hint(text, pattern):
            return arch
    return None


def _extract_tags(text: str, source_label: str) -> HintTags:
    normalized = _hint_text(text)
    tags = HintTags()

    for value, pattern in _ECOSYSTEM_PATTERNS:
        if _matches_hint(normalized, pattern):
            tags.ecosystems.append(value)
    for value, pattern in _VARIANT_PATTERNS:
        if _matches_hint(normalized, pattern):
            tags.variants.append(value)
    for value, pattern in _UNRESOLVED_HINT_PATTERNS:
        if _matches_hint(normalized, pattern):
            tags.unresolved_hints.append(value)
    for value, pattern in _TECHNIQUE_PATTERNS:
        if _matches_hint(normalized, pattern):
            tags.techniques.append(value)

    if _matches_hint(
        normalized,
        r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*7",
    ):
        tags.model_version = "7"
    elif _matches_hint(
        normalized,
        r"pony(?:[ ._-]*diffusion)?[ ._-]*(?:v|version)?[ ._-]*6",
    ):
        tags.model_version = "6"
    elif _matches_hint(normalized, r"bernini(?:[ ._-]*r)?"):
        # The released Bernini-R renderer is based on Wan 2.2.
        tags.model_version = "2.2"
    else:
        anima_match = re.search(
            r"(?<![a-z0-9])anima(?!gine)(?:[ ._-]*(?:base|turbo|aesthetic))?"
            r"(?:[ ._-]*(?:v|version))?[ ._-]*([0-9]+(?:[ ._-][0-9]+)+)",
            normalized,
        )
        if anima_match:
            tags.model_version = re.sub(r"[ _-]", ".", anima_match.group(1))

    if not tags.model_version and re.search(
        r"(?<![a-z0-9])wan[ ._-]*2[ ._-]*2(?![a-z0-9])", normalized
    ):
        tags.model_version = "2.2"
    elif not tags.model_version and re.search(
        r"(?<![a-z0-9])wan[ ._-]*2[ ._-]*1(?![a-z0-9])", normalized
    ):
        tags.model_version = "2.1"
    elif not tags.model_version and re.search(
        r"(?<![a-z0-9])sd3[ ._-]*5(?![a-z0-9])", normalized
    ):
        tags.model_version = "3.5"

    # _camel_split deliberately separates acronym-to-word transitions. Accept
    # optional separators inside compact task names so ``FL2VA`` and ``I2V``
    # survive that normalization as well as punctuation-heavy filenames.
    task_patterns = [
        ("Ref2VA", r"ref2[ ._-]*va|reference[ ._-]*to[ ._-]*video[ ._-]*(?:and|with)?[ ._-]*audio"),
        ("FL2VA", r"fl2[ ._-]*va|first[ ._-]*last[ ._-]*frame[ ._-]*to[ ._-]*video[ ._-]*(?:and|with)?[ ._-]*audio"),
        ("T2VA", r"t2[ ._-]*va|text[ ._-]*to[ ._-]*video[ ._-]*(?:and|with)?[ ._-]*audio"),
        ("TI2V", r"ti2[ ._-]*v|text[ ._-]*image[ ._-]*to[ ._-]*video"),
        ("I2V", r"i2[ ._-]*v|image[ ._-]*to[ ._-]*video"),
        ("T2V", r"t2[ ._-]*v|text[ ._-]*to[ ._-]*video"),
        ("V2V", r"v2[ ._-]*v|video[ ._-]*to[ ._-]*video"),
        ("I2I", r"i2[ ._-]*i|image[ ._-]*to[ ._-]*image"),
        ("T2I", r"t2[ ._-]*i|text[ ._-]*to[ ._-]*image"),
        ("Edit", r"edit"),
        ("Animate", r"animate"),
    ]
    for value, pattern in task_patterns:
        if _matches_hint(normalized, pattern):
            tags.task_variant = value
            break

    if _matches_hint(normalized, r"high[ ._-]*noise"):
        tags.noise_stage = "high"
    elif _matches_hint(normalized, r"low[ ._-]*noise"):
        tags.noise_stage = "low"

    found = tags.ecosystems + tags.variants + tags.unresolved_hints + tags.techniques
    if tags.model_version:
        found.append(f"version={tags.model_version}")
    if tags.task_variant:
        found.append(f"task={tags.task_variant}")
    if tags.noise_stage:
        found.append(f"noise={tags.noise_stage}")
    if found:
        tags.evidence.append(
            f"{source_label} tags: {', '.join(_unique(found))}"
        )
    return tags


def metadata_signals(meta: Mapping[str, Any]) -> Tuple[List[ArchSignal], HintTags, List[str]]:
    signals: List[ArchSignal] = []
    evidence: List[str] = []
    tag_texts: List[str] = []

    for field_name, confidence, is_identity in _METADATA_FIELDS:
        raw = meta.get(field_name, "")
        if raw is None:
            continue
        value = str(raw).strip()
        if not value:
            continue
        tag_texts.append(value)
        arch = _first_arch(value, "metadata" if is_identity else "filename")
        if arch:
            signals.append(
                ArchSignal(
                    arch=arch,
                    confidence=confidence,
                    source=f"metadata:{field_name}",
                    detail=value[:120],
                    priority=80 if confidence == "high" else 60,
                )
            )
            evidence.append(f"metadata {field_name} -> {arch}: {value[:80]}")

    if str(meta.get("ss_v2", "")).strip().lower() in {"true", "1", "yes"}:
        signals.append(
            ArchSignal(
                arch="SD2",
                confidence="high",
                source="metadata:ss_v2",
                detail="ss_v2=True",
                priority=75,
            )
        )
        evidence.append("metadata ss_v2=True -> SD2")

    tags = _extract_tags(" ".join(tag_texts), "metadata")

    # The undistilled foundation checkpoint is officially named simply
    # ``Tongyi-MAI/Z-Image`` rather than ``Z-Image-Base``. Only infer the Base
    # variant from trusted base-model identity fields whose final component is
    # exactly Z-Image. A filename such as zimage_character remains unqualified.
    zimage_base_fields = (
        "ss_base_model_version",
        "ss_base_model",
        "base_model",
        "base_model_name",
        "base_model_name_or_path",
        "pretrained_model_name_or_path",
        "ss_sd_model_name",
    )
    for field_name in zimage_base_fields:
        raw = meta.get(field_name, "")
        if raw is None:
            continue
        value = str(raw).strip().rstrip("/\\")
        if not value:
            continue
        leaf = re.split(r"[/\\]", value)[-1]
        leaf = re.sub(
            r"\.(?:safetensors|ckpt|pt|pth|bin)$",
            "",
            leaf,
            flags=re.IGNORECASE,
        )
        if re.fullmatch(r"z[ _-]*image", leaf, re.IGNORECASE):
            tags.variants.append("Base")
            evidence.append(
                f"metadata {field_name} identifies the Z-Image foundation/Base checkpoint"
            )
            break

    evidence.extend(tags.evidence)
    return signals, tags, evidence


def name_signals(filename: str) -> Tuple[List[ArchSignal], HintTags, List[str]]:
    stem = Path(filename).stem
    arch = _first_arch(stem, "filename")
    signals: List[ArchSignal] = []
    evidence: List[str] = []
    if arch:
        signals.append(
            ArchSignal(
                arch=arch,
                confidence="low",
                source="filename",
                detail=stem[:120],
                priority=20,
            )
        )
        evidence.append(f"filename -> {arch}: {stem[:80]}")

    normalized = _hint_text(stem)
    for label, implied_arch, pattern in _DEMOTED_NAME_TOKENS:
        if arch != implied_arch and _matches_hint(normalized, pattern):
            evidence.append(
                f"filename token {label!r} alone is not precise enough to classify"
            )

    tags = _extract_tags(stem, "filename")
    evidence.extend(tags.evidence)
    return signals, tags, evidence


def _merge_tags(primary: HintTags, secondary: HintTags) -> HintTags:
    return HintTags(
        ecosystems=_unique(primary.ecosystems + secondary.ecosystems),
        variants=_unique(primary.variants + secondary.variants),
        unresolved_hints=_unique(
            primary.unresolved_hints + secondary.unresolved_hints
        ),
        techniques=_unique(primary.techniques + secondary.techniques),
        model_version=primary.model_version or secondary.model_version,
        task_variant=primary.task_variant or secondary.task_variant,
        noise_stage=primary.noise_stage or secondary.noise_stage,
        evidence=_unique(primary.evidence + secondary.evidence),
    )


def _implied_signals(
    tags: HintTags,
    source: str,
    confidence: str,
    priority: int,
) -> List[ArchSignal]:
    signals: List[ArchSignal] = []
    for ecosystem in tags.ecosystems:
        arch = _ECOSYSTEM_IMPLIES.get(ecosystem)
        detail = f"{ecosystem} normally targets {arch}" if arch else ""
        if ecosystem == "Pony" and tags.model_version:
            arch = _PONY_VERSION_IMPLIES.get(tags.model_version)
            if arch:
                detail = f"Pony V{tags.model_version} targets {arch}"
        if arch:
            signals.append(
                ArchSignal(
                    arch=arch,
                    confidence=confidence,
                    source=f"{source}:ecosystem",
                    detail=detail,
                    priority=priority,
                )
            )
    for variant in tags.variants:
        arch = _VARIANT_IMPLIES.get(variant)
        if arch:
            signals.append(
                ArchSignal(
                    arch=arch,
                    confidence=confidence,
                    source=f"{source}:variant",
                    detail=f"{variant} targets {arch}",
                    priority=priority,
                )
            )
    return signals


def _signal_summary(signals: Sequence[ArchSignal], source_prefix: str) -> str:
    values = _unique(signal.arch for signal in signals if signal.source.startswith(source_prefix))
    return ";".join(values)


def _family_compatible(family: str, arch: str) -> bool:
    # These families are intentionally non-exclusive when a LoRA is sparse.
    if family in {"Unknown", "TextEncoderOnly", "TransformerDiT"}:
        return True

    expected = _ARCH_FAMILY.get(arch)
    if expected is None:
        return True
    if family == expected:
        return True

    compatible_sets = {
        "SingleStreamSwiGLUDiT": {"ZImage", "Ideogram4"},
        "JointAttentionDiT": {"SD3", "QwenImage"},
        "DualSingleStreamDiT": {"Flux1", "Flux2", "Chroma", "LongCat", "Hunyuan"},
        "VideoTransformerDiT": {"LTX"},
        "DualAttentionTransformerDiT": {"LTX", "Anima"},
        "CosmosPredict2DiT": {"Anima"},
        "MiniMaxH3OmniDiT": {"MiniMaxH3"},
        "AuraFlowDiT": {"AuraFlow"},
        "Krea2DiT": {"Krea2"},
        "ZImageDiT": {"ZImage"},
        "Ideogram4DiT": {"Ideogram4"},
        "ErnieImageDiT": {"ErnieImage"},
        "WanLikeVideoDiT": {"Wan"},
        "SDUNet": {"SD15", "SD2", "SDXL"},
    }
    return arch in compatible_sets.get(family, set())


def classify(
    header: Mapping[str, Mapping[str, Any]],
    meta: Mapping[str, Any],
    filename: str,
) -> Dict[str, str]:
    structure = arch_from_structure(header)
    meta_signals, meta_tags, meta_evidence = metadata_signals(meta)
    file_signals, file_tags, file_evidence = name_signals(filename)
    tags = _merge_tags(meta_tags, file_tags)

    signals: List[ArchSignal] = []
    evidence = [f"structure: {item}" for item in structure.evidence]
    evidence.extend(meta_evidence)
    evidence.extend(file_evidence)

    if structure.arch:
        signals.append(
            ArchSignal(
                arch=structure.arch,
                confidence=structure.confidence,
                source="structure",
                detail="; ".join(structure.evidence),
                priority=100,
            )
        )

    signals.extend(meta_signals)
    metadata_implied = _implied_signals(meta_tags, "metadata", "medium", 55)
    filename_implied = _implied_signals(file_tags, "filename", "low", 15)
    signals.extend(metadata_implied)
    signals.extend(file_signals)
    signals.extend(filename_implied)
    evidence.extend(
        f"{signal.source} -> {signal.arch}: {signal.detail}"
        for signal in metadata_implied + filename_implied
    )
    if tags.unresolved_hints:
        evidence.append(
            "unresolved name hint does not classify architecture: "
            + ", ".join(tags.unresolved_hints)
        )

    conflicts: List[str] = []
    exact_arches = _unique(signal.arch for signal in signals)
    if len(exact_arches) > 1:
        rendered = ", ".join(
            f"{signal.source}={signal.arch}" for signal in sorted(signals, key=lambda item: -item.priority)
        )
        conflicts.append(f"architecture evidence disagrees: {rendered}")

    for signal in signals:
        if not _family_compatible(structure.family, signal.arch):
            conflicts.append(
                f"structural family {structure.family} is incompatible with "
                f"{signal.source}={signal.arch}"
            )

    for ecosystem in tags.ecosystems:
        allowed = _ECOSYSTEM_ALLOWED_ARCHES.get(ecosystem)
        if not allowed:
            continue
        incompatible = _unique(
            signal.arch for signal in signals if signal.arch not in allowed
        )
        if incompatible:
            conflicts.append(
                f"ecosystem {ecosystem} supports {','.join(sorted(allowed))}, "
                f"not {','.join(incompatible)}"
            )

    # Choose a display candidate by evidence priority. Conflict status prevents moves.
    ordered = sorted(
        signals,
        key=lambda item: (CONF_RANK.get(item.confidence, 0), item.priority),
        reverse=True,
    )
    candidate = ordered[0].arch if ordered else "Unknown"
    confidence = ordered[0].confidence if ordered else "low"

    if candidate != "Unknown" and not conflicts:
        if any(
            signal.arch == candidate and signal.confidence == "high"
            for signal in signals
        ):
            confidence = "high"
        elif structure.arch == candidate and any(
            signal.arch == candidate and signal.source.startswith("metadata")
            for signal in signals
        ):
            confidence = "high"
            evidence.append(f"structure and metadata agree -> {candidate}")
        elif structure.arch == candidate:
            confidence = structure.confidence
        elif meta_signals:
            confidence = max(
                (signal.confidence for signal in meta_signals if signal.arch == candidate),
                key=lambda value: CONF_RANK[value],
                default="medium",
            )
        elif structure.family != "Unknown" and any(
            signal.arch == candidate for signal in file_signals
        ):
            confidence = "medium"
        else:
            confidence = "low"

    candidate_metadata = [
        signal for signal in meta_signals if signal.arch == candidate
    ]
    if conflicts:
        status = "conflict"
        evidence.extend(f"CONFLICT: {item}" for item in _unique(conflicts))
    elif candidate == "Unknown":
        status = "ambiguous"
    elif ordered and all(signal.source.startswith("filename") for signal in ordered):
        status = "ambiguous"
        evidence.append("filename-only architecture hint requires review")
    elif structure.arch == candidate and structure.ambiguous and not candidate_metadata:
        status = "ambiguous"
        evidence.append("structural signature is shared; metadata confirmation required")
    else:
        status = "ok"

    ecosystem = _join_values(tags.ecosystems)
    model_variant = _join_values(tags.variants)
    unresolved_hint = _join_values(tags.unresolved_hints)

    technique_values = list(tags.techniques)
    # "Turbo" is a release variant for these named model families. Keep generic
    # portrait_turbo as a technique, but avoid reporting Z-Image-Turbo twice.
    if "Turbo" in tags.variants and candidate in {
        "ZImage", "Anima", "ErnieImage"
    }:
        technique_values = [value for value in technique_values if value != "Turbo"]
        evidence.append(f"Turbo interpreted as a {candidate} model variant")
    techniques = _join_values(technique_values)

    # Legacy-friendly secondary folder. Compatibility-sensitive version/task
    # qualifiers are retained without collapsing the richer CSV columns.
    base_subfolder = ""
    if ecosystem:
        base_subfolder = ecosystem.split(";", 1)[0]
    elif model_variant:
        base_subfolder = model_variant.split(";", 1)[0]

    qualifiers: List[str] = []
    if tags.model_version:
        qualifiers.append(f"v{tags.model_version}")
    if candidate in {"Wan", "LTX", "QwenImage", "Krea2", "MiniMaxH3"}:
        if tags.task_variant:
            qualifiers.append(tags.task_variant)
        if tags.noise_stage:
            qualifiers.append(f"{tags.noise_stage}-noise")

    if base_subfolder and qualifiers:
        sort_subfolder = base_subfolder + "__" + "+".join(qualifiers)
    elif base_subfolder:
        sort_subfolder = base_subfolder
    elif qualifiers:
        sort_subfolder = "+".join(qualifiers)
    elif techniques:
        sort_subfolder = "+".join(techniques.split(";"))
    else:
        sort_subfolder = ""

    return {
        "arch": candidate,
        "structural_family": structure.family,
        "model_version": tags.model_version,
        "task_variant": tags.task_variant,
        "noise_stage": tags.noise_stage,
        "confidence": confidence,
        "status": status,
        "ecosystem": ecosystem,
        "model_variant": model_variant,
        "unresolved_hint": unresolved_hint,
        "techniques": techniques,
        "lineage": sort_subfolder,
        "sort_subfolder": sort_subfolder,
        "structure_arch": structure.arch or "",
        "metadata_arch": _signal_summary(meta_signals, "metadata"),
        "name_arch": _signal_summary(file_signals, "filename"),
        "evidence": "; ".join(_unique(evidence)),
    }


# ---------------------------------------------------------------------------
# File inspection and CLI
# ---------------------------------------------------------------------------

def _blank_record() -> Dict[str, str]:
    return {field_name: "" for field_name in FIELDNAMES}


def inspect_one_file(path: Path, follow_symlinks: bool = False) -> Dict[str, str]:
    input_path = _absolute_without_resolving(path)
    record = _blank_record()
    record.update(
        input_path=str(input_path),
        path=str(input_path),
        source_was_symlink=str(input_path.is_symlink()),
        arch="Unknown",
        structural_family="Unknown",
        confidence="low",
        status="ambiguous",
    )

    if input_path.is_symlink():
        if not follow_symlinks:
            record.update(
                status="error",
                error="symlink skipped; use --follow-symlinks to inspect and record its target",
            )
            return record
        try:
            actual_path = input_path.resolve(strict=True)
        except OSError as exc:
            record.update(status="error", error=f"broken or unreadable symlink: {exc}")
            return record
        record["path"] = str(actual_path)
    else:
        actual_path = input_path

    try:
        before = actual_path.stat()
        header_data = read_header(actual_path)
        after = actual_path.stat()
    except Exception as exc:  # each file gets an independent failure record
        record.update(status="error", error=str(exc))
        return record

    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        record.update(
            status="error",
            error="file changed while its header was being inspected; retry",
        )
        return record

    record.update(
        file_size=str(after.st_size),
        mtime_ns=str(after.st_mtime_ns),
        header_sha256=header_data.header_sha256,
        header_bytes=str(header_data.header_bytes),
        total_tensors=str(len(header_data.tensors)),
        format=detect_format(list(header_data.tensors)),
    )

    try:
        record.update(classify(header_data.tensors, header_data.metadata, actual_path.name))
    except Exception as exc:
        record.update(status="error", error=f"classification failed: {exc}")
    return record


def expand_arg(arg: str) -> List[Path]:
    path = Path(arg)
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.suffix.lower() == ".safetensors"
            and (candidate.is_file() or candidate.is_symlink())
        )
    # Existing literal paths win over glob parsing. This matters for legal Unix
    # names containing '[' and for bracketed library folders copied between hosts.
    if path.exists() or path.is_symlink():
        return [path]
    if any(character in arg for character in "*?["):
        # A directory can legally be named foo.safetensors. Globs return only
        # existing paths, so real directories are dropped here; symlinks pass
        # through and receive their normal per-file symlink handling.
        return [
            Path(match)
            for match in glob.glob(arg, recursive=True)
            if Path(match).is_symlink() or not Path(match).is_dir()
        ]
    return [path]


def _dedupe_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def collect_files(arguments: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    seen: Set[str] = set()
    for argument in arguments:
        expanded = expand_arg(argument)
        if not expanded:
            print(f"No matches for: {argument}")
        for path in expanded:
            if path.suffix.lower() != ".safetensors":
                continue
            key = _dedupe_key(path)
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def infer_common_base(files: Sequence[Path]) -> Optional[Path]:
    if not files:
        return None
    parents = [str(_absolute_without_resolving(path).parent) for path in files]
    try:
        return Path(os.path.commonpath(parents))
    except ValueError:
        return None


def write_csv(rows: Sequence[Mapping[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect LoRA safetensors headers and write a conservative CSV index."
    )
    parser.add_argument("targets", nargs="+", help="file, glob, or folder to inspect")
    parser.add_argument(
        "--output",
        default=CSV_NAME,
        help=f"index CSV path (default: ./{CSV_NAME})",
    )
    parser.add_argument(
        "--base",
        help="base folder for the automatic dry-run sort plan (default: common parent)",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="inspect symlink targets and record their resolved target paths",
    )
    parser.add_argument(
        "--no-sort-plan",
        action="store_true",
        help="write only the inspection CSV; do not invoke inspect_sort_lora.py",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    files = collect_files(args.targets)
    if not files:
        print("No .safetensors files found.")
        return 1

    rows: List[Dict[str, str]] = []
    for path in files:
        record = inspect_one_file(path, follow_symlinks=args.follow_symlinks)
        rows.append(record)
        subtype = record.get("sort_subfolder", "")
        tag = f"{record['arch']}/{subtype}" if subtype else record["arch"]
        suffix = f" ERROR: {record['error'][:80]}" if record.get("error") else ""
        print(
            f"{path.name}: {tag} [{record['confidence']}; {record['status']}; "
            f"{record['structural_family']}]{suffix}"
        )

    output_path = _absolute_without_resolving(Path(args.output))
    write_csv(rows, output_path)
    print(f"\nWrote {len(rows)} records to {output_path}")

    if args.no_sort_plan:
        return 0

    try:
        import inspect_sort_lora
    except ImportError:
        print("inspect_sort_lora.py not found; skipping the dry-run sort plan.")
        return 0

    base = Path(args.base) if args.base else infer_common_base(files)
    if base is None:
        print("Could not infer a common base folder; skipping the dry-run sort plan.")
        return 0

    return int(
        inspect_sort_lora.main(
            csv_arg=str(output_path),
            base_dir_arg=str(base),
            apply_moves=False,
            min_conf="high",
        )
        or 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
