#!/usr/bin/env python3
"""Regression tests and a real-file smoke inspector for the LoRA tools.

Run the regression suite:
    python test_lora_tools.py
    python -m unittest -v test_lora_tools.py

Smoke-inspect one or more real LoRAs without writing a CSV or moving anything:
    python test_lora_tools.py /path/to/model.safetensors
    python test_lora_tools.py --inspect /path/to/model.safetensors
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import struct
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import inspect_lora
import inspect_sort_lora


def tensor(shape: Iterable[int]) -> Dict[str, object]:
    return {"dtype": "F16", "shape": list(shape), "data_offsets": [0, 0]}


def write_fake_safetensors(
    path: Path,
    tensors: Mapping[str, Mapping[str, object]],
    metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = dict(tensors)
    if metadata:
        header["__metadata__"] = {key: str(value) for key, value in metadata.items()}
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
    return path


def write_index(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=inspect_lora.FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class ClassifierTests(unittest.TestCase):
    def classify(self, tensors, metadata=None, filename="model.safetensors"):
        return inspect_lora.classify(tensors, metadata or {}, filename)

    def test_sd15_by_cross_attention_width(self):
        result = self.classify(
            {
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 768])
            }
        )
        self.assertEqual(result["arch"], "SD15")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_sd2_by_cross_attention_width(self):
        result = self.classify(
            {
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 1024])
            }
        )
        self.assertEqual(result["arch"], "SD2")
        self.assertEqual(result["status"], "ok")

    def test_sdxl_by_cross_attention_width(self):
        result = self.classify(
            {
                "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 2048])
            }
        )
        self.assertEqual(result["arch"], "SDXL")
        self.assertEqual(result["status"], "ok")

    def test_sparse_down_blocks_is_not_forced_to_sd15(self):
        result = self.classify(
            {"unet.down_blocks.0.resnets.0.conv1.lora_A.weight": tensor([8, 320])}
        )
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["structural_family"], "SDUNet")
        self.assertEqual(result["status"], "ambiguous")

    def test_sparse_sdxl_uses_trusted_metadata(self):
        result = self.classify(
            {"unet.down_blocks.0.resnets.0.conv1.lora_A.weight": tensor([8, 320])},
            {"modelspec.architecture": "stable-diffusion-xl-v1-base"},
        )
        self.assertEqual(result["arch"], "SDXL")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_sdxl_second_text_encoder_signature(self):
        result = self.classify(
            {
                "lora_unet_input_blocks_4_1_proj_in.lora_down.weight": tensor([8, 640]),
                "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": tensor([8, 1280]),
            }
        )
        self.assertEqual(result["arch"], "SDXL")
        self.assertEqual(result["confidence"], "high")

    def test_flux2_fused_projection_is_distinctive(self):
        result = self.classify(
            {
                "transformer.single_transformer_blocks.47.attn."
                "to_qkv_mlp_proj.lora_A.weight": tensor([8, 6144])
            }
        )
        self.assertEqual(result["arch"], "Flux2")
        self.assertEqual(result["confidence"], "high")

    def test_high_observed_block_index_does_not_mean_hunyuan(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.7.attn.to_q.lora_A.weight": tensor([8, 3072]),
                "transformer.single_transformer_blocks.47.attn.to_q.lora_A.weight": tensor([8, 3072]),
            }
        )
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["structural_family"], "DualSingleStreamDiT")
        self.assertEqual(result["status"], "ambiguous")

    def test_generic_transformer_blocks_are_not_assumed_flux_like(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
            }
        )
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["structural_family"], "TransformerDiT")

    def test_qwen_image_modulation_pair_is_distinctive(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.0.img_mod.1.lora_A.weight": tensor([8, 3072]),
                "transformer.transformer_blocks.0.txt_mod.1.lora_A.weight": tensor([8, 3072]),
            }
        )
        self.assertEqual(result["arch"], "QwenImage")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_hunyuan_token_refiner_signature(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
                "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
                "transformer.context_embedder.token_refiner.refiner_blocks.0.self_attn.q_proj.lora_A.weight": tensor([8, 3072]),
            }
        )
        self.assertEqual(result["arch"], "Hunyuan")
        self.assertEqual(result["confidence"], "high")

    def test_sparse_hunyuan_token_refiner_signature(self):
        result = self.classify(
            {
                "lora_transformer_context_embedder_token_refiner_refiner_blocks_0_"
                "self_attn_q_proj.lora_A.weight": tensor([8, 3072]),
            }
        )
        self.assertEqual(result["arch"], "Hunyuan")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_flux_like_layout_can_be_settled_by_hunyuan_metadata(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
                "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
            },
            {"modelspec.architecture": "hunyuan-video"},
        )
        self.assertEqual(result["arch"], "Hunyuan")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_generic_hunyuan_name_is_not_forced_to_video(self):
        result = self.classify({}, filename="hunyuan_style.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["name_arch"], "")

    def test_sd15_structure_and_sd2_metadata_is_a_conflict(self):
        result = self.classify(
            {
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 768])
            },
            {"ss_base_model_version": "stable-diffusion-v2-1"},
        )
        self.assertEqual(result["arch"], "SD15")
        self.assertEqual(result["status"], "conflict")
        self.assertIn("CONFLICT", result["evidence"])

    def test_explicit_pony_v6_xl_conflicts_with_sd15_structure(self):
        result = self.classify(
            {
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 768])
            },
            filename="MyPonyV6XLStyle.safetensors",
        )
        self.assertEqual(result["arch"], "SD15")
        self.assertEqual(result["ecosystem"], "Pony")
        self.assertEqual(result["status"], "conflict")

    def test_generic_pony_can_follow_sd15_structure(self):
        result = self.classify(
            {
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 768])
            },
            filename="MyPonyStyle.safetensors",
        )
        self.assertEqual(result["arch"], "SD15")
        self.assertEqual(result["ecosystem"], "Pony")
        self.assertEqual(result["status"], "ok")

    def test_ss_sd_model_name_is_metadata_not_filename(self):
        result = self.classify(
            {},
            {"ss_sd_model_name": "sd_xl_base_1.0.safetensors"},
        )
        self.assertEqual(result["arch"], "SDXL")
        self.assertEqual(result["confidence"], "medium")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metadata_arch"], "SDXL")

    def test_multiple_techniques_are_preserved(self):
        result = self.classify({}, filename="ponyLightningDMD2Turbo.safetensors")
        self.assertEqual(result["ecosystem"], "Pony")
        self.assertEqual(result["techniques"], "Turbo;Lightning;DMD")
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["status"], "ambiguous")

    def test_swan_does_not_match_wan(self):
        result = self.classify({}, filename="SwanLakeStyle.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["name_arch"], "")

    def test_wan_version_task_and_noise_form_a_compatibility_subfolder(self):
        result = self.classify(
            {
                "blocks.0.self_attn.k.lora_A.weight": tensor([8, 5120]),
                "blocks.0.cross_attn.k.lora_A.weight": tensor([8, 5120]),
                "blocks.0.ffn.0.lora_A.weight": tensor([8, 5120]),
            },
            {"ss_base_model_version": "wan-2.2-i2v-high-noise"},
        )
        self.assertEqual(result["arch"], "Wan")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sort_subfolder"], "v2.2+I2V+high-noise")

    def test_generic_flux_filename_is_not_forced_to_flux1(self):
        result = self.classify({}, filename="flux_character_style.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["name_arch"], "")

    def test_diffusers_flux1_class_name_is_explicit(self):
        result = self.classify(
            {},
            {"modelspec.architecture": "FluxTransformer2DModel"},
        )
        self.assertEqual(result["arch"], "Flux1")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_camel_case_longcat_metadata_is_recognised(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
                "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
            },
            {"modelspec.architecture": "LongCatImageTransformer2DModel"},
        )
        self.assertEqual(result["arch"], "LongCat")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_zimage_separate_qkv_width_signature(self):
        result = self.classify(
            {
                "transformer.layers.0.attention.to_q.lora_A.weight": tensor([8, 3840]),
                "transformer.layers.0.feed_forward.w2.lora_A.weight": tensor([8, 10240]),
            }
        )
        self.assertEqual(result["arch"], "ZImage")
        self.assertEqual(result["structural_family"], "ZImageDiT")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_zimage_comfy_fused_qkv_width_signature(self):
        result = self.classify(
            {
                "diffusion_model.layers.0.attention.qkv.lora_down.weight": tensor([8, 3840]),
                "diffusion_model.layers.0.feed_forward.w2.lora_down.weight": tensor([8, 10240]),
            }
        )
        self.assertEqual(result["arch"], "ZImage")
        self.assertEqual(result["structural_family"], "ZImageDiT")
        self.assertIn("fused-qkv", result["evidence"])

    def test_zimage_refiner_path_is_exact(self):
        result = self.classify(
            {
                "transformer.noise_refiner.0.attention.to_q.lora_A.weight": tensor([8, 3840]),
            }
        )
        self.assertEqual(result["arch"], "ZImage")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_zimage_base_is_a_model_variant(self):
        result = self.classify(
            {},
            {"modelspec.architecture": "Z-Image-Base"},
        )
        self.assertEqual(result["arch"], "ZImage")
        self.assertEqual(result["model_variant"], "Base")
        self.assertEqual(result["techniques"], "")
        self.assertEqual(result["sort_subfolder"], "Base")
        self.assertEqual(result["status"], "ok")

    def test_official_zimage_repo_name_is_the_base_variant(self):
        result = self.classify(
            {},
            {"pretrained_model_name_or_path": "Tongyi-MAI/Z-Image"},
        )
        self.assertEqual(result["arch"], "ZImage")
        self.assertEqual(result["model_variant"], "Base")
        self.assertEqual(result["sort_subfolder"], "Base")
        self.assertEqual(result["status"], "ok")

    def test_zimage_filename_without_variant_does_not_assume_base(self):
        result = self.classify({}, filename="zimage_character.safetensors")
        self.assertEqual(result["arch"], "ZImage")
        self.assertEqual(result["model_variant"], "")
        self.assertEqual(result["status"], "ambiguous")

    def test_zimage_turbo_is_not_reported_twice_as_a_technique(self):
        result = self.classify(
            {},
            {"modelspec.architecture": "Z-Image-Turbo"},
        )
        self.assertEqual(result["arch"], "ZImage")
        self.assertEqual(result["model_variant"], "Turbo")
        self.assertEqual(result["techniques"], "")
        self.assertEqual(result["sort_subfolder"], "Turbo")

    def test_zimage_filename_only_requires_review(self):
        result = self.classify({}, filename="portrait_Z-Image-Turbo.safetensors")
        self.assertEqual(result["arch"], "ZImage")
        self.assertEqual(result["model_variant"], "Turbo")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["confidence"], "low")

    def test_ideogram4_separate_qkv_width_signature(self):
        result = self.classify(
            {
                "transformer.layers.0.attention.to_q.lora_A.weight": tensor([8, 4608]),
                "transformer.layers.0.feed_forward.w2.lora_A.weight": tensor([8, 12288]),
            }
        )
        self.assertEqual(result["arch"], "Ideogram4")
        self.assertEqual(result["structural_family"], "Ideogram4DiT")
        self.assertEqual(result["status"], "ok")

    def test_ideogram4_comfy_fused_qkv_width_signature(self):
        result = self.classify(
            {
                "diffusion_model.layers.0.attention.qkv.lora_down.weight": tensor([8, 4608]),
                "diffusion_model.layers.0.feed_forward.w2.lora_down.weight": tensor([8, 12288]),
            }
        )
        self.assertEqual(result["arch"], "Ideogram4")
        self.assertIn("fused-qkv", result["evidence"])

    def test_ideogram4_llm_condition_module_is_exact(self):
        result = self.classify(
            {
                "diffusion_model.llm_cond_proj.lora_A.weight": tensor([8, 53248]),
            }
        )
        self.assertEqual(result["arch"], "Ideogram4")
        self.assertEqual(result["confidence"], "high")

    def test_shared_nextdit_layout_without_known_width_stays_ambiguous(self):
        result = self.classify(
            {
                "transformer.layers.0.attention.to_q.lora_A.weight": tensor([8, 4096]),
                "transformer.layers.0.feed_forward.w2.lora_A.weight": tensor([8, 11008]),
            }
        )
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["structural_family"], "SingleStreamSwiGLUDiT")
        self.assertEqual(result["status"], "ambiguous")

    def test_minimax_h3_structure_is_exact(self):
        result = self.classify(
            {
                "blocks.0.attn.qkv_proj.lora_A.weight": tensor([8, 5376]),
                "blocks.0.adaln_proj.linear.lora_A.weight": tensor([8, 5376]),
                "blocks.0.mlp.fc1.lora_A.weight": tensor([8, 5376]),
                "blocks.0.mlp.fc2.lora_A.weight": tensor([8, 14336]),
                "video_patch_proj.lora_A.weight": tensor([8, 16]),
                "audio_patch_proj.lora_A.weight": tensor([8, 16]),
            }
        )
        self.assertEqual(result["arch"], "MiniMaxH3")
        self.assertEqual(result["structural_family"], "MiniMaxH3OmniDiT")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_minimax_h3_fl2va_task_survives_camel_splitting(self):
        result = self.classify(
            {
                "blocks.0.attn.qkv_proj.lora_A.weight": tensor([8, 5376]),
                "blocks.0.mlp.fc2.lora_A.weight": tensor([8, 14336]),
            },
            {"modelspec.architecture": "MiniMax-H3-FL2VA"},
        )
        self.assertEqual(result["arch"], "MiniMaxH3")
        self.assertEqual(result["task_variant"], "FL2VA")
        self.assertEqual(result["sort_subfolder"], "FL2VA")
        self.assertEqual(result["unresolved_hint"], "")

    def test_generic_minimax_name_is_unresolved(self):
        result = self.classify({}, filename="minimax_character.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["unresolved_hint"], "MiniMax (family unspecified)")

    def test_anima_llm_adapter_path_is_exact(self):
        result = self.classify(
            {
                "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight": tensor([8, 2048]),
            }
        )
        self.assertEqual(result["arch"], "Anima")
        self.assertEqual(result["structural_family"], "CosmosPredict2DiT")
        self.assertEqual(result["confidence"], "high")

    def test_cosmos_style_denoiser_alone_does_not_claim_anima(self):
        result = self.classify(
            {
                "blocks.0.self_attn.q_proj.lora_A.weight": tensor([8, 2048]),
                "blocks.0.cross_attn.q_proj.lora_A.weight": tensor([8, 2048]),
                "blocks.0.mlp.layer1.lora_A.weight": tensor([8, 2048]),
            }
        )
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["structural_family"], "CosmosPredict2DiT")
        self.assertEqual(result["status"], "ambiguous")

    def test_anima_metadata_settles_cosmos_style_denoiser(self):
        result = self.classify(
            {
                "blocks.0.self_attn.q_proj.lora_A.weight": tensor([8, 2048]),
                "blocks.0.cross_attn.q_proj.lora_A.weight": tensor([8, 2048]),
                "blocks.0.mlp.layer1.lora_A.weight": tensor([8, 2048]),
            },
            {"modelspec.architecture": "Anima-Base-v1.0"},
        )
        self.assertEqual(result["arch"], "Anima")
        self.assertEqual(result["model_variant"], "Base")
        self.assertEqual(result["model_version"], "1.0")
        self.assertEqual(result["status"], "ok")

    def test_anima_turbo_is_not_animagine_or_generic_turbo(self):
        result = self.classify(
            {},
            {"modelspec.architecture": "Anima-Turbo-v1.0"},
        )
        self.assertEqual(result["arch"], "Anima")
        self.assertEqual(result["ecosystem"], "")
        self.assertEqual(result["model_variant"], "Turbo")
        self.assertEqual(result["techniques"], "")

    def test_ernie_image_structure_is_exact(self):
        result = self.classify(
            {
                "transformer.layers.0.self_attention.to_q.lora_A.weight": tensor([8, 3072]),
                "transformer.layers.0.mlp.linear_fc2.lora_A.weight": tensor([8, 8192]),
            }
        )
        self.assertEqual(result["arch"], "ErnieImage")
        self.assertEqual(result["structural_family"], "ErnieImageDiT")
        self.assertEqual(result["status"], "ok")

    def test_literal_bernie_is_not_guessed(self):
        result = self.classify({}, filename="bernie_style_v1.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["status"], "ambiguous")
        self.assertIn("Bernini or ERNIE-Image", result["unresolved_hint"])

    def test_bernini_r_is_wan_22_variant(self):
        result = self.classify(
            {
                "blocks.0.self_attn.k.lora_A.weight": tensor([8, 5120]),
                "blocks.0.cross_attn.k.lora_A.weight": tensor([8, 5120]),
                "blocks.0.ffn.0.lora_A.weight": tensor([8, 5120]),
            },
            {"modelspec.architecture": "Bernini-R-I2V-high-noise"},
        )
        self.assertEqual(result["arch"], "Wan")
        self.assertEqual(result["model_variant"], "Bernini-R")
        self.assertEqual(result["model_version"], "2.2")
        self.assertEqual(result["task_variant"], "I2V")
        self.assertEqual(result["noise_stage"], "high")
        self.assertEqual(
            result["sort_subfolder"],
            "Bernini-R__v2.2+I2V+high-noise",
        )

    def test_bernini_filename_only_requires_review(self):
        result = self.classify({}, filename="Bernini-R-character.safetensors")
        self.assertEqual(result["arch"], "Wan")
        self.assertEqual(result["model_variant"], "Bernini-R")
        self.assertEqual(result["status"], "ambiguous")

    def test_pony_v6_xl_metadata_targets_sdxl(self):
        result = self.classify(
            {},
            {"ss_base_model_version": "Pony-Diffusion-V6-XL"},
        )
        self.assertEqual(result["arch"], "SDXL")
        self.assertEqual(result["ecosystem"], "Pony")
        self.assertEqual(result["model_version"], "6")
        self.assertEqual(result["sort_subfolder"], "Pony__v6")
        self.assertEqual(result["status"], "ok")

    def test_bare_pony_v6_does_not_choose_between_15_and_xl(self):
        result = self.classify(
            {},
            {"ss_base_model_version": "Pony-Diffusion-V6"},
        )
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["ecosystem"], "Pony")
        self.assertEqual(result["model_version"], "6")
        self.assertEqual(result["status"], "ambiguous")

    def test_explicit_pony_v6_15_targets_sd15(self):
        result = self.classify(
            {},
            {"ss_base_model_version": "Pony-Diffusion-V6-1.5"},
        )
        self.assertEqual(result["arch"], "SD15")
        self.assertEqual(result["ecosystem"], "Pony")
        self.assertEqual(result["model_version"], "6")
        self.assertEqual(result["status"], "ok")

    def test_pony_v7_metadata_targets_auraflow(self):
        result = self.classify(
            {},
            {"ss_base_model_version": "Pony-Diffusion-V7"},
        )
        self.assertEqual(result["arch"], "AuraFlow")
        self.assertEqual(result["ecosystem"], "Pony")
        self.assertEqual(result["model_version"], "7")
        self.assertEqual(result["sort_subfolder"], "Pony__v7")
        self.assertEqual(result["status"], "ok")

    def test_pony_v7_and_auraflow_structure_agree(self):
        result = self.classify(
            {
                "transformer.joint_transformer_blocks.0.attn.add_k_proj.lora_A.weight": tensor([8, 3072]),
                "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
            },
            {"ss_base_model_version": "Pony-Diffusion-V7"},
        )
        self.assertEqual(result["arch"], "AuraFlow")
        self.assertEqual(result["structural_family"], "AuraFlowDiT")
        self.assertEqual(result["status"], "ok")

    def test_generic_pony_does_not_imply_one_generation(self):
        result = self.classify({}, filename="pony_portrait.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["ecosystem"], "Pony")
        self.assertEqual(result["status"], "ambiguous")

    def test_illustrious_metadata_targets_sdxl(self):
        result = self.classify(
            {},
            {"ss_base_model_version": "Illustrious-XL-v2"},
        )
        self.assertEqual(result["arch"], "SDXL")
        self.assertEqual(result["ecosystem"], "Illustrious")
        self.assertEqual(result["status"], "ok")

    def test_illustrious_conflicts_with_sd15_structure(self):
        result = self.classify(
            {
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 768])
            },
            {"modelspec.architecture": "Illustrious-XL"},
        )
        self.assertEqual(result["arch"], "SD15")
        self.assertEqual(result["ecosystem"], "Illustrious")
        self.assertEqual(result["status"], "conflict")

    def test_generic_turbo_remains_a_technique(self):
        result = self.classify(
            {
                "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 2048])
            },
            filename="portrait_turbo.safetensors",
        )
        self.assertEqual(result["arch"], "SDXL")
        self.assertEqual(result["model_variant"], "")
        self.assertEqual(result["techniques"], "Turbo")

    def test_krea2_diffusers_structure_is_exact(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.0.attn.to_gate.lora_A.weight": tensor([8, 6144]),
                "transformer.transformer_blocks.0.ff.down.lora_A.weight": tensor([8, 16384]),
            }
        )
        self.assertEqual(result["arch"], "Krea2")
        self.assertEqual(result["structural_family"], "Krea2DiT")
        self.assertEqual(result["status"], "ok")

    def test_krea2_comfy_structure_is_exact(self):
        result = self.classify(
            {
                "diffusion_model.blocks.0.attn.wq.lora_down.weight": tensor([8, 6144]),
                "diffusion_model.blocks.0.attn.gate.lora_down.weight": tensor([8, 6144]),
                "diffusion_model.blocks.0.mlp.down.lora_down.weight": tensor([8, 16384]),
            }
        )
        self.assertEqual(result["arch"], "Krea2")
        self.assertEqual(result["structural_family"], "Krea2DiT")
        self.assertIn("ComfyUI", result["evidence"])

    def test_krea2_text_fusion_is_exact_even_when_main_blocks_are_sparse(self):
        result = self.classify(
            {
                "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_A.weight": tensor([8, 2560]),
            }
        )
        self.assertEqual(result["arch"], "Krea2")
        self.assertEqual(result["confidence"], "high")

    # --- 0.5.0: bare filename tokens no longer manufacture conflicts ---

    def test_bare_klein_filename_does_not_conflict_with_sdxl_structure(self):
        result = self.classify(
            {
                "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 2048])
            },
            filename="klein_haruka_character.safetensors",
        )
        self.assertEqual(result["arch"], "SDXL")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["model_variant"], "")
        self.assertIn("'Klein' alone is not precise enough", result["evidence"])

    def test_flux_scoped_klein_still_classifies_and_tags(self):
        result = self.classify({}, filename="Flux2_Klein_detail.safetensors")
        self.assertEqual(result["arch"], "Flux2")
        self.assertEqual(result["model_variant"], "Klein")
        self.assertEqual(result["status"], "ambiguous")

    def test_bare_schnell_without_flux_context_is_inert(self):
        result = self.classify({}, filename="schnell_zeichnen_sketch.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["model_variant"], "")
        self.assertIn("'Schnell' alone is not precise enough", result["evidence"])

    def test_flux1_schnell_metadata_keeps_variant(self):
        result = self.classify(
            {},
            metadata={"ss_base_model": "black-forest-labs/FLUX.1-schnell"},
        )
        self.assertEqual(result["arch"], "Flux1")
        self.assertEqual(result["model_variant"], "Schnell")
        self.assertEqual(result["status"], "ok")

    def test_obi_wan_is_not_wan(self):
        result = self.classify({}, filename="Obi-Wan_Kenobi_style.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertIn("'Wan' alone is not precise enough", result["evidence"])

    def test_bare_wan_filename_does_not_conflict_with_sdxl_structure(self):
        result = self.classify(
            {
                "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 2048])
            },
            filename="wan_style_pack.safetensors",
        )
        self.assertEqual(result["arch"], "SDXL")
        self.assertEqual(result["status"], "ok")

    def test_versioned_wan_filename_still_classifies(self):
        result = self.classify({}, filename="wan2.2_i2v_smooth_motion.safetensors")
        self.assertEqual(result["arch"], "Wan")
        self.assertEqual(result["model_version"], "2.2")
        self.assertEqual(result["task_variant"], "I2V")
        self.assertEqual(result["status"], "ambiguous")

    def test_bare_chroma_filename_is_inert_but_noted(self):
        result = self.classify({}, filename="chroma_glow_style.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertIn("'Chroma' alone is not precise enough", result["evidence"])

    def test_chroma_identity_metadata_still_classifies(self):
        result = self.classify(
            {},
            metadata={"ss_base_model": "chroma-unlocked-v38"},
        )
        self.assertEqual(result["arch"], "Chroma")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_chroma_in_free_text_title_does_not_classify(self):
        result = self.classify(
            {},
            metadata={"modelspec.title": "chroma aberration photo look"},
        )
        self.assertEqual(result["arch"], "Unknown")

    def test_bare_anima_filename_is_inert_but_qualified_form_classifies(self):
        inert = self.classify({}, filename="anima_animus_concept.safetensors")
        self.assertEqual(inert["arch"], "Unknown")
        self.assertIn("'Anima' alone is not precise enough", inert["evidence"])

        qualified = self.classify({}, filename="Anima-Base_v1.0_style.safetensors")
        self.assertEqual(qualified["arch"], "Anima")
        self.assertEqual(qualified["status"], "ambiguous")

    # --- 0.5.0: SD3, Chroma, and LongCat structural rules ---

    def test_sd3_mmdit_joint_blocks_naming_is_exact(self):
        result = self.classify(
            {
                "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight": tensor([8, 1536]),
                "lora_unet_joint_blocks_0_context_block_attn_qkv.lora_down.weight": tensor([8, 1536]),
            }
        )
        self.assertEqual(result["arch"], "SD3")
        self.assertEqual(result["structural_family"], "JointAttentionDiT")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_sd3_diffusers_width_1536_is_exact(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight": tensor([8, 1536]),
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 1536]),
            }
        )
        self.assertEqual(result["arch"], "SD3")
        self.assertEqual(result["confidence"], "high")

    def test_sd35_large_width_2432_is_exact(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight": tensor([8, 2432]),
                "transformer.transformer_blocks.0.attn2.to_q.lora_A.weight": tensor([8, 2432]),
            }
        )
        self.assertEqual(result["arch"], "SD3")
        self.assertEqual(result["confidence"], "high")

    def test_joint_layout_width_3072_stays_ambiguous(self):
        result = self.classify(
            {
                "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight": tensor([8, 3072]),
            }
        )
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["structural_family"], "JointAttentionDiT")
        self.assertEqual(result["status"], "ambiguous")

    def test_chroma_distilled_guidance_layer_is_exact(self):
        result = self.classify(
            {
                "transformer.distilled_guidance_layer.layers.0.in_proj.lora_A.weight": tensor([8, 5120]),
            }
        )
        self.assertEqual(result["arch"], "Chroma")
        self.assertEqual(result["structural_family"], "DualSingleStreamDiT")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_chroma_structure_against_flux1_metadata_conflicts(self):
        result = self.classify(
            {
                "transformer.distilled_guidance_layer.layers.0.in_proj.lora_A.weight": tensor([8, 5120]),
            },
            metadata={"modelspec.architecture": "FluxTransformer2DModel"},
        )
        self.assertEqual(result["status"], "conflict")

    def test_longcat_context_width_inside_dual_single_layout(self):
        result = self.classify(
            {
                "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
                "transformer.context_embedder.lora_A.weight": tensor([8, 3584]),
            }
        )
        self.assertEqual(result["arch"], "LongCat")
        self.assertEqual(result["structural_family"], "DualSingleStreamDiT")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["status"], "ok")

    def test_4096_context_width_stays_shared_between_flux1_and_chroma(self):
        result = self.classify(
            {
                "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 3072]),
                "transformer.context_embedder.lora_A.weight": tensor([8, 4096]),
            }
        )
        self.assertEqual(result["arch"], "Unknown")
        self.assertIn("shared by Flux1 and Chroma", result["evidence"])


class InspectorFileTests(unittest.TestCase):
    def test_directory_scan_accepts_uppercase_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = write_fake_safetensors(root / "MODEL.SAFETENSORS", {})
            self.assertEqual(inspect_lora.expand_arg(str(root)), [model])

    def test_existing_bracketed_path_is_not_treated_as_a_glob(self):
        with tempfile.TemporaryDirectory() as temp:
            model = write_fake_safetensors(Path(temp) / "style[final].safetensors", {})
            self.assertEqual(inspect_lora.expand_arg(str(model)), [model])

    def test_glob_skips_directory_named_like_a_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = write_fake_safetensors(root / "real.safetensors", {})
            (root / "fake.safetensors").mkdir()
            matches = inspect_lora.expand_arg(str(root / "*.safetensors"))
            self.assertEqual(matches, [model])

    def test_header_reader_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            write_fake_safetensors(
                path,
                {
                    "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_"
                    "attn2_to_k.lora_down.weight": tensor([8, 2048])
                },
            )
            record = inspect_lora.inspect_one_file(path)
            self.assertEqual(record["arch"], "SDXL")
            self.assertEqual(len(record["header_sha256"]), 64)
            self.assertEqual(record["status"], "ok")

    @unittest.skipIf(not hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_is_skipped_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = write_fake_safetensors(root / "target.safetensors", {})
            link = root / "link.safetensors"
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is not permitted")
            record = inspect_lora.inspect_one_file(link)
            self.assertEqual(record["status"], "error")
            self.assertIn("symlink skipped", record["error"])


class SorterTests(unittest.TestCase):
    def make_sdxl_record(self, path: Path):
        write_fake_safetensors(
            path,
            {
                "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                "attn2_to_k.lora_down.weight": tensor([8, 2048])
            },
        )
        return inspect_lora.inspect_one_file(path)

    def test_safe_sidecars_exclude_another_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(root / "foo.safetensors", {})
            (root / "foo.preview.png").write_bytes(b"preview")
            (root / "foo.civitai.info").write_text("{}", encoding="utf-8")
            (root / "foo.json").write_text("{}", encoding="utf-8")
            write_fake_safetensors(root / "foo.v2.safetensors", {})
            (root / "foobar.png").write_bytes(b"not a sidecar")

            names = [path.name for path in inspect_sort_lora.discover_sidecars(source)]
            self.assertEqual(names, ["foo.civitai.info", "foo.json", "foo.preview.png"])
            self.assertNotIn("foo.v2.safetensors", names)

    def test_sidecar_prefix_matching_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(root / "Character.safetensors", {})
            (root / "character.preview.PNG").write_bytes(b"preview")
            names = [path.name for path in inspect_sort_lora.discover_sidecars(source)]
            self.assertEqual(names, ["character.preview.PNG"])

    def test_destination_beyond_max_path_length_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            index = root / "index.csv"
            write_index(index, [record])
            base = root / "sorted"
            limit = len(str(base))  # any destination under base must exceed this
            plan = inspect_sort_lora.build_sort_plan(index, base, max_path_length=limit)
            self.assertEqual(plan[0]["action"], "skip_path_too_long")
            self.assertEqual(plan[0]["move"], "False")
            self.assertIn(str(limit), plan[0]["reason"])

    def test_long_sidecar_destination_blocks_whole_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            # ".civitai.info" is one character longer than ".safetensors", so a
            # limit sized exactly for the primary rejects only the sidecar.
            (root / "model.civitai.info").write_text("{}", encoding="utf-8")
            index = root / "index.csv"
            write_index(index, [record])
            base = root / "sorted"
            unlimited = inspect_sort_lora.build_sort_plan(index, base, max_path_length=0)
            self.assertEqual(unlimited[0]["action"], "move")
            primary_length = len(unlimited[0]["destination"])
            plan = inspect_sort_lora.build_sort_plan(
                index, base, max_path_length=primary_length
            )
            self.assertEqual(plan[0]["action"], "skip_path_too_long")
            # Only the sidecar exceeds a limit sized exactly for the primary.
            self.assertIn("model.civitai.info", plan[0]["reason"])
            self.assertNotIn("model.safetensors", plan[0]["reason"])

    def test_max_path_length_zero_disables_the_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            index = root / "index.csv"
            write_index(index, [record])
            plan = inspect_sort_lora.build_sort_plan(
                index, root / "sorted", max_path_length=0
            )
            self.assertEqual(plan[0]["action"], "move")

    def test_conflict_status_never_enters_move_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "pony.safetensors"
            record = self.make_sdxl_record(source)
            record["status"] = "conflict"
            index = root / "index.csv"
            write_index(index, [record])
            plan = inspect_sort_lora.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_conflict")

    def test_unrecognised_status_never_enters_move_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            record["status"] = "banana"
            index = root / "index.csv"
            write_index(index, [record])
            plan = inspect_sort_lora.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_invalid_status")

    def test_medium_confidence_is_not_moved_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(
                root / "wan.safetensors",
                {
                    "blocks.0.self_attn.k.lora_A.weight": tensor([8, 5120]),
                    "blocks.0.cross_attn.k.lora_A.weight": tensor([8, 5120]),
                    "blocks.0.ffn.0.lora_A.weight": tensor([8, 5120]),
                },
            )
            record = inspect_lora.inspect_one_file(source)
            self.assertEqual(record["confidence"], "medium")
            index = root / "index.csv"
            write_index(index, [record])
            plan = inspect_sort_lora.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_ambiguous")
            self.assertEqual(plan[0]["move"], "False")

    def test_metadata_only_medium_confidence_is_not_moved_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(
                root / "model.safetensors",
                {},
                {"ss_sd_model_name": "sd_xl_base_1.0.safetensors"},
            )
            record = inspect_lora.inspect_one_file(source)
            self.assertEqual(record["confidence"], "medium")
            self.assertEqual(record["status"], "ok")
            index = root / "index.csv"
            write_index(index, [record])
            plan = inspect_sort_lora.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_low_confidence")

    def test_destination_outside_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = self.make_sdxl_record(root / "model.safetensors")
            record["destination"] = str(root.parent / "escape" / "model.safetensors")
            index = root / "index.csv"
            write_index(index, [record])
            plan = inspect_sort_lora.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_outside_base")

    def test_changed_file_is_rejected_as_stale(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            with source.open("ab") as handle:
                handle.write(b"changed")
            index = root / "index.csv"
            write_index(index, [record])
            plan = inspect_sort_lora.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_stale_or_unverified")

    def test_sidecar_collision_blocks_whole_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            (root / "model.preview.png").write_bytes(b"preview")
            destination_dir = root / "sorted" / "SDXL"
            destination_dir.mkdir(parents=True)
            (destination_dir / "model.preview.png").write_bytes(b"occupied")
            index = root / "index.csv"
            write_index(index, [record])
            plan = inspect_sort_lora.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_collision")
            self.assertTrue(source.exists())

    def test_plan_detects_two_rows_with_same_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.make_sdxl_record(root / "a" / "same.safetensors")
            second = self.make_sdxl_record(root / "b" / "same.safetensors")
            index = root / "index.csv"
            write_index(index, [first, second])
            plan = inspect_sort_lora.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_plan_collision")
            self.assertEqual(plan[1]["action"], "skip_plan_collision")

    def test_apply_and_undo_move_primary_and_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            sidecar = root / "model.preview.png"
            sidecar.write_bytes(b"preview")
            index = root / "index.csv"
            write_index(index, [record])
            base = root / "sorted"
            plan = inspect_sort_lora.build_sort_plan(index, base)
            self.assertEqual(plan[0]["action"], "move")

            manifest = root / "manifest.json"
            with contextlib.redirect_stdout(io.StringIO()):
                counts = inspect_sort_lora.apply_plan(
                    plan,
                    csv_path=index,
                    base_dir=base,
                    dry_run=False,
                    manifest_path=manifest,
                )
            destination = base / "SDXL" / "model.safetensors"
            destination_sidecar = base / "SDXL" / "model.preview.png"
            self.assertEqual(counts["moved"], 1)
            self.assertTrue(destination.exists())
            self.assertTrue(destination_sidecar.exists())
            self.assertFalse(source.exists())
            self.assertFalse(sidecar.exists())

            with contextlib.redirect_stdout(io.StringIO()):
                rc = inspect_sort_lora.undo_manifest(manifest, apply_changes=True)
            self.assertEqual(rc, 0)
            self.assertTrue(source.exists())
            self.assertTrue(sidecar.exists())
            self.assertFalse(destination.exists())
            self.assertFalse(destination_sidecar.exists())


    def test_sidecar_change_after_planning_blocks_apply(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            index = root / "index.csv"
            write_index(index, [record])
            base = root / "sorted"
            plan = inspect_sort_lora.build_sort_plan(index, base)
            (root / "model.preview.png").write_bytes(b"appeared later")

            with contextlib.redirect_stdout(io.StringIO()):
                counts = inspect_sort_lora.apply_plan(
                    plan,
                    csv_path=index,
                    base_dir=base,
                    dry_run=False,
                    manifest_path=root / "manifest.json",
                )
            self.assertEqual(counts["moved"], 0)
            self.assertEqual(plan[0]["action"], "skip_stale_sidecars")
            self.assertTrue(source.exists())

    def test_failed_group_move_rolls_back_completed_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            sidecar = root / "model.preview.png"
            sidecar.write_bytes(b"preview")
            index = root / "index.csv"
            write_index(index, [record])
            base = root / "sorted"
            plan = inspect_sort_lora.build_sort_plan(index, base)
            manifest = root / "manifest.json"

            real_move = inspect_sort_lora.shutil.move
            calls = {"count": 0}

            def flaky_move(src, dst):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("synthetic primary move failure")
                return real_move(src, dst)

            with mock.patch.object(inspect_sort_lora.shutil, "move", side_effect=flaky_move):
                with contextlib.redirect_stdout(io.StringIO()):
                    counts = inspect_sort_lora.apply_plan(
                        plan,
                        csv_path=index,
                        base_dir=base,
                        dry_run=False,
                        manifest_path=manifest,
                    )

            self.assertEqual(counts["rolled_back"], 1)
            self.assertEqual(counts["errors"], 0)
            self.assertTrue(source.exists())
            self.assertTrue(sidecar.exists())
            self.assertFalse((base / "SDXL" / "model.safetensors").exists())
            self.assertFalse((base / "SDXL" / "model.preview.png").exists())
            operation = json.loads(manifest.read_text(encoding="utf-8"))["operations"][0]
            self.assertEqual(operation["status"], "rolled_back")

    def test_existing_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            index = root / "index.csv"
            write_index(index, [record])
            plan = inspect_sort_lora.build_sort_plan(index, root / "sorted")
            manifest = root / "manifest.json"
            manifest.write_text("do not replace", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                counts = inspect_sort_lora.apply_plan(
                    plan,
                    csv_path=index,
                    base_dir=root / "sorted",
                    dry_run=False,
                    manifest_path=manifest,
                )
            self.assertEqual(counts["errors"], 1)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "do not replace")
            self.assertTrue(source.exists())

    def test_manifest_cannot_overlap_a_planned_sidecar_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "model.safetensors"
            record = self.make_sdxl_record(source)
            (root / "model.preview.png").write_bytes(b"preview")
            index = root / "index.csv"
            write_index(index, [record])
            base = root / "sorted"
            plan = inspect_sort_lora.build_sort_plan(index, base)
            manifest = base / "SDXL" / "model.preview.png"

            with contextlib.redirect_stdout(io.StringIO()):
                counts = inspect_sort_lora.apply_plan(
                    plan,
                    csv_path=index,
                    base_dir=base,
                    dry_run=False,
                    manifest_path=manifest,
                )
            self.assertEqual(counts["errors"], 1)
            self.assertFalse(manifest.exists())
            self.assertTrue(source.exists())

    def test_undo_flags_pending_operations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"operations": [{"status": "pending", "transfers": []}]}),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rc = inspect_sort_lora.undo_manifest(manifest, apply_changes=False)
            self.assertEqual(rc, 1)

    def test_windows_reserved_bucket_name_is_sanitised(self):
        self.assertEqual(inspect_sort_lora._safe_component("CON"), "_CON")


# ---------------------------------------------------------------------------
# Real-file smoke mode
# ---------------------------------------------------------------------------

def _display_value(record, field_name: str, fallback: str = "-") -> str:
    value = str(record.get(field_name, "") or "").strip()
    return value or fallback


def _print_smoke_record(record) -> None:
    path = _display_value(record, "path")
    print("\n" + "=" * 80)
    print(f"LoRA: {path}")
    print("=" * 80)

    rows = (
        ("Architecture", _display_value(record, "arch", "Unknown")),
        ("Structural family", _display_value(record, "structural_family", "Unknown")),
        ("Status", _display_value(record, "status", "unknown")),
        ("Confidence", _display_value(record, "confidence", "low")),
        ("Model version", _display_value(record, "model_version")),
        ("Task variant", _display_value(record, "task_variant")),
        ("Noise stage", _display_value(record, "noise_stage")),
        ("Ecosystem", _display_value(record, "ecosystem")),
        ("Model variant", _display_value(record, "model_variant")),
        ("Unresolved hint", _display_value(record, "unresolved_hint")),
        ("Techniques", _display_value(record, "techniques")),
        ("Suggested subfolder", _display_value(record, "sort_subfolder")),
        ("Tensor-key format", _display_value(record, "format", "unknown")),
        ("Tensor records", _display_value(record, "total_tensors", "0")),
        ("Header bytes", _display_value(record, "header_bytes", "0")),
        ("File bytes", _display_value(record, "file_size", "0")),
        ("Structure says", _display_value(record, "structure_arch")),
        ("Metadata says", _display_value(record, "metadata_arch")),
        ("Filename says", _display_value(record, "name_arch")),
    )
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")

    error = str(record.get("error", "") or "").strip()
    if error:
        print("\nERROR")
        print(textwrap.fill(error, width=78, subsequent_indent="  "))

    evidence = str(record.get("evidence", "") or "").strip()
    if evidence:
        print("\nEvidence")
        for item in evidence.split("; "):
            item = item.strip()
            if item:
                print(textwrap.fill(item, width=76, initial_indent="  - ", subsequent_indent="    "))


def run_real_file_smoke(
    inputs,
    *,
    follow_symlinks: bool = False,
    json_output: bool = False,
) -> int:
    """Inspect literal files, folders, or globs and print classification records.

    This is intentionally read-only. It does not write the inspection CSV, invoke the
    sorter, or move files. The return code is non-zero only for unreadable/invalid
    inputs, not merely for an ambiguous or conflicting classification.
    """
    candidates = []
    seen = set()
    for raw in inputs:
        expanded = inspect_lora.expand_arg(os.path.expanduser(str(raw)))
        for candidate in expanded:
            marker = str(candidate.absolute())
            if marker not in seen:
                seen.add(marker)
                candidates.append(candidate)

    if not candidates:
        print("No files matched the supplied input.", file=sys.stderr)
        return 1

    records = [
        inspect_lora.inspect_one_file(candidate, follow_symlinks=follow_symlinks)
        for candidate in candidates
    ]

    if json_output:
        print(json.dumps(records, indent=2, ensure_ascii=False))
    else:
        for record in records:
            _print_smoke_record(record)

        counts = {}
        for record in records:
            status = str(record.get("status", "unknown") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        print(f"\nSmoke inspection complete: files={len(records)}; {rendered}")
        print("No CSV was written and no files were moved.")

    return 1 if any(record.get("status") == "error" for record in records) else 0


def _looks_like_smoke_invocation(argv) -> bool:
    if not argv:
        return False
    if "--inspect" in argv:
        return True

    # Preserve normal unittest selectors such as
    # ClassifierTests.test_sd15_by_cross_attention_width. A literal existing path,
    # a safetensors suffix, or a glob is instead treated as smoke-inspection input.
    for argument in argv:
        if argument.startswith("-"):
            continue
        expanded = os.path.expanduser(argument)
        path = Path(expanded)
        if (
            path.exists()
            or path.is_symlink()
            or path.suffix.lower() == ".safetensors"
            or any(character in expanded for character in "*?[")
        ):
            return True
    return False


def _smoke_cli(argv) -> int:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description=(
            "Read-only smoke inspection of real LoRA safetensors files. "
            "With no file input, this script runs its unittest regression suite."
        ),
    )
    parser.add_argument("inputs", nargs="+", help="safetensors file, folder, or glob")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="explicitly select smoke-inspection mode (optional when a path is supplied)",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="inspect symlink targets (default: report symlinks as errors)",
    )
    parser.add_argument("--json", action="store_true", help="emit full records as JSON")
    args = parser.parse_args(argv)
    return run_real_file_smoke(
        args.inputs,
        follow_symlinks=args.follow_symlinks,
        json_output=args.json,
    )


class SmokeRunnerTests(unittest.TestCase):
    def test_real_file_argument_is_inspected_instead_of_sent_to_unittest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "NoraX_Krea2.safetensors",
                {
                    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": tensor([8, 4096]),
                },
                {"modelspec.architecture": "krea-2"},
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = run_real_file_smoke([str(path)])
            rendered = output.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("Architecture", rendered)
            self.assertIn("Krea2", rendered)
            self.assertIn("No CSV was written and no files were moved", rendered)

    def test_missing_real_file_returns_nonzero_without_unittest_failedtest(self):
        missing = "/definitely/not/here/model.safetensors"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = run_real_file_smoke([missing])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR", output.getvalue())
        self.assertNotIn("_FailedTest", output.getvalue())

    def test_unittest_selector_is_not_mistaken_for_a_path(self):
        selector = "ClassifierTests.test_sd15_by_cross_attention_width"
        self.assertFalse(_looks_like_smoke_invocation([selector]))

    def test_safetensors_path_selects_smoke_mode_even_when_missing(self):
        self.assertTrue(_looks_like_smoke_invocation(["missing_model.safetensors"]))


if __name__ == "__main__":
    command_line = sys.argv[1:]
    if _looks_like_smoke_invocation(command_line):
        raise SystemExit(_smoke_cli(command_line))
    unittest.main(verbosity=2)
