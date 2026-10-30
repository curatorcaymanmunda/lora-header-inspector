"""Positive, sparse, conflicting, and common-word traps for new image models."""
import unittest

from lora_header_inspector.classifier import classify


SIGNATURES = {
    "Flux3": {
        "diffusion_model.content_mode_blocks.image.3.q_proj.lora_A.weight": 3072,
        "diffusion_model.txt_mode_blocks.1.attn_out.lora_A.weight": 3072,
    },
    "MOSSImage": {
        "transformer.layers.7.proj.ffn_up_gate.lora_A.weight": 3072,
        "transformer.shared_block_modulation.lora_A.weight": 3072,
    },
    "FIBO2": {
        "diffusion_model.perceiver.layers.1.cross_attention.to_kv.lora_A.weight": 7680,
        "diffusion_model.layers.14.cross_attention.to_kv.lora_A.weight": 7680,
    },
}


def header(widths):
    return {key: {"dtype": "BF16", "shape": [8, width]} for key, width in widths.items()}


class RecentImageSignatureTests(unittest.TestCase):
    def test_distinctive_sparse_structures(self):
        for arch, widths in SIGNATURES.items():
            with self.subTest(arch=arch):
                result = classify(header(widths), {}, "neutral.safetensors")
                self.assertEqual(result["arch"], arch)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["confidence"], "high")

    def test_kohya_separators_and_down_weights(self):
        for arch, widths in SIGNATURES.items():
            converted = {"lora_unet_" + key.split(".", 1)[1].split(".lora_A")[0].replace(".", "_") + ".lora_down.weight": width
                         for key, width in widths.items()}
            with self.subTest(arch=arch):
                self.assertEqual(classify(header(converted), {}, "neutral.safetensors")["arch"], arch)

    def test_each_anchor_alone_is_insufficient(self):
        for arch, widths in SIGNATURES.items():
            for key, width in widths.items():
                with self.subTest(arch=arch, key=key):
                    self.assertNotEqual(classify(header({key: width}), {}, "neutral.safetensors")["arch"], arch)

    def test_wrong_width_is_insufficient(self):
        for arch, widths in SIGNATURES.items():
            for key in widths:
                altered = dict(widths, **{key: 1024})
                with self.subTest(arch=arch, key=key):
                    self.assertNotEqual(classify(header(altered), {}, "neutral.safetensors")["arch"], arch)

    def test_conflicting_identity_is_blocked(self):
        for arch, widths in SIGNATURES.items():
            with self.subTest(arch=arch):
                result = classify(header(widths), {"modelspec.architecture": "sdxl"}, "neutral.safetensors")
                self.assertNotEqual(result["status"], "ok")

    def test_explicit_metadata_identities(self):
        for name, arch in (("black-forest-labs/flux-3-image-klein-base", "Flux3"),
                           ("OpenMOSS-Team/MOSS-Image-SFT", "MOSSImage"),
                           ("briaai/FIBO", "FIBO"), ("briaai/Fibo-2.0-dev", "FIBO2")):
            with self.subTest(name=name):
                result = classify({}, {"base_model": name}, "neutral.safetensors")
                self.assertEqual(result["arch"], arch)
                self.assertEqual(result["status"], "ok")

    def test_common_words_do_not_classify(self):
        for name in ("moss_on_stone", "fibo_portrait", "fibonacci", "flux30", "klein"):
            with self.subTest(name=name):
                self.assertEqual(classify({}, {}, name + ".safetensors")["arch"], "Unknown")

    def test_flux3_klein_does_not_imply_flux2(self):
        result = classify(header(SIGNATURES["Flux3"]), {}, "flux3_image_klein.safetensors")
        self.assertEqual(result["arch"], "Flux3")
        self.assertEqual(result["status"], "ok")

    def test_generic_flux2_single_block_does_not_become_flux3(self):
        result = classify(header({"transformer.single_transformer_blocks.4.attn.to_qkv_mlp_proj.lora_A.weight": 6144}), {}, "neutral.safetensors")
        self.assertEqual(result["arch"], "Flux2")

    def test_qualified_filenames_still_require_review(self):
        for name in ("flux3_image_klein", "moss-image", "fibo-2.0"):
            with self.subTest(name=name):
                self.assertEqual(classify({}, {}, name + ".safetensors")["status"], "ambiguous")

    def test_mixed_exact_signatures_are_blocked(self):
        mixed = dict(SIGNATURES["Flux3"], **SIGNATURES["FIBO2"])
        self.assertEqual(classify(header(mixed), {}, "neutral.safetensors")["status"], "conflict")

    def test_flux3_generic_single_stream_is_not_a_conflict(self):
        widths = dict(SIGNATURES["Flux3"])
        widths["diffusion_model.single_blocks.27.mlp_in.lora_A.weight"] = 3072
        result = classify(header(widths), {}, "neutral.safetensors")
        self.assertEqual(result["arch"], "Flux3")
        self.assertEqual(result["status"], "ok")
