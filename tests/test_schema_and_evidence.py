from __future__ import annotations

import csv
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from lora_header_inspector import classifier, inspection, planner, schema
from lora_header_inspector.version import (
    INSPECTION_SCHEMA_NAME,
    INSPECTION_SCHEMA_VERSION,
    PLAN_SCHEMA_NAME,
)

from lora_header_inspector.classifier import classify
from tests.helpers import tensor, write_csv, write_fake_safetensors


class EvidenceTests(unittest.TestCase):
    def test_metadata_fields_are_parsed_independently(self):
        result = classifier.classify(
            {},
            {
                "modelspec.architecture": "Z-",
                "ss_base_model_version": "Image",
            },
            "neutral.safetensors",
        )
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["metadata_arch"], "")
        self.assertEqual(result["status"], "ambiguous")

    def test_conflicting_metadata_identity_fields_are_a_hard_conflict(self):
        result = classifier.classify(
            {},
            {
                "modelspec.architecture": "stable-diffusion-xl-v1-base",
                "ss_base_model_version": "stable-diffusion-v1-5",
            },
            "neutral.safetensors",
        )
        self.assertEqual(result["status"], "conflict")
        conflicts = json.loads(result["conflicts_json"])
        self.assertTrue(any(item["category"] == "architecture" for item in conflicts))
        self.assertIn("SDXL", result["metadata_arch"])
        self.assertIn("SD15", result["metadata_arch"])

    def test_version_task_and_noise_disagreements_are_structured_conflicts(self):
        result = classifier.classify(
            {},
            {"ss_base_model_version": "wan-2.1-i2v-high-noise"},
            "wan-2.2-t2v-low-noise.safetensors",
        )
        self.assertEqual(result["arch"], "Wan")
        self.assertEqual(result["status"], "conflict")
        categories = {item["category"] for item in json.loads(result["conflicts_json"])}
        self.assertTrue({"model_version", "task_variant", "noise_stage"}.issubset(categories))

    def test_structured_evidence_has_stable_rule_and_source_fields(self):
        result = classifier.classify(
            {
                "transformer.layers.0.attention.to_q.lora_A.weight": tensor([8, 3840]),
                "transformer.layers.0.feed_forward.w2.lora_A.weight": tensor([8, 10240]),
            },
            {"modelspec.architecture": "Z-Image-Base"},
            "portrait.safetensors",
        )
        evidence = json.loads(result["evidence_json"])
        self.assertTrue(evidence)
        for item in evidence:
            self.assertIn("rule_id", item)
            self.assertIn("source", item)
            self.assertIn("category", item)
            self.assertIn("message", item)
        self.assertTrue(any(item["rule_id"] == "zimage.width-signature" for item in evidence))
        self.assertTrue(
            any(item["source"] == "metadata" and item["field"] == "modelspec.architecture" for item in evidence)
        )

    def test_bare_qwen_is_not_assumed_to_be_qwen_image(self):
        result = classifier.classify({}, {}, "qwen_character_style.safetensors")
        self.assertEqual(result["arch"], "Unknown")
        self.assertEqual(result["name_arch"], "")
        self.assertIn("family unspecified", result["unresolved_hint"])


class SchemaTests(unittest.TestCase):
    def test_duplicate_column_names_are_rejected(self):
        valid, reason = schema.validate_inspection_schema(
            ["schema_name", "schema_version", "path", "path", "arch", "confidence"]
        )
        self.assertFalse(valid)
        self.assertIn("duplicate column", reason)

    def test_versioned_index_missing_required_columns_is_rejected(self):
        valid, reason = schema.validate_inspection_schema(
            ["schema_name", "schema_version", "path", "arch", "confidence"]
        )
        self.assertFalse(valid)
        self.assertIn("missing required", reason)

    def test_unversioned_csv_requires_explicit_legacy_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(root / "model.safetensors", {})
            index = root / "legacy.csv"
            fields = ["path", "arch", "confidence", "status", "error", "move", "destination"]
            write_csv(
                index,
                [{"path": str(source), "arch": "SDXL", "confidence": "high", "status": "ok"}],
                fields,
            )
            with self.assertRaisesRegex(ValueError, "no supported schema markers"):
                planner.build_sort_plan(index, root / "sorted", allow_unverified=True)
            plan = planner.build_sort_plan(
                index,
                root / "sorted",
                allow_unverified=True,
                allow_legacy_plan=True,
            )
            self.assertEqual(plan[0]["legacy_imported"], "True")
            self.assertEqual(plan[0]["plan_schema_name"], PLAN_SCHEMA_NAME)

    def test_unsupported_versioned_plan_is_rejected_even_with_legacy_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(root / "model.safetensors", {})
            row = inspection.inspect_one_file(source)
            row.update(
                plan_schema_name=PLAN_SCHEMA_NAME,
                plan_schema_version="1.0",
                plan_tool_version="0.1",
                plan_id="old-plan",
                action="move",
                move="True",
            )
            path = root / "old_plan.csv"
            write_csv(path, [row], schema.PLAN_FIELDS)
            with self.assertRaisesRegex(ValueError, "unsupported plan schema"):
                planner.build_sort_plan(
                    path,
                    root / "sorted",
                    allow_legacy_plan=True,
                )

    def test_blank_schema_values_in_versioned_index_are_not_treated_as_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(root / "model.safetensors", {})
            row = inspection.inspect_one_file(source)
            row["schema_name"] = ""
            row["schema_version"] = ""
            path = root / "corrupt.csv"
            write_csv(path, [row], schema.INSPECTION_FIELDS)
            with self.assertRaisesRegex(ValueError, "row 2: row is unversioned"):
                planner.build_sort_plan(
                    path,
                    root / "sorted",
                    allow_legacy_plan=True,
                )


    def test_move_false_in_inspection_index_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(
                root / "model.safetensors",
                {
                    "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                    "attn2_to_k.lora_down.weight": tensor([8, 2048])
                },
            )
            row = inspection.inspect_one_file(source)
            row["move"] = "False"
            index = root / "index.csv"
            write_csv(index, [row], schema.INSPECTION_FIELDS)
            plan = planner.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_user_false")
            self.assertEqual(plan[0]["move"], "False")

    def test_move_false_edit_in_reviewed_plan_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(
                root / "model.safetensors",
                {
                    "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                    "attn2_to_k.lora_down.weight": tensor([8, 2048])
                },
            )
            row = inspection.inspect_one_file(source)
            index = root / "index.csv"
            write_csv(index, [row], schema.INSPECTION_FIELDS)
            first = planner.build_sort_plan(index, root / "sorted")
            self.assertEqual(first[0]["action"], "move")
            first[0]["move"] = "False"
            reviewed = root / "index_sort_plan.csv"
            write_csv(reviewed, first, schema.PLAN_FIELDS)
            second = planner.build_sort_plan(reviewed, root / "sorted")
            self.assertEqual(second[0]["action"], "skip_user_false")
            self.assertEqual(second[0]["move"], "False")

    def test_reviewed_plan_rejects_mixed_plan_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for name in ("a.safetensors", "b.safetensors"):
                source = write_fake_safetensors(
                    root / name,
                    {
                        "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                        "attn2_to_k.lora_down.weight": tensor([8, 2048])
                    },
                )
                rows.append(inspection.inspect_one_file(source))
            index = root / "index.csv"
            write_csv(index, rows, schema.INSPECTION_FIELDS)
            plan = planner.build_sort_plan(index, root / "sorted")
            plan[1]["plan_id"] = "another-plan"
            reviewed = root / "mixed.csv"
            write_csv(reviewed, plan, schema.PLAN_FIELDS)
            with self.assertRaisesRegex(ValueError, "inconsistent plan_id"):
                planner.build_sort_plan(reviewed, root / "sorted")

    def test_reviewed_plan_rejects_different_base(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(
                root / "model.safetensors",
                {
                    "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                    "attn2_to_k.lora_down.weight": tensor([8, 2048])
                },
            )
            row = inspection.inspect_one_file(source)
            index = root / "index.csv"
            write_csv(index, [row], schema.INSPECTION_FIELDS)
            plan = planner.build_sort_plan(index, root / "sorted-a")
            reviewed = root / "reviewed.csv"
            write_csv(reviewed, plan, schema.PLAN_FIELDS)
            with self.assertRaisesRegex(ValueError, "different resolved base"):
                planner.build_sort_plan(reviewed, root / "sorted-b")

    @unittest.skipIf(os.name == "nt", "Windows st_ctime is not a metadata-change clock")
    def test_ctime_rejects_same_size_payload_rewrite_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = write_fake_safetensors(
                root / "model.safetensors",
                {
                    "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                    "attn2_to_k.lora_down.weight": tensor([8, 2048])
                },
                payload=b"payload-one",
            )
            record = inspection.inspect_one_file(source)
            before = source.stat()
            time.sleep(0.002)
            with source.open("r+b") as handle:
                handle.seek(-len(b"payload-one"), os.SEEK_END)
                handle.write(b"payload-two")
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            after = source.stat()
            if after.st_ctime_ns == int(record["ctime_ns"]):
                self.skipTest("filesystem did not expose a distinct ctime")

            index = root / "index.csv"
            write_csv(index, [record], schema.INSPECTION_FIELDS)
            plan = planner.build_sort_plan(index, root / "sorted")
            self.assertEqual(plan[0]["action"], "skip_stale_or_unverified")
            self.assertIn("ctime", plan[0]["reason"])

    def test_current_inspection_record_carries_schema_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            source = write_fake_safetensors(Path(temp) / "model.safetensors", {})
            row = inspection.inspect_one_file(source)
            self.assertEqual(row["schema_name"], INSPECTION_SCHEMA_NAME)
            self.assertEqual(row["schema_version"], INSPECTION_SCHEMA_VERSION)
            self.assertTrue(row["ctime_ns"])


if __name__ == "__main__":
    unittest.main()


class FolderSafeVersionTests(unittest.TestCase):
    """Version-shaped metadata noise must not mint one bucket per float."""

    ANIMA_KEY = {
        "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight": tensor([8, 2048])
    }

    def test_folderable_version_gate(self):
        from lora_header_inspector.normalization import folderable_version

        for raw, expected in [
            ("1.0", "1.0"), ("v6", "6"), ("2.2", "2.2"), ("0.1", "0.1"),
            ("10.000002", ""), ("1.1024.8", ""), ("3.000009", ""),
            ("1.2.3.4", ""), ("", ""), ("v", ""), ("banana", ""),
        ]:
            self.assertEqual(folderable_version(raw), expected, raw)

    def test_float_noise_version_keeps_field_but_not_folder(self):
        result = classify(self.ANIMA_KEY, {"ss_base_model": "anima_base_v10.000002"}, "x.safetensors")
        self.assertEqual(result["model_version"], "10.000002")
        self.assertEqual(result["sort_subfolder"], "Base")
        self.assertTrue(
            any("folder-safe" in w for w in json.loads(result["warnings_json"])),
            "dropped version should be surfaced as a warning",
        )

    def test_sane_version_still_mints_a_qualifier(self):
        result = classify(self.ANIMA_KEY, {"ss_base_model": "anima_base_v1.0"}, "x.safetensors")
        self.assertEqual(result["sort_subfolder"], "Base__v1.0")
        self.assertFalse(any("folder-safe" in w for w in json.loads(result["warnings_json"])))
