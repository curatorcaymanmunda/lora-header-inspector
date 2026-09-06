from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from lora_header_inspector import inspection, smoke
from lora_header_inspector.version import (
    INSPECTION_SCHEMA_NAME,
    INSPECTION_SCHEMA_VERSION,
    SIGNATURE_REPORT_SCHEMA_NAME,
)

from tests.helpers import tensor, write_fake_safetensors


class InspectionCliTests(unittest.TestCase):
    def test_cli_writes_versioned_index_without_sort_when_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = write_fake_safetensors(
                root / "model.safetensors",
                {
                    "lora_unet_input_blocks_4_1_transformer_blocks_0_"
                    "attn2_to_k.lora_down.weight": tensor([8, 2048])
                },
            )
            output = root / "index.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = inspection.main(
                    [str(model), "--output", str(output), "--no-sort-plan"]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(output.exists())
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["schema_name"], INSPECTION_SCHEMA_NAME)
            self.assertEqual(rows[0]["schema_version"], INSPECTION_SCHEMA_VERSION)
            self.assertEqual(rows[0]["arch"], "SDXL")
            self.assertFalse((root / "index_sort_plan.csv").exists())

    def test_hash_content_is_optional(self):
        with tempfile.TemporaryDirectory() as temp:
            model = write_fake_safetensors(Path(temp) / "model.safetensors", {}, payload=b"payload")
            quick = inspection.inspect_one_file(model)
            hashed = inspection.inspect_one_file(model, include_full_hash=True)
            self.assertEqual(quick["full_sha256"], "")
            self.assertEqual(len(hashed["full_sha256"]), 64)


class SignatureReportTests(unittest.TestCase):
    def test_report_contains_no_absolute_paths_offsets_or_url_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = write_fake_safetensors(
                root / "private-name.safetensors",
                {"transformer.layers.0.attention.to_q.lora_A.weight": tensor([8, 3840])},
                metadata={
                    "base_model_name_or_path": "/home/alice/private/Z-Image.safetensors",
                    "pretrained_model_name_or_path": "https://alice:secret@example.org/org/model?token=very-secret#frag",
                },
            )
            report = root / "signatures.json"
            inspection.write_signature_report([model], report)
            text = report.read_text(encoding="utf-8")
            payload = json.loads(text)
            self.assertEqual(payload["schema_name"], SIGNATURE_REPORT_SCHEMA_NAME)
            self.assertNotIn(str(root), text)
            self.assertNotIn("/home/alice/private", text)
            self.assertNotIn("alice:secret", text)
            self.assertNotIn("very-secret", text)
            self.assertNotIn("data_offsets", text)
            record = payload["records"][0]
            self.assertEqual(record["filename"], "")
            self.assertEqual(
                record["identity_metadata"]["base_model_name_or_path"],
                "local-path:Z-Image.safetensors",
            )
            self.assertEqual(
                record["identity_metadata"]["pretrained_model_name_or_path"],
                "https://example.org/org/model",
            )
            self.assertIn("base_model_name_or_path", record["metadata_redactions"])

    def test_filename_inclusion_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            model = write_fake_safetensors(Path(temp) / "named.safetensors", {})
            hidden = inspection.anonymized_signature_record(model)
            visible = inspection.anonymized_signature_record(model, include_filename=True)
            self.assertEqual(hidden["filename"], "")
            self.assertEqual(visible["filename"], "named.safetensors")


class SmokeTests(unittest.TestCase):
    def test_smoke_mode_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = write_fake_safetensors(root / "model.safetensors", {})
            before = sorted(path.name for path in root.iterdir())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = smoke.run([str(model)])
            after = sorted(path.name for path in root.iterdir())
            self.assertEqual(rc, 0)
            self.assertEqual(before, after)
            self.assertIn("No CSV was written and no files were moved", output.getvalue())


class SortCliSettingTests(unittest.TestCase):
    def test_reviewed_plan_settings_are_inferred_when_not_overridden(self):
        from unittest import mock
        from lora_header_inspector import cli_sort, planner, schema
        from tests.helpers import write_csv

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_path = root / "reviewed.csv"
            row = {field: "" for field in schema.PLAN_FIELDS}
            row.update(
                schema_name="lora-inspection",
                schema_version="2.0",
                path=str(root / "model.safetensors"),
                arch="Wan",
                confidence="medium",
                status="ambiguous",
                file_size="1",
                mtime_ns="1",
                header_sha256="0" * 64,
                plan_schema_name="lora-sort-plan",
                plan_schema_version="2.0",
                plan_id="plan-1",
                plan_base_dir=str(root / "library"),
                plan_base_resolved=str((root / "library").resolve()),
                use_lineage="False",
                min_confidence="medium",
                allow_ambiguous="True",
                allow_outside_base="False",
                allow_unverified="False",
                allow_symlink_parents="True",
                max_path_length="321",
                duplicate_policy="hardlink",
                quarantine_dir="_ReviewDuplicates",
                action="move",
                move="True",
            )
            write_csv(plan_path, [row], schema.PLAN_FIELDS)
            with mock.patch.object(cli_sort, "run_plan", return_value=0) as called:
                rc = cli_sort.main([str(plan_path)])
            self.assertEqual(rc, 0)
            kwargs = called.call_args.kwargs
            self.assertEqual(kwargs["base_dir"], root / "library")
            self.assertFalse(kwargs["use_lineage"])
            self.assertEqual(kwargs["min_confidence"], "medium")
            self.assertTrue(kwargs["allow_ambiguous"])
            self.assertTrue(kwargs["allow_symlink_parents"])
            self.assertEqual(kwargs["max_path_length"], 321)
            self.assertEqual(kwargs["duplicate_policy"], "hardlink")
            self.assertEqual(kwargs["quarantine_dir"], "_ReviewDuplicates")


    def test_apply_requires_relaxed_plan_flags_to_be_reacknowledged(self):
        from unittest import mock
        from lora_header_inspector import cli_sort, schema
        from tests.helpers import write_csv

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "library"
            plan_path = root / "reviewed.csv"
            row = {field: "" for field in schema.PLAN_FIELDS}
            row.update(
                schema_name="lora-inspection", schema_version="2.0",
                path=str(root / "model.safetensors"), arch="Wan",
                confidence="medium", status="ambiguous", file_size="1",
                mtime_ns="1", header_sha256="0" * 64,
                plan_schema_name="lora-sort-plan", plan_schema_version="2.0",
                plan_id="plan-relaxed", plan_base_dir=str(library),
                plan_base_resolved=str(library.resolve()), use_lineage="True",
                min_confidence="medium", allow_ambiguous="True",
                allow_outside_base="False", allow_unverified="False",
                allow_symlink_parents="False", max_path_length="0",
                duplicate_policy="report", quarantine_dir="_Duplicates",
                action="move", move="True",
            )
            write_csv(plan_path, [row], schema.PLAN_FIELDS)
            output = io.StringIO()
            with mock.patch.object(cli_sort, "run_plan", return_value=0) as called:
                with contextlib.redirect_stdout(output):
                    rc = cli_sort.main([str(plan_path), "--base", str(library), "--apply"])
            self.assertEqual(rc, 2)
            called.assert_not_called()
            self.assertIn("repeat --allow-ambiguous explicitly", output.getvalue())

            with mock.patch.object(cli_sort, "run_plan", return_value=0) as called:
                rc = cli_sort.main([
                    str(plan_path), "--base", str(library), "--apply",
                    "--allow-ambiguous",
                ])
            self.assertEqual(rc, 0)
            self.assertTrue(called.call_args.kwargs["allow_ambiguous"])


class CompatibilityTestRunnerTests(unittest.TestCase):
    def test_root_runner_help_does_not_execute_suite(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parents[1] / "test_lora_tools.py"), "--help"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Modes:", result.stdout)
        self.assertNotIn("Ran ", result.stderr + result.stdout)

    def test_root_runner_accepts_legacy_short_selector(self):
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parents[1] / "test_lora_tools.py"),
                "ClassifierTests.test_sd15_by_cross_attention_width",
                "--quiet",
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("OK", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()


class EnumerateOnlyTests(unittest.TestCase):
    def _library(self, root: Path) -> None:
        from tests.helpers import write_fake_safetensors
        tensor = {"a.lora_down.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
        write_fake_safetensors(root / "SDXL" / "one.safetensors", tensor, payload=b"\x00\x00")
        write_fake_safetensors(root / "SDXL" / "characters" / "two.SAFETENSORS", tensor, payload=b"\x00\x00")
        write_fake_safetensors(root / "Flux2" / "three.safetensors", tensor, payload=b"\x00\x00")
        write_fake_safetensors(root / "loose.safetensors", tensor, payload=b"\x00\x00")
        (root / "SDXL" / "notes.txt").write_text("not a model", encoding="utf-8")

    def test_counts_per_bucket_without_reading_headers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "library"
            self._library(root)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), mock.patch.object(
                inspection, "read_header", side_effect=AssertionError("header was read")
            ):
                rc = inspection.main([str(root), "--enumerate-only"])
            self.assertEqual(rc, 0)
            output = buffer.getvalue()
            self.assertIn("2  %s" % (root / "SDXL"), output)
            self.assertIn("1  %s" % (root / "Flux2"), output)
            self.assertIn("1  %s" % root, output)
            self.assertIn("4  total", output)
            self.assertFalse(list(Path(temp).rglob("*.csv")), "enumeration wrote a CSV")

    def test_deduplicates_overlapping_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "library"
            self._library(root)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                rc = inspection.main(
                    [str(root), str(root / "SDXL" / "one.safetensors"), "--enumerate-only"]
                )
            self.assertEqual(rc, 0)
            self.assertIn("4  total", buffer.getvalue())


class ParallelInspectionTests(unittest.TestCase):
    def test_parallel_rows_are_identical_to_serial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {"a.lora_down.weight": {"dtype": "F16", "shape": [8, 2048], "data_offsets": [0, 32768]}}
            files = []
            for index in range(12):
                files.append(
                    write_fake_safetensors(
                        root / ("bucket_%d" % (index % 3)) / ("model_%02d.safetensors" % index),
                        spec,
                        payload=bytes([index % 251]) * 32768,
                    )
                )
            serial = inspection.inspect_files(files, include_full_hash=True, jobs=1)
            parallel = inspection.inspect_files(files, include_full_hash=True, jobs=4)
            self.assertEqual(serial, parallel)
            self.assertEqual(
                [row["path"] for row in parallel], [str(path) for path in files],
                "parallel inspection reordered rows",
            )

    def test_jobs_flag_is_exposed_and_defaults_to_serial(self):
        args = inspection.build_parser().parse_args(["x.safetensors"])
        self.assertEqual(args.jobs, 1)
        args = inspection.build_parser().parse_args(["x.safetensors", "--jobs", "8"])
        self.assertEqual(args.jobs, 8)
