from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from lora_header_inspector import fixtures


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "signatures"


class SignatureFixtureTests(unittest.TestCase):
    def test_bundled_signature_fixtures_match(self):
        paths = fixtures.collect_fixture_paths([str(FIXTURE_DIR)])
        results = fixtures.validate_fixture_paths(paths)
        self.assertEqual(len(results), 13)
        self.assertTrue(all(result.valid for result in results))
        self.assertTrue(all(result.matched for result in results))


    def test_fixture_file_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate-json.json"
            path.write_text(
                '{"schema_name":"lora-signature-fixture",'
                '"schema_name":"other","schema_version":"1.0"}',
                encoding="utf-8",
            )
            result = fixtures.validate_fixture(path)
        self.assertFalse(result.valid)
        self.assertFalse(result.matched)
        self.assertIn("duplicate JSON key", result.errors[0])

    def test_fixture_rejects_payload_offsets(self):
        payload = {
            "schema_name": "lora-signature-fixture",
            "schema_version": "1.0",
            "fixture_id": "unsafe-offsets",
            "provenance": {"source_type": "synthetic"},
            "metadata": {},
            "tensors": [
                {
                    "key": "transformer.block.weight",
                    "dtype": "F16",
                    "shape": [8, 8],
                    "data_offsets": [0, 128],
                }
            ],
            "expected": {"arch": "Unknown"},
        }
        with self.assertRaisesRegex(fixtures.FixtureError, "payload/offset"):
            fixtures.validate_fixture_payload(payload)

    def test_fixture_rejects_filename_path(self):
        payload = {
            "schema_name": "lora-signature-fixture",
            "schema_version": "1.0",
            "fixture_id": "unsafe-name",
            "provenance": {"source_type": "synthetic"},
            "filename": "/home/alice/private/model.safetensors",
            "metadata": {},
            "tensors": [],
            "expected": {"arch": "Unknown"},
        }
        with self.assertRaisesRegex(fixtures.FixtureError, "basename"):
            fixtures.validate_fixture_payload(payload)

    def test_fixture_rejects_duplicate_tensor_key(self):
        tensor = {"key": "same.weight", "dtype": "F16", "shape": [8, 8]}
        payload = {
            "schema_name": "lora-signature-fixture",
            "schema_version": "1.0",
            "fixture_id": "duplicate-key",
            "provenance": {"source_type": "synthetic"},
            "metadata": {},
            "tensors": [tensor, tensor],
            "expected": {"arch": "Unknown"},
        }
        with self.assertRaisesRegex(fixtures.FixtureError, "duplicate tensor key"):
            fixtures.validate_fixture_payload(payload)

    def test_expected_mismatch_is_valid_but_not_matched(self):
        source = FIXTURE_DIR / "sdxl-kohya.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["expected"]["arch"] = "SD15"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mismatch.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = fixtures.validate_fixture(path)
        self.assertTrue(result.valid)
        self.assertFalse(result.matched)
        self.assertIn("expected 'SD15'", result.errors[0])

    def test_fixture_cli_writes_validation_report(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "report.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = fixtures.main([str(FIXTURE_DIR), "--output", str(report)])
            self.assertEqual(rc, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["fixture_count"], 13)
            self.assertEqual(payload["failed"], 0)
            self.assertIn("passed=13", output.getvalue())


if __name__ == "__main__":
    unittest.main()
