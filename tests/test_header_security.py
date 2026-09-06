from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from lora_header_inspector.constants import DEFAULT_MAX_HEADER_BYTES
from lora_header_inspector.header import DuplicateJSONKeyError, read_header, read_header_hash

from tests.helpers import tensor, write_fake_safetensors, write_raw_header


class HeaderSecurityTests(unittest.TestCase):
    def test_short_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "short.safetensors"
            path.write_bytes(b"123")
            with self.assertRaisesRegex(ValueError, "too short"):
                read_header(path)

    def test_implausibly_large_header_is_rejected_before_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "huge.safetensors"
            path.write_bytes(struct.pack("<Q", 2**40))
            with self.assertRaisesRegex(ValueError, "safety limit"):
                read_header(path)

    def test_declared_header_beyond_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "truncated.safetensors"
            path.write_bytes(struct.pack("<Q", 100) + b"{}")
            with self.assertRaisesRegex(ValueError, "extends beyond"):
                read_header(path)

    def test_duplicate_json_tensor_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_raw_header(
                Path(temp) / "duplicate.safetensors",
                b'{"x":{"dtype":"F16","shape":[],"data_offsets":[0,0]},'
                b'"x":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}',
            )
            with self.assertRaises(DuplicateJSONKeyError):
                read_header(path)

    def test_non_object_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_raw_header(Path(temp) / "root.safetensors", b"[]")
            with self.assertRaisesRegex(ValueError, "does not begin with '{'|root is not"):
                read_header(path)

    def test_tensor_offset_beyond_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "offset.safetensors",
                {"x": tensor([1], offsets=(0, 10))},
                payload=b"12",
            )
            with self.assertRaisesRegex(ValueError, "beyond the file payload"):
                read_header(path)

    def test_boolean_shape_dimension_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "shape.safetensors",
                {"x": {"dtype": "F16", "shape": [True], "data_offsets": [0, 0]}},
            )
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                read_header(path)

    def test_non_string_metadata_is_normalized_with_warning(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "meta.safetensors",
                {},
                metadata={"integer": 7, "nested": {"a": 1}, "none": None},
            )
            header = read_header(path)
            self.assertEqual(header.metadata["integer"], "7")
            self.assertEqual(json.loads(header.metadata["nested"]), {"a": 1})
            self.assertEqual(header.metadata["none"], "")
            self.assertGreaterEqual(len(header.warnings), 3)

    def test_non_object_metadata_is_ignored_with_warning(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_raw_header(
                Path(temp) / "meta-list.safetensors",
                b'{"__metadata__":[],"x":{"dtype":"F16","shape":[],"data_offsets":[0,0]}}',
            )
            header = read_header(path)
            self.assertEqual(header.metadata, {})
            self.assertIn("not an object", header.warnings[0])

    def test_header_hash_reader_uses_same_length_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", 100) + b"{}")
            with self.assertRaisesRegex(ValueError, "extends beyond"):
                read_header_hash(path)


if __name__ == "__main__":
    unittest.main()


class HeaderParsingIncrementTests(unittest.TestCase):
    """Checks added after the 0.6.1 parsing review: first byte, padding
    discipline, the official header cap, dtype allowlisting, and the opt-in
    strict layout validation."""

    def test_leading_whitespace_before_object_is_rejected(self):
        # json.loads would accept this; the spec requires byte zero to be '{'.
        with tempfile.TemporaryDirectory() as temp:
            path = write_raw_header(Path(temp) / "pad.safetensors", b" {}")
            with self.assertRaisesRegex(ValueError, "does not begin with"):
                read_header(path)

    def test_trailing_space_padding_is_accepted_and_hashed(self):
        with tempfile.TemporaryDirectory() as temp:
            padded = write_raw_header(Path(temp) / "padded.safetensors", b"{}    ")
            bare = write_raw_header(Path(temp) / "bare.safetensors", b"{}")
            first = read_header(padded)
            second = read_header(bare)
            self.assertEqual(first.tensors, {})
            # Raw bytes are hashed, so alignment padding changes the digest.
            self.assertNotEqual(first.header_sha256, second.header_sha256)

    def test_non_space_trailing_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_raw_header(Path(temp) / "tail.safetensors", b"{}\n\n")
            with self.assertRaisesRegex(ValueError, "space padding"):
                read_header(path)

    def test_default_cap_matches_official_limit(self):
        self.assertEqual(DEFAULT_MAX_HEADER_BYTES, 100 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "huge.safetensors"
            declared = DEFAULT_MAX_HEADER_BYTES + 1
            path.write_bytes(struct.pack("<Q", declared) + b"{}")
            with self.assertRaisesRegex(ValueError, "exceeds safety limit"):
                read_header(path)

    def test_unrecognised_dtype_warns_but_still_parses(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "dtype.safetensors",
                {"a.lora_down.weight": {"dtype": "fp16", "shape": [1], "data_offsets": [0, 2]}},
                payload=b"\x00\x00",
            )
            header = read_header(path)
            self.assertIn("a.lora_down.weight", header.tensors)
            self.assertTrue(any("unrecognised dtype" in w for w in header.warnings))

    def test_strict_layout_rejects_unrecognised_dtype(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "dtype.safetensors",
                {"a.lora_down.weight": {"dtype": "fp16", "shape": [1], "data_offsets": [0, 2]}},
                payload=b"\x00\x00",
            )
            with self.assertRaisesRegex(ValueError, "recognised dtype"):
                read_header(path, strict_layout=True)

    def test_strict_layout_accepts_exact_tiling_with_empty_tensor(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "tiled.safetensors",
                {
                    "b": {"dtype": "F16", "shape": [2], "data_offsets": [4, 8]},
                    "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                    "empty": {"dtype": "F16", "shape": [0], "data_offsets": [8, 8]},
                },
                payload=b"\x00" * 8,
            )
            header = read_header(path, strict_layout=True)
            self.assertEqual(len(header.tensors), 3)

    def test_strict_layout_rejects_payload_hole(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "hole.safetensors",
                {
                    "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                    "b": {"dtype": "F16", "shape": [2], "data_offsets": [8, 12]},
                },
                payload=b"\x00" * 12,
            )
            with self.assertRaisesRegex(ValueError, "hole"):
                read_header(path, strict_layout=True)

    def test_strict_layout_rejects_overlapping_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "overlap.safetensors",
                {
                    "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                    "b": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
                },
                payload=b"\x00" * 6,
            )
            with self.assertRaisesRegex(ValueError, "overlaps"):
                read_header(path, strict_layout=True)

    def test_strict_layout_rejects_shape_dtype_size_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "lie.safetensors",
                {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
                payload=b"\x00" * 4,
            )
            with self.assertRaisesRegex(ValueError, "shape and dtype require"):
                read_header(path, strict_layout=True)

    def test_strict_layout_rejects_uncovered_payload_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "tail.safetensors",
                {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
                payload=b"\x00" * 8,
            )
            with self.assertRaisesRegex(ValueError, "cover 4 bytes of a 8 byte payload"):
                read_header(path, strict_layout=True)

    def test_lenient_default_still_accepts_holes(self):
        # Classification must not start refusing sloppy but loadable files.
        with tempfile.TemporaryDirectory() as temp:
            path = write_fake_safetensors(
                Path(temp) / "hole.safetensors",
                {
                    "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                    "b": {"dtype": "F16", "shape": [2], "data_offsets": [8, 12]},
                },
                payload=b"\x00" * 12,
            )
            header = read_header(path)
            self.assertEqual(len(header.tensors), 2)

    def test_inspection_cli_exposes_the_new_flags(self):
        from lora_header_inspector.inspection import build_parser

        args = build_parser().parse_args(
            ["x.safetensors", "--strict-layout", "--max-header-bytes", "1048576"]
        )
        self.assertTrue(args.strict_layout)
        self.assertEqual(args.max_header_bytes, 1048576)
