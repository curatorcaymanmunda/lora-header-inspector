from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lora_header_inspector import inspection, planner, schema, transactions
from lora_header_inspector.fingerprints import file_identity, full_file_sha256, same_inode
from lora_header_inspector.version import (
    MANIFEST_SCHEMA_NAME,
    MANIFEST_SCHEMA_VERSION,
    UNDO_SCHEMA_NAME,
    UNDO_SCHEMA_VERSION,
)

from tests.helpers import tensor, write_csv, write_fake_safetensors


def sdxl_model(path: Path, payload: bytes = b"") -> Path:
    return write_fake_safetensors(
        path,
        {
            "lora_unet_input_blocks_4_1_transformer_blocks_0_"
            "attn2_to_k.lora_down.weight": tensor([8, 2048])
        },
        payload=payload,
    )


def write_index(path: Path, models) -> None:
    rows = [inspection.inspect_one_file(model) for model in models]
    write_csv(path, rows, schema.INSPECTION_FIELDS)


def write_apply_manifest(path: Path, operation) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_name": MANIFEST_SCHEMA_NAME,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "tool_version": "0.6.0",
                "manifest_type": "apply",
                "status": "attention_required",
                "operations": [operation],
            }
        ),
        encoding="utf-8",
    )


def write_undo_manifest(path: Path, operation) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_name": UNDO_SCHEMA_NAME,
                "schema_version": UNDO_SCHEMA_VERSION,
                "tool_version": "0.6.0",
                "manifest_type": "undo",
                "status": "attention_required",
                "operations": [operation],
            }
        ),
        encoding="utf-8",
    )


class PublicationJournalTests(unittest.TestCase):
    def test_exclusive_json_creation_never_replaces_an_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "journal.json"
            path.write_text("race winner", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                transactions.write_json_exclusive({"status": "new"}, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "race winner")

    def test_apply_manifest_creation_race_is_reported_before_any_move(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sdxl_model(root / "model.safetensors")
            index = root / "index.csv"
            write_index(index, [source])
            base = root / "sorted"
            plan = planner.build_sort_plan(index, base)
            manifest = root / "apply.json"
            real_create = transactions.write_json_exclusive

            def race(payload, path):
                Path(path).write_text("race winner", encoding="utf-8")
                return real_create(payload, path)

            output = io.StringIO()
            with mock.patch.object(transactions, "write_json_exclusive", side_effect=race):
                with contextlib.redirect_stdout(output):
                    counts = transactions.apply_plan(
                        plan, index, base, dry_run=False, manifest_path=manifest
                    )
            self.assertEqual(counts["errors"], 1)
            self.assertEqual(counts["moved"], 0)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "race winner")
            self.assertTrue(source.exists())
            self.assertFalse((base / "SDXL" / source.name).exists())
            self.assertIn("could not create apply manifest", output.getvalue())

    def test_undo_manifest_creation_race_leaves_applied_state_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sdxl_model(root / "model.safetensors")
            index = root / "index.csv"
            write_index(index, [source])
            base = root / "sorted"
            plan = planner.build_sort_plan(index, base)
            apply_manifest = root / "apply.json"
            with contextlib.redirect_stdout(io.StringIO()):
                counts = transactions.apply_plan(
                    plan, index, base, dry_run=False, manifest_path=apply_manifest
                )
            self.assertEqual(counts["moved"], 1)
            destination = base / "SDXL" / source.name
            undo_manifest = root / "undo.json"
            real_create = transactions.write_json_exclusive

            def race(payload, path):
                Path(path).write_text("race winner", encoding="utf-8")
                return real_create(payload, path)

            output = io.StringIO()
            with mock.patch.object(transactions, "write_json_exclusive", side_effect=race):
                with contextlib.redirect_stdout(output):
                    rc = transactions.undo_manifest(
                        apply_manifest, apply_changes=True, output_manifest=undo_manifest
                    )
            self.assertEqual(rc, 1)
            self.assertEqual(undo_manifest.read_text(encoding="utf-8"), "race winner")
            self.assertFalse(source.exists())
            self.assertTrue(destination.exists())
            self.assertIn("could not create undo manifest", output.getvalue())

    def test_recovery_manifest_creation_race_makes_no_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            source.write_bytes(b"payload")
            interrupted = root / "interrupted.json"
            write_apply_manifest(
                interrupted,
                {
                    "operation_id": "one",
                    "record_path": str(source),
                    "type": "move",
                    "status": "pending",
                    "steps": [],
                },
            )
            recovery_manifest = root / "recovery.json"
            real_create = transactions.write_json_exclusive

            def race(payload, path):
                Path(path).write_text("race winner", encoding="utf-8")
                return real_create(payload, path)

            output = io.StringIO()
            with mock.patch.object(transactions, "write_json_exclusive", side_effect=race):
                with contextlib.redirect_stdout(output):
                    rc = transactions.recover_manifest(
                        interrupted, apply_changes=True, output_manifest=recovery_manifest
                    )
            self.assertEqual(rc, 1)
            self.assertEqual(recovery_manifest.read_text(encoding="utf-8"), "race winner")
            self.assertEqual(source.read_bytes(), b"payload")
            self.assertIn("could not create recovery manifest", output.getvalue())

    def test_exclusive_destination_is_journalled_before_fallback_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"payload" * 100)
            identities = []

            def no_links(*_args, **_kwargs):
                raise OSError(errno.EXDEV, "synthetic no-hardlink filesystem")

            def journal_and_fail(identity):
                identities.append(dict(identity))
                raise RuntimeError("synthetic journal failure")

            with mock.patch.object(transactions.os, "link", side_effect=no_links):
                with self.assertRaisesRegex(RuntimeError, "journal failure"):
                    transactions.safe_move(
                        source,
                        destination,
                        on_destination_created=journal_and_fail,
                    )
            self.assertEqual(len(identities), 1)
            self.assertIn("device", identities[0])
            self.assertIn("inode", identities[0])
            self.assertTrue(source.exists())
            self.assertFalse(destination.exists())

    def test_recovery_removes_owned_partial_fallback_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"complete payload")
            destination.write_bytes(b"partial")
            manifest = root / "apply.json"
            write_apply_manifest(
                manifest,
                {
                    "operation_id": "one",
                    "record_path": str(source),
                    "type": "move",
                    "status": "pending",
                    "steps": [
                        {
                            "type": "move",
                            "from": str(source),
                            "to": str(destination),
                            "status": "publishing",
                            "sha256": full_file_sha256(source),
                            "destination_identity": file_identity(destination).to_dict(),
                            "staging_path": "",
                        }
                    ],
                },
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rc = transactions.recover_manifest(manifest, apply_changes=True)
            self.assertEqual(rc, 0)
            self.assertTrue(source.exists())
            self.assertFalse(destination.exists())

    @unittest.skipIf(not hasattr(os, "link"), "hard links unavailable")
    def test_relink_race_preserves_intruder_and_private_backup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "canonical.bin"
            duplicate = root / "duplicate.bin"
            canonical.write_bytes(b"original")
            duplicate.write_bytes(b"original")
            expected = file_identity(duplicate).to_dict()
            real_atomic_hardlink = transactions.atomic_hardlink

            def race(source, target):
                Path(target).write_bytes(b"intruder")
                raise FileExistsError("synthetic relink race")

            with mock.patch.object(transactions, "atomic_hardlink", side_effect=race):
                with self.assertRaisesRegex(RuntimeError, "rollback errors"):
                    transactions.atomic_relink(
                        duplicate,
                        canonical,
                        expected_identity=expected,
                        expected_sha256=full_file_sha256(canonical),
                    )
            self.assertEqual(duplicate.read_bytes(), b"intruder")
            backups = list(root.glob(".duplicate.bin.relink-backup.*.tmp"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"original")
            # Keep the local variable live so static tools do not mistake the
            # intentionally patched callable for an unused test artefact.
            self.assertTrue(callable(real_atomic_hardlink))


class UndoJournalTests(unittest.TestCase):
    def test_partial_move_undo_is_rolled_back_to_applied_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sdxl_model(root / "model.safetensors")
            sidecar = root / "model.preview.png"
            sidecar.write_bytes(b"preview")
            index = root / "index.csv"
            write_index(index, [source])
            base = root / "sorted"
            plan = planner.build_sort_plan(index, base)
            apply_manifest = root / "apply.json"
            with contextlib.redirect_stdout(io.StringIO()):
                counts = transactions.apply_plan(
                    plan,
                    index,
                    base,
                    dry_run=False,
                    manifest_path=apply_manifest,
                )
            self.assertEqual(counts["moved"], 1)
            destination = base / "SDXL" / "model.safetensors"
            destination_sidecar = base / "SDXL" / "model.preview.png"

            real_safe_move = transactions.safe_move
            calls = {"count": 0}

            def fail_second(*args, **kwargs):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("synthetic second undo move failure")
                return real_safe_move(*args, **kwargs)

            undo_manifest = root / "undo.json"
            with mock.patch.object(transactions, "safe_move", side_effect=fail_second):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = transactions.undo_manifest(
                        apply_manifest,
                        apply_changes=True,
                        output_manifest=undo_manifest,
                    )
            self.assertEqual(rc, 1)
            self.assertFalse(source.exists())
            self.assertFalse(sidecar.exists())
            self.assertTrue(destination.exists())
            self.assertTrue(destination_sidecar.exists())
            journal = json.loads(undo_manifest.read_text(encoding="utf-8"))
            self.assertEqual(journal["status"], "completed_with_rollbacks")
            self.assertEqual(journal["operations"][0]["status"], "rolled_back")

            with contextlib.redirect_stdout(io.StringIO()):
                rc = transactions.undo_manifest(apply_manifest, apply_changes=True)
            self.assertEqual(rc, 0)
            self.assertTrue(source.exists())
            self.assertTrue(sidecar.exists())

    def test_undo_output_manifest_cannot_overlap_an_operation_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sdxl_model(root / "model.safetensors")
            index = root / "index.csv"
            write_index(index, [source])
            base = root / "sorted"
            plan = planner.build_sort_plan(index, base)
            apply_manifest = root / "apply.json"
            with contextlib.redirect_stdout(io.StringIO()):
                transactions.apply_plan(
                    plan,
                    index,
                    base,
                    dry_run=False,
                    manifest_path=apply_manifest,
                )
            destination = base / "SDXL" / "model.safetensors"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = transactions.undo_manifest(
                    apply_manifest,
                    apply_changes=True,
                    output_manifest=destination,
                )
            self.assertEqual(rc, 1)
            self.assertIn("overlaps", output.getvalue())
            self.assertTrue(destination.exists())
            self.assertFalse(source.exists())


class UndoRecoveryTests(unittest.TestCase):
    def test_interrupted_undo_move_is_recovered_to_applied_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            applied = root / "sorted.bin"
            original = root / "original.bin"
            original.write_bytes(b"payload")
            digest = full_file_sha256(original)
            # Simulate an undo step that already moved the applied file back to
            # its original name but died before completing the undo journal.
            undo = root / "undo.json"
            write_undo_manifest(
                undo,
                {
                    "operation_id": "undo-one",
                    "record_path": str(original),
                    "type": "undo_move",
                    "status": "pending",
                    "steps": [
                        {
                            "type": "move",
                            "from": str(applied),
                            "to": str(original),
                            "status": "done",
                            "sha256": digest,
                            "destination_identity": file_identity(original).to_dict(),
                            "staging_path": "",
                        }
                    ],
                },
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rc = transactions.recover_manifest(undo, apply_changes=True)
            self.assertEqual(rc, 0)
            self.assertTrue(applied.exists())
            self.assertFalse(original.exists())

    @unittest.skipIf(not hasattr(os, "link"), "hard links unavailable")
    def test_interrupted_hardlink_remove_is_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "canonical.bin"
            target = root / "alias.bin"
            canonical.write_bytes(b"payload")
            undo = root / "undo.json"
            write_undo_manifest(
                undo,
                {
                    "operation_id": "undo-one",
                    "record_path": str(canonical),
                    "type": "undo_hardlink",
                    "status": "pending",
                    "steps": [
                        {
                            "type": "hardlink-remove",
                            "canonical": str(canonical),
                            "target": str(target),
                            "status": "done",
                        }
                    ],
                },
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rc = transactions.recover_manifest(undo, apply_changes=True)
            self.assertEqual(rc, 0)
            self.assertTrue(target.exists())
            self.assertTrue(same_inode(canonical, target))

    @unittest.skipIf(not hasattr(os, "link"), "hard links unavailable")
    def test_interrupted_hardlink_break_is_relinked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "canonical.bin"
            independent = root / "alias.bin"
            canonical.write_bytes(b"payload")
            independent.write_bytes(b"payload")
            digest = full_file_sha256(canonical)
            undo = root / "undo.json"
            write_undo_manifest(
                undo,
                {
                    "operation_id": "undo-one",
                    "record_path": str(canonical),
                    "type": "undo_hardlink",
                    "status": "pending",
                    "steps": [
                        {
                            "type": "hardlink-break",
                            "path": str(independent),
                            "canonical": str(canonical),
                            "status": "independent_published",
                            "sha256": digest,
                            "backup_path": "",
                            "staging_path": "",
                            "replacement_identity": file_identity(independent).to_dict(),
                        }
                    ],
                },
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rc = transactions.recover_manifest(undo, apply_changes=True)
            self.assertEqual(rc, 0)
            self.assertTrue(same_inode(canonical, independent))

    def test_recovery_dry_run_reports_unrecoverable_content_race(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"one")
            destination.write_bytes(b"two")
            manifest = root / "apply.json"
            write_apply_manifest(
                manifest,
                {
                    "operation_id": "one",
                    "record_path": str(source),
                    "type": "move",
                    "status": "pending",
                    "steps": [
                        {
                            "type": "move",
                            "from": str(source),
                            "to": str(destination),
                            "status": "published",
                            "sha256": "",
                            "destination_identity": {},
                            "staging_path": "",
                        }
                    ],
                },
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = transactions.recover_manifest(manifest, apply_changes=False)
            self.assertEqual(rc, 1)
            self.assertIn("blocked=1", output.getvalue())
            self.assertEqual(source.read_bytes(), b"one")
            self.assertEqual(destination.read_bytes(), b"two")


if __name__ == "__main__":
    unittest.main()
