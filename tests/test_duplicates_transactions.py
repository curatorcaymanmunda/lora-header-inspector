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

from lora_header_inspector import duplicates, inspection, planner, schema, transactions
from lora_header_inspector.fingerprints import same_inode
from lora_header_inspector.version import MANIFEST_SCHEMA_NAME, MANIFEST_SCHEMA_VERSION

from tests.helpers import tensor, write_csv, write_fake_safetensors


def sdxl_model(path: Path, payload: bytes = b""):
    return write_fake_safetensors(
        path,
        {
            "lora_unet_input_blocks_4_1_transformer_blocks_0_"
            "attn2_to_k.lora_down.weight": tensor([8, 2048])
        },
        payload=payload,
    )


def index_for(path: Path, models):
    rows = [inspection.inspect_one_file(model) for model in models]
    write_csv(path, rows, schema.INSPECTION_FIELDS)
    return rows


class DuplicatePlanningTests(unittest.TestCase):
    def test_exact_duplicate_group_is_content_aware(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = sdxl_model(root / "a" / "same.safetensors")
            second = sdxl_model(root / "b" / "same.safetensors")
            index = root / "index.csv"
            index_for(index, [first, second])
            plan = planner.build_sort_plan(index, root / "sorted")
            self.assertEqual({row["action"] for row in plan}, {"skip_exact_duplicate_group"})
            self.assertEqual({row["collision_kind"] for row in plan}, {"exact_duplicate"})
            self.assertEqual(len({row["content_sha256"] for row in plan}), 1)
            self.assertEqual(len(plan[0]["content_sha256"]), 64)
            self.assertEqual(plan[0]["distinct_inode_count"], "2")

    def test_same_name_different_bytes_remains_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = sdxl_model(root / "a" / "same.safetensors", payload=b"A")
            second = sdxl_model(root / "b" / "same.safetensors", payload=b"B")
            index = root / "index.csv"
            index_for(index, [first, second])
            plan = planner.build_sort_plan(index, root / "sorted", duplicate_policy="hardlink")
            self.assertEqual({row["action"] for row in plan}, {"skip_content_collision"})
            self.assertEqual({row["collision_kind"] for row in plan}, {"different_content"})
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    @unittest.skipIf(not hasattr(os, "link"), "hard links unavailable")
    def test_same_inode_group_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = sdxl_model(root / "a" / "same.safetensors")
            second = root / "b" / "same.safetensors"
            second.parent.mkdir(parents=True)
            os.link(first, second)
            index = root / "index.csv"
            index_for(index, [first, second])
            plan = planner.build_sort_plan(index, root / "sorted")
            self.assertEqual({row["collision_kind"] for row in plan}, {"same_inode"})
            self.assertEqual({row["action"] for row in plan}, {"skip_same_inode_group"})
            self.assertEqual(plan[0]["distinct_inode_count"], "1")

    def test_reviewed_canonical_choice_is_honoured(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = sdxl_model(root / "longer-folder" / "same.safetensors")
            second = sdxl_model(root / "b" / "same.safetensors")
            rows = [inspection.inspect_one_file(first), inspection.inspect_one_file(second)]
            rows[0]["canonical"] = "True"
            index = root / "index.csv"
            write_csv(index, rows, schema.INSPECTION_FIELDS)
            plan = planner.build_sort_plan(index, root / "sorted", duplicate_policy="hardlink")
            leader = next(row for row in plan if row["action"] == "hardlink_duplicate_group")
            self.assertEqual(Path(leader["canonical_path"]), first)
            self.assertEqual(Path(leader["path"]), first)

    def test_differing_duplicate_sidecars_are_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = sdxl_model(root / "a" / "same.safetensors")
            second = sdxl_model(root / "b" / "same.safetensors")
            (first.parent / "same.preview.png").write_bytes(b"one")
            (second.parent / "same.preview.png").write_bytes(b"two")
            index = root / "index.csv"
            index_for(index, [first, second])
            plan = planner.build_sort_plan(index, root / "sorted", duplicate_policy="hardlink")
            leader = next(row for row in plan if row["action"] == "hardlink_duplicate_group")
            notes = json.loads(leader["sidecar_notes"])
            self.assertTrue(any("different content" in note for note in notes))
            self.assertTrue((first.parent / "same.preview.png").exists())
            self.assertTrue((second.parent / "same.preview.png").exists())


@unittest.skipIf(not hasattr(os, "link"), "hard links unavailable")
class DuplicateApplyTests(unittest.TestCase):
    def test_hardlink_policy_preserves_old_paths_and_undo_restores_independent_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = sdxl_model(root / "a" / "same.safetensors")
            second = sdxl_model(root / "b" / "same.safetensors")
            index = root / "index.csv"
            index_for(index, [first, second])
            base = root / "sorted"
            plan = planner.build_sort_plan(index, base, duplicate_policy="hardlink")
            manifest = root / "manifest.json"
            with contextlib.redirect_stdout(io.StringIO()):
                counts = transactions.apply_plan(
                    plan,
                    csv_path=index,
                    base_dir=base,
                    dry_run=False,
                    manifest_path=manifest,
                )
            destination = base / "SDXL" / "same.safetensors"
            self.assertEqual(counts["moved"], 1)
            self.assertTrue(first.exists() and second.exists() and destination.exists())
            self.assertTrue(same_inode(first, second))
            self.assertTrue(same_inode(first, destination))

            with contextlib.redirect_stdout(io.StringIO()):
                rc = transactions.undo_manifest(manifest, apply_changes=True)
            self.assertEqual(rc, 0)
            self.assertTrue(first.exists() and second.exists())
            self.assertFalse(destination.exists())
            self.assertFalse(same_inode(first, second))
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_existing_identical_destination_can_become_canonical(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sdxl_model(root / "old" / "same.safetensors")
            destination = sdxl_model(root / "sorted" / "SDXL" / "same.safetensors")
            index = root / "index.csv"
            index_for(index, [source])
            plan = planner.build_sort_plan(index, root / "sorted", duplicate_policy="hardlink")
            leader = next(row for row in plan if row["action"] == "hardlink_duplicate_group")
            self.assertEqual(Path(leader["canonical_path"]), destination)
            with contextlib.redirect_stdout(io.StringIO()):
                counts = transactions.apply_plan(
                    plan,
                    index,
                    root / "sorted",
                    dry_run=False,
                    manifest_path=root / "manifest.json",
                )
            self.assertEqual(counts["moved"], 1)
            self.assertTrue(same_inode(source, destination))

    def test_quarantine_requires_acknowledgement_and_is_undoable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = sdxl_model(root / "a" / "same.safetensors")
            second = sdxl_model(root / "b" / "same.safetensors")
            index = root / "index.csv"
            index_for(index, [first, second])
            base = root / "sorted"
            plan = planner.build_sort_plan(index, base, duplicate_policy="quarantine")
            with contextlib.redirect_stdout(io.StringIO()):
                denied = transactions.apply_plan(
                    plan,
                    index,
                    base,
                    dry_run=False,
                    manifest_path=root / "denied.json",
                )
            self.assertEqual(denied["moved"], 0)
            self.assertTrue(first.exists() and second.exists())

            plan = planner.build_sort_plan(index, base, duplicate_policy="quarantine")
            manifest = root / "quarantine.json"
            with contextlib.redirect_stdout(io.StringIO()):
                allowed = transactions.apply_plan(
                    plan,
                    index,
                    base,
                    dry_run=False,
                    manifest_path=manifest,
                    allow_workflow_path_breakage=True,
                )
            self.assertEqual(allowed["moved"], 1)
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertTrue((base / "SDXL" / "same.safetensors").exists())
            quarantined = list((base / "_Duplicates").rglob("same.safetensors"))
            self.assertEqual(len(quarantined), 1)

            with contextlib.redirect_stdout(io.StringIO()):
                rc = transactions.undo_manifest(manifest, apply_changes=True)
            self.assertEqual(rc, 0)
            self.assertTrue(first.exists() and second.exists())
            self.assertFalse((base / "SDXL" / "same.safetensors").exists())


class PathSafetyTests(unittest.TestCase):
    @unittest.skipIf(not hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlinked_source_subdirectory_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            external = root / "external"
            source_real = sdxl_model(external / "model.safetensors")
            link = root / "library" / "linked"
            link.parent.mkdir(parents=True)
            os.symlink(external, link)
            source = link / source_real.name
            index = root / "index.csv"
            index_for(index, [source])
            plan = planner.build_sort_plan(index, root / "library")
            self.assertEqual(plan[0]["action"], "skip_path_safety")
            self.assertIn("symlinked subdirectory", plan[0]["reason"])

    @unittest.skipIf(not hasattr(os, "symlink"), "symlinks unavailable")
    def test_changed_source_symlink_target_is_caught_at_apply(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            external_one = root / "external-one"
            external_two = root / "external-two"
            first = sdxl_model(external_one / "model.safetensors")
            sdxl_model(external_two / "model.safetensors")
            library = root / "library"
            library.mkdir()
            linked = library / "linked"
            os.symlink(external_one, linked)
            source = linked / first.name
            index = root / "index.csv"
            index_for(index, [source])
            plan = planner.build_sort_plan(
                index,
                library,
                allow_symlink_parents=True,
            )
            self.assertEqual(plan[0]["action"], "move")
            linked.unlink()
            os.symlink(external_two, linked)
            with contextlib.redirect_stdout(io.StringIO()):
                counts = transactions.apply_plan(
                    plan,
                    index,
                    library,
                    dry_run=False,
                    manifest_path=root / "manifest.json",
                    allow_symlink_parents=True,
                )
            self.assertEqual(counts["moved"], 0)
            self.assertEqual(plan[0]["action"], "skip_path_safety_changed")
            self.assertIn("different location", plan[0]["reason"])

    @unittest.skipIf(not hasattr(os, "symlink"), "symlinks unavailable")
    def test_new_destination_symlink_is_caught_before_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = sdxl_model(root / "model.safetensors")
            index = root / "index.csv"
            index_for(index, [source])
            base = root / "sorted"
            plan = planner.build_sort_plan(index, base)
            base.mkdir()
            outside = root / "outside"
            outside.mkdir()
            os.symlink(outside, base / "SDXL")
            with contextlib.redirect_stdout(io.StringIO()):
                counts = transactions.apply_plan(
                    plan,
                    index,
                    base,
                    dry_run=False,
                    manifest_path=root / "manifest.json",
                )
            self.assertEqual(counts["moved"], 0)
            self.assertEqual(plan[0]["action"], "skip_path_safety_changed")
            self.assertTrue(source.exists())
            self.assertFalse((outside / "model.safetensors").exists())


class SafeMoveTests(unittest.TestCase):
    def test_same_filesystem_move_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"source")
            destination.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                transactions.safe_move(source, destination)
            self.assertEqual(source.read_bytes(), b"source")
            self.assertEqual(destination.read_bytes(), b"existing")

    def test_cross_device_fallback_is_verified_and_removes_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"payload" * 1000)
            real_link = transactions.os.link

            def no_links(*args, **kwargs):
                raise OSError(errno.EXDEV, "synthetic cross-device link failure")

            with mock.patch.object(transactions.os, "link", side_effect=no_links):
                result = transactions.safe_move(source, destination)
            self.assertFalse(source.exists())
            self.assertEqual(destination.read_bytes(), b"payload" * 1000)
            self.assertEqual(result["method"], "copy-verify-exclusive-finalize-unlink")
            self.assertEqual(len(result["sha256"]), 64)

    def test_destination_race_leaves_source_and_intruder_intact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"source")

            def race(_source, target, **_kwargs):
                Path(target).write_bytes(b"intruder")
                raise OSError(errno.EEXIST, "synthetic race")

            with mock.patch.object(transactions.os, "link", side_effect=race):
                with self.assertRaises(FileExistsError):
                    transactions.safe_move(source, destination)
            self.assertEqual(source.read_bytes(), b"source")
            self.assertEqual(destination.read_bytes(), b"intruder")

    def test_published_callback_failure_rolls_back_same_filesystem_link(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"payload")

            def fail(_result):
                raise RuntimeError("journal write failed")

            with self.assertRaisesRegex(RuntimeError, "journal write failed"):
                transactions.safe_move(source, destination, on_published=fail)
            self.assertTrue(source.exists())
            self.assertFalse(destination.exists())


class RecoveryTests(unittest.TestCase):
    def _manifest(self, path: Path, operation):
        payload = {
            "schema_name": MANIFEST_SCHEMA_NAME,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tool_version": "0.6.0",
            "manifest_type": "apply",
            "operations": [operation],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_pending_published_move_is_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"payload")
            os.link(source, destination)
            manifest = root / "manifest.json"
            self._manifest(
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

    def test_pending_hardlink_create_is_recovered_even_without_done_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "canonical.bin"
            target = root / "alias.bin"
            source.write_bytes(b"payload")
            os.link(source, target)
            manifest = root / "manifest.json"
            self._manifest(
                manifest,
                {
                    "operation_id": "one",
                    "record_path": str(source),
                    "type": "hardlink_duplicate_group",
                    "status": "pending",
                    "steps": [
                        {
                            "type": "hardlink-create",
                            "source": str(source),
                            "target": str(target),
                            "status": "pending",
                        }
                    ],
                },
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rc = transactions.recover_manifest(manifest, apply_changes=True)
            self.assertEqual(rc, 0)
            self.assertTrue(source.exists())
            self.assertFalse(target.exists())

    def test_manifest_with_wrong_schema_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_name": "something-else",
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "manifest_type": "apply",
                        "operations": [],
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = transactions.undo_manifest(manifest)
            self.assertEqual(rc, 1)
            self.assertIn("unrecognised manifest schema", output.getvalue())


if __name__ == "__main__":
    unittest.main()


class PlanTimeHashReuseTests(unittest.TestCase):
    def _pair(self, root: Path):
        first = sdxl_model(root / "a" / "same.safetensors", payload=b"X" * 4096)
        second = sdxl_model(root / "b" / "same.safetensors", payload=b"X" * 4096)
        return first, second

    def test_default_plan_rehashes_and_catches_a_lying_csv(self):
        # Without the trust flag, a falsified full_sha256 on an untouched file
        # is detected by plan-time content re-verification. This pins the
        # pre-existing paranoid default.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = self._pair(root)
            rows = inspection.inspect_files([first, second], include_full_hash=True)
            rows[0]["full_sha256"] = "0" * 64
            index = root / "index.csv"
            write_csv(index, rows, schema.INSPECTION_FIELDS)
            plan = planner.build_sort_plan(index, root / "sorted", duplicate_policy="hardlink")
            liar = next(r for r in plan if r["path"] == str(first))
            self.assertEqual(liar["action"], "skip_stale_or_unverified")
            self.assertIn("content hash", liar["reason"])

    def test_trust_flag_reuses_identity_verified_hashes(self):
        # Same distinctive fake digest on both records with truthful identity
        # fields: under the flag the plan trusts them, so the group carries
        # the fake instead of a freshly computed real digest.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = self._pair(root)
            rows = inspection.inspect_files([first, second], include_full_hash=True)
            fake = "f" * 64
            for row in rows:
                row["full_sha256"] = fake
            index = root / "index.csv"
            write_csv(index, rows, schema.INSPECTION_FIELDS)
            plan = planner.build_sort_plan(
                index, root / "sorted", duplicate_policy="hardlink",
                trust_recorded_hashes=True,
            )
            actions = sorted(row["action"] for row in plan)
            self.assertEqual(actions, ["hardlink_duplicate_group", "hardlink_duplicate_member"])
            leader = next(row for row in plan if row["action"] == "hardlink_duplicate_group")
            self.assertEqual(
                leader["content_sha256"], fake,
                "planner re-hashed despite --trust-recorded-hashes",
            )

    def test_trust_flag_still_rehashes_after_identity_change(self):
        # Touching the file breaks the identity key, so the trusted-cache
        # entry can never be consulted; verification sees the drift and the
        # record is skipped, exactly as without the flag.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = self._pair(root)
            rows = inspection.inspect_files([first, second], include_full_hash=True)
            index = root / "index.csv"
            write_csv(index, rows, schema.INSPECTION_FIELDS)
            os.utime(first, ns=(1, 1))
            plan = planner.build_sort_plan(
                index, root / "sorted", duplicate_policy="hardlink",
                trust_recorded_hashes=True,
            )
            touched = next(r for r in plan if r["path"] == str(first))
            self.assertEqual(touched["action"], "skip_stale_or_unverified")

    def test_missing_recorded_hash_still_hashes_lazily(self):
        # Without --hash-content the CSV has no full_sha256: collision
        # planning falls back to lazy hashing and still groups.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = self._pair(root)
            rows = inspection.inspect_files([first, second], include_full_hash=False)
            self.assertFalse(any(row["full_sha256"] for row in rows))
            index = root / "index.csv"
            write_csv(index, rows, schema.INSPECTION_FIELDS)
            calls = []
            real = duplicates.full_file_sha256

            def counting(path, cache=None, **kwargs):
                calls.append(path)
                return real(path, cache, **kwargs)

            with mock.patch.object(duplicates, "full_file_sha256", side_effect=counting):
                plan = planner.build_sort_plan(
                    index, root / "sorted", duplicate_policy="hardlink",
                    trust_recorded_hashes=True,
                )
            actions = sorted(row["action"] for row in plan)
            self.assertEqual(actions, ["hardlink_duplicate_group", "hardlink_duplicate_member"])
            self.assertTrue(calls, "lazy hashing fallback never ran")
