"""Fault injection: kill the process at every journalled phase, then recover.

Each test runs an apply (or undo) in a subprocess whose ``_save_manifest`` is
wrapped so the process hard-exits with ``os._exit(9)`` immediately *after* the
journal reaches a chosen phase on disk. That simulates a genuine crash: no
Python unwinding, no ``except`` blocks, no local rollback. The parent then
asserts the documented invariants:

1. No payload is ever lost: the model content exists at the source or the
   destination (or both) after the crash.
2. ``recover_manifest`` dry run reports the interruption without errors.
3. ``recover_manifest --apply`` returns the tree to the documented state:
   pre-operation names for an interrupted apply, applied state for an
   interrupted undo.
4. No staging temporaries survive recovery.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lora_header_inspector import transactions

REPO_ROOT = Path(__file__).resolve().parent.parent

DRIVER = r"""
import errno, io, json, os, sys, contextlib
from pathlib import Path

sys.path.insert(0, os.environ["FI_ROOT"])

from lora_header_inspector import planner, transactions
from tests.test_transaction_journals import sdxl_model, write_index

WORK = Path(os.environ["FI_WORKDIR"])
SCENARIO = os.environ["FI_SCENARIO"]
PHASE = os.environ["FI_PHASE"]
COUNT = int(os.environ.get("FI_COUNT", "1"))
EXDEV = os.environ.get("FI_EXDEV") == "1"
SIDECAR = os.environ.get("FI_SIDECAR") == "1"

state = {"hits": 0, "armed": False}
_real_save = transactions._save_manifest


def _steps(payload):
    for op in payload.get("operations", []):
        steps = op.get("steps") or []
        if not isinstance(steps, list):
            continue
        for step in steps:
            yield step
            backup = step.get("backup_move")
            if isinstance(backup, dict):
                yield backup


def _phase_reached(payload):
    statuses = [str(s.get("status", "")) for s in _steps(payload)]
    if PHASE == "all_pending":
        return bool(statuses) and all(s == "pending" for s in statuses)
    if PHASE == "first_done_rest_pending":
        return "done" in statuses and "pending" in statuses
    return PHASE in statuses


def killing_save(payload, path):
    _real_save(payload, path)
    if state["armed"] and _phase_reached(payload):
        state["hits"] += 1
        if state["hits"] >= COUNT:
            try:  # os._exit skips atexit: flush coverage data when measuring.
                import coverage
                current = coverage.Coverage.current()
                if current is not None:
                    current.save()
            except Exception:
                pass
            os._exit(9)


transactions._save_manifest = killing_save


def deny_hardlinks():
    def refuse(*args, **kwargs):
        raise OSError(errno.EXDEV, "cross-device link (injected)")

    os.link = refuse


def note(payload):
    (WORK / "expected.json").write_text(json.dumps(payload), encoding="utf-8")


PAYLOAD = b"payload-bytes-" * 64


def apply_move():
    library = WORK / "library"
    model = sdxl_model(library / "model.safetensors", payload=PAYLOAD)
    sidecars = []
    if SIDECAR:
        preview = library / "model.preview.png"
        preview.write_bytes(b"preview-bytes")
        sidecars.append(str(preview))
    index = WORK / "index.csv"
    write_index(index, [model])
    base = WORK / "sorted"
    plan = planner.build_sort_plan(index, base)
    destination = base / "SDXL" / "model.safetensors"
    note({"sources": [str(model)], "destination": str(destination), "sidecars": sidecars})
    if EXDEV:
        deny_hardlinks()
    state["armed"] = True
    with contextlib.redirect_stdout(io.StringIO()):
        transactions.apply_plan(
            plan, csv_path=index, base_dir=base, dry_run=False,
            manifest_path=WORK / "manifest.json",
        )
    raise SystemExit(0)


def apply_duplicates(policy):
    first = sdxl_model(WORK / "a" / "same.safetensors", payload=PAYLOAD)
    second = sdxl_model(WORK / "b" / "same.safetensors", payload=PAYLOAD)
    index = WORK / "index.csv"
    write_index(index, [first, second])
    base = WORK / "sorted"
    plan = planner.build_sort_plan(index, base, duplicate_policy=policy)
    note({
        "sources": [str(first), str(second)],
        "destination": str(base / "SDXL" / "same.safetensors"),
        "sidecars": [],
    })
    state["armed"] = True
    with contextlib.redirect_stdout(io.StringIO()):
        transactions.apply_plan(
            plan, csv_path=index, base_dir=base, dry_run=False,
            manifest_path=WORK / "manifest.json",
            allow_workflow_path_breakage=True,
        )
    raise SystemExit(0)


def undo_move():
    library = WORK / "library"
    model = sdxl_model(library / "model.safetensors", payload=PAYLOAD)
    index = WORK / "index.csv"
    write_index(index, [model])
    base = WORK / "sorted"
    plan = planner.build_sort_plan(index, base)
    destination = base / "SDXL" / "model.safetensors"
    with contextlib.redirect_stdout(io.StringIO()):
        transactions.apply_plan(
            plan, csv_path=index, base_dir=base, dry_run=False,
            manifest_path=WORK / "manifest.json",
        )
    note({"sources": [str(model)], "destination": str(destination), "sidecars": []})
    if EXDEV:
        deny_hardlinks()
    state["armed"] = True
    with contextlib.redirect_stdout(io.StringIO()):
        transactions.undo_manifest(
            WORK / "manifest.json", apply_changes=True,
            output_manifest=WORK / "undo.json",
        )
    raise SystemExit(0)


if SCENARIO == "move":
    apply_move()
elif SCENARIO == "hardlink":
    apply_duplicates("hardlink")
elif SCENARIO == "quarantine":
    apply_duplicates("quarantine")
elif SCENARIO == "undo_move":
    undo_move()
else:
    raise SystemExit("unknown scenario: %s" % SCENARIO)
"""


@unittest.skipIf(not hasattr(os, "link"), "hard links unavailable")
class FaultInjectionTests(unittest.TestCase):
    maxDiff = None

    def crash_at(self, scenario, phase, *, count=1, exdev=False, sidecar=False):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        work = Path(self.temp.name)
        env = dict(os.environ)
        env.update(
            FI_ROOT=str(REPO_ROOT),
            FI_WORKDIR=str(work),
            FI_SCENARIO=scenario,
            FI_PHASE=phase,
            FI_COUNT=str(count),
            FI_EXDEV="1" if exdev else "",
            FI_SIDECAR="1" if sidecar else "",
        )
        proc = subprocess.run(
            [sys.executable, "-c", DRIVER],
            env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(
            proc.returncode, 9,
            "expected an injected crash at phase %r, got rc=%d\nstdout:\n%s\nstderr:\n%s"
            % (phase, proc.returncode, proc.stdout, proc.stderr),
        )
        expected = json.loads((work / "expected.json").read_text(encoding="utf-8"))
        return work, expected

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def payload_digests(self, work: Path):
        found = {}
        for path in work.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix != ".json":
                if path.name.endswith(".csv"):
                    continue
                found.setdefault(self.digest(path.read_bytes()), []).append(path)
        return found

    def assert_payload_survives(self, work: Path, expected):
        """The model content must exist somewhere after the crash."""
        sources = [Path(p) for p in expected["sources"]]
        destination = Path(expected["destination"])
        reference = None
        for candidate in sources + [destination]:
            if candidate.is_file():
                reference = candidate
                break
        self.assertIsNotNone(
            reference,
            "payload vanished: none of %s exist after crash" % ([str(s) for s in sources + [destination]]),
        )

    def recover(self, work: Path, journal: Path):
        with contextlib.redirect_stdout(io.StringIO()) as dry_out:
            dry_rc = transactions.recover_manifest(journal, apply_changes=False)
        self.assertEqual(dry_rc, 0, "dry-run recovery reported failures:\n%s" % dry_out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as apply_out:
            rc = transactions.recover_manifest(
                journal, apply_changes=True,
                output_manifest=work / ("recovery_%s.json" % journal.stem),
            )
        self.assertEqual(rc, 0, "recovery reported failures:\n%s" % apply_out.getvalue())

    def assert_no_staging_left(self, work: Path):
        strays = [str(p) for p in work.rglob("*.tmp") if p.is_file()]
        self.assertEqual(strays, [], "staging temporaries survived recovery")

    def assert_restored(self, work: Path, expected):
        for source in expected["sources"]:
            self.assertTrue(Path(source).is_file(), "source not restored: %s" % source)
        for sidecar in expected["sidecars"]:
            self.assertTrue(Path(sidecar).is_file(), "sidecar not restored: %s" % sidecar)
        self.assertFalse(
            Path(expected["destination"]).exists(),
            "destination survived rollback: %s" % expected["destination"],
        )
        self.assert_no_staging_left(work)

    # ------------------------------------------------------------------ moves

    def test_crash_with_all_steps_pending_recovers_to_untouched_tree(self):
        work, expected = self.crash_at("move", "all_pending")
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "manifest.json")
        self.assert_restored(work, expected)

    def test_crash_after_samefs_publication_before_unlink(self):
        # link-unlink path: destination link exists, old name still present.
        work, expected = self.crash_at("move", "published")
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "manifest.json")
        self.assert_restored(work, expected)

    def test_crash_during_crossfs_staging(self):
        work, expected = self.crash_at("move", "staging", exdev=True)
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "manifest.json")
        self.assert_restored(work, expected)

    def test_crash_after_crossfs_destination_created(self):
        work, expected = self.crash_at("move", "publishing", exdev=True)
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "manifest.json")
        self.assert_restored(work, expected)

    def test_crash_after_crossfs_publication_before_source_removal(self):
        work, expected = self.crash_at("move", "published", exdev=True)
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "manifest.json")
        self.assert_restored(work, expected)

    def test_crash_between_group_steps_rolls_back_completed_primary(self):
        # Model step done, sidecar step still pending: recovery must undo the
        # completed primary move so the group never lands half-applied.
        work, expected = self.crash_at("move", "first_done_rest_pending", sidecar=True)
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "manifest.json")
        self.assert_restored(work, expected)

    # ------------------------------------------------------- duplicate groups

    def test_crash_after_duplicate_backup_published(self):
        work, expected = self.crash_at("hardlink", "backup_published")
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "manifest.json")
        for source in expected["sources"]:
            path = Path(source)
            self.assertTrue(path.is_file(), "duplicate not restored: %s" % source)
        self.assert_no_staging_left(work)

    def test_crash_after_duplicate_relinked(self):
        work, expected = self.crash_at("hardlink", "linked")
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "manifest.json")
        for source in expected["sources"]:
            self.assertTrue(Path(source).is_file(), "duplicate not restored: %s" % source)
        self.assert_no_staging_left(work)

    def test_crash_during_quarantine_publication(self):
        work, expected = self.crash_at("quarantine", "published")
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "manifest.json")
        for source in expected["sources"]:
            self.assertTrue(Path(source).is_file(), "quarantined file not restored: %s" % source)
        self.assert_no_staging_left(work)

    # -------------------------------------------------------------------- undo

    def test_crash_mid_undo_recovers_to_applied_state(self):
        # Recovery of an interrupted *undo* journal returns to the applied
        # state, not the original one: destination present, source absent.
        work, expected = self.crash_at("undo_move", "published", exdev=True)
        self.assert_payload_survives(work, expected)
        self.recover(work, work / "undo.json")
        self.assertTrue(Path(expected["destination"]).is_file())
        for source in expected["sources"]:
            self.assertFalse(Path(source).exists(), "undo recovery resurrected the source")
        self.assert_no_staging_left(work)


if __name__ == "__main__":
    unittest.main()
