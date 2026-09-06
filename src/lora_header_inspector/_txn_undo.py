"""Whole-manifest undo with preflight and rollback to applied state."""

# This module is part of the transactions layer. Do not import it directly:
# use ``lora_header_inspector.transactions``, which assembles the public API.
# Cross-cutting primitives are called as ``_txn.<name>`` so that patching an
# attribute on the transactions facade behaves exactly as it did when the
# engine was a single module.

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .fingerprints import (
    HashCache,
    file_identity,
    full_file_sha256,
    identity_matches,
    same_inode,
    verify_record_fingerprint,
)
from .paths import (
    absolute_without_resolving,
    normal_path_key,
    verify_path_safety_snapshot,
)
from .sidecars import verify_sidecar_snapshot
from .version import (
    MANIFEST_SCHEMA_NAME,
    UNDO_SCHEMA_NAME,
    UNDO_SCHEMA_VERSION,
)
from .schema import MANIFEST_SCHEMA_VERSION

from . import transactions as _txn

COPY_CHUNK_BYTES = 8 * 1024 * 1024

def _prepare_undo_operation(
    source_operation: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Build and preflight one journalled undo operation."""
    operation_type = str(source_operation.get("type", "") or "")
    source_steps = source_operation.get("steps", [])
    if source_operation.get("status") != "completed":
        return None, "operation was not completed"
    if not isinstance(source_steps, list):
        return None, "operation steps are invalid"

    journal: Dict[str, Any] = {
        "operation_id": str(uuid.uuid4()),
        "source_operation_id": str(source_operation.get("operation_id", "") or ""),
        "source_type": operation_type,
        "record_path": str(source_operation.get("record_path", "") or ""),
        "type": "",
        "status": "pending",
        "error": "",
        "rollback_errors": [],
        "steps": [],
    }

    if operation_type in {"move", "quarantine_duplicate_group"}:
        reverse_steps: List[Dict[str, Any]] = []
        transfers: List[Tuple[Path, Path]] = []
        for source_step in reversed(source_steps):
            if source_step.get("type") != "move" or source_step.get("status") != "done":
                continue
            source = Path(str(source_step.get("to", "")))
            destination = Path(str(source_step.get("from", "")))
            if not source.is_file() or source.is_symlink():
                return None, f"undo source missing or unsafe: {source}"
            if destination.exists() or destination.is_symlink():
                return None, f"undo destination occupied: {destination}"
            digest = str(source_step.get("sha256", "") or "")
            if not digest:
                digest = full_file_sha256(source)
            elif full_file_sha256(source) != digest:
                return None, f"undo source content no longer matches the apply journal: {source}"
            identity = file_identity(source)
            reverse_steps.append(
                {
                    "type": "move",
                    "from": str(source),
                    "to": str(destination),
                    "status": "pending",
                    "method": "",
                    "sha256": digest,
                    "size": str(identity.size),
                    "source_identity": identity.to_dict(),
                    "destination_identity": {},
                    "staging_path": "",
                }
            )
            transfers.append((source, destination))
        if not reverse_steps:
            return None, "completed operation contains no reversible move steps"
        ready, reason = _txn._preflight_transfers(transfers)
        if not ready:
            return None, reason
        journal["type"] = "undo_move"
        journal["steps"] = reverse_steps
        return journal, "reversible move operation"

    if operation_type == "hardlink_duplicate_group":
        undo_steps: List[Dict[str, Any]] = []
        for source_step in reversed(source_steps):
            if source_step.get("status") != "done":
                continue
            step_type = str(source_step.get("type", "") or "")
            if step_type == "hardlink-create":
                canonical = Path(str(source_step.get("source", "")))
                target = Path(str(source_step.get("target", "")))
                if canonical.is_symlink() or not canonical.is_file():
                    return None, f"canonical hard-link source is unavailable: {canonical}"
                if target.is_symlink() or not target.is_file():
                    return None, f"created hard-link target is unavailable: {target}"
                if not same_inode(canonical, target):
                    return None, f"created hard-link target was replaced: {target}"
                undo_steps.append(
                    {
                        "type": "hardlink-remove",
                        "canonical": str(canonical),
                        "target": str(target),
                        "status": "pending",
                        "linked_identity": file_identity(target).to_dict(),
                        "sha256": str(source_step.get("sha256", "") or ""),
                    }
                )
            elif step_type == "hardlink-replace":
                path = Path(str(source_step.get("path", "")))
                canonical = Path(str(source_step.get("canonical", "")))
                if path.is_symlink() or not path.is_file():
                    return None, f"relinked path is unavailable: {path}"
                if canonical.is_symlink() or not canonical.is_file():
                    return None, f"canonical path is unavailable: {canonical}"
                if not same_inode(path, canonical):
                    return None, f"relinked path no longer points to canonical inode: {path}"
                digest = str(source_step.get("sha256", "") or "")
                if digest and full_file_sha256(path) != digest:
                    return None, f"relinked path content changed after apply: {path}"
                undo_steps.append(
                    {
                        "type": "hardlink-break",
                        "path": str(path),
                        "canonical": str(canonical),
                        "status": "pending",
                        "linked_identity": file_identity(path).to_dict(),
                        "sha256": digest,
                        "staging_path": "",
                        "staging_identity": {},
                        "backup_path": "",
                        "backup_move": {},
                        "replacement_identity": {},
                        "backup_removed": False,
                    }
                )
            else:
                return None, f"unsupported hard-link journal step: {step_type}"
        if not undo_steps:
            return None, "completed hard-link operation contains no reversible steps"
        journal["type"] = "undo_hardlink"
        journal["steps"] = undo_steps
        return journal, "reversible hard-link operation"

    return None, f"unsupported operation type: {operation_type}"


def _execute_undo_move_operation(
    operation: MutableMapping[str, Any],
    undo: MutableMapping[str, Any],
    undo_path: Path,
) -> Tuple[bool, List[str]]:
    steps = operation.get("steps", [])
    try:
        for step in steps:
            def staged(path: Path, current=step) -> None:
                current["staging_path"] = str(path)
                current["status"] = "staging"
                _txn._save_manifest(undo, undo_path)

            def destination_created(identity: Mapping[str, Any], current=step) -> None:
                current["destination_identity"] = dict(identity)
                current["status"] = "publishing"
                _txn._save_manifest(undo, undo_path)

            def published(result: Mapping[str, Any], current=step) -> None:
                current.update(result)
                current["status"] = "published"
                _txn._save_manifest(undo, undo_path)

            result = _txn.safe_move(
                Path(step["from"]),
                Path(step["to"]),
                expected_sha256=str(step.get("sha256", "") or ""),
                on_staged=staged,
                on_destination_created=destination_created,
                on_published=published,
            )
            step.update(status="done", **result)
            _txn._save_manifest(undo, undo_path)
    except Exception as exc:
        rollback_errors = _txn._rollback_move_steps(steps)
        operation.update(
            status="rollback_error" if rollback_errors else "rolled_back",
            error=str(exc),
            rollback_errors=rollback_errors,
        )
        _txn._save_manifest(undo, undo_path)
        return False, rollback_errors
    operation["status"] = "completed"
    _txn._save_manifest(undo, undo_path)
    return True, []


def _rollback_undo_hardlink_steps(
    steps: Sequence[MutableMapping[str, Any]],
) -> List[str]:
    """Restore the post-apply hard-link state after an interrupted undo."""
    errors: List[str] = []
    for step in reversed(steps):
        step_type = str(step.get("type", "") or "")
        try:
            if step_type == "hardlink-remove":
                canonical = Path(str(step.get("canonical", "")))
                target = Path(str(step.get("target", "")))
                if canonical.is_symlink() or not canonical.is_file():
                    errors.append(f"canonical path is unavailable: {canonical}")
                    continue
                if target.is_symlink():
                    errors.append(f"removed hard-link target became a symlink: {target}")
                    continue
                if target.exists():
                    if not same_inode(canonical, target):
                        errors.append(f"hard-link target was replaced during undo: {target}")
                        continue
                else:
                    _txn.atomic_hardlink(canonical, target)
                step["status"] = "rolled_back"
            elif step_type == "hardlink-break":
                errors.extend(_txn._rollback_break_link_step(step))
            else:
                errors.append(f"unsupported undo hard-link step: {step_type}")
        except Exception as exc:
            errors.append(f"{step}: {exc}")
    return errors


def _execute_undo_hardlink_operation(
    operation: MutableMapping[str, Any],
    undo: MutableMapping[str, Any],
    undo_path: Path,
) -> Tuple[bool, List[str]]:
    steps = operation.get("steps", [])
    try:
        for step in steps:
            step_type = str(step.get("type", "") or "")
            if step_type == "hardlink-remove":
                canonical = Path(str(step["canonical"]))
                target = Path(str(step["target"]))
                linked_identity = step.get("linked_identity", {})
                if linked_identity and not _txn._matches_stable_identity(target, linked_identity):
                    raise ValueError(f"hard-link target changed before undo: {target}")
                if not same_inode(canonical, target):
                    raise ValueError(f"hard-link target no longer points to canonical inode: {target}")
                step["status"] = "removing"
                _txn._save_manifest(undo, undo_path)
                target.unlink()
                _txn._fsync_directory(target.parent)
                step["status"] = "done"
                _txn._save_manifest(undo, undo_path)
            elif step_type == "hardlink-break":
                _txn._journaled_break_link_step(
                    step,
                    lambda: _txn._save_manifest(undo, undo_path),
                )
            else:
                raise ValueError(f"unsupported undo hard-link step: {step_type}")
    except Exception as exc:
        rollback_errors = _txn._rollback_undo_hardlink_steps(steps)
        operation.update(
            status="rollback_error" if rollback_errors else "rolled_back",
            error=str(exc),
            rollback_errors=rollback_errors,
        )
        _txn._save_manifest(undo, undo_path)
        return False, rollback_errors
    operation["status"] = "completed"
    _txn._save_manifest(undo, undo_path)
    return True, []


def undo_manifest(
    manifest_path: Path,
    apply_changes: bool = False,
    output_manifest: Optional[Path] = None,
) -> int:
    try:
        manifest = _txn._load_manifest(manifest_path)
    except Exception as exc:
        print(f"ERROR: could not read manifest: {exc}")
        return 1

    prepared: List[Dict[str, Any]] = []
    failures: List[str] = []
    for source_operation in reversed(manifest.get("operations", [])):
        label = source_operation.get("record_path", source_operation.get("operation_id", "unknown"))
        journal, reason = _txn._prepare_undo_operation(source_operation)
        print(f"[{'UNDO-CHECK' if apply_changes else 'DRY-UNDO'}] {label}")
        if journal is None:
            failures.append(f"{label}: {reason}")
            print(f"       blocked: {reason}")
        else:
            prepared.append(journal)
            print(f"       {reason}")

    # Undo is preflighted as a whole. A known-bad operation must not leave an
    # earlier operation reversed while a later one remains applied.
    if failures:
        print(f"Undo preflight failed. reversible={len(prepared)} blocked={len(failures)}")
        return 1
    if not apply_changes:
        print(f"Undo dry run complete. reversible={len(prepared)} failures=0")
        return 0

    undo = _txn.undo_header()
    undo.update(
        manifest_type="undo",
        manifest_id=str(uuid.uuid4()),
        created_utc=datetime.now(timezone.utc).isoformat(),
        updated_utc=datetime.now(timezone.utc).isoformat(),
        undo_of=str(manifest_path),
        status="in_progress",
        operations=prepared,
    )
    output_manifest = absolute_without_resolving(
        output_manifest
        or manifest_path.with_name(f"{manifest_path.stem}_undo_{_txn._utc_stamp()}.json")
    )
    protected = _txn._journal_paths(manifest) | _txn._journal_paths(undo) | {normal_path_key(manifest_path)}
    if normal_path_key(output_manifest) in protected:
        print("ERROR: undo manifest overlaps the source journal or an operation path")
        return 1
    if output_manifest.exists() or output_manifest.is_symlink():
        print(f"ERROR: undo manifest already exists: {output_manifest}")
        return 1
    try:
        _txn._create_manifest(undo, output_manifest)
    except Exception as exc:
        print(f"ERROR: could not create undo manifest: {exc}")
        return 1

    completed = 0
    rolled_back = 0
    errors = 0
    for operation in undo["operations"]:
        print(f"[UNDO] {operation.get('record_path', operation.get('operation_id', 'unknown'))}")
        if operation.get("type") == "undo_move":
            success, rollback_errors = _txn._execute_undo_move_operation(
                operation, undo, output_manifest
            )
        elif operation.get("type") == "undo_hardlink":
            success, rollback_errors = _txn._execute_undo_hardlink_operation(
                operation, undo, output_manifest
            )
        else:
            operation.update(status="rollback_error", error="unsupported undo operation")
            _txn._save_manifest(undo, output_manifest)
            success, rollback_errors = False, ["unsupported undo operation"]
        if success:
            completed += 1
        elif rollback_errors:
            errors += 1
        else:
            rolled_back += 1

    if any(op.get("status") in {"pending", "rollback_error"} for op in undo["operations"]):
        undo["status"] = "attention_required"
    elif rolled_back:
        undo["status"] = "completed_with_rollbacks"
    else:
        undo["status"] = "completed"
    _txn._save_manifest(undo, output_manifest)
    print(
        f"Undo complete. completed={completed} rolled_back={rolled_back} errors={errors}"
    )
    print(f"Undo manifest: {output_manifest}")
    return 1 if errors or rolled_back else 0
