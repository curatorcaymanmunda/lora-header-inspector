"""Interrupted-journal analysis and recovery."""

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

def _analyze_move_rollback_steps(steps: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    for step in reversed(steps):
        source = Path(str(step.get("from", "")))
        destination = Path(str(step.get("to", "")))
        if source.is_symlink() or destination.is_symlink():
            errors.append(f"rollback path is a symlink: {destination} -> {source}")
            continue
        source_exists = source.is_file()
        destination_exists = destination.is_file()
        if source_exists and destination_exists:
            recorded_destination = step.get("destination_identity", {})
            owned = (
                isinstance(recorded_destination, Mapping)
                and bool(recorded_destination)
                and identity_matches(destination, recorded_destination)
            )
            if not _txn._same_regular_content(source, destination) and not owned:
                errors.append(f"both rollback paths have different content: {source} and {destination}")
        elif not source_exists and destination_exists:
            if source.exists() or source.is_symlink():
                errors.append(f"rollback destination is occupied: {source}")
        elif source_exists and not destination_exists:
            pass
        else:
            errors.append(f"neither rollback path exists: {source} / {destination}")
        staging_text = str(step.get("staging_path", "") or "")
        if staging_text and Path(staging_text).is_symlink():
            errors.append(f"staging path became a symlink: {staging_text}")
    return errors


def _analyze_relink_rollback_step(step: Mapping[str, Any]) -> List[str]:
    path = Path(str(step.get("path", "")))
    canonical = Path(str(step.get("canonical", "")))
    backup_text = str(step.get("backup_path", "") or "")
    backup = Path(backup_text) if backup_text else None
    if canonical.is_symlink() or not canonical.is_file():
        return [f"canonical path is unavailable: {canonical}"]
    if path.is_symlink() or (backup is not None and backup.is_symlink()):
        return [f"relink rollback contains a symlink path: {path} / {backup}"]
    path_exists = path.is_file()
    backup_exists = bool(backup and backup.is_file())
    path_is_link = path_exists and same_inode(path, canonical)
    if backup_exists:
        if path_exists and not path_is_link:
            before = step.get("before_identity", {})
            if (
                same_inode(path, backup)
                and isinstance(before, Mapping)
                and _txn._inode_unchanged(path, before)
            ):
                # Interrupted backup publication: both names still reference the
                # original inode, so rollback only removes the extra backup name.
                return []
            if not (isinstance(before, Mapping) and identity_matches(path, before)):
                return [f"cannot restore relink backup because path is occupied: {path}"]
        return []
    if path_is_link:
        return []
    before = step.get("before_identity", {})
    if path_exists and isinstance(before, Mapping) and identity_matches(path, before):
        return []
    return [f"cannot determine a safe relink rollback for {path}"]


def _analyze_hardlink_rollback_steps(steps: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    for step in reversed(steps):
        step_type = str(step.get("type", "") or "")
        if step_type == "hardlink-create":
            source = Path(str(step.get("source", "")))
            target = Path(str(step.get("target", "")))
            if target.is_symlink():
                errors.append(f"hard-link target became a symlink: {target}")
            elif target.exists() and (
                source.is_symlink() or not source.is_file() or not same_inode(source, target)
            ):
                errors.append(f"hard-link target no longer points to its source: {target}")
        elif step_type == "hardlink-replace":
            errors.extend(_txn._analyze_relink_rollback_step(step))
        else:
            errors.append(f"unsupported hard-link journal step: {step_type}")
    return errors


def _analyze_break_link_rollback_step(step: Mapping[str, Any]) -> List[str]:
    path = Path(str(step.get("path", "")))
    canonical = Path(str(step.get("canonical", "")))
    backup_text = str(step.get("backup_path", "") or "")
    backup = Path(backup_text) if backup_text else None
    if canonical.is_symlink() or not canonical.is_file():
        return [f"canonical path is unavailable: {canonical}"]
    if path.is_symlink() or (backup is not None and backup.is_symlink()):
        return [f"undo rollback contains a symlink path: {path} / {backup}"]
    path_exists = path.is_file()
    path_is_link = path_exists and same_inode(path, canonical)
    backup_exists = bool(backup and backup.is_file())
    if backup_exists:
        if path_exists and not path_is_link:
            replacement = step.get("replacement_identity", {})
            digest = str(step.get("sha256", "") or "")
            owned = isinstance(replacement, Mapping) and bool(replacement) and identity_matches(path, replacement)
            if not owned and digest:
                try:
                    owned = full_file_sha256(path) == digest
                except OSError:
                    owned = False
            if not owned:
                return [f"public path was replaced during undo: {path}"]
        return []
    if path_is_link:
        return []
    if path_exists:
        digest = str(step.get("sha256", "") or "")
        if digest:
            try:
                if full_file_sha256(path) != digest:
                    return [f"independent path content changed during undo: {path}"]
            except OSError as exc:
                return [f"could not inspect independent path {path}: {exc}"]
        return []
    return [f"neither linked path nor undo backup exists: {path}"]


def _analyze_undo_hardlink_rollback_steps(steps: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    for step in reversed(steps):
        step_type = str(step.get("type", "") or "")
        if step_type == "hardlink-remove":
            canonical = Path(str(step.get("canonical", "")))
            target = Path(str(step.get("target", "")))
            if canonical.is_symlink() or not canonical.is_file():
                errors.append(f"canonical path is unavailable: {canonical}")
            elif target.is_symlink():
                errors.append(f"hard-link target became a symlink: {target}")
            elif target.exists() and not same_inode(canonical, target):
                errors.append(f"hard-link target was replaced during undo: {target}")
        elif step_type == "hardlink-break":
            errors.extend(_txn._analyze_break_link_rollback_step(step))
        else:
            errors.append(f"unsupported undo hard-link step: {step_type}")
    return errors


def recover_manifest(
    manifest_path: Path,
    apply_changes: bool = False,
    output_manifest: Optional[Path] = None,
) -> int:
    """Roll back journaled completed steps from interrupted operations."""
    try:
        manifest, journal_kind = _txn._load_recoverable_journal(manifest_path)
    except Exception as exc:
        print(f"ERROR: could not read manifest: {exc}")
        return 1

    candidates = [
        operation
        for operation in manifest.get("operations", [])
        if operation.get("status") in {"pending", "rollback_error"}
    ]
    if not candidates:
        print("No interrupted operations require recovery.")
        return 0

    recovery = _txn.manifest_header("recovery")
    recovery.update(
        manifest_id=str(uuid.uuid4()),
        created_utc=datetime.now(timezone.utc).isoformat(),
        recovery_of=str(manifest_path),
        status="in_progress" if apply_changes else "dry_run",
        operations=[],
    )
    if apply_changes:
        output_manifest = absolute_without_resolving(
            output_manifest
            or manifest_path.with_name(f"{manifest_path.stem}_recovery_{_txn._utc_stamp()}.json")
        )
        protected = _txn._journal_paths(manifest) | {normal_path_key(manifest_path)}
        if normal_path_key(output_manifest) in protected:
            print("ERROR: recovery manifest overlaps the source journal or an operation path")
            return 1
        if output_manifest.exists() or output_manifest.is_symlink():
            print(f"ERROR: recovery manifest already exists: {output_manifest}")
            return 1
        try:
            _txn._create_manifest(recovery, output_manifest)
        except Exception as exc:
            print(f"ERROR: could not create recovery manifest: {exc}")
            return 1

    failures = 0
    for operation in reversed(candidates):
        op_type = operation.get("type", "")
        steps = operation.get("steps", []) if isinstance(operation.get("steps", []), list) else []
        print(f"{'[RECOVER]' if apply_changes else '[DRY-RECOVER]'} {operation.get('record_path', '')}")
        errors: List[str] = []
        if journal_kind == "apply" and op_type in {"move", "quarantine_duplicate_group"}:
            errors = _txn._rollback_move_steps(steps) if apply_changes else _txn._analyze_move_rollback_steps(steps)
        elif journal_kind == "apply" and op_type == "hardlink_duplicate_group":
            errors = _txn._rollback_hardlink_steps(steps) if apply_changes else _txn._analyze_hardlink_rollback_steps(steps)
        elif journal_kind == "undo" and op_type == "undo_move":
            errors = _txn._rollback_move_steps(steps) if apply_changes else _txn._analyze_move_rollback_steps(steps)
        elif journal_kind == "undo" and op_type == "undo_hardlink":
            errors = (
                _txn._rollback_undo_hardlink_steps(steps)
                if apply_changes
                else _txn._analyze_undo_hardlink_rollback_steps(steps)
            )
        else:
            errors = [f"unsupported interrupted {journal_kind} operation type: {op_type}"]
        status = "recovered" if not errors else "failed"
        recovery["operations"].append(
            {
                "source_operation_id": operation.get("operation_id", ""),
                "status": status if apply_changes else ("recoverable" if not errors else "blocked"),
                "errors": errors,
            }
        )
        failures += bool(errors)
        if apply_changes:
            _txn.write_json_atomic(recovery, output_manifest)

    if apply_changes:
        recovery["status"] = "completed" if not failures else "attention_required"
        _txn.write_json_atomic(recovery, output_manifest)
        print(f"Recovery complete. failures={failures}")
        print(f"Recovery manifest: {output_manifest}")
    else:
        print(
            f"Recovery dry run complete. candidates={len(candidates)} "
            f"blocked={failures}"
        )
    return 1 if failures else 0
