"""Journalled hard-link replace and break steps with rollback."""

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

def _journaled_relink_step(
    step: MutableMapping[str, Any],
    persist: Callable[[], None],
) -> None:
    """Execute one duplicate relink while persisting every recoverable phase."""
    path = Path(str(step["path"]))
    canonical = Path(str(step["canonical"]))
    digest = str(step.get("sha256", "") or "")
    expected_identity = step.get("before_identity", {})
    if not identity_matches(path, expected_identity):
        raise ValueError(f"duplicate path changed before relink: {path}")
    if not canonical.is_file() or canonical.is_symlink():
        raise ValueError(f"canonical path is unavailable: {canonical}")
    if digest:
        if full_file_sha256(path) != digest:
            raise ValueError(f"duplicate path content changed before relink: {path}")
        if full_file_sha256(canonical) != digest:
            raise ValueError("canonical duplicate content changed before relink")
    if same_inode(path, canonical):
        step["status"] = "done"
        step["already_linked"] = True
        persist()
        return

    backup = Path(str(step.get("backup_path", "") or ""))
    if not str(step.get("backup_path", "") or ""):
        backup = _txn._unique_sibling(path, "relink-backup")
        step["backup_path"] = str(backup)
    backup_move = step.get("backup_move")
    if not isinstance(backup_move, MutableMapping):
        backup_move = {
            "type": "move",
            "from": str(path),
            "to": str(backup),
            "status": "pending",
            "method": "",
            "sha256": digest,
            "size": str(file_identity(path).size),
            "source_identity": file_identity(path).to_dict(),
            "destination_identity": {},
            "staging_path": "",
        }
        step["backup_move"] = backup_move
    step["status"] = "backup_planned"
    persist()

    def staged(staging_path: Path) -> None:
        backup_move["staging_path"] = str(staging_path)
        backup_move["status"] = "staging"
        step["status"] = "backup_staging"
        persist()

    def destination_created(identity: Mapping[str, Any]) -> None:
        backup_move["destination_identity"] = dict(identity)
        backup_move["status"] = "publishing"
        step["status"] = "backup_publishing"
        persist()

    def published(result: Mapping[str, Any]) -> None:
        backup_move.update(result)
        backup_move["status"] = "published"
        step["status"] = "backup_published"
        persist()

    result = _txn.safe_move(
        path,
        backup,
        expected_sha256=digest,
        on_staged=staged,
        on_destination_created=destination_created,
        on_published=published,
    )
    backup_move.update(result)
    backup_move["status"] = "done"
    step["status"] = "backup_moved"
    persist()

    _txn.atomic_hardlink(canonical, path)
    step["replacement_identity"] = file_identity(path).to_dict()
    step["status"] = "linked"
    persist()
    if not same_inode(path, canonical):
        raise RuntimeError("relinked path does not share the canonical inode")

    backup.unlink()
    _txn._fsync_directory(backup.parent)
    step["backup_removed"] = True
    step["status"] = "done"
    persist()


def _rollback_relink_step(step: MutableMapping[str, Any]) -> List[str]:
    """Restore one journaled relink step to an independent public path."""
    errors: List[str] = []
    path = Path(str(step.get("path", "")))
    canonical = Path(str(step.get("canonical", "")))
    digest = str(step.get("sha256", "") or "")
    backup_text = str(step.get("backup_path", "") or "")
    backup = Path(backup_text) if backup_text else None

    try:
        path_is_link = (
            path.exists()
            and not path.is_symlink()
            and canonical.exists()
            and not canonical.is_symlink()
            and same_inode(path, canonical)
        )
        backup_exists = bool(backup and backup.exists() and not backup.is_symlink())
        if backup is not None and backup.is_symlink():
            errors.append(f"relink backup became a symlink: {backup}")
            return errors

        if backup_exists:
            if path_is_link:
                path.unlink()
                _txn._fsync_directory(path.parent)
            elif path.exists() or path.is_symlink():
                before_identity = step.get("before_identity", {})
                if (
                    not path.is_symlink()
                    and same_inode(path, backup)
                    and isinstance(before_identity, Mapping)
                    and _txn._inode_unchanged(path, before_identity)
                ):
                    # Backup publication was interrupted before the old name was
                    # unlinked: both names reference the original inode (only its
                    # ctime moved when the backup link was created). Removing the
                    # extra backup name restores the pre-operation tree.
                    backup.unlink()
                    _txn._fsync_directory(backup.parent)
                    step["status"] = "rolled_back"
                    return errors
                if isinstance(before_identity, Mapping) and identity_matches(path, before_identity):
                    # The original path is already restored. The private backup is
                    # another link/copy of the same original and may be removed.
                    if _txn._same_regular_content(path, backup):
                        backup.unlink()
                        _txn._fsync_directory(backup.parent)
                        step["status"] = "rolled_back"
                        return errors
                errors.append(f"cannot restore relink backup because path is occupied: {path}")
                return errors
            errors.extend(_txn._restore_backup(path, backup, digest))
            if not errors:
                step["status"] = "rolled_back"
            return errors

        if path_is_link:
            # The private backup was already removed. Re-create an independent
            # byte-identical inode, which is the strongest reversible state left.
            _txn.break_hardlink(path, canonical=canonical, expected_sha256=digest)
            step["status"] = "rolled_back"
            return errors

        before_identity = step.get("before_identity", {})
        if path.exists() and not path.is_symlink() and isinstance(before_identity, Mapping):
            if identity_matches(path, before_identity):
                step["status"] = "rolled_back"
                return errors
        errors.append(f"cannot determine a safe relink rollback for {path}")
    except Exception as exc:
        errors.append(f"could not roll back relink {path}: {exc}")
    return errors


def _journaled_break_link_step(
    step: MutableMapping[str, Any],
    persist: Callable[[], None],
) -> None:
    """Break one canonical hard link into an independent copy with a journal."""
    path = Path(str(step["path"]))
    canonical = Path(str(step["canonical"]))
    digest = str(step.get("sha256", "") or "")
    expected_identity = step.get("linked_identity", {})
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"hard-link path is unavailable: {path}")
    if canonical.is_symlink() or not canonical.is_file():
        raise ValueError(f"canonical path is unavailable: {canonical}")
    if expected_identity and not _txn._matches_stable_identity(path, expected_identity):
        raise ValueError(f"hard-link path changed before undo: {path}")
    if not same_inode(path, canonical):
        raise ValueError(f"hard-link path no longer shares the canonical inode: {path}")
    if digest and full_file_sha256(path) != digest:
        raise ValueError(f"hard-link content changed before undo: {path}")

    staging_text = str(step.get("staging_path", "") or "")
    if staging_text:
        staging = Path(staging_text)
    else:
        def staged_created(temp_path: Path) -> None:
            step["staging_path"] = str(temp_path)
            step["status"] = "copy_staging"
            persist()

        staging = _txn._copy_to_temp(path, path.parent, on_staged=staged_created)
        if digest and full_file_sha256(staging) != digest:
            raise IOError("independent-copy staging SHA-256 mismatch")
        step["staging_identity"] = file_identity(staging).to_dict()
        step["status"] = "copy_staged"
        persist()

    backup_text = str(step.get("backup_path", "") or "")
    backup = Path(backup_text) if backup_text else _txn._unique_sibling(path, "undo-break-backup")
    if not backup_text:
        step["backup_path"] = str(backup)

    backup_move = step.get("backup_move")
    if not isinstance(backup_move, MutableMapping):
        backup_move = {
            "type": "move",
            "from": str(path),
            "to": str(backup),
            "status": "pending",
            "method": "",
            "sha256": digest,
            "size": str(file_identity(path).size),
            "source_identity": file_identity(path).to_dict(),
            "destination_identity": {},
            "staging_path": "",
        }
        step["backup_move"] = backup_move
    step["status"] = "backup_planned"
    persist()

    def backup_staged(temp_path: Path) -> None:
        backup_move["staging_path"] = str(temp_path)
        backup_move["status"] = "staging"
        step["status"] = "backup_staging"
        persist()

    def backup_destination_created(identity: Mapping[str, Any]) -> None:
        backup_move["destination_identity"] = dict(identity)
        backup_move["status"] = "publishing"
        step["status"] = "backup_publishing"
        persist()

    def backup_published(result: Mapping[str, Any]) -> None:
        backup_move.update(result)
        backup_move["status"] = "published"
        step["status"] = "backup_published"
        persist()

    backup_result = _txn.safe_move(
        path,
        backup,
        expected_sha256=digest,
        on_staged=backup_staged,
        on_destination_created=backup_destination_created,
        on_published=backup_published,
    )
    backup_move.update(backup_result)
    backup_move["status"] = "done"
    step["status"] = "backup_moved"
    persist()

    def independent_destination_created(identity: Mapping[str, Any]) -> None:
        step["replacement_identity"] = dict(identity)
        step["status"] = "copy_publishing"
        persist()

    publish_method = _txn._publish_staging_no_replace(
        staging,
        path,
        on_destination_created=independent_destination_created,
    )
    step["publish_method"] = publish_method
    step["replacement_identity"] = file_identity(path).to_dict()
    step["staging_path"] = ""
    step["status"] = "independent_published"
    persist()
    if digest and full_file_sha256(path) != digest:
        raise IOError("independent hard-link copy SHA-256 mismatch")
    if same_inode(path, canonical):
        raise RuntimeError("hard link was not broken")

    backup.unlink()
    _txn._fsync_directory(backup.parent)
    step["backup_removed"] = True
    step["status"] = "done"
    persist()


def _rollback_break_link_step(step: MutableMapping[str, Any]) -> List[str]:
    """Restore an interrupted hard-link break to the original linked state."""
    errors: List[str] = []
    path = Path(str(step.get("path", "")))
    canonical = Path(str(step.get("canonical", "")))
    digest = str(step.get("sha256", "") or "")
    backup_text = str(step.get("backup_path", "") or "")
    backup = Path(backup_text) if backup_text else None

    try:
        if canonical.is_symlink() or not canonical.is_file():
            errors.append(f"canonical path is unavailable: {canonical}")
            return errors
        path_exists = path.exists() and not path.is_symlink()
        path_is_link = path_exists and same_inode(path, canonical)
        backup_exists = bool(backup and backup.exists() and not backup.is_symlink())
        if backup is not None and backup.is_symlink():
            errors.append(f"undo backup became a symlink: {backup}")
            return errors

        if backup_exists:
            if path_is_link:
                # The apply state already exists. The backup is another link to the
                # same inode and can be discarded after a content check.
                if not _txn._same_regular_content(path, backup):
                    errors.append(f"undo backup content disagrees with linked path: {backup}")
                    return errors
                backup.unlink()
                _txn._fsync_directory(backup.parent)
            elif path_exists:
                replacement_identity = step.get("replacement_identity", {})
                owned = (
                    isinstance(replacement_identity, Mapping)
                    and bool(replacement_identity)
                    and identity_matches(path, replacement_identity)
                )
                if not owned and digest:
                    owned = full_file_sha256(path) == digest
                if not owned:
                    errors.append(f"public path was replaced during undo: {path}")
                    return errors
                path.unlink()
                _txn._fsync_directory(path.parent)
                errors.extend(_txn._restore_backup(path, backup, digest))
            else:
                errors.extend(_txn._restore_backup(path, backup, digest))
        elif path_is_link:
            pass
        elif path_exists:
            if digest and full_file_sha256(path) != digest:
                errors.append(f"independent path content changed during undo: {path}")
                return errors
            _txn.atomic_relink(
                path,
                canonical,
                expected_identity=file_identity(path).to_dict(),
                expected_sha256=digest,
            )
        else:
            errors.append(f"neither linked path nor undo backup exists: {path}")
            return errors

        if path.exists() and not same_inode(path, canonical):
            errors.append(f"rollback did not restore canonical hard link: {path}")
        if not errors:
            step["status"] = "rolled_back"
    except Exception as exc:
        errors.append(f"could not roll back hard-link break {path}: {exc}")
    finally:
        staging_text = str(step.get("staging_path", "") or "")
        if staging_text:
            staging = Path(staging_text)
            try:
                if staging.exists() and not staging.is_symlink():
                    staging.unlink()
                    _txn._fsync_directory(staging.parent)
                elif staging.is_symlink():
                    errors.append(f"undo staging path became a symlink: {staging}")
            except Exception as exc:
                errors.append(f"could not remove undo staging file {staging}: {exc}")
    return errors
