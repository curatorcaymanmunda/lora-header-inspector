"""Plan application: move, hard-link, and quarantine operations."""

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

def _preflight_transfers(transfers: Sequence[Tuple[Path, Path]]) -> Tuple[bool, str]:
    sources: Set[str] = set()
    destinations: Set[str] = set()
    for source, destination in transfers:
        source_key = normal_path_key(source)
        destination_key = normal_path_key(destination)
        if source_key in sources:
            return False, f"duplicate source in transaction: {source}"
        if destination_key in destinations:
            return False, f"duplicate destination in transaction: {destination}"
        sources.add(source_key)
        destinations.add(destination_key)
        if source.is_symlink() or not source.is_file():
            return False, f"source missing or is a symlink: {source}"
        if destination.exists() or destination.is_symlink():
            return False, f"destination already exists: {destination}"
    return True, ""


def _current_regular_transfers(record: Mapping[str, str]) -> Tuple[List[Tuple[Path, Path]], str]:
    source = Path(record["path"])
    destination = Path(record["destination"])
    valid, reason, sidecars = verify_sidecar_snapshot(
        source,
        record.get("sidecar_fingerprints_json", ""),
        require_hash=True,
    )
    if not valid:
        return [], reason
    transfers = [(item, destination.parent / item.name) for item in sidecars]
    transfers.append((source, destination))
    return transfers, ""


def _records_for_group(plan: Sequence[Mapping[str, str]], group_id: str) -> List[Mapping[str, str]]:
    return [record for record in plan if record.get("collision_group") == group_id]


def _verify_group_records(
    records: Sequence[Mapping[str, str]], cache: HashCache, expected_sha256: str
) -> Tuple[bool, str]:
    for record in records:
        source = Path(record.get("path", ""))
        valid, reason = verify_record_fingerprint(record, source, cache=cache)
        if not valid:
            return False, f"{source}: {reason}"
        if expected_sha256:
            try:
                digest = full_file_sha256(source, cache)
            except Exception as exc:
                return False, f"{source}: could not rehash duplicate member: {exc}"
            if digest != expected_sha256:
                return False, f"{source}: duplicate content changed after planning"
        valid_sidecars, sidecar_reason, _ = verify_sidecar_snapshot(
            source,
            record.get("sidecar_fingerprints_json", ""),
            require_hash=True,
        )
        if not valid_sidecars:
            return False, f"{source}: {sidecar_reason}"
    return True, ""


def _same_regular_content(first: Path, second: Path) -> bool:
    try:
        if first.is_symlink() or second.is_symlink():
            return False
        if not first.is_file() or not second.is_file():
            return False
        if same_inode(first, second):
            return True
        if first.stat().st_size != second.stat().st_size:
            return False
        return full_file_sha256(first) == full_file_sha256(second)
    except Exception:
        return False


def _remove_staging(step: MutableMapping[str, Any], errors: List[str]) -> None:
    raw = str(step.get("staging_path", "") or "")
    if not raw:
        return
    staging = Path(raw)
    try:
        if staging.exists() and not staging.is_symlink():
            staging.unlink()
            _txn._fsync_directory(staging.parent)
        elif staging.is_symlink():
            errors.append("staging path became a symlink: %s" % staging)
    except Exception as exc:
        errors.append("could not remove staging file %s: %s" % (staging, exc))


def _rollback_move_steps(
    steps: Sequence[MutableMapping[str, Any]],
) -> List[str]:
    """Restore pre-operation names, including a process killed mid-step."""
    errors: List[str] = []
    for step in reversed(steps):
        status = str(step.get("status", "pending") or "pending")
        if status in {"rolled_back", "not_started"}:
            _txn._remove_staging(step, errors)
            continue
        source = Path(str(step.get("from", "")))
        destination = Path(str(step.get("to", "")))
        if not str(source) or not str(destination):
            errors.append("move step is missing a source or destination")
            continue
        try:
            source_exists = source.exists() and not source.is_symlink()
            destination_exists = destination.exists() and not destination.is_symlink()
            if source.is_symlink() or destination.is_symlink():
                errors.append(
                    "rollback refused a symlink path: %s -> %s" % (destination, source)
                )
                continue

            if source_exists and destination_exists:
                # The process may have published a no-overwrite link/copy but
                # died before unlinking the old name or marking the step done.
                recorded_destination = step.get("destination_identity", {})
                owned_partial = (
                    isinstance(recorded_destination, Mapping)
                    and bool(recorded_destination)
                    and identity_matches(destination, recorded_destination)
                )
                if not _txn._same_regular_content(source, destination) and not owned_partial:
                    errors.append(
                        "both rollback paths exist with different content: %s and %s"
                        % (source, destination)
                    )
                    continue
                destination.unlink()
                _txn._fsync_directory(destination.parent)
                step["status"] = "rolled_back"
            elif not source_exists and destination_exists:
                source.parent.mkdir(parents=True, exist_ok=True)
                _txn.safe_move(
                    destination,
                    source,
                    expected_sha256=str(step.get("sha256", "") or ""),
                )
                step["status"] = "rolled_back"
            elif source_exists and not destination_exists:
                # No published mutation remains.  A staging file may still do.
                step["status"] = "rolled_back"
            else:
                errors.append(
                    "neither rollback source nor original path exists: %s / %s"
                    % (destination, source)
                )
                continue
        except Exception as exc:
            errors.append("%s -> %s: %s" % (destination, source, exc))
        finally:
            _txn._remove_staging(step, errors)
    return errors


def _apply_move_operation(
    record: MutableMapping[str, str],
    operation: MutableMapping[str, Any],
    manifest: MutableMapping[str, Any],
    manifest_path: Path,
    cache: HashCache,
) -> Tuple[bool, List[str]]:
    source = Path(record["path"])
    valid, reason = verify_record_fingerprint(record, source, cache=cache)
    if not valid:
        record.update(move="False", action="skip_stale_or_unverified", reason=reason, verified="False")
        operation.update(status="skipped", error=reason)
        _txn._save_manifest(manifest, manifest_path)
        return False, []
    transfers, error = _txn._current_regular_transfers(record)
    if error:
        record.update(move="False", action="skip_stale_sidecars", reason=error)
        operation.update(status="skipped", error=error)
        _txn._save_manifest(manifest, manifest_path)
        return False, []
    ready, error = _txn._preflight_transfers(transfers)
    if not ready:
        record.update(move="False", action="skip_preflight", reason=error)
        operation.update(status="skipped", error=error)
        _txn._save_manifest(manifest, manifest_path)
        return False, []

    operation["steps"] = [
        {
            "type": "move",
            "from": str(src),
            "to": str(dst),
            "status": "pending",
            "method": "",
            "sha256": "",
            "size": str(file_identity(src).size),
            "source_identity": file_identity(src).to_dict(),
            "destination_identity": {},
            "staging_path": "",
        }
        for src, dst in transfers
    ]
    _txn._save_manifest(manifest, manifest_path)
    try:
        for step in operation["steps"]:
            def staged(path: Path, current=step) -> None:
                current["staging_path"] = str(path)
                current["status"] = "staging"
                _txn._save_manifest(manifest, manifest_path)

            def published(result: Mapping[str, Any], current=step) -> None:
                current.update(result)
                current["status"] = "published"
                _txn._save_manifest(manifest, manifest_path)

            def destination_created(identity: Mapping[str, Any], current=step) -> None:
                current["destination_identity"] = dict(identity)
                current["status"] = "publishing"
                _txn._save_manifest(manifest, manifest_path)

            result = _txn.safe_move(
                Path(step["from"]),
                Path(step["to"]),
                on_staged=staged,
                on_destination_created=destination_created,
                on_published=published,
            )
            step.update(status="done", **result)
            _txn._save_manifest(manifest, manifest_path)
    except Exception as exc:
        rollback_errors = _txn._rollback_move_steps(operation["steps"])
        operation.update(
            status="rollback_error" if rollback_errors else "rolled_back",
            error=str(exc),
            rollback_errors=rollback_errors,
        )
        _txn._save_manifest(manifest, manifest_path)
        return False, rollback_errors

    operation["status"] = "completed"
    record.update(action="moved", reason="transaction completed", verified="True")
    _txn._save_manifest(manifest, manifest_path)
    return True, []


def _rollback_hardlink_steps(
    steps: Sequence[MutableMapping[str, Any]],
) -> List[str]:
    """Undo hard-link steps, including a process killed before status persistence."""
    errors: List[str] = []
    for step in reversed(steps):
        status = str(step.get("status", "pending") or "pending")
        if status in {"rolled_back", "not_started"}:
            continue
        step_type = str(step.get("type", "") or "")
        try:
            if step_type == "hardlink-create":
                source = Path(str(step.get("source", "")))
                target = Path(str(step.get("target", "")))
                if target.is_symlink():
                    errors.append("hard-link target became a symlink: %s" % target)
                    continue
                if not target.exists():
                    step["status"] = "rolled_back"
                    continue
                if not source.exists() or source.is_symlink():
                    errors.append(
                        "cannot remove created hard link because its source is unavailable: %s"
                        % source
                    )
                    continue
                if not same_inode(source, target):
                    errors.append(
                        "hard-link target no longer points to its recorded source: %s"
                        % target
                    )
                    continue
                target.unlink()
                _txn._fsync_directory(target.parent)
                step["status"] = "rolled_back"

            elif step_type == "hardlink-replace":
                errors.extend(_txn._rollback_relink_step(step))
            else:
                errors.append("unsupported hard-link journal step: %s" % step_type)
        except Exception as exc:
            errors.append("%s: %s" % (step, exc))
    return errors


def _apply_hardlink_operation(
    record: MutableMapping[str, str],
    group_records: Sequence[Mapping[str, str]],
    spec: Mapping[str, Any],
    operation: MutableMapping[str, Any],
    manifest: MutableMapping[str, Any],
    manifest_path: Path,
    cache: HashCache,
) -> Tuple[bool, List[str]]:
    digest = str(spec.get("content_sha256", ""))
    valid, reason = _txn._verify_group_records(group_records, cache, digest)
    if not valid:
        record.update(move="False", action="skip_stale_duplicate_group", reason=reason)
        operation.update(status="skipped", error=reason)
        _txn._save_manifest(manifest, manifest_path)
        return False, []

    canonical = Path(spec["canonical_path"])
    if not canonical.is_file() or canonical.is_symlink():
        reason = f"canonical path is unavailable: {canonical}"
        operation.update(status="skipped", error=reason)
        record.update(move="False", action="skip_preflight", reason=reason)
        _txn._save_manifest(manifest, manifest_path)
        return False, []
    if digest and full_file_sha256(canonical, cache) != digest:
        reason = "canonical duplicate content changed after planning"
        operation.update(status="skipped", error=reason)
        record.update(move="False", action="skip_stale_duplicate_group", reason=reason)
        _txn._save_manifest(manifest, manifest_path)
        return False, []

    steps: List[Dict[str, Any]] = []
    for item in spec.get("links", []):
        source = Path(item["source"])
        target = Path(item["target"])
        if not identity_matches(source, item["source_identity"]):
            reason = f"hard-link source changed after planning: {source}"
            operation.update(status="skipped", error=reason)
            record.update(move="False", action="skip_stale_duplicate_group", reason=reason)
            _txn._save_manifest(manifest, manifest_path)
            return False, []
        if target.exists() or target.is_symlink():
            reason = f"hard-link target appeared after planning: {target}"
            operation.update(status="skipped", error=reason)
            record.update(move="False", action="skip_preflight", reason=reason)
            _txn._save_manifest(manifest, manifest_path)
            return False, []
        steps.append(
            {
                "type": "hardlink-create",
                "source": str(source),
                "target": str(target),
                "status": "pending",
                "sha256": item.get("source_sha256", ""),
            }
        )
    for item in spec.get("replacements", []):
        path = Path(item["path"])
        if not identity_matches(path, item["source_identity"]):
            reason = f"duplicate path changed after planning: {path}"
            operation.update(status="skipped", error=reason)
            record.update(move="False", action="skip_stale_duplicate_group", reason=reason)
            _txn._save_manifest(manifest, manifest_path)
            return False, []
        if digest and full_file_sha256(path, cache) != digest:
            reason = f"duplicate path content changed after planning: {path}"
            operation.update(status="skipped", error=reason)
            record.update(move="False", action="skip_stale_duplicate_group", reason=reason)
            _txn._save_manifest(manifest, manifest_path)
            return False, []
        steps.append(
            {
                "type": "hardlink-replace",
                "path": str(path),
                "canonical": str(canonical),
                "status": "pending",
                "before_identity": item["source_identity"],
                "sha256": digest,
                "backup_path": "",
                "backup_move": {},
                "replacement_identity": {},
                "backup_removed": False,
            }
        )

    operation["steps"] = steps
    _txn._save_manifest(manifest, manifest_path)
    try:
        for step in steps:
            if step["type"] == "hardlink-create":
                _txn.atomic_hardlink(Path(step["source"]), Path(step["target"]))
                step["target_identity"] = file_identity(Path(step["target"])).to_dict()
                step["status"] = "done"
                _txn._save_manifest(manifest, manifest_path)
            else:
                _txn._journaled_relink_step(
                    step,
                    lambda: _txn._save_manifest(manifest, manifest_path),
                )
    except Exception as exc:
        rollback_errors = _txn._rollback_hardlink_steps(steps)
        operation.update(
            status="rollback_error" if rollback_errors else "rolled_back",
            error=str(exc),
            rollback_errors=rollback_errors,
        )
        _txn._save_manifest(manifest, manifest_path)
        return False, rollback_errors

    operation["status"] = "completed"
    record.update(action="hardlinked_duplicate_group", reason="duplicate group hard-linked", verified="True")
    _txn._save_manifest(manifest, manifest_path)
    return True, []


def _apply_quarantine_operation(
    record: MutableMapping[str, str],
    group_records: Sequence[Mapping[str, str]],
    spec: Mapping[str, Any],
    operation: MutableMapping[str, Any],
    manifest: MutableMapping[str, Any],
    manifest_path: Path,
    cache: HashCache,
    allow_path_breakage: bool,
) -> Tuple[bool, List[str]]:
    if not allow_path_breakage:
        reason = "quarantine changes existing model paths; pass --allow-workflow-path-breakage"
        operation.update(status="skipped", error=reason)
        record.update(move="False", action="skip_path_breakage_not_acknowledged", reason=reason)
        _txn._save_manifest(manifest, manifest_path)
        return False, []
    digest = str(spec.get("content_sha256", ""))
    valid, reason = _txn._verify_group_records(group_records, cache, digest)
    if not valid:
        operation.update(status="skipped", error=reason)
        record.update(move="False", action="skip_stale_duplicate_group", reason=reason)
        _txn._save_manifest(manifest, manifest_path)
        return False, []
    transfers = [(Path(one), Path(two)) for one, two in spec.get("transfers", [])]
    ready, reason = _txn._preflight_transfers(transfers)
    if not ready:
        operation.update(status="skipped", error=reason)
        record.update(move="False", action="skip_preflight", reason=reason)
        _txn._save_manifest(manifest, manifest_path)
        return False, []
    operation["steps"] = [
        {
            "type": "move",
            "from": str(src),
            "to": str(dst),
            "status": "pending",
            "method": "",
            "sha256": "",
            "size": str(file_identity(src).size),
            "source_identity": file_identity(src).to_dict(),
            "destination_identity": {},
            "staging_path": "",
        }
        for src, dst in transfers
    ]
    _txn._save_manifest(manifest, manifest_path)
    try:
        for step in operation["steps"]:
            def staged(path: Path, current=step) -> None:
                current["staging_path"] = str(path)
                current["status"] = "staging"
                _txn._save_manifest(manifest, manifest_path)

            def published(result: Mapping[str, Any], current=step) -> None:
                current.update(result)
                current["status"] = "published"
                _txn._save_manifest(manifest, manifest_path)

            def destination_created(identity: Mapping[str, Any], current=step) -> None:
                current["destination_identity"] = dict(identity)
                current["status"] = "publishing"
                _txn._save_manifest(manifest, manifest_path)

            result = _txn.safe_move(
                Path(step["from"]),
                Path(step["to"]),
                on_staged=staged,
                on_destination_created=destination_created,
                on_published=published,
            )
            step.update(status="done", **result)
            _txn._save_manifest(manifest, manifest_path)
    except Exception as exc:
        rollback_errors = _txn._rollback_move_steps(operation["steps"])
        operation.update(
            status="rollback_error" if rollback_errors else "rolled_back",
            error=str(exc),
            rollback_errors=rollback_errors,
        )
        _txn._save_manifest(manifest, manifest_path)
        return False, rollback_errors
    operation["status"] = "completed"
    record.update(action="quarantined_duplicate_group", reason="duplicate group normalized and quarantined", verified="True")
    _txn._save_manifest(manifest, manifest_path)
    return True, []


def apply_plan(
    plan: Sequence[MutableMapping[str, str]],
    csv_path: Path,
    base_dir: Path,
    dry_run: bool = True,
    allow_unverified: bool = False,
    manifest_path: Optional[Path] = None,
    allow_workflow_path_breakage: bool = False,
    allow_outside_base: bool = False,
    allow_symlink_parents: bool = False,
) -> Dict[str, int]:
    counts = {"moved": 0, "skipped": 0, "errors": 0, "rolled_back": 0}
    if dry_run:
        for record in plan:
            source = Path(record.get("path", ""))
            action = record.get("action", "unknown")
            if action in {"move", "hardlink_duplicate_group", "quarantine_duplicate_group"} and str(record.get("move", "")).lower() == "true":
                print(f"[DRY] {action:28s} {source} -> {record.get('destination', '')}")
            else:
                if action != "already_in_place":
                    print(f"[SKIP] {action:28s} {source.name}  {record.get('reason', '')}")
                counts["skipped"] += 1
        print("\nDry run complete. No files were moved or relinked.")
        return counts

    base_dir = absolute_without_resolving(base_dir)
    if base_dir.exists() or base_dir.is_symlink():
        if not base_dir.is_dir():
            print(f"ERROR: --base exists but is not a directory: {base_dir}")
            counts["errors"] += 1
            return counts
    else:
        # The explicit --apply flag and explicit --base value are the user's
        # authorization to create the bucket root. Individual operation journals
        # still cover every file mutation below it.
        try:
            base_dir.mkdir(parents=True, exist_ok=False)
            _txn._fsync_directory(base_dir.parent)
        except Exception as exc:
            print(f"ERROR: could not create --base directory {base_dir}: {exc}")
            counts["errors"] += 1
            return counts

    manifest_path = absolute_without_resolving(manifest_path or _txn._default_manifest_path(csv_path))
    if normal_path_key(manifest_path) in _txn._protected_manifest_paths(plan, csv_path):
        print("ERROR: manifest path overlaps the CSV or a planned transfer")
        counts["errors"] += 1
        return counts
    if manifest_path.exists() or manifest_path.is_symlink():
        print(f"ERROR: manifest already exists and will not be overwritten: {manifest_path}")
        counts["errors"] += 1
        return counts

    manifest = _txn._new_manifest(
        csv_path,
        base_dir,
        allow_outside_base=allow_outside_base,
        allow_symlink_parents=allow_symlink_parents,
    )
    try:
        _txn._create_manifest(manifest, manifest_path)
    except Exception as exc:
        print(f"ERROR: could not create apply manifest: {exc}")
        counts["errors"] += 1
        return counts
    cache = HashCache()

    for record in plan:
        action = record.get("action", "")
        source = Path(record.get("path", ""))
        if action not in {"move", "hardlink_duplicate_group", "quarantine_duplicate_group"} or str(record.get("move", "")).lower() != "true":
            if action != "already_in_place":
                print(f"[SKIP] {action:28s} {source.name}  {record.get('reason', '')}")
            counts["skipped"] += 1
            continue

        operation: Dict[str, Any] = {
            "operation_id": str(uuid.uuid4()),
            "record_path": str(source),
            "type": action,
            "status": "pending",
            "error": "",
            "rollback_errors": [],
            "steps": [],
        }
        manifest["operations"].append(operation)
        _txn._save_manifest(manifest, manifest_path)
        print(f"[APPLY] {action:28s} {source}")

        paths_ok, path_reason = verify_path_safety_snapshot(
            record.get("path_safety_json", ""),
            base=base_dir,
            allow_outside=allow_outside_base,
            allow_symlink_parents=allow_symlink_parents,
        )
        if not paths_ok:
            record.update(
                move="False",
                action="skip_path_safety_changed",
                reason=path_reason,
                verified="False",
            )
            operation.update(status="skipped", error=path_reason)
            _txn._save_manifest(manifest, manifest_path)
            print(f"       skipped: {path_reason}")
            counts["skipped"] += 1
            continue

        if action == "move":
            success, rollback_errors = _txn._apply_move_operation(record, operation, manifest, manifest_path, cache)
        else:
            try:
                spec = json.loads(record.get("_duplicate_spec", "{}") or "{}")
            except json.JSONDecodeError as exc:
                success, rollback_errors = False, [f"invalid duplicate spec: {exc}"]
                operation.update(status="rollback_error", error=str(exc), rollback_errors=rollback_errors)
                _txn._save_manifest(manifest, manifest_path)
            else:
                group_records = _txn._records_for_group(plan, record.get("collision_group", ""))
                if action == "hardlink_duplicate_group":
                    success, rollback_errors = _txn._apply_hardlink_operation(
                        record, group_records, spec, operation, manifest, manifest_path, cache
                    )
                else:
                    success, rollback_errors = _txn._apply_quarantine_operation(
                        record,
                        group_records,
                        spec,
                        operation,
                        manifest,
                        manifest_path,
                        cache,
                        allow_workflow_path_breakage,
                    )

        if success:
            counts["moved"] += 1
        elif rollback_errors:
            counts["errors"] += 1
        elif operation.get("status") == "rolled_back":
            counts["rolled_back"] += 1
        else:
            counts["skipped"] += 1

    if any(operation.get("status") in {"pending", "rollback_error"} for operation in manifest["operations"]):
        manifest["status"] = "attention_required"
    elif any(operation.get("status") == "rolled_back" for operation in manifest["operations"]):
        manifest["status"] = "completed_with_rollbacks"
    else:
        manifest["status"] = "completed"
    _txn._save_manifest(manifest, manifest_path)
    print(
        "\nApply complete. "
        f"completed={counts['moved']} skipped={counts['skipped']} "
        f"rolled_back={counts['rolled_back']} errors={counts['errors']}"
    )
    print(f"Manifest: {manifest_path}")
    return counts
