"""Manifest and journal creation, persistence, and loading."""

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

def _default_manifest_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_manifest_{_txn._utc_stamp()}.json")


def _new_manifest(
    csv_path: Path,
    base_dir: Path,
    *,
    allow_outside_base: bool,
    allow_symlink_parents: bool,
) -> Dict[str, Any]:
    manifest = _txn.manifest_header("apply")
    manifest.update(
        manifest_id=str(uuid.uuid4()),
        created_utc=datetime.now(timezone.utc).isoformat(),
        updated_utc=datetime.now(timezone.utc).isoformat(),
        status="in_progress",
        base_dir=str(base_dir),
        base_resolved=str(base_dir.resolve(strict=False)),
        allow_outside_base=bool(allow_outside_base),
        allow_symlink_parents=bool(allow_symlink_parents),
        source_csv=str(csv_path),
        operations=[],
    )
    return manifest


def _create_manifest(manifest: MutableMapping[str, Any], path: Path) -> None:
    """Create the first journal image without replacing a race winner."""
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _txn.write_json_exclusive(manifest, path)


def _save_manifest(manifest: MutableMapping[str, Any], path: Path) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _txn.write_json_atomic(manifest, path)


def _protected_manifest_paths(plan: Sequence[Mapping[str, str]], csv_path: Path) -> Set[str]:
    protected = {normal_path_key(csv_path)}
    for record in plan:
        text = str(record.get("_transfers", "") or "")
        try:
            parsed = json.loads(text) if text else []
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    protected.add(normal_path_key(Path(str(item[0]))))
                    protected.add(normal_path_key(Path(str(item[1]))))
    return protected


def _load_manifest(
    path: Path,
    *,
    expected_types: Sequence[str] = ("apply",),
) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("manifest root is not an object")
    if str(data.get("schema_name", "")) != MANIFEST_SCHEMA_NAME:
        raise ValueError(
            "unrecognised manifest schema name %r" % data.get("schema_name")
        )
    if str(data.get("schema_version", "")) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported manifest schema %r; expected %s"
            % (data.get("schema_version"), MANIFEST_SCHEMA_VERSION)
        )
    manifest_type = str(data.get("manifest_type", "") or "")
    if expected_types and manifest_type not in set(expected_types):
        raise ValueError(
            "manifest type %r is not valid for this operation; expected %s"
            % (manifest_type, ", ".join(expected_types))
        )
    if not isinstance(data.get("operations", []), list):
        raise ValueError("manifest operations field is invalid")
    return data


def _journal_paths(payload: Mapping[str, Any]) -> Set[str]:
    """Return all public paths mentioned by a manifest or undo journal."""
    paths: Set[str] = set()
    for top_level in ("source_csv", "undo_of", "recovery_of"):
        value = str(payload.get(top_level, "") or "")
        if value:
            paths.add(normal_path_key(Path(value)))
    for operation in payload.get("operations", []):
        if not isinstance(operation, Mapping):
            continue
        for step in operation.get("steps", []):
            if not isinstance(step, Mapping):
                continue
            for field in (
                "from", "to", "source", "target", "path", "canonical",
                "backup_path", "staging_path",
            ):
                value = str(step.get(field, "") or "")
                if value:
                    paths.add(normal_path_key(Path(value)))
            backup_move = step.get("backup_move", {})
            if isinstance(backup_move, Mapping):
                for field in ("from", "to", "staging_path"):
                    value = str(backup_move.get(field, "") or "")
                    if value:
                        paths.add(normal_path_key(Path(value)))
    return paths


def _load_undo_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("undo manifest root is not an object")
    if str(data.get("schema_name", "")) != UNDO_SCHEMA_NAME:
        raise ValueError("unrecognised undo schema name %r" % data.get("schema_name"))
    if str(data.get("schema_version", "")) != UNDO_SCHEMA_VERSION:
        raise ValueError(
            "unsupported undo schema %r; expected %s"
            % (data.get("schema_version"), UNDO_SCHEMA_VERSION)
        )
    if str(data.get("manifest_type", "") or "") != "undo":
        raise ValueError("undo manifest type is not 'undo'")
    if not isinstance(data.get("operations", []), list):
        raise ValueError("undo manifest operations field is invalid")
    return data


def _load_recoverable_journal(path: Path) -> Tuple[Dict[str, Any], str]:
    with path.open("r", encoding="utf-8") as handle:
        probe = json.load(handle)
    if not isinstance(probe, dict):
        raise ValueError("journal root is not an object")
    schema_name = str(probe.get("schema_name", "") or "")
    if schema_name == MANIFEST_SCHEMA_NAME:
        return _txn._load_manifest(path, expected_types=("apply",)), "apply"
    if schema_name == UNDO_SCHEMA_NAME:
        return _txn._load_undo_manifest(path), "undo"
    raise ValueError(f"unrecognised recoverable journal schema {schema_name!r}")
