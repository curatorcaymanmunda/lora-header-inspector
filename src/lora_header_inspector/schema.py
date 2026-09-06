"""Versioned CSV and JSON schema helpers.

Inspection indexes and reviewed sort plans are different contracts.  A plan can
be regenerated from an index, but an unsupported version is never silently
reinterpreted as the current one.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .utils import (
    absolute_without_resolving,
    atomic_json_write,
    create_json_exclusive,
    fsync_directory,
)
from .version import (
    COLLISION_SCHEMA_NAME,
    COLLISION_SCHEMA_VERSION,
    INSPECTION_SCHEMA_NAME,
    INSPECTION_SCHEMA_VERSION,
    MANIFEST_SCHEMA_NAME,
    MANIFEST_SCHEMA_VERSION,
    PLAN_SCHEMA_NAME,
    PLAN_SCHEMA_VERSION,
    TOOL_VERSION,
    UNDO_SCHEMA_NAME,
    UNDO_SCHEMA_VERSION,
)


class SchemaError(ValueError):
    """Raised when an on-disk artifact cannot be interpreted safely."""


INSPECTION_FIELDS = [
    "schema_name", "schema_version", "tool_version", "record_id",
    "input_path", "path", "source_was_symlink",
    "file_size", "mtime_ns", "ctime_ns", "device", "inode",
    "header_sha256", "full_sha256", "header_bytes", "payload_bytes",
    "header_warnings_json",
    "arch", "structural_family", "model_version", "task_variant", "noise_stage",
    "confidence", "status", "ecosystem", "model_variant", "unresolved_hint",
    "techniques", "lineage", "sort_subfolder", "format",
    "structure_arch", "metadata_arch", "name_arch",
    "evidence", "evidence_json", "conflicts_json", "warnings_json",
    "total_tensors", "error",
    # Deliberately hand-editable review fields.
    "move", "destination", "canonical", "duplicate_preference",
]
INSPECTION_FIELDNAMES = INSPECTION_FIELDS

PLAN_EXTRA_FIELDS = [
    "plan_schema_name", "plan_schema_version", "plan_tool_version", "plan_id",
    "plan_base_dir", "plan_base_resolved", "use_lineage", "min_confidence",
    "allow_ambiguous", "allow_outside_base", "allow_unverified",
    "allow_symlink_parents", "max_path_length", "legacy_imported",
    "generated_move", "move_override", "bucket", "action", "reason", "verified",
    "sidecars", "sidecar_fingerprints_json", "sidecar_count", "path_safety_json",
    "collision_group", "collision_kind", "collision_target", "collision_sources",
    "collision_member_count", "canonical_path", "duplicate_of", "duplicate_role",
    "duplicate_destination", "content_sha256", "distinct_inode_count",
    "logical_duplicate_bytes", "duplicate_policy", "quarantine_dir",
    "sidecar_notes", "collision_details",
]
PLAN_FIELDS = INSPECTION_FIELDS + [
    value for value in PLAN_EXTRA_FIELDS if value not in INSPECTION_FIELDS
]
PLAN_FIELDNAMES = PLAN_FIELDS

RUNTIME_FIELDS = {
    "_source", "_destination", "_sidecar_sources", "_transfers", "_duplicate_spec",
}


def blank_inspection_record() -> Dict[str, str]:
    return {field: "" for field in INSPECTION_FIELDS}


def inspection_schema_fields() -> Dict[str, str]:
    return {
        "schema_name": INSPECTION_SCHEMA_NAME,
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
    }


def plan_schema_fields() -> Dict[str, str]:
    return {
        "plan_schema_name": PLAN_SCHEMA_NAME,
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "plan_tool_version": TOOL_VERSION,
    }


def ensure_plan_fields(row: Mapping[str, Any]) -> Dict[str, str]:
    """Create a current plan row after the source row has been validated."""
    result = {field: str(row.get(field, "") or "") for field in PLAN_FIELDS}
    # Preserve unknown public columns so reviewers do not lose their own notes.
    for key, value in row.items():
        key = str(key)
        if key not in result and not key.startswith("_"):
            result[key] = str(value or "")
    result.update(plan_schema_fields())
    return result


def validate_inspection_schema(
    fieldnames: Sequence[str], *, allow_legacy: bool = False
) -> Tuple[bool, str]:
    """Validate the CSV header before any row is normalised or acted upon."""
    rendered = [str(value or "").strip() for value in fieldnames]
    if not all(rendered):
        return False, "CSV contains a blank column name"
    duplicates = sorted({value for value in rendered if rendered.count(value) > 1})
    if duplicates:
        return False, "CSV contains duplicate column names: " + ", ".join(duplicates)

    fields = set(rendered)
    common_required = {"path", "arch", "confidence"}
    if {"plan_schema_name", "plan_schema_version"}.issubset(fields):
        required = common_required | {
            "schema_name", "schema_version", "status", "file_size", "mtime_ns",
            "ctime_ns", "header_sha256", "plan_id", "action", "move",
        }
        missing = sorted(required - fields)
        if missing:
            return False, "versioned plan is missing required columns: " + ", ".join(missing)
        return True, "versioned plan columns present"
    if {"schema_name", "schema_version"}.issubset(fields):
        required = common_required | {
            "status", "file_size", "mtime_ns", "ctime_ns", "header_sha256",
        }
        missing = sorted(required - fields)
        if missing:
            return False, "versioned inspection index is missing required columns: " + ", ".join(missing)
        return True, "versioned inspection columns present"
    if allow_legacy and common_required.issubset(fields):
        return True, "unversioned legacy columns explicitly allowed"
    return False, (
        "CSV has no supported schema markers; rerun inspection or pass "
        "--allow-legacy-plan for a deliberately reviewed unversioned file"
    )


def validate_source_row(
    row: Mapping[str, Any], *, allow_legacy: bool = False
) -> Tuple[bool, str]:
    """Validate a source index or reviewed plan before normalising its fields."""
    plan_name = str(row.get("plan_schema_name", "") or "").strip()
    plan_version = str(row.get("plan_schema_version", "") or "").strip()
    index_name = str(row.get("schema_name", "") or "").strip()
    index_version = str(row.get("schema_version", "") or "").strip()

    if plan_name or plan_version:
        if plan_name != PLAN_SCHEMA_NAME:
            return False, "unrecognised plan schema name %r" % plan_name
        if plan_version != PLAN_SCHEMA_VERSION:
            return False, (
                "unsupported plan schema %r; expected %s"
                % (plan_version, PLAN_SCHEMA_VERSION)
            )
        # A current plan generated from a versioned index must retain a compatible
        # underlying inspection schema.  Plans imported from an explicitly reviewed
        # unversioned CSV have both inspection fields blank and set legacy_imported.
        if index_name or index_version:
            if index_name != INSPECTION_SCHEMA_NAME:
                return False, "unrecognised embedded inspection schema name %r" % index_name
            if index_version != INSPECTION_SCHEMA_VERSION:
                return False, (
                    "unsupported embedded inspection schema %r; expected %s"
                    % (index_version, INSPECTION_SCHEMA_VERSION)
                )
        elif str(row.get("legacy_imported", "") or "").strip().lower() not in {
            "true", "1", "yes", "on"
        }:
            return False, "versioned plan has neither an inspection schema nor an explicit legacy-import marker"
    elif index_name or index_version:
        if index_name != INSPECTION_SCHEMA_NAME:
            return False, "unrecognised inspection schema name %r" % index_name
        if index_version != INSPECTION_SCHEMA_VERSION:
            return False, (
                "unsupported inspection schema %r; expected %s"
                % (index_version, INSPECTION_SCHEMA_VERSION)
            )
    elif not allow_legacy:
        return False, "row is unversioned; pass --allow-legacy-plan only after review"

    return True, "schema accepted"


def validate_plan_collection(
    rows: Sequence[Mapping[str, Any]], *, expected_base_resolved: str = ""
) -> Tuple[bool, str]:
    """Validate invariants shared by every row of a reviewed plan.

    Combining rows from different plans, or applying a reviewed plan against a
    different library root, changes the meaning of destinations and collision
    groups.  Those conditions are rejected before any row is normalised.
    """
    if not rows:
        return True, "empty collection"
    is_plan = any(
        str(row.get("plan_schema_name", "") or "").strip()
        or str(row.get("plan_schema_version", "") or "").strip()
        for row in rows
    )
    if not is_plan:
        return True, "inspection index collection"

    required_uniform = (
        "plan_id",
        "plan_base_resolved",
        "use_lineage",
        "min_confidence",
        "allow_ambiguous",
        "allow_outside_base",
        "allow_unverified",
        "allow_symlink_parents",
        "max_path_length",
        "duplicate_policy",
        "quarantine_dir",
    )
    for field in required_uniform:
        values = {str(row.get(field, "") or "").strip() for row in rows}
        if len(values) > 1:
            return False, f"reviewed plan contains inconsistent {field} values"
        if field in {"plan_id", "plan_base_resolved"} and (not values or not next(iter(values))):
            return False, f"reviewed plan has an empty {field}"

    if expected_base_resolved:
        recorded = str(rows[0].get("plan_base_resolved", "") or "").strip()
        expected = os.path.normcase(os.path.abspath(expected_base_resolved))
        actual = os.path.normcase(os.path.abspath(recorded))
        if actual != expected:
            return False, (
                "reviewed plan was generated for a different resolved base: "
                f"recorded={recorded!r}, supplied={expected_base_resolved!r}"
            )
    return True, "reviewed plan collection accepted"


# Compatibility name used by early modular builds.
validate_plan_row = validate_source_row


def write_csv_atomic(
    rows: Sequence[Mapping[str, Any]], path: Path, fieldnames: Sequence[str]
) -> None:
    path = absolute_without_resolving(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    atomic_json_write(payload, path)


def write_json_exclusive(payload: Mapping[str, Any], path: Path) -> None:
    create_json_exclusive(payload, path)


def collision_report_header() -> Dict[str, str]:
    return {
        "schema_name": COLLISION_SCHEMA_NAME,
        "schema_version": COLLISION_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
    }


def manifest_header(operation: str = "apply") -> Dict[str, str]:
    return {
        "schema_name": MANIFEST_SCHEMA_NAME,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "manifest_type": str(operation or "apply"),
    }


def undo_header() -> Dict[str, str]:
    return {
        "schema_name": UNDO_SCHEMA_NAME,
        "schema_version": UNDO_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
    }


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], Tuple[str, ...]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SchemaError("CSV has no header row")
        rows = [
            {str(key): str(value or "") for key, value in row.items() if key is not None}
            for row in reader
        ]
        return rows, tuple(reader.fieldnames)
