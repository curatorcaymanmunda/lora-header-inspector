"""Cross-platform path normalization, containment, and symlink-chain checks."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .constants import WINDOWS_RESERVED_NAMES


def absolute_without_resolving(path: Path) -> Path:
    """Return an absolute lexical path without dereferencing its final component."""
    return Path(os.path.abspath(os.fspath(path)))


def normal_path_key(path: Path) -> str:
    """Return the platform-normalized lexical comparison key for a path."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def resolved_destination(path: Path) -> Path:
    """Resolve existing parent components without requiring the leaf to exist."""
    path = absolute_without_resolving(path)
    return path.parent.resolve(strict=False) / path.name


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def safe_component(value: str, fallback: str = "Unsorted", max_length: int = 120) -> str:
    """Create one portable filename component without silently creating subpaths."""
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", str(value or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    if cleaned.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        cleaned = "_" + cleaned
    return cleaned[:max_length]


def nearest_existing_parent(path: Path) -> Path:
    candidate = absolute_without_resolving(path)
    while not candidate.exists() and not candidate.is_symlink():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError("no existing ancestor for %s" % path)
        candidate = parent
    return candidate


def check_destination_containment(
    destination: Path,
    base: Path,
    allow_outside: bool = False,
) -> Tuple[bool, str, Path]:
    """Check a destination after resolving every currently existing parent link."""
    base_resolved = absolute_without_resolving(base).resolve(strict=False)
    destination_resolved = resolved_destination(destination)
    if not allow_outside and not is_relative_to(destination_resolved.parent, base_resolved):
        return (
            False,
            "destination is outside --base after resolving existing parent links",
            destination_resolved,
        )
    return True, "", destination_resolved


def _component_scope(path: Path, base: Optional[Path]) -> Path:
    """Choose a lexical scan root while allowing the user-supplied base itself to be a link."""
    absolute = absolute_without_resolving(path)
    if base is not None:
        lexical_base = absolute_without_resolving(base)
        if is_relative_to(absolute, lexical_base):
            return lexical_base
    return Path(absolute.anchor)


def existing_symlink_components(path: Path, stop_at: Optional[Path] = None) -> List[Path]:
    """Return existing symlink components between a scan root and ``path.parent``.

    The ``stop_at`` component itself is deliberately not inspected. This lets a user
    nominate a symlinked library root as ``--base`` while still refusing unexpected
    symlinked subdirectories beneath that root by default.
    """
    absolute = absolute_without_resolving(path)
    stop = absolute_without_resolving(stop_at) if stop_at else Path(absolute.anchor)
    try:
        relative = absolute.relative_to(stop)
        current = stop
        parts = relative.parts[:-1]
    except ValueError:
        current = Path(absolute.anchor)
        parts = absolute.parts[1:-1]

    output: List[Path] = []
    for part in parts:
        current = current / part
        try:
            if current.is_symlink():
                output.append(current)
        except OSError:
            # A later stat/resolve operation will report the inaccessible component.
            continue
    return output


def symlink_component_snapshot(path: Path, *, base: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Capture the identity and target text of every existing parent symlink."""
    scope = _component_scope(path, base)
    snapshots: List[Dict[str, Any]] = []
    for component in existing_symlink_components(path, stop_at=scope):
        stat = component.lstat()
        snapshots.append(
            {
                "path": str(component),
                "target": os.readlink(component),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return snapshots


def _snapshot_marker(values: Sequence[Mapping[str, Any]]) -> Tuple[Tuple[str, str, int, int, int], ...]:
    output: List[Tuple[str, str, int, int, int]] = []
    for item in values:
        output.append(
            (
                str(item.get("path", "")),
                str(item.get("target", "")),
                int(item.get("device", -1)),
                int(item.get("inode", -1)),
                int(item.get("mtime_ns", -1)),
            )
        )
    return tuple(output)


def build_path_safety_snapshot(
    transfers: Sequence[Mapping[str, Any]],
    *,
    base: Path,
    allow_outside: bool,
    allow_symlink_parents: bool,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Build an apply-time-verifiable path snapshot for one planned operation.

    Each transfer mapping must contain ``source`` and ``target``. A truthy
    ``managed_target`` means the target is a new managed destination and must remain
    inside ``base`` unless ``allow_outside`` was explicitly selected.
    """
    lexical_base = absolute_without_resolving(base)
    resolved_base = lexical_base.resolve(strict=False)
    captured: List[Dict[str, Any]] = []
    for raw in transfers:
        source = absolute_without_resolving(Path(str(raw.get("source", ""))))
        target = absolute_without_resolving(Path(str(raw.get("target", ""))))
        managed_target = bool(raw.get("managed_target", True))
        if not str(raw.get("source", "")).strip() or not str(raw.get("target", "")).strip():
            return None, "path-safety transfer is missing a source or target"

        try:
            source_resolved = source.resolve(strict=True)
            target_resolved = resolved_destination(target)
            source_links = symlink_component_snapshot(source, base=lexical_base)
            target_links = symlink_component_snapshot(target, base=lexical_base)
        except (OSError, RuntimeError, ValueError) as exc:
            return None, "could not snapshot transfer paths: %s" % exc

        if not allow_symlink_parents:
            if source_links:
                return None, "source path crosses a symlinked subdirectory: %s" % source_links[0]["path"]
            if target_links:
                return None, "destination path crosses a symlinked subdirectory: %s" % target_links[0]["path"]

        if managed_target and not allow_outside and not is_relative_to(
            target_resolved.parent, resolved_base
        ):
            return None, "managed destination resolves outside --base: %s" % target

        captured.append(
            {
                "source": str(source),
                "source_resolved": str(source_resolved),
                "source_parent_symlinks": source_links,
                "target": str(target),
                "target_resolved": str(target_resolved),
                "target_parent_symlinks": target_links,
                "managed_target": managed_target,
            }
        )

    return (
        {
            "base": str(lexical_base),
            "base_resolved": str(resolved_base),
            "allow_outside_base": bool(allow_outside),
            "allow_symlink_parents": bool(allow_symlink_parents),
            "transfers": captured,
        },
        "",
    )


def verify_path_safety_snapshot(
    serialized: str,
    *,
    base: Path,
    allow_outside: bool,
    allow_symlink_parents: bool,
) -> Tuple[bool, str]:
    """Reject path redirection or containment changes between planning and apply."""
    try:
        payload = json.loads(str(serialized or ""))
    except json.JSONDecodeError as exc:
        return False, "invalid path-safety snapshot JSON: %s" % exc
    if not isinstance(payload, Mapping):
        return False, "path-safety snapshot is not an object"
    entries = payload.get("transfers", [])
    if not isinstance(entries, list):
        return False, "path-safety transfer list is invalid"

    lexical_base = absolute_without_resolving(base)
    resolved_base = lexical_base.resolve(strict=False)
    recorded_base = str(payload.get("base", ""))
    recorded_resolved = str(payload.get("base_resolved", ""))
    if normal_path_key(Path(recorded_base)) != normal_path_key(lexical_base):
        return False, "apply --base differs from the planned base"
    if normal_path_key(Path(recorded_resolved)) != normal_path_key(resolved_base):
        return False, "the resolved --base path changed after planning"
    if bool(payload.get("allow_outside_base", False)) != bool(allow_outside):
        return False, "--allow-outside-base differs from the planned policy"
    if bool(payload.get("allow_symlink_parents", False)) != bool(allow_symlink_parents):
        return False, "--allow-symlink-parents differs from the planned policy"

    for raw in entries:
        if not isinstance(raw, Mapping):
            return False, "path-safety transfer entry is not an object"
        source = absolute_without_resolving(Path(str(raw.get("source", ""))))
        target = absolute_without_resolving(Path(str(raw.get("target", ""))))
        managed_target = bool(raw.get("managed_target", True))
        try:
            current_source_resolved = source.resolve(strict=True)
            current_target_resolved = resolved_destination(target)
            current_source_links = symlink_component_snapshot(source, base=lexical_base)
            current_target_links = symlink_component_snapshot(target, base=lexical_base)
        except (OSError, RuntimeError, ValueError) as exc:
            return False, "could not revalidate transfer paths: %s" % exc

        if normal_path_key(current_source_resolved) != normal_path_key(
            Path(str(raw.get("source_resolved", "")))
        ):
            return False, "source path now resolves to a different location: %s" % source
        if normal_path_key(current_target_resolved) != normal_path_key(
            Path(str(raw.get("target_resolved", "")))
        ):
            return False, "destination path now resolves to a different location: %s" % target
        try:
            recorded_source_links = raw.get("source_parent_symlinks", [])
            recorded_target_links = raw.get("target_parent_symlinks", [])
            if not isinstance(recorded_source_links, list) or not isinstance(recorded_target_links, list):
                return False, "recorded parent-symlink snapshot is invalid"
            if _snapshot_marker(recorded_source_links) != _snapshot_marker(current_source_links):
                return False, "source parent symlink chain changed after planning: %s" % source
            if _snapshot_marker(recorded_target_links) != _snapshot_marker(current_target_links):
                return False, "destination parent symlink chain changed after planning: %s" % target
        except (TypeError, ValueError):
            return False, "recorded parent-symlink snapshot contains invalid values"

        if not allow_symlink_parents and (current_source_links or current_target_links):
            return False, "a transfer path crosses a symlinked subdirectory"
        if managed_target and not allow_outside and not is_relative_to(
            current_target_resolved.parent, resolved_base
        ):
            return False, "managed destination now resolves outside --base: %s" % target
    return True, "path-safety snapshot matches"


def split_semicolon(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def default_subfolder(record: Mapping[str, Any]) -> str:
    explicit = str(record.get("sort_subfolder", "") or "").strip()
    if explicit:
        return explicit
    lineage = str(record.get("lineage", "") or "").strip()
    if lineage:
        return lineage
    ecosystems = split_semicolon(record.get("ecosystem", ""))
    if ecosystems:
        return ecosystems[0]
    variants = split_semicolon(record.get("model_variant", ""))
    if variants:
        return variants[0]
    techniques = split_semicolon(record.get("techniques", ""))
    if techniques:
        return "+".join(techniques)
    return ""
