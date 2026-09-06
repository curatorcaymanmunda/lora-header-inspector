"""Explicit sidecar discovery, snapshots, and reconciliation helpers.

A filename-prefix test is intentionally not enough.  ``foo.v2.safetensors`` is
another model, not a sidecar of ``foo.safetensors``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .constants import MODEL_LIKE_SUFFIXES
from .fingerprints import CacheKey, full_file_sha256
from .models import SidecarSnapshot


@dataclass(frozen=True)
class SidecarRule:
    rule_id: str
    role: str
    scope: str  # "stem" or "full-name"
    pattern: re.Pattern


RULES: Tuple[SidecarRule, ...] = (
    SidecarRule("sidecar.preview.simple", "preview", "stem", re.compile(r"\.(?:png|jpe?g|webp|gif)$", re.I)),
    SidecarRule("sidecar.preview.named", "preview", "stem", re.compile(r"\.(?:preview|thumb|thumbnail|cover)\.(?:png|jpe?g|webp|gif)$", re.I)),
    SidecarRule("sidecar.civitai", "civitai_metadata", "stem", re.compile(r"\.civitai\.info$", re.I)),
    SidecarRule("sidecar.metadata_json", "metadata", "stem", re.compile(r"\.(?:metadata|info)\.json$", re.I)),
    SidecarRule("sidecar.json", "metadata", "stem", re.compile(r"\.json$", re.I)),
    SidecarRule("sidecar.caption", "caption", "stem", re.compile(r"\.(?:txt|md)$", re.I)),
    SidecarRule("sidecar.notes", "notes", "stem", re.compile(r"\.(?:activation|trigger|description|notes)\.txt$", re.I)),
    SidecarRule("sidecar.yaml", "metadata", "stem", re.compile(r"\.ya?ml$", re.I)),
    SidecarRule("sidecar.checksum", "checksum", "stem", re.compile(r"\.(?:sha256|sha512)$", re.I)),
    SidecarRule("sidecar.full_preview.simple", "preview", "full-name", re.compile(r"\.(?:png|jpe?g|webp|gif)$", re.I)),
    SidecarRule("sidecar.full_preview.named", "preview", "full-name", re.compile(r"\.(?:preview|thumb|thumbnail|cover)\.(?:png|jpe?g|webp|gif)$", re.I)),
    SidecarRule("sidecar.full_json", "metadata", "full-name", re.compile(r"\.json$", re.I)),
    SidecarRule("sidecar.full_txt", "caption", "full-name", re.compile(r"\.(?:txt|md)$", re.I)),
    SidecarRule("sidecar.full_yaml", "metadata", "full-name", re.compile(r"\.ya?ml$", re.I)),
    SidecarRule("sidecar.full_checksum", "checksum", "full-name", re.compile(r"\.(?:sha256|sha512)$", re.I)),
    SidecarRule("sidecar.full_civitai", "civitai_metadata", "full-name", re.compile(r"\.civitai\.info$", re.I)),
)


def _casefold_remainder(name: str, prefix: str) -> Optional[str]:
    if len(name) < len(prefix):
        return None
    if name[: len(prefix)].casefold() != prefix.casefold():
        return None
    return name[len(prefix) :]


def sidecar_rule(source: Path, candidate: Path) -> Optional[SidecarRule]:
    try:
        if candidate == source or candidate.is_symlink() or not candidate.is_file():
            return None
    except OSError:
        return None
    if candidate.suffix.casefold() in MODEL_LIKE_SUFFIXES:
        return None
    for rule in RULES:
        prefix = source.stem if rule.scope == "stem" else source.name
        remainder = _casefold_remainder(candidate.name, prefix)
        if remainder is not None and rule.pattern.fullmatch(remainder):
            return rule
    return None


def sidecar_role(source: Path, candidate: Path) -> Optional[str]:
    rule = sidecar_rule(source, candidate)
    return rule.role if rule else None


def is_safe_sidecar(source: Path, candidate: Path) -> bool:
    return sidecar_rule(source, candidate) is not None


def discover_sidecars(source: Path) -> List[Path]:
    try:
        siblings = list(source.parent.iterdir())
    except OSError:
        return []
    return sorted(
        (candidate for candidate in siblings if is_safe_sidecar(source, candidate)),
        key=lambda value: (value.name.casefold(), value.name),
    )


def snapshot_sidecar(
    source: Path,
    sidecar: Path,
    *,
    include_hash: bool = False,
    hash_cache: Optional[MutableMapping[CacheKey, str]] = None,
) -> SidecarSnapshot:
    rule = sidecar_rule(source, sidecar)
    if rule is None:
        raise ValueError("not a recognised safe sidecar: %s" % sidecar)
    before = sidecar.stat()
    digest = full_file_sha256(sidecar, hash_cache) if include_hash else ""
    after = sidecar.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError("sidecar changed while being inspected: %s" % sidecar)
    return SidecarSnapshot(
        path=sidecar,
        name=sidecar.name,
        role=rule.role,
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        sha256=digest,
    )


def snapshot_sidecars(
    source: Path,
    *,
    include_hash: bool = False,
    hash_cache: Optional[MutableMapping[CacheKey, str]] = None,
) -> List[SidecarSnapshot]:
    return [
        snapshot_sidecar(
            source,
            value,
            include_hash=include_hash,
            hash_cache=hash_cache,
        )
        for value in discover_sidecars(source)
    ]


def sidecar_snapshot_json(snapshots: Sequence[SidecarSnapshot]) -> str:
    return json.dumps(
        [snapshot.to_dict() for snapshot in snapshots],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_snapshot_json(value: str) -> Tuple[List[Mapping[str, object]], str]:
    text = str(value or "").strip()
    if not text:
        return [], ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], "invalid sidecar snapshot JSON: %s" % exc
    if not isinstance(parsed, list):
        return [], "sidecar snapshot is not a JSON list"
    output: List[Mapping[str, object]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, Mapping):
            return [], "sidecar snapshot item %d is not an object" % index
        output.append(item)
    return output, ""


def verify_sidecar_snapshot(
    source: Path,
    serialized: str,
    *,
    require_hash: bool = True,
    hash_cache: Optional[MutableMapping[CacheKey, str]] = None,
) -> Tuple[bool, str, List[Path]]:
    """Verify that a model's sidecar set and every recorded file are unchanged."""
    expected, error = _parse_snapshot_json(serialized)
    if error:
        return False, error, []

    current = discover_sidecars(source)
    expected_names = [str(item.get("name", "") or "") for item in expected]
    current_names = [path.name for path in current]
    if expected_names != current_names:
        return (
            False,
            "sidecar set changed after planning: planned=%s, current=%s"
            % (expected_names, current_names),
            current,
        )

    current_by_name = {path.name: path for path in current}
    for item in expected:
        name = str(item.get("name", "") or "")
        path = current_by_name.get(name)
        if path is None:
            return False, "planned sidecar is missing: %s" % name, current
        try:
            stat = path.stat()
        except OSError as exc:
            return False, "could not stat sidecar %s: %s" % (path, exc), current
        checks = (
            ("size", stat.st_size),
            ("mtime_ns", stat.st_mtime_ns),
            ("device", stat.st_dev),
            ("inode", stat.st_ino),
        )
        for field, actual in checks:
            raw = item.get(field, "")
            try:
                expected_value = int(raw)
            except (TypeError, ValueError):
                return False, "invalid recorded %s for sidecar %s" % (field, name), current
            if int(actual) != expected_value:
                return False, "sidecar changed after planning: %s (%s)" % (name, field), current

        expected_hash = str(item.get("sha256", "") or "").lower()
        if require_hash and not expected_hash:
            return False, "sidecar content hash is missing: %s" % name, current
        if expected_hash:
            try:
                actual_hash = full_file_sha256(path, hash_cache)
            except Exception as exc:
                return False, "could not hash sidecar %s: %s" % (name, exc), current
            if actual_hash.lower() != expected_hash:
                return False, "sidecar changed after planning: %s (content hash)" % name, current

    return True, "sidecar snapshot matches", current


def compare_sidecar_sets(
    members: Sequence[Tuple[str, Sequence[SidecarSnapshot]]]
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Describe same-name sidecars that differ across duplicate model copies."""
    by_name: Dict[str, List[Tuple[str, SidecarSnapshot]]] = {}
    for source, snapshots in members:
        for snapshot in snapshots:
            by_name.setdefault(snapshot.name.casefold(), []).append((source, snapshot))
    conflicts: List[Dict[str, object]] = []
    notes: List[str] = []
    for _, values in sorted(by_name.items()):
        if len(values) < 2:
            continue
        hashes = {snapshot.sha256 for _, snapshot in values if snapshot.sha256}
        identities = {(snapshot.device, snapshot.inode) for _, snapshot in values}
        if len(identities) == 1:
            notes.append("sidecar %s already shares one inode" % values[0][1].name)
        elif len(hashes) == 1 and hashes:
            notes.append("sidecar %s is an exact duplicate" % values[0][1].name)
        else:
            conflicts.append(
                {
                    "name": values[0][1].name,
                    "role": values[0][1].role,
                    "members": [
                        {
                            "source_model": source,
                            "path": str(snapshot.path),
                            "size": snapshot.size,
                            "sha256": snapshot.sha256,
                            "device": snapshot.device,
                            "inode": snapshot.inode,
                        }
                        for source, snapshot in values
                    ],
                    "reason": "same sidecar name has different content or could not be verified",
                }
            )
    return conflicts, notes
