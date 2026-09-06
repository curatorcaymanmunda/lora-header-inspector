#!/usr/bin/env python3
"""Build, review, apply, or undo a conservative LoRA sort plan.

The default is a dry run. Automatic moves require all of the following unless an
explicit override is supplied:

* exact architecture is known
* status is ``ok``
* confidence meets the threshold (default: high)
* the source fingerprint still matches the inspection CSV
* the destination remains inside ``--base``
* the primary file and every recognised sidecar pass collision preflight

Each LoRA plus its sidecars is moved as one small transaction. A JSON manifest is
written during apply runs and can later be used with ``--undo``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

TOOL_VERSION = "0.5.0"
CONF_RANK = {"low": 0, "medium": 1, "high": 2}
TRUTHY = {"true", "1", "yes", "y", "on"}
FALSY = {"false", "0", "no", "n", "off"}
MAX_HEADER_BYTES = 256 * 1024 * 1024

# These patterns intentionally describe known metadata/preview companions only.
# A prefix match alone is unsafe because foo.v2.safetensors is not a sidecar of
# foo.safetensors.
STEM_SIDECAR_PATTERNS = [
    re.compile(r"\.(?:png|jpe?g|webp|gif|txt|json|ya?ml)$", re.IGNORECASE),
    re.compile(r"\.(?:preview|thumb|thumbnail|cover)\.(?:png|jpe?g|webp|gif)$", re.IGNORECASE),
    re.compile(r"\.civitai\.info$", re.IGNORECASE),
    re.compile(r"\.(?:metadata|info)\.json$", re.IGNORECASE),
    re.compile(r"\.(?:activation|trigger|description|notes)\.txt$", re.IGNORECASE),
    re.compile(r"\.sha256$", re.IGNORECASE),
]
FULL_NAME_SIDECAR_PATTERNS = [
    re.compile(r"\.(?:json|txt|civitai\.info)$", re.IGNORECASE),
]
MODEL_LIKE_SUFFIXES = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".onnx",
}

PLAN_RUNTIME_KEYS = {
    "_source",
    "_destination",
    "_sidecar_sources",
    "_transfers",
}

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def explicitly_false(value: Any) -> bool:
    return str(value).strip().lower() in FALSY


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _normal_path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _resolved_destination(path: Path) -> Path:
    """Resolve existing parents without requiring the destination to exist."""
    return path.parent.resolve(strict=False) / path.name


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_component(value: str, fallback: str = "Unsorted") -> str:
    """Create a folder component valid on Windows, macOS, and Linux."""
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", str(value).strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    if cleaned.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        cleaned = "_" + cleaned
    return cleaned[:120]


def _split_semicolon(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def _default_subfolder(record: Mapping[str, str]) -> str:
    explicit = str(record.get("sort_subfolder", "")).strip()
    if explicit:
        return explicit

    # Backward compatibility with the first revision and older plan CSVs.
    lineage = str(record.get("lineage", "")).strip()
    if lineage:
        return lineage

    ecosystems = _split_semicolon(record.get("ecosystem", ""))
    if ecosystems:
        return ecosystems[0]
    variants = _split_semicolon(record.get("model_variant", ""))
    if variants:
        return variants[0]
    techniques = _split_semicolon(record.get("techniques", ""))
    if techniques:
        return "+".join(techniques)
    return ""


def _read_header_hash(path: Path) -> str:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError("file too short to contain a safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len < 2 or header_len > MAX_HEADER_BYTES:
            raise ValueError("invalid or implausible safetensors header length")
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError("truncated safetensors header")
    return hashlib.sha256(raw_header).hexdigest()


def verify_fingerprint(
    record: Mapping[str, str], source: Path, allow_unverified: bool = False
) -> Tuple[bool, str]:
    expected_size = str(record.get("file_size", "")).strip()
    expected_mtime = str(record.get("mtime_ns", "")).strip()
    expected_header = str(record.get("header_sha256", "")).strip().lower()

    if not expected_size or not expected_mtime or not expected_header:
        if allow_unverified:
            return True, "fingerprint absent; explicitly allowed"
        return False, "inspection fingerprint is missing"

    try:
        stat = source.stat()
        current_header = _read_header_hash(source)
    except Exception as exc:
        return False, f"could not verify source fingerprint: {exc}"

    mismatches: List[str] = []
    try:
        if stat.st_size != int(expected_size):
            mismatches.append("size")
    except ValueError:
        mismatches.append("recorded size is invalid")
    try:
        if stat.st_mtime_ns != int(expected_mtime):
            mismatches.append("mtime")
    except ValueError:
        mismatches.append("recorded mtime is invalid")
    if current_header.lower() != expected_header:
        mismatches.append("header hash")

    if mismatches:
        return False, "source changed after inspection: " + ", ".join(mismatches)
    return True, "fingerprint matches"


def _json_list(value: str) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Compatibility with semicolon-separated plans from the earlier script.
        return [part for part in text.split(";") if part]
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _atomic_json_write(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------

def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.fullmatch(text) for pattern in patterns)


def _case_insensitive_remainder(name: str, prefix: str) -> Optional[str]:
    """Return the literal remainder when an equal-length prefix matches by case."""
    if len(name) < len(prefix):
        return None
    if name[: len(prefix)].casefold() != prefix.casefold():
        return None
    return name[len(prefix) :]


def is_safe_sidecar(source: Path, candidate: Path) -> bool:
    if candidate == source or not candidate.is_file() or candidate.is_symlink():
        return False
    if candidate.suffix.lower() in MODEL_LIKE_SUFFIXES:
        return False

    name = candidate.name
    stem_prefix = source.stem
    remainder = _case_insensitive_remainder(name, stem_prefix)
    if remainder is not None and _matches_any(remainder, STEM_SIDECAR_PATTERNS):
        return True

    full_prefix = source.name
    remainder = _case_insensitive_remainder(name, full_prefix)
    if remainder is not None and _matches_any(remainder, FULL_NAME_SIDECAR_PATTERNS):
        return True
    return False


def discover_sidecars(source: Path) -> List[Path]:
    try:
        siblings = source.parent.iterdir()
    except OSError:
        return []
    return sorted(
        (candidate for candidate in siblings if is_safe_sidecar(source, candidate)),
        key=lambda path: path.name.lower(),
    )


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _skip(record: MutableMapping[str, str], action: str, reason: str) -> None:
    record.update(move="False", action=action, reason=reason)


def _candidate_destination(
    record: Mapping[str, str], base_dir: Path, use_lineage: bool
) -> Path:
    raw_destination = str(record.get("destination", "")).strip()
    if raw_destination:
        destination = Path(raw_destination)
        return destination if destination.is_absolute() else base_dir / destination

    arch = _safe_component(record.get("arch", "Unknown"), "Unknown")
    bucket = Path(arch)
    if use_lineage:
        subfolder = _default_subfolder(record)
        if subfolder:
            bucket /= _safe_component(subfolder)
    return base_dir / bucket / Path(record["path"]).name


def _preflight_existing_targets(
    source: Path, destination: Path, sidecar_sources: Sequence[Path]
) -> List[str]:
    collisions: List[str] = []
    transfers = [(source, destination)] + [
        (sidecar, destination.parent / sidecar.name) for sidecar in sidecar_sources
    ]
    for src, dst in transfers:
        if _normal_path_key(src) == _normal_path_key(dst):
            continue
        if dst.exists() or dst.is_symlink():
            collisions.append(dst.name)
    return collisions


def build_sort_plan(
    csv_path: Path,
    base_dir: Path,
    use_lineage: bool = True,
    min_conf: str = "high",
    allow_ambiguous: bool = False,
    allow_outside_base: bool = False,
    allow_unverified: bool = False,
    max_path_length: int = 0,
) -> List[Dict[str, str]]:
    plan: List[Dict[str, str]] = []
    base_dir = _absolute_without_resolving(base_dir)
    base_resolved = base_dir.resolve(strict=False)
    min_rank = CONF_RANK.get(min_conf, CONF_RANK["high"])
    seen_sources: Set[str] = set()

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")

        for original in reader:
            record: Dict[str, str] = {key: str(value or "") for key, value in original.items()}
            record.setdefault("move", "")
            record.setdefault("destination", "")
            record.update(
                bucket="",
                action="",
                reason="",
                sidecars="[]",
                sidecar_count="0",
                verified="False",
            )

            path_text = record.get("path", "").strip()
            if not path_text:
                _skip(record, "skip_no_path", "missing source path")
                plan.append(record)
                continue

            try:
                source = _absolute_without_resolving(Path(path_text))
                source_key = _normal_path_key(source)
                source_is_symlink = source.is_symlink()
                source_is_file = source.is_file()
            except (OSError, ValueError, RuntimeError) as exc:
                _skip(record, "skip_invalid_path", f"invalid source path: {exc}")
                plan.append(record)
                continue

            record["path"] = str(source)
            if source_key in seen_sources:
                _skip(record, "skip_duplicate_source", "source appears more than once in the CSV")
                plan.append(record)
                continue
            seen_sources.add(source_key)

            if source_is_symlink:
                _skip(record, "skip_symlink", "source path is a symlink")
                plan.append(record)
                continue
            if not source_is_file:
                _skip(record, "skip_missing", "source file was not found")
                plan.append(record)
                continue
            if record.get("error", "").strip():
                _skip(record, "skip_error", record["error"][:160])
                plan.append(record)
                continue

            arch = record.get("arch", "Unknown").strip() or "Unknown"
            confidence = record.get("confidence", "low").strip().lower() or "low"
            status = record.get("status", "ambiguous").strip().lower() or "ambiguous"

            if arch == "Unknown":
                record["bucket"] = "Unknown"
                _skip(record, "skip_unknown", record.get("evidence", "")[:160])
                plan.append(record)
                continue
            if status == "conflict":
                _skip(record, "skip_conflict", "classification evidence conflicts")
                plan.append(record)
                continue
            if status == "error":
                _skip(record, "skip_error", "classification status is error")
                plan.append(record)
                continue
            if status == "ambiguous" and not allow_ambiguous:
                _skip(record, "skip_ambiguous", "classification requires review")
                plan.append(record)
                continue
            if status not in {"ok", "ambiguous"}:
                _skip(record, "skip_invalid_status", f"unrecognised status={status!r}")
                plan.append(record)
                continue
            if CONF_RANK.get(confidence, -1) < min_rank:
                _skip(
                    record,
                    "skip_low_confidence",
                    f"confidence={confidence} is below {min_conf}",
                )
                plan.append(record)
                continue
            if explicitly_false(record.get("move", "")):
                _skip(record, "skip_user_false", "move=False in CSV")
                plan.append(record)
                continue

            try:
                destination = _candidate_destination(record, base_dir, use_lineage)
                destination = _absolute_without_resolving(destination)
                destination_resolved = _resolved_destination(destination)
                destination_parent_resolved = destination_resolved.parent
            except (OSError, ValueError, RuntimeError) as exc:
                _skip(record, "skip_invalid_destination", f"invalid destination: {exc}")
                plan.append(record)
                continue

            if not allow_outside_base and not _is_relative_to(destination_parent_resolved, base_resolved):
                record["destination"] = str(destination)
                _skip(record, "skip_outside_base", "destination is outside --base")
                plan.append(record)
                continue

            record["destination"] = str(destination)
            try:
                record["bucket"] = str(destination_parent_resolved.relative_to(base_resolved))
            except ValueError:
                record["bucket"] = str(destination_parent_resolved)

            if _normal_path_key(source) == _normal_path_key(destination):
                _skip(record, "already_in_place", "source already equals destination")
                plan.append(record)
                continue

            verified, verify_reason = verify_fingerprint(
                record, source, allow_unverified=allow_unverified
            )
            record["verified"] = str(verified)
            if not verified:
                _skip(record, "skip_stale_or_unverified", verify_reason)
                plan.append(record)
                continue

            sidecar_sources = discover_sidecars(source)
            sidecar_names = [sidecar.name for sidecar in sidecar_sources]
            record["sidecars"] = json.dumps(sidecar_names, ensure_ascii=False)
            record["sidecar_count"] = str(len(sidecar_names))

            if max_path_length:
                planned_targets = [destination] + [
                    destination.parent / sidecar.name for sidecar in sidecar_sources
                ]
                too_long = [
                    target.name
                    for target in planned_targets
                    if len(str(target)) > max_path_length
                ]
                if too_long:
                    _skip(
                        record,
                        "skip_path_too_long",
                        f"destination path exceeds {max_path_length} characters: "
                        + ", ".join(too_long),
                    )
                    plan.append(record)
                    continue

            collisions = _preflight_existing_targets(source, destination, sidecar_sources)
            if collisions:
                _skip(
                    record,
                    "skip_collision",
                    "destination collision(s): " + ", ".join(collisions),
                )
                plan.append(record)
                continue

            record.update(move="True", action="move", reason=verify_reason)
            record["_source"] = str(source)
            record["_destination"] = str(destination)
            record["_sidecar_sources"] = json.dumps(
                [str(sidecar) for sidecar in sidecar_sources], ensure_ascii=False
            )
            transfers = [(str(source), str(destination))] + [
                (str(sidecar), str(destination.parent / sidecar.name))
                for sidecar in sidecar_sources
            ]
            record["_transfers"] = json.dumps(transfers, ensure_ascii=False)
            plan.append(record)

    # Detect collisions between rows in the same plan, including sidecars.
    target_to_rows: Dict[str, List[int]] = {}
    for index, record in enumerate(plan):
        if record.get("action") != "move":
            continue
        for _, destination in json.loads(record.get("_transfers", "[]")):
            target_to_rows.setdefault(_normal_path_key(Path(destination)), []).append(index)

    conflicting_rows: Set[int] = set()
    for indexes in target_to_rows.values():
        if len(indexes) > 1:
            conflicting_rows.update(indexes)

    for index in conflicting_rows:
        record = plan[index]
        _skip(
            record,
            "skip_plan_collision",
            "two or more plan rows target the same destination",
        )

    return plan


# ---------------------------------------------------------------------------
# Plan display and CSV output
# ---------------------------------------------------------------------------

def summarize_plan(plan: Sequence[Mapping[str, str]], label: str = "Sort plan") -> None:
    buckets = Counter(record.get("bucket") or "Unknown" for record in plan)
    actions = Counter(record.get("action") or "unknown" for record in plan)
    statuses = Counter(record.get("status") or "legacy/unknown" for record in plan)

    print(f"\n{label}: {len(plan)} entries")
    print("By status:")
    for status, count in sorted(statuses.items()):
        print(f"  {status:28s} {count}")
    print("By bucket:")
    for bucket, count in sorted(buckets.items()):
        print(f"  {bucket:28s} {count}")
    print("By action:")
    for action, count in sorted(actions.items()):
        print(f"  {action:28s} {count}")


def _public_fieldnames(plan: Sequence[Mapping[str, str]]) -> List[str]:
    fieldnames: List[str] = []
    seen: Set[str] = set()
    for record in plan:
        for key in record:
            if key in PLAN_RUNTIME_KEYS or key.startswith("_") or key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def write_plan_csv(plan: Sequence[Mapping[str, str]], output_path: Path) -> None:
    if not plan:
        print("No records to write.")
        return
    fieldnames = _public_fieldnames(plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(plan)
    print(f"Sort plan written to: {output_path}")


def _plan_output_path(csv_path: Path) -> Path:
    """Avoid producing *_sort_plan_sort_plan.csv when a reviewed plan is reused."""
    if csv_path.stem.endswith("_sort_plan"):
        return csv_path
    return csv_path.with_name(csv_path.stem + "_sort_plan.csv")


# ---------------------------------------------------------------------------
# Transactional movement
# ---------------------------------------------------------------------------

def _planned_sidecar_names(record: Mapping[str, str]) -> List[str]:
    return _json_list(record.get("sidecars", ""))


def _current_transfers(record: Mapping[str, str]) -> Tuple[List[Tuple[Path, Path]], str]:
    source = Path(record["path"])
    destination = Path(record["destination"])
    current_sidecars = discover_sidecars(source)
    current_names = [path.name for path in current_sidecars]
    planned_names = _planned_sidecar_names(record)
    if current_names != planned_names:
        return [], (
            "sidecar set changed after planning: planned="
            f"{planned_names}, current={current_names}"
        )

    # Sidecars first and primary last. If an ordinary exception occurs, every
    # completed transfer is rolled back in reverse order.
    transfers = [
        (sidecar, destination.parent / sidecar.name) for sidecar in current_sidecars
    ]
    transfers.append((source, destination))
    return transfers, ""


def _preflight_transaction(transfers: Sequence[Tuple[Path, Path]]) -> Tuple[bool, str]:
    sources: Set[str] = set()
    destinations: Set[str] = set()
    for source, destination in transfers:
        source_key = _normal_path_key(source)
        destination_key = _normal_path_key(destination)
        if source_key in sources:
            return False, f"duplicate source in transaction: {source}"
        if destination_key in destinations:
            return False, f"duplicate destination in transaction: {destination}"
        sources.add(source_key)
        destinations.add(destination_key)
        if not source.is_file() or source.is_symlink():
            return False, f"source missing or is a symlink: {source}"
        if destination.exists() or destination.is_symlink():
            return False, f"destination already exists: {destination}"
    return True, ""


def _rollback_transfers(completed: Sequence[Tuple[Path, Path]]) -> List[str]:
    errors: List[str] = []
    for original, moved_to in reversed(completed):
        try:
            if not moved_to.exists():
                errors.append(f"rollback source missing: {moved_to}")
                continue
            if original.exists() or original.is_symlink():
                errors.append(f"rollback destination occupied: {original}")
                continue
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(moved_to), str(original))
        except Exception as exc:
            errors.append(f"{moved_to} -> {original}: {exc}")
    return errors


def _default_manifest_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_move_manifest_{_utc_stamp()}.json")


def apply_plan(
    plan: Sequence[MutableMapping[str, str]],
    csv_path: Path,
    base_dir: Path,
    dry_run: bool = True,
    allow_unverified: bool = False,
    manifest_path: Optional[Path] = None,
) -> Dict[str, int]:
    counts = {"moved": 0, "skipped": 0, "errors": 0, "rolled_back": 0}
    manifest: Dict[str, Any] = {
        "tool": "inspect_sort_lora.py",
        "tool_version": TOOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_dir": str(base_dir),
        "source_csv": str(csv_path),
        "operations": [],
    }

    if not dry_run:
        manifest_path = _absolute_without_resolving(
            manifest_path or _default_manifest_path(csv_path)
        )
        protected_paths = {_normal_path_key(csv_path)}
        for record in plan:
            raw_transfers = record.get("_transfers", "")
            try:
                parsed_transfers = json.loads(raw_transfers) if raw_transfers else []
            except json.JSONDecodeError:
                parsed_transfers = []
            for item in parsed_transfers:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                protected_paths.add(_normal_path_key(Path(item[0])))
                protected_paths.add(_normal_path_key(Path(item[1])))

        manifest_key = _normal_path_key(manifest_path)
        if manifest_key in protected_paths:
            print("ERROR: manifest path overlaps the CSV or a planned transfer")
            counts["errors"] += 1
            return counts
        if manifest_path.exists() or manifest_path.is_symlink():
            print(f"ERROR: manifest already exists and will not be overwritten: {manifest_path}")
            counts["errors"] += 1
            return counts
        try:
            _atomic_json_write(manifest, manifest_path)
        except Exception as exc:
            print(f"ERROR: could not create move manifest: {exc}")
            counts["errors"] += 1
            return counts

    for record in plan:
        source = Path(record.get("path", ""))
        if record.get("action") != "move" or not truthy(record.get("move", "")):
            if record.get("action") != "already_in_place":
                print(
                    f"[SKIP] {record.get('action', 'unknown'):24s} "
                    f"{source.name}  {record.get('reason', '')}"
                )
            counts["skipped"] += 1
            continue

        destination = Path(record["destination"])
        sidecar_count = len(_planned_sidecar_names(record))
        tag = "[DRY] " if dry_run else "[MOVE]"
        print(
            f"{tag} {source} -> {destination}"
            + (f"  (+{sidecar_count} sidecar)" if sidecar_count else "")
        )
        if dry_run:
            continue

        verified, verify_reason = verify_fingerprint(
            record, source, allow_unverified=allow_unverified
        )
        if not verified:
            record.update(
                move="False",
                action="skip_stale_or_unverified",
                reason=verify_reason,
                verified="False",
            )
            print(f"       skipped: {verify_reason}")
            counts["skipped"] += 1
            continue

        transfers, transfer_error = _current_transfers(record)
        if transfer_error:
            record.update(move="False", action="skip_stale_sidecars", reason=transfer_error)
            print(f"       skipped: {transfer_error}")
            counts["skipped"] += 1
            continue

        ready, preflight_error = _preflight_transaction(transfers)
        if not ready:
            record.update(move="False", action="skip_preflight", reason=preflight_error)
            print(f"       skipped: {preflight_error}")
            counts["skipped"] += 1
            continue

        operation: Dict[str, Any] = {
            "record_path": str(source),
            "status": "pending",
            "transfers": [
                {"from": str(src), "to": str(dst)} for src, dst in transfers
            ],
            "error": "",
            "rollback_errors": [],
        }
        manifest["operations"].append(operation)
        _atomic_json_write(manifest, manifest_path)

        completed: List[Tuple[Path, Path]] = []
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            for transfer_source, transfer_destination in transfers:
                transfer_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(transfer_source), str(transfer_destination))
                completed.append((transfer_source, transfer_destination))
        except Exception as exc:
            rollback_errors = _rollback_transfers(completed)
            operation["status"] = "rollback_error" if rollback_errors else "rolled_back"
            operation["error"] = str(exc)
            operation["rollback_errors"] = rollback_errors
            if rollback_errors:
                record.update(
                    action="rollback_error",
                    reason=f"move failed: {exc}; rollback errors: {' | '.join(rollback_errors)}",
                )
                counts["errors"] += 1
            else:
                record.update(action="rolled_back", reason=f"move failed and was rolled back: {exc}")
                counts["rolled_back"] += 1
            print(f"[ERR]  {source} -> {destination}: {record['reason']}")
        else:
            operation["status"] = "moved"
            record.update(action="moved", reason="transaction completed", verified="True")
            counts["moved"] += 1
        finally:
            _atomic_json_write(manifest, manifest_path)

    print()
    if dry_run:
        print("Dry run complete. No files were moved.")
    else:
        print(
            "Move complete. "
            f"moved={counts['moved']} skipped={counts['skipped']} "
            f"rolled_back={counts['rolled_back']} errors={counts['errors']}"
        )
        print(f"Move manifest: {manifest_path}")
    return counts


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

def undo_manifest(
    manifest_path: Path,
    apply_changes: bool = False,
    output_manifest: Optional[Path] = None,
) -> int:
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        print(f"ERROR: could not read manifest: {exc}")
        return 1

    operations = manifest.get("operations", [])
    if not isinstance(operations, list):
        print("ERROR: manifest operations field is invalid")
        return 1

    undo_log: Dict[str, Any] = {
        "tool": "inspect_sort_lora.py",
        "tool_version": TOOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "undo_of": str(manifest_path),
        "operations": [],
    }
    if apply_changes:
        output_manifest = _absolute_without_resolving(
            output_manifest
            or manifest_path.with_name(
                f"{manifest_path.stem}_undo_{_utc_stamp()}.json"
            )
        )
        protected_paths = {_normal_path_key(manifest_path)}
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            raw_transfers = operation.get("transfers", [])
            if not isinstance(raw_transfers, list):
                continue
            for item in raw_transfers:
                if not isinstance(item, dict):
                    continue
                for field_name in ("from", "to"):
                    if field_name in item:
                        protected_paths.add(_normal_path_key(Path(item[field_name])))

        if _normal_path_key(output_manifest) in protected_paths:
            print("ERROR: undo manifest overlaps the source manifest or a transfer path")
            return 1
        if output_manifest.exists() or output_manifest.is_symlink():
            print(f"ERROR: undo manifest already exists: {output_manifest}")
            return 1
        try:
            _atomic_json_write(undo_log, output_manifest)
        except Exception as exc:
            print(f"ERROR: could not create undo manifest: {exc}")
            return 1

    unresolved = [
        operation
        for operation in operations
        if isinstance(operation, dict)
        and operation.get("status") in {"pending", "rollback_error"}
    ]
    failures = len(unresolved)
    if unresolved:
        print(
            "WARNING: manifest contains pending or rollback-error operations; "
            "inspect those paths manually"
        )
    processed = 0
    for operation in reversed(operations):
        if operation.get("status") != "moved":
            continue
        raw_transfers = operation.get("transfers", [])
        if not isinstance(raw_transfers, list):
            failures += 1
            continue

        transfers: List[Tuple[Path, Path]] = []
        for item in reversed(raw_transfers):
            if not isinstance(item, dict) or "from" not in item or "to" not in item:
                transfers = []
                break
            transfers.append((Path(item["to"]), Path(item["from"])))
        if not transfers:
            failures += 1
            continue

        source_name = transfers[0][0].name if transfers else "unknown"
        print(f"{'[UNDO]' if apply_changes else '[DRY-UNDO]'} {source_name}")

        ready, error = _preflight_transaction(transfers)
        if not ready:
            print(f"       skipped: {error}")
            failures += 1
            continue
        if not apply_changes:
            processed += 1
            continue

        undo_operation: Dict[str, Any] = {
            "status": "pending",
            "transfers": [
                {"from": str(src), "to": str(dst)} for src, dst in transfers
            ],
            "error": "",
            "rollback_errors": [],
        }
        undo_log["operations"].append(undo_operation)
        _atomic_json_write(undo_log, output_manifest)

        completed: List[Tuple[Path, Path]] = []
        try:
            for transfer_source, transfer_destination in transfers:
                transfer_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(transfer_source), str(transfer_destination))
                completed.append((transfer_source, transfer_destination))
        except Exception as exc:
            rollback_errors = _rollback_transfers(completed)
            undo_operation["status"] = "rollback_error" if rollback_errors else "rolled_back"
            undo_operation["error"] = str(exc)
            undo_operation["rollback_errors"] = rollback_errors
            failures += 1
        else:
            undo_operation["status"] = "undone"
            processed += 1
        finally:
            _atomic_json_write(undo_log, output_manifest)

    if apply_changes:
        print(f"Undo complete. undone={processed} failures={failures}")
        print(f"Undo manifest: {output_manifest}")
    else:
        print(f"Undo dry run complete. reversible operations={processed} failures={failures}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(
    csv_arg: Optional[str] = None,
    base_dir_arg: Optional[str] = None,
    apply_moves: bool = False,
    use_lineage: bool = True,
    min_conf: str = "high",
    allow_ambiguous: bool = False,
    allow_outside_base: bool = False,
    allow_unverified: bool = False,
    manifest_arg: Optional[str] = None,
    max_path_length: Optional[int] = None,
) -> int:
    csv_path = _absolute_without_resolving(Path(csv_arg or "lora_inspect_index.csv"))
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    if apply_moves and not base_dir_arg:
        print("ERROR: --apply requires an explicit --base folder")
        return 2

    # Windows caps ordinary paths at 260 characters (259 usable) unless long
    # paths are enabled system-wide. Default the guard on there; elsewhere it
    # is off (0) but can be forced for staging drives destined for Windows.
    if max_path_length is None:
        max_path_length = 259 if os.name == "nt" else 0

    base_dir = _absolute_without_resolving(Path(base_dir_arg)) if base_dir_arg else Path.cwd()
    print(
        f"inspect_sort_lora: CSV={csv_path} base={base_dir} "
        f"subfolders={'on' if use_lineage else 'off'} min-confidence={min_conf} "
        f"allow-ambiguous={'yes' if allow_ambiguous else 'no'} "
        f"max-path={max_path_length if max_path_length else 'off'} "
        f"apply={'YES' if apply_moves else 'no (dry run)'}"
    )

    try:
        plan = build_sort_plan(
            csv_path=csv_path,
            base_dir=base_dir,
            use_lineage=use_lineage,
            min_conf=min_conf,
            allow_ambiguous=allow_ambiguous,
            allow_outside_base=allow_outside_base,
            allow_unverified=allow_unverified,
            max_path_length=max_path_length,
        )
    except Exception as exc:
        print(f"ERROR: could not build sort plan: {exc}")
        return 1

    summarize_plan(plan)
    plan_path = _plan_output_path(csv_path)
    write_plan_csv(plan, plan_path)
    if not apply_moves:
        print(
            "Review the plan CSV, then apply that plan file with an explicit "
            f"--base: {plan_path}"
        )

    manifest_path = Path(manifest_arg) if manifest_arg else None
    try:
        counts = apply_plan(
            plan=plan,
            csv_path=csv_path,
            base_dir=base_dir,
            dry_run=not apply_moves,
            allow_unverified=allow_unverified,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        print(f"ERROR: sort execution failed: {exc}")
        return 1

    # On apply, persist final actions such as moved, rolled_back, or stale.
    if apply_moves:
        write_plan_csv(plan, plan_path)
        if counts["errors"] or counts["rolled_back"]:
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sort LoRAs into <base>/<arch>[/<subfolder>] from an inspection CSV."
    )
    parser.add_argument("csv", nargs="?", default="lora_inspect_index.csv")
    parser.add_argument(
        "--base",
        help="root folder for architecture buckets; required with --apply",
    )
    parser.add_argument("--apply", action="store_true", help="execute moves; default is dry run")
    parser.add_argument(
        "--no-lineage",
        action="store_true",
        help="use flat <base>/<arch>/ folders without a secondary subtype folder",
    )
    parser.add_argument(
        "--min-confidence",
        choices=("low", "medium", "high"),
        default="high",
        help="minimum confidence eligible for movement (default: high)",
    )
    parser.add_argument(
        "--allow-ambiguous",
        action="store_true",
        help="permit status=ambiguous rows if they meet the confidence threshold",
    )
    parser.add_argument(
        "--allow-outside-base",
        action="store_true",
        help="permit hand-edited destinations outside --base",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="permit legacy CSV rows without inspection fingerprints",
    )
    parser.add_argument("--manifest", help="custom path for the apply or undo manifest")
    parser.add_argument(
        "--max-path-length",
        type=int,
        default=None,
        metavar="N",
        help=(
            "skip moves whose destination path exceeds N characters "
            "(default: 259 on Windows, disabled elsewhere; 0 disables)"
        ),
    )
    parser.add_argument(
        "--undo",
        metavar="MANIFEST",
        help="dry-run reversal of a previous move manifest; add --apply to execute",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.undo:
        return undo_manifest(
            manifest_path=_absolute_without_resolving(Path(args.undo)),
            apply_changes=args.apply,
            output_manifest=Path(args.manifest) if args.manifest else None,
        )

    return main(
        csv_arg=args.csv,
        base_dir_arg=args.base,
        apply_moves=args.apply,
        use_lineage=not args.no_lineage,
        min_conf=args.min_confidence,
        allow_ambiguous=args.allow_ambiguous,
        allow_outside_base=args.allow_outside_base,
        allow_unverified=args.allow_unverified,
        manifest_arg=args.manifest,
        max_path_length=args.max_path_length,
    )


if __name__ == "__main__":
    raise SystemExit(cli())
