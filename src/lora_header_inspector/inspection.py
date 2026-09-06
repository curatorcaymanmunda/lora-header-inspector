"""File collection, header-only inspection, and the inspection CLI."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from urllib.parse import urlsplit, urlunsplit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set

from .classifier import classify
from .fingerprints import HashCache, full_file_sha256
from .constants import DEFAULT_MAX_HEADER_BYTES
from .header import HeaderData, read_header
from .normalization import KeyIndex, detect_format
from .schema import (
    INSPECTION_FIELDNAMES,
    blank_inspection_record,
    inspection_schema_fields,
    write_csv_atomic,
    write_json_atomic,
)
from .utils import absolute_without_resolving, stable_id
from .version import (
    SIGNATURE_REPORT_SCHEMA_NAME,
    SIGNATURE_REPORT_SCHEMA_VERSION,
    TOOL_VERSION,
)

CSV_NAME = "lora_inspect_index.csv"
FIELDNAMES = INSPECTION_FIELDNAMES

IDENTITY_METADATA_FIELDS = (
    "modelspec.architecture",
    "architecture",
    "model_type",
    "ss_base_model_version",
    "ss_base_model",
    "base_model",
    "base_model_name",
    "base_model_name_or_path",
    "pretrained_model_name_or_path",
    "ss_sd_model_name",
)


def _changed(before, after) -> bool:
    return (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    )


def inspect_one_file(
    path: Path,
    follow_symlinks: bool = False,
    include_full_hash: bool = False,
    hash_cache: Optional[HashCache] = None,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    strict_layout: bool = False,
    **compatibility,
) -> Dict[str, str]:
    """Inspect one safetensors file without loading its tensor payload.

    ``include_content_hash`` is accepted as a compatibility spelling for early
    v0.6 prototypes.  A full hash is optional because reading every multi-GB
    payload would defeat the normal header-only fast path.
    """
    if "include_content_hash" in compatibility:
        include_full_hash = bool(compatibility.pop("include_content_hash"))
    if compatibility:
        unexpected = ", ".join(sorted(compatibility))
        raise TypeError("unexpected inspect_one_file option(s): %s" % unexpected)

    input_path = absolute_without_resolving(path)
    record = blank_inspection_record()
    record.update(inspection_schema_fields())
    record.update(
        input_path=str(input_path),
        path=str(input_path),
        source_was_symlink=str(input_path.is_symlink()),
        arch="Unknown",
        structural_family="Unknown",
        confidence="low",
        status="ambiguous",
    )

    if input_path.is_symlink():
        if not follow_symlinks:
            record.update(
                status="error",
                error="symlink skipped; use --follow-symlinks to inspect and record its target",
            )
            record["record_id"] = stable_id("record", input_path, "symlink")
            return record
        try:
            actual_path = input_path.resolve(strict=True)
        except OSError as exc:
            record.update(status="error", error="broken or unreadable symlink: %s" % exc)
            record["record_id"] = stable_id("record", input_path, "broken-symlink")
            return record
        record["path"] = str(actual_path)
    else:
        actual_path = input_path

    try:
        before = actual_path.stat()
        header_data = read_header(
            actual_path,
            max_header_bytes=max_header_bytes,
            strict_layout=strict_layout,
        )
        after = actual_path.stat()
    except Exception as exc:
        record.update(status="error", error=str(exc))
        record["record_id"] = stable_id("record", actual_path, record["error"])
        return record

    if _changed(before, after):
        record.update(
            status="error",
            error="file changed while its header was being inspected; retry",
        )
        record["record_id"] = stable_id("record", actual_path, "changed-during-read")
        return record

    content_hash = ""
    if include_full_hash:
        try:
            cache = hash_cache if hash_cache is not None else HashCache()
            content_hash = full_file_sha256(actual_path, cache)
            final = actual_path.stat()
            if _changed(after, final):
                raise ValueError("file changed while its full hash was being calculated")
        except Exception as exc:
            record.update(status="error", error=str(exc))
            record["record_id"] = stable_id("record", actual_path, record["error"])
            return record

    record.update(
        file_size=str(after.st_size),
        mtime_ns=str(after.st_mtime_ns),
        ctime_ns=str(after.st_ctime_ns),
        device=str(after.st_dev),
        inode=str(after.st_ino),
        header_sha256=header_data.header_sha256,
        full_sha256=content_hash,
        header_bytes=str(header_data.header_bytes),
        payload_bytes=str(header_data.payload_bytes),
        header_warnings_json=json.dumps(list(header_data.warnings), ensure_ascii=False),
        total_tensors=str(len(header_data.tensors)),
        format=detect_format(list(header_data.tensors)),
    )
    record["record_id"] = stable_id(
        "record",
        actual_path,
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        header_data.header_sha256,
    )

    try:
        record.update(classify(header_data.tensors, header_data.metadata, actual_path.name))
    except Exception as exc:
        record.update(status="error", error="classification failed: %s" % exc)
    return record


def expand_arg(argument: str) -> List[Path]:
    argument = os.path.expanduser(argument)
    path = Path(argument)
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.suffix.lower() == ".safetensors"
            and (candidate.is_file() or candidate.is_symlink())
        )
    # A literal existing path wins over glob parsing.  Unix filenames may
    # legally contain square brackets.
    if path.exists() or path.is_symlink():
        return [path]
    if any(character in argument for character in "*?["):
        return [
            Path(match)
            for match in glob.glob(argument, recursive=True)
            if Path(match).is_symlink() or not Path(match).is_dir()
        ]
    return [path]


def _dedupe_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def collect_files(arguments: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    seen: Set[str] = set()
    for argument in arguments:
        expanded = expand_arg(argument)
        if not expanded:
            print("No matches for: %s" % argument)
        for path in expanded:
            if path.suffix.lower() != ".safetensors":
                continue
            key = _dedupe_key(path)
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def enumerate_only_report(arguments: Sequence[str]) -> int:
    """Count matching files per bucket without reading a single header.

    A directory target groups its matches by first path component below the
    target, so a sorted library reports one line per bucket. Loose files at
    the target root count under ".". Non-directory targets (files, globs)
    group under the argument that matched them. Deduplication matches
    collect_files exactly, so the total is the number of records a full
    inspection of the same targets would produce.
    """
    seen: Set[str] = set()
    buckets: Dict[str, int] = {}
    total = 0
    for argument in arguments:
        expanded = expand_arg(argument)
        if not expanded:
            print("No matches for: %s" % argument)
            continue
        root = Path(os.path.expanduser(argument))
        is_directory = root.is_dir()
        for path in expanded:
            if path.suffix.lower() != ".safetensors":
                continue
            key = _dedupe_key(path)
            if key in seen:
                continue
            seen.add(key)
            total += 1
            if is_directory:
                relative = absolute_without_resolving(path).relative_to(
                    absolute_without_resolving(root)
                )
                bucket = relative.parts[0] if len(relative.parts) > 1 else "."
                label = "%s/%s" % (root, bucket) if bucket != "." else str(root)
            else:
                label = argument
            buckets[label] = buckets.get(label, 0) + 1
    for label in sorted(buckets, key=lambda name: (-buckets[name], name)):
        print("%6d  %s" % (buckets[label], label))
    print("%6d  total" % total)
    return 0


def infer_common_base(files: Sequence[Path]) -> Optional[Path]:
    if not files:
        return None
    parents = [str(absolute_without_resolving(path).parent) for path in files]
    try:
        return Path(os.path.commonpath(parents))
    except ValueError:
        return None


def inspect_files(
    files: Sequence[Path],
    *,
    follow_symlinks: bool = False,
    include_full_hash: bool = False,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    strict_layout: bool = False,
    jobs: int = 1,
) -> List[Dict[str, str]]:
    """Inspect files serially, or with a thread pool when jobs > 1.

    Threads are effective here because hashlib and file reads release the GIL.
    The shared HashCache needs no lock: collect_files deduplicates paths, so
    no two workers ever compute the same cache key, and single-key dict access
    is atomic in CPython. Row order always matches the input order, so serial
    and parallel runs produce byte-identical CSVs.
    """
    cache = HashCache()

    def worker(path: Path) -> Dict[str, str]:
        return inspect_one_file(
            path,
            follow_symlinks=follow_symlinks,
            include_full_hash=include_full_hash,
            hash_cache=cache,
            max_header_bytes=max_header_bytes,
            strict_layout=strict_layout,
        )

    if jobs <= 1 or len(files) <= 1:
        return [worker(path) for path in files]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(worker, files))


def write_inspection_csv(rows: Sequence[Mapping[str, str]], output_path: Path) -> None:
    write_csv_atomic(rows, output_path, INSPECTION_FIELDNAMES)


# Compatibility alias used by the original standalone script.
write_csv = write_inspection_csv


def _sanitize_fixture_metadata_value(value: str) -> tuple[str, str]:
    """Preserve model identity while removing local paths and URL credentials."""
    text = str(value or "").strip()
    if not text:
        return "", ""
    if len(text) > 2048:
        text = text[:2048]
        note = "value truncated to 2048 characters"
    else:
        note = ""

    # URL query strings and fragments frequently contain signed credentials.
    parsed = urlsplit(text)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        hostname = parsed.hostname or parsed.netloc.split("@")[-1]
        netloc = hostname
        if parsed.port:
            netloc += ":%d" % parsed.port
        sanitized = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        redaction = "URL credentials/query/fragment removed" if sanitized != text else note
        return sanitized, redaction or note

    # Local trainer paths are not architecture evidence and can expose usernames.
    windows_absolute = re.match(r"^[A-Za-z]:[\\/]", text) is not None
    unc_path = text.startswith("\\\\")
    posix_absolute = text.startswith("/")
    file_uri = text.lower().startswith("file://")
    if windows_absolute or unc_path or posix_absolute or file_uri:
        normalized = text.replace("\\", "/").rstrip("/")
        leaf = normalized.rsplit("/", 1)[-1] or "redacted"
        return "local-path:" + leaf, "local filesystem path reduced to its final component"
    return text, note


def anonymized_signature_record(
    path: Path,
    *,
    include_filename: bool = False,
    include_identity_metadata: bool = True,
) -> Dict[str, object]:
    """Produce a payload-free fixture candidate for signature research.

    No tensor data offsets or payload bytes are included.  Absolute paths are
    never emitted.  Filenames are omitted unless the caller opts in.
    """
    actual = absolute_without_resolving(path)
    header = read_header(actual)
    classification = classify(header.tensors, header.metadata, actual.name)
    index = KeyIndex(header.tensors)
    metadata: Dict[str, str] = {}
    metadata_redactions: Dict[str, str] = {}
    if include_identity_metadata:
        for key in IDENTITY_METADATA_FIELDS:
            if key not in header.metadata or not header.metadata[key]:
                continue
            sanitized, redaction = _sanitize_fixture_metadata_value(header.metadata[key])
            metadata[key] = sanitized
            if redaction:
                metadata_redactions[key] = redaction
    return {
        "schema_name": SIGNATURE_REPORT_SCHEMA_NAME,
        "schema_version": SIGNATURE_REPORT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "file_id": stable_id("fixture", header.header_sha256, header.file_bytes),
        "filename": actual.name if include_filename else "",
        "file_bytes": header.file_bytes,
        "header_sha256": header.header_sha256,
        "tensor_count": len(header.tensors),
        "format": classification.get("format", "unknown"),
        "classification": {
            key: classification.get(key, "")
            for key in (
                "arch",
                "structural_family",
                "confidence",
                "status",
                "ecosystem",
                "model_variant",
                "model_version",
                "task_variant",
                "noise_stage",
            )
        },
        "identity_metadata": metadata,
        "metadata_redactions": metadata_redactions,
        "module_prefix_histogram": index.module_prefix_histogram(depth=6)[:200],
        "tensors": [
            {
                "key": key,
                "dtype": value.get("dtype"),
                "shape": value.get("shape"),
            }
            for key, value in header.tensors.items()
        ],
    }


def write_signature_report(
    files: Sequence[Path],
    output_path: Path,
    *,
    include_filenames: bool = False,
    include_identity_metadata: bool = True,
) -> None:
    records = [
        anonymized_signature_record(
            path,
            include_filename=include_filenames,
            include_identity_metadata=include_identity_metadata,
        )
        for path in files
    ]
    payload: Dict[str, object] = {
        "schema_name": SIGNATURE_REPORT_SCHEMA_NAME,
        "schema_version": SIGNATURE_REPORT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "record_count": len(records),
        "records": records,
    }
    write_json_atomic(payload, output_path)


def _print_record(record: Mapping[str, str], source: Path) -> None:
    subtype = str(record.get("sort_subfolder", "") or "")
    architecture = str(record.get("arch", "Unknown") or "Unknown")
    tag = "%s/%s" % (architecture, subtype) if subtype else architecture
    error = str(record.get("error", "") or "")
    suffix = " ERROR: %s" % error[:100] if error else ""
    print(
        "%s: %s [%s; %s; %s]%s"
        % (
            source.name,
            tag,
            record.get("confidence", "low"),
            record.get("status", "ambiguous"),
            record.get("structural_family", "Unknown"),
            suffix,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect LoRA safetensors headers and write a conservative, versioned CSV index."
    )
    parser.add_argument("targets", nargs="+", help="safetensors file, glob, or folder")
    parser.add_argument("--output", default=CSV_NAME, help="inspection CSV path")
    parser.add_argument("--base", help="base folder for the automatic dry-run plan")
    parser.add_argument("--follow-symlinks", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="inspect and hash with this many threads (default 1; try your "
        "physical core count on SSD/NVMe, keep 1 on spinning disks)",
    )
    parser.add_argument(
        "--enumerate-only",
        action="store_true",
        help="count matching safetensors files per bucket and exit without "
        "reading any headers",
    )
    parser.add_argument(
        "--max-header-bytes",
        type=int,
        default=DEFAULT_MAX_HEADER_BYTES,
        help="reject headers whose declared length exceeds this many bytes "
        "(default matches the official safetensors limit)",
    )
    parser.add_argument(
        "--strict-layout",
        action="store_true",
        help="additionally require recognised dtypes, shape-consistent sizes, "
        "and a payload with no holes or overlapping tensors",
    )
    parser.add_argument(
        "--hash-content",
        action="store_true",
        help="also stream a full SHA-256 for every model (slow; collision planning hashes lazily)",
    )
    parser.add_argument(
        "--no-sort-plan",
        action="store_true",
        help="write only the inspection CSV; do not generate a dry-run sort plan",
    )
    parser.add_argument(
        "--signature-report",
        metavar="PATH",
        help="also write a payload-free structural fixture report",
    )
    parser.add_argument(
        "--include-fixture-filenames",
        action="store_true",
        help="include model filenames in --signature-report (paths are never included)",
    )
    parser.add_argument(
        "--omit-fixture-metadata",
        action="store_true",
        help="omit trusted identity metadata from --signature-report",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.enumerate_only:
        return enumerate_only_report(args.targets)
    files = collect_files(args.targets)
    if not files:
        print("No .safetensors files found.")
        return 1

    rows = inspect_files(
        files,
        follow_symlinks=args.follow_symlinks,
        include_full_hash=args.hash_content,
        max_header_bytes=args.max_header_bytes,
        strict_layout=args.strict_layout,
        jobs=max(1, args.jobs),
    )
    for source, record in zip(files, rows):
        _print_record(record, source)

    output_path = absolute_without_resolving(Path(args.output))
    write_inspection_csv(rows, output_path)
    print("\nWrote %d records to %s" % (len(rows), output_path))

    counts = Counter(str(row.get("status", "unknown") or "unknown") for row in rows)
    print("Status summary: " + ", ".join("%s=%d" % item for item in sorted(counts.items())))

    if args.signature_report:
        signature_path = absolute_without_resolving(Path(args.signature_report))
        write_signature_report(
            files,
            signature_path,
            include_filenames=args.include_fixture_filenames,
            include_identity_metadata=not args.omit_fixture_metadata,
        )
        print("Signature report written to: %s" % signature_path)

    if args.no_sort_plan:
        return 1 if counts.get("error", 0) else 0

    try:
        from .planner import run_plan
    except ImportError:
        print("Planner package not available; skipping the dry-run sort plan.")
        return 1 if counts.get("error", 0) else 0

    base = absolute_without_resolving(Path(args.base)) if args.base else infer_common_base(files)
    if base is None:
        print("Could not infer a common base folder; skipping the dry-run sort plan.")
        return 1 if counts.get("error", 0) else 0

    plan_result = run_plan(
        csv_path=output_path,
        base_dir=base,
        apply_changes=False,
        min_confidence="high",
        duplicate_policy="report",
        collision_detail="summary",
    )
    return 1 if counts.get("error", 0) or plan_result else 0


__all__ = [
    "CSV_NAME",
    "FIELDNAMES",
    "HeaderData",
    "anonymized_signature_record",
    "collect_files",
    "expand_arg",
    "infer_common_base",
    "inspect_files",
    "inspect_one_file",
    "main",
    "read_header",
    "write_csv",
    "write_inspection_csv",
    "write_signature_report",
]
