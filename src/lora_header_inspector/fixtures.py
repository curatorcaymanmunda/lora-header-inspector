"""Portable, payload-free structural signature fixtures.

Fixtures are deliberately data, not Python.  A contributor can submit a
sanitised safetensors-header description and the classification it is expected
to produce without redistributing model weights, prompts, local paths, or tensor
offsets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .classifier import classify
from .utils import atomic_json_write
from .version import (
    SIGNATURE_FIXTURE_SCHEMA_NAME,
    SIGNATURE_FIXTURE_SCHEMA_VERSION,
    TOOL_VERSION,
)

MAX_FIXTURE_TENSORS = 250_000
_EXPECTED_FIELDS = (
    "arch",
    "structural_family",
    "status",
    "confidence",
    "ecosystem",
    "model_variant",
    "model_version",
    "task_variant",
    "noise_stage",
    "techniques",
)
_SOURCE_TYPES = {"synthetic", "real-header-derived", "community-submitted"}


class FixtureError(ValueError):
    """Raised when a signature fixture is unsafe or structurally invalid."""


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FixtureError(f"duplicate JSON key in fixture: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class FixtureValidation:
    path: str
    fixture_id: str
    valid: bool
    matched: bool
    errors: Tuple[str, ...]
    expected: Mapping[str, str]
    actual: Mapping[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "fixture_id": self.fixture_id,
            "valid": self.valid,
            "matched": self.matched,
            "errors": list(self.errors),
            "expected": dict(self.expected),
            "actual": dict(self.actual),
        }


def _is_absolute_or_pathlike(value: str) -> bool:
    text = str(value or "")
    if not text:
        return False
    return bool(
        os.path.isabs(text)
        or re.match(r"^[A-Za-z]:[\\/]", text)
        or text.startswith("\\\\")
        or "/" in text
        or "\\" in text
    )


def _string_mapping(value: Any, field: str) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise FixtureError(f"{field} must be a JSON object")
    result: Dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise FixtureError(f"{field} contains an empty or non-string key")
        if isinstance(item, (dict, list)):
            raise FixtureError(f"{field}.{key} must be a scalar value")
        result[key] = "" if item is None else str(item)
    return result


def _validate_provenance(value: Any) -> Dict[str, str]:
    provenance = _string_mapping(value, "provenance")
    source_type = provenance.get("source_type", "").strip()
    if source_type not in _SOURCE_TYPES:
        raise FixtureError(
            "provenance.source_type must be one of: " + ", ".join(sorted(_SOURCE_TYPES))
        )
    for key, item in provenance.items():
        if key.endswith("path") and _is_absolute_or_pathlike(item):
            raise FixtureError(f"provenance.{key} must not contain a filesystem path")
    return provenance


def _validate_tensor_records(value: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not isinstance(value, list):
        raise FixtureError("tensors must be a JSON array")
    if len(value) > MAX_FIXTURE_TENSORS:
        raise FixtureError(f"fixture contains more than {MAX_FIXTURE_TENSORS:,} tensors")

    records: List[Dict[str, Any]] = []
    reconstructed: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise FixtureError(f"tensors[{index}] must be a JSON object")
        forbidden = sorted(set(raw).intersection({"data_offsets", "data", "payload", "bytes"}))
        if forbidden:
            raise FixtureError(
                f"tensors[{index}] contains payload/offset field(s): {', '.join(forbidden)}"
            )
        key = raw.get("key")
        dtype = raw.get("dtype", "F16")
        shape = raw.get("shape")
        if not isinstance(key, str) or not key.strip():
            raise FixtureError(f"tensors[{index}].key must be a non-empty string")
        if _is_absolute_or_pathlike(key) and (key.startswith("/") or re.match(r"^[A-Za-z]:", key)):
            raise FixtureError(f"tensors[{index}].key looks like a filesystem path")
        if key in reconstructed:
            raise FixtureError(f"duplicate tensor key in fixture: {key}")
        if not isinstance(dtype, str) or not dtype.strip():
            raise FixtureError(f"tensors[{index}].dtype must be a non-empty string")
        if not isinstance(shape, list) or not shape:
            raise FixtureError(f"tensors[{index}].shape must be a non-empty integer array")
        clean_shape: List[int] = []
        for axis, dimension in enumerate(shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                raise FixtureError(
                    f"tensors[{index}].shape[{axis}] must be a non-negative integer"
                )
            clean_shape.append(int(dimension))
        record = {"key": key, "dtype": dtype, "shape": clean_shape}
        records.append(record)
        reconstructed[key] = {
            "dtype": dtype,
            "shape": clean_shape,
            # Classifiers never inspect offsets, but the in-memory tensor record
            # follows the safetensors header shape used by the production reader.
            "data_offsets": [0, 0],
        }
    return records, reconstructed


def validate_fixture_payload(payload: Any, *, source: str = "<memory>") -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Validate and normalise one fixture, returning it and its classification."""
    if not isinstance(payload, Mapping):
        raise FixtureError("fixture root must be a JSON object")
    schema_name = str(payload.get("schema_name", "") or "")
    schema_version = str(payload.get("schema_version", "") or "")
    if schema_name != SIGNATURE_FIXTURE_SCHEMA_NAME:
        raise FixtureError(f"unsupported fixture schema name {schema_name!r}")
    if schema_version != SIGNATURE_FIXTURE_SCHEMA_VERSION:
        raise FixtureError(
            f"unsupported fixture schema version {schema_version!r}; "
            f"expected {SIGNATURE_FIXTURE_SCHEMA_VERSION}"
        )

    fixture_id = str(payload.get("fixture_id", "") or "").strip()
    if not fixture_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", fixture_id):
        raise FixtureError("fixture_id must be 1-128 portable identifier characters")

    provenance = _validate_provenance(payload.get("provenance"))
    filename = str(payload.get("filename", "") or "").strip()
    if filename and _is_absolute_or_pathlike(filename):
        raise FixtureError("filename must be a basename, not a path")
    if filename and not filename.lower().endswith(".safetensors"):
        raise FixtureError("filename must end in .safetensors when supplied")

    metadata = _string_mapping(payload.get("metadata", {}), "metadata")
    for key, item in metadata.items():
        if key.lower().endswith(("path", "directory", "dir")) and _is_absolute_or_pathlike(item):
            raise FixtureError(f"metadata.{key} must not expose a local filesystem path")

    tensor_records, tensors = _validate_tensor_records(payload.get("tensors"))
    expected = _string_mapping(payload.get("expected", {}), "expected")
    if not expected:
        raise FixtureError("expected must contain at least one asserted result field")
    unknown_expected = sorted(set(expected) - set(_EXPECTED_FIELDS))
    if unknown_expected:
        raise FixtureError("expected contains unsupported field(s): " + ", ".join(unknown_expected))

    actual = classify(tensors, metadata, filename or "fixture.safetensors")
    normalised: Dict[str, Any] = {
        "schema_name": SIGNATURE_FIXTURE_SCHEMA_NAME,
        "schema_version": SIGNATURE_FIXTURE_SCHEMA_VERSION,
        "tool_version": str(payload.get("tool_version", TOOL_VERSION) or TOOL_VERSION),
        "fixture_id": fixture_id,
        "provenance": provenance,
        "filename": filename,
        "metadata": metadata,
        "tensors": tensor_records,
        "expected": expected,
    }
    return normalised, actual


def load_fixture(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixtureError(f"could not read fixture {path}: {exc}") from exc


def validate_fixture(path: Path) -> FixtureValidation:
    errors: List[str] = []
    fixture_id = ""
    expected: Dict[str, str] = {}
    actual: Dict[str, str] = {}
    try:
        payload = load_fixture(path)
        fixture_id = str(payload.get("fixture_id", "") or "") if isinstance(payload, Mapping) else ""
        normalised, actual = validate_fixture_payload(payload, source=str(path))
        expected = dict(normalised["expected"])
        for field, value in expected.items():
            if str(actual.get(field, "") or "") != value:
                errors.append(
                    f"{field}: expected {value!r}, got {str(actual.get(field, '') or '')!r}"
                )
    except FixtureError as exc:
        errors.append(str(exc))
    return FixtureValidation(
        path=str(path),
        fixture_id=fixture_id,
        valid=not errors or bool(actual),
        matched=not errors,
        errors=tuple(errors),
        expected=expected,
        actual={field: str(actual.get(field, "") or "") for field in _EXPECTED_FIELDS},
    )


def collect_fixture_paths(arguments: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for argument in arguments:
        candidate = Path(os.path.expanduser(argument))
        matches: Iterable[Path]
        if candidate.is_dir():
            matches = sorted(candidate.rglob("*.json"))
        else:
            matches = [candidate]
        for path in matches:
            marker = os.path.normcase(os.path.abspath(os.fspath(path)))
            if marker not in seen:
                seen.add(marker)
                paths.append(path)
    return paths


def validate_fixture_paths(paths: Sequence[Path]) -> List[FixtureValidation]:
    return [validate_fixture(path) for path in paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-fixtures",
        description="Validate payload-free LoRA structural signature fixtures.",
    )
    parser.add_argument("fixtures", nargs="+", help="fixture JSON file or directory")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable validation report")
    parser.add_argument("--output", help="write the JSON report atomically to this path")
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    paths = collect_fixture_paths(args.fixtures)
    if not paths:
        print("No fixture JSON files found.")
        return 1
    results = validate_fixture_paths(paths)
    payload = {
        "schema_name": "lora-signature-fixture-validation",
        "schema_version": "1.0",
        "tool_version": TOOL_VERSION,
        "fixture_count": len(results),
        "passed": sum(result.matched for result in results),
        "failed": sum(not result.matched for result in results),
        "results": [result.to_dict() for result in results],
    }
    if args.output:
        atomic_json_write(payload, Path(args.output))
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for result in results:
            label = "PASS" if result.matched else "FAIL"
            print(f"[{label}] {result.fixture_id or Path(result.path).name}")
            for error in result.errors:
                print(f"       {error}")
        print(
            f"Fixture validation complete: total={len(results)} "
            f"passed={payload['passed']} failed={payload['failed']}"
        )
        if args.output:
            print(f"Validation report: {args.output}")
    return 1 if payload["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
