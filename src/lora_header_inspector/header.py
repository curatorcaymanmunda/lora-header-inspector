"""Bounded safetensors-header parsing without loading tensor payloads."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .constants import (
    ALLOWED_DTYPES,
    DEFAULT_MAX_HEADER_BYTES,
    DTYPE_SIZES,
    MAX_TENSOR_DIMENSION,
)
from .models import HeaderData


class DuplicateJSONKeyError(ValueError):
    pass


def _element_count(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _validate_layout(tensors: Mapping[str, Mapping[str, Any]], payload_bytes: int) -> None:
    """Strict layout: dtype-consistent sizes, no holes, no overlaps.

    Zero-length tensors are exempt from tiling but must still sit inside the
    payload bounds, which per-entry validation already guarantees.
    """
    spans: List[Tuple[int, int, str]] = []
    for key, entry in tensors.items():
        dtype = str(entry.get("dtype", ""))
        shape = entry.get("shape", [])
        start, end = entry.get("data_offsets", [0, 0])
        if dtype not in DTYPE_SIZES:
            raise ValueError(
                "strict layout requires a recognised dtype, got %r on %r" % (dtype, key)
            )
        if any(dimension > MAX_TENSOR_DIMENSION for dimension in shape):
            raise ValueError("tensor entry %r has a dimension beyond the schema bound" % key)
        expected = _element_count(shape) * DTYPE_SIZES[dtype]
        if end - start != expected:
            raise ValueError(
                "tensor entry %r declares %d bytes but its shape and dtype require %d"
                % (key, end - start, expected)
            )
        if end > start:
            spans.append((start, end, key))
    spans.sort()
    cursor = 0
    for start, end, key in spans:
        if start < cursor:
            raise ValueError("tensor entry %r overlaps the previous tensor" % key)
        if start > cursor:
            raise ValueError("payload hole before tensor entry %r" % key)
        cursor = end
    if cursor != payload_bytes:
        raise ValueError(
            "declared tensors cover %d bytes of a %d byte payload" % (cursor, payload_bytes)
        )


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError("duplicate JSON key in safetensors header: %r" % key)
        result[key] = value
    return result


def _normalise_metadata(raw: Any, warnings: List[str]) -> Dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        warnings.append("__metadata__ is not an object and was ignored")
        return {}
    result: Dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            warnings.append("non-string metadata key was ignored")
            continue
        if isinstance(value, str):
            result[key] = value
        elif value is None:
            result[key] = ""
            warnings.append("metadata %s was null and was normalised to an empty string" % key)
        else:
            try:
                result[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                result[key] = str(value)
            warnings.append("metadata %s was not a string and was normalised" % key)
    return result


def _validate_tensor_entry(
    key: str,
    value: Any,
    payload_bytes: int,
    warnings: List[str],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("tensor entry %r is not an object" % key)
    dtype = value.get("dtype")
    shape = value.get("shape")
    offsets = value.get("data_offsets")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("tensor entry %r has an invalid dtype" % key)
    if dtype not in ALLOWED_DTYPES:
        warnings.append(
            "tensor entry %s uses unrecognised dtype %s; official loaders may reject this file"
            % (key, dtype)
        )
    if not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape
    ):
        raise ValueError("tensor entry %r has an invalid shape" % key)
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
    ):
        raise ValueError("tensor entry %r has invalid data_offsets" % key)
    start, end = offsets
    if start < 0 or end < start:
        raise ValueError("tensor entry %r has reversed or negative data_offsets" % key)
    if end > payload_bytes:
        raise ValueError(
            "tensor entry %r ends beyond the file payload (%d > %d)"
            % (key, end, payload_bytes)
        )
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [start, end]}


def read_header(
    path: Path,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    validate_tensor_entries: bool = True,
    strict_layout: bool = False,
) -> HeaderData:
    """Read and validate a safetensors header without reading tensor payload bytes."""
    stat = path.stat()
    file_bytes = stat.st_size
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError("file too short to contain a safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len < 2:
            raise ValueError("invalid safetensors header length")
        if header_len > max_header_bytes:
            raise ValueError(
                "header length %d exceeds safety limit %d"
                % (header_len, max_header_bytes)
            )
        if 8 + header_len > file_bytes:
            raise ValueError("declared safetensors header extends beyond end of file")
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError("truncated safetensors header")

    if raw_header[:1] != b"{":
        raise ValueError("safetensors header does not begin with '{'")
    try:
        decoded = raw_header.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("safetensors header is not valid UTF-8") from exc
    try:
        parsed, value_end = json.JSONDecoder(object_pairs_hook=_unique_object).raw_decode(decoded)
    except DuplicateJSONKeyError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError("invalid safetensors JSON header: %s" % exc) from exc
    if decoded[value_end:].strip(" "):
        raise ValueError("bytes after the safetensors JSON value are not space padding")
    if not isinstance(parsed, dict):
        raise ValueError("safetensors header root is not a JSON object")

    warnings: List[str] = []
    metadata = _normalise_metadata(parsed.get("__metadata__"), warnings)
    payload_bytes = file_bytes - 8 - header_len
    tensors: Dict[str, Dict[str, Any]] = {}
    for key, value in parsed.items():
        if key == "__metadata__":
            continue
        if not isinstance(key, str) or not key:
            raise ValueError("safetensors tensor key is not a non-empty string")
        if validate_tensor_entries:
            tensors[key] = _validate_tensor_entry(key, value, payload_bytes, warnings)
        elif isinstance(value, Mapping):
            tensors[key] = dict(value)
        else:
            warnings.append("non-object tensor entry %r was ignored" % key)

    if strict_layout:
        if not validate_tensor_entries:
            raise ValueError("strict layout checking requires tensor entry validation")
        _validate_layout(tensors, payload_bytes)

    return HeaderData(
        tensors=tensors,
        metadata=metadata,
        header_sha256=hashlib.sha256(raw_header).hexdigest(),
        header_bytes=header_len,
        file_bytes=file_bytes,
        warnings=tuple(warnings),
    )


def read_header_hash(
    path: Path, max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES
) -> str:
    """Return only the raw header SHA-256, with the same length guards."""
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError("file too short to contain a safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len < 2 or header_len > max_header_bytes:
            raise ValueError("invalid or implausible safetensors header length")
        if 8 + header_len > file_bytes:
            raise ValueError("declared safetensors header extends beyond end of file")
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError("truncated safetensors header")
    return hashlib.sha256(raw_header).hexdigest()
