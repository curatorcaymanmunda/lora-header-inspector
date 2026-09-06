"""Shared test helpers for small, payload-safe safetensors fixtures."""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path
from typing import Iterable, Mapping, Optional


def tensor(shape: Iterable[int], offsets=(0, 0), dtype: str = "F16"):
    return {
        "dtype": dtype,
        "shape": list(shape),
        "data_offsets": list(offsets),
    }


def write_fake_safetensors(
    path: Path,
    tensors: Mapping[str, Mapping[str, object]],
    metadata: Optional[Mapping[str, object]] = None,
    payload: bytes = b"",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = dict(tensors)
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(payload)
    return path


def write_raw_header(path: Path, raw_header: bytes, payload: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        handle.write(payload)
    return path


def write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
