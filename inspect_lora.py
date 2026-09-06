#!/usr/bin/env python3
"""Compatibility launcher for the packaged inspector."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from lora_header_inspector.classifier import classify, classify_result
from lora_header_inspector.header import HeaderData, read_header
from lora_header_inspector.inspection import (
    CSV_NAME, FIELDNAMES, collect_files, expand_arg, infer_common_base,
    inspect_files, inspect_one_file, main, write_csv,
)
from lora_header_inspector.normalization import detect_format
from lora_header_inspector.version import TOOL_VERSION
__all__ = [
    "CSV_NAME", "FIELDNAMES", "HeaderData", "TOOL_VERSION", "classify",
    "classify_result", "collect_files", "detect_format", "expand_arg",
    "infer_common_base", "inspect_files", "inspect_one_file", "main", "read_header", "write_csv",
]
if __name__ == "__main__":
    raise SystemExit(main())
