"""Read-only real-file smoke inspection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Optional, Sequence

from .inspection import expand_arg, inspect_one_file


def _value(record, field: str, fallback: str = "-") -> str:
    value = str(record.get(field, "") or "").strip()
    return value or fallback


def print_record(record) -> None:
    print("\n" + "=" * 80)
    print(f"LoRA: {_value(record, 'path')}")
    print("=" * 80)
    rows = (
        ("Architecture", _value(record, "arch", "Unknown")),
        ("Structural family", _value(record, "structural_family", "Unknown")),
        ("Status", _value(record, "status", "unknown")),
        ("Confidence", _value(record, "confidence", "low")),
        ("Model version", _value(record, "model_version")),
        ("Task variant", _value(record, "task_variant")),
        ("Noise stage", _value(record, "noise_stage")),
        ("Ecosystem", _value(record, "ecosystem")),
        ("Model variant", _value(record, "model_variant")),
        ("Unresolved hint", _value(record, "unresolved_hint")),
        ("Techniques", _value(record, "techniques")),
        ("Suggested subfolder", _value(record, "sort_subfolder")),
        ("Tensor-key format", _value(record, "format", "unknown")),
        ("Tensor records", _value(record, "total_tensors", "0")),
        ("Header bytes", _value(record, "header_bytes", "0")),
        ("Payload bytes", _value(record, "payload_bytes", "0")),
        ("File bytes", _value(record, "file_size", "0")),
        ("Structure says", _value(record, "structure_arch")),
        ("Metadata says", _value(record, "metadata_arch")),
        ("Filename says", _value(record, "name_arch")),
    )
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")
    if record.get("error"):
        print("\nERROR")
        print(textwrap.fill(str(record["error"]), width=78, subsequent_indent="  "))
    evidence_json = str(record.get("evidence_json", "") or "")
    if evidence_json:
        try:
            evidence = json.loads(evidence_json)
        except json.JSONDecodeError:
            evidence = []
        if evidence:
            print("\nEvidence")
            for item in evidence:
                message = str(item.get("message", "") or "").strip()
                if message:
                    print(textwrap.fill(message, width=76, initial_indent="  - ", subsequent_indent="    "))


def run(inputs: Sequence[str], *, follow_symlinks: bool = False, json_output: bool = False, hash_content: bool = False) -> int:
    candidates = []
    seen = set()
    for raw in inputs:
        for candidate in expand_arg(os.path.expanduser(str(raw))):
            marker = str(candidate.absolute())
            if marker not in seen:
                seen.add(marker)
                candidates.append(candidate)
    if not candidates:
        print("No files matched the supplied input.", file=sys.stderr)
        return 1
    records = [
        inspect_one_file(candidate, follow_symlinks=follow_symlinks, include_content_hash=hash_content)
        for candidate in candidates
    ]
    if json_output:
        print(json.dumps(records, indent=2, ensure_ascii=False))
    else:
        for record in records:
            print_record(record)
        counts = {}
        for record in records:
            status = str(record.get("status", "unknown") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        print(f"\nSmoke inspection complete: files={len(records)}; {rendered}")
        print("No CSV was written and no files were moved.")
    return 1 if any(record.get("status") == "error" for record in records) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only smoke inspection of LoRA safetensors headers.")
    parser.add_argument("inputs", nargs="+", help="safetensors file, folder, or glob")
    parser.add_argument("--follow-symlinks", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--hash-content", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return run(
        args.inputs,
        follow_symlinks=args.follow_symlinks,
        json_output=args.json,
        hash_content=args.hash_content,
    )
