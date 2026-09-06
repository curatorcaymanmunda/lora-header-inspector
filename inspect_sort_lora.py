#!/usr/bin/env python3
"""Compatibility launcher for the packaged planner and transaction engine."""
from __future__ import annotations
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from lora_header_inspector.cli_sort import main as cli
from lora_header_inspector.duplicates import DUPLICATE_POLICIES, compare_file_group
from lora_header_inspector.fingerprints import verify_record_fingerprint as verify_fingerprint
from lora_header_inspector.paths import safe_component as _safe_component
from lora_header_inspector.planner import (
    build_sort_plan, collision_groups, plan_output_path as _plan_output_path,
    print_collision_report, run_plan, summarize_plan, write_plan_csv,
)
from lora_header_inspector.sidecars import discover_sidecars, is_safe_sidecar
from lora_header_inspector.transactions import apply_plan, recover_manifest, undo_manifest
from lora_header_inspector.version import TOOL_VERSION
sidecars = discover_sidecars
truthy = lambda value: str(value or "").strip().lower() in {"true", "1", "yes", "y", "on"}

def main(
    csv_arg=None,
    base_dir_arg=None,
    apply_moves=False,
    use_lineage=True,
    min_conf="high",
    allow_ambiguous=False,
    allow_outside_base=False,
    allow_unverified=False,
    manifest_arg=None,
    max_path_length=None,
    duplicate_policy="report",
    quarantine_dir="_Duplicates",
    allow_legacy_plan=False,
    allow_symlink_parents=False,
    collision_detail="summary",
    allow_workflow_path_breakage=False,
):
    csv_path = Path(csv_arg or "lora_inspect_index.csv")
    base_dir = Path(base_dir_arg) if base_dir_arg else Path.cwd()
    return run_plan(
        csv_path=csv_path,
        base_dir=base_dir,
        apply_changes=apply_moves,
        use_lineage=use_lineage,
        min_confidence=min_conf,
        allow_ambiguous=allow_ambiguous,
        allow_outside_base=allow_outside_base,
        allow_unverified=allow_unverified,
        max_path_length=max_path_length,
        duplicate_policy=duplicate_policy,
        quarantine_dir=quarantine_dir,
        allow_legacy_plan=allow_legacy_plan,
        allow_symlink_parents=allow_symlink_parents,
        collision_detail=collision_detail,
        manifest_path=Path(manifest_arg) if manifest_arg else None,
        allow_workflow_path_breakage=allow_workflow_path_breakage,
    )

if __name__ == "__main__":
    raise SystemExit(cli())
