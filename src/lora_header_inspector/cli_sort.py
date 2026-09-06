"""Command-line interface for plan, apply, undo, and recovery."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

from .duplicates import DUPLICATE_POLICIES
from .paths import absolute_without_resolving
from .planner import run_plan
from .transactions import recover_manifest, undo_manifest
from .version import TOOL_VERSION


def _infer_setting(csv_path: Path, field: str) -> str:
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            values = {
                str(row.get(field, "") or "").strip()
                for row in csv.DictReader(handle)
                if str(row.get(field, "") or "").strip()
            }
    except OSError:
        return ""
    return next(iter(values)) if len(values) == 1 else ""


def _parse_bool(value: str, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _infer_bool(csv_path: Path, field: str, default: bool) -> bool:
    return _parse_bool(_infer_setting(csv_path, field), default)


def _infer_int(csv_path: Path, field: str, default: Optional[int]) -> Optional[int]:
    value = _infer_setting(csv_path, field)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default




def _resolve_relaxation(
    *,
    cli_value: Optional[bool],
    inferred_value: bool,
    apply_changes: bool,
    flag: str,
) -> tuple[Optional[bool], str]:
    """Require dangerous plan relaxations to be re-acknowledged on apply."""
    if cli_value is not None:
        return cli_value, ""
    if apply_changes and inferred_value:
        return None, (
            f"reviewed plan enables {flag}; repeat {flag} explicitly with --apply "
            "or use its rejecting counterpart to tighten the plan"
        )
    return inferred_value, ""

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, review, apply, undo, or recover a conservative LoRA library plan."
    )
    parser.add_argument("csv", nargs="?", default="lora_inspect_index.csv")
    parser.add_argument("--base", help="root folder for architecture buckets; required with --apply")
    parser.add_argument("--apply", action="store_true", help="execute operations; default is dry run")
    lineage = parser.add_mutually_exclusive_group()
    lineage.add_argument("--lineage", dest="use_lineage", action="store_true", help="use <base>/<arch>/<subfolder> folders")
    lineage.add_argument("--no-lineage", dest="use_lineage", action="store_false", help="use flat <base>/<arch>/ folders")
    parser.set_defaults(use_lineage=None)
    parser.add_argument("--min-confidence", choices=("low", "medium", "high"), default=None)

    ambiguous = parser.add_mutually_exclusive_group()
    ambiguous.add_argument("--allow-ambiguous", dest="allow_ambiguous", action="store_true")
    ambiguous.add_argument("--reject-ambiguous", dest="allow_ambiguous", action="store_false")
    parser.set_defaults(allow_ambiguous=None)

    containment = parser.add_mutually_exclusive_group()
    containment.add_argument("--allow-outside-base", dest="allow_outside_base", action="store_true")
    containment.add_argument("--confine-to-base", dest="allow_outside_base", action="store_false")
    parser.set_defaults(allow_outside_base=None)

    symlinks = parser.add_mutually_exclusive_group()
    symlinks.add_argument(
        "--allow-symlink-parents",
        dest="allow_symlink_parents",
        action="store_true",
        help="permit source/destination subdirectories that are symlinks; topology is still snapshotted",
    )
    symlinks.add_argument("--reject-symlink-parents", dest="allow_symlink_parents", action="store_false")
    parser.set_defaults(allow_symlink_parents=None)

    verification = parser.add_mutually_exclusive_group()
    verification.add_argument("--allow-unverified", dest="allow_unverified", action="store_true", help="allow legacy rows without fingerprints")
    verification.add_argument("--require-verification", dest="allow_unverified", action="store_false")
    verification.add_argument(
        "--trust-recorded-hashes",
        action="store_true",
        help="skip plan-time re-hashing when a record's full hash was taken at "
        "the file's current device, inode, size, mtime, and ctime; identity "
        "changes still force a re-hash, and apply-time duplicate relinks still "
        "verify digests against real bytes",
    )
    parser.set_defaults(allow_unverified=None)
    parser.add_argument("--allow-legacy-plan", action="store_true", help="read an unversioned v0.x CSV")
    parser.add_argument(
        "--duplicate-policy",
        choices=DUPLICATE_POLICIES,
        default=None,
        help="report, hardlink, or quarantine; reviewed plans retain this setting",
    )
    parser.add_argument("--quarantine-dir", default=None, help="quarantine folder under --base")
    parser.add_argument(
        "--collision-detail",
        choices=("none", "summary", "full"),
        default="summary",
    )
    parser.add_argument("--allow-workflow-path-breakage", action="store_true")
    parser.add_argument("--manifest", help="custom apply, undo, or recovery manifest path")
    parser.add_argument("--max-path-length", type=int, default=None, metavar="N")
    parser.add_argument("--undo", metavar="MANIFEST", help="dry-run undo; add --apply to execute")
    parser.add_argument("--recover", metavar="MANIFEST", help="dry-run interrupted-operation recovery; add --apply")
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.undo and args.recover:
        print("ERROR: --undo and --recover are mutually exclusive")
        return 2
    if args.undo:
        return undo_manifest(
            absolute_without_resolving(Path(args.undo)),
            apply_changes=args.apply,
            output_manifest=Path(args.manifest) if args.manifest else None,
        )
    if args.recover:
        return recover_manifest(
            absolute_without_resolving(Path(args.recover)),
            apply_changes=args.apply,
            output_manifest=Path(args.manifest) if args.manifest else None,
        )

    csv_path = absolute_without_resolving(Path(args.csv))
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    if args.apply and not args.base:
        print("ERROR: --apply requires an explicit --base folder")
        return 2
    inferred_base = _infer_setting(csv_path, "plan_base_dir")
    base_dir = absolute_without_resolving(
        Path(args.base or inferred_base or Path.cwd())
    )

    use_lineage = args.use_lineage if args.use_lineage is not None else _infer_bool(csv_path, "use_lineage", True)
    min_confidence = args.min_confidence or _infer_setting(csv_path, "min_confidence") or "high"

    relaxation_specs = (
        ("allow_ambiguous", args.allow_ambiguous, "--allow-ambiguous"),
        ("allow_outside_base", args.allow_outside_base, "--allow-outside-base"),
        ("allow_unverified", args.allow_unverified, "--allow-unverified"),
        ("allow_symlink_parents", args.allow_symlink_parents, "--allow-symlink-parents"),
    )
    relaxation_values = {}
    for field, cli_value, flag in relaxation_specs:
        resolved, error = _resolve_relaxation(
            cli_value=cli_value,
            inferred_value=_infer_bool(csv_path, field, False),
            apply_changes=args.apply,
            flag=flag,
        )
        if error:
            print("ERROR: " + error)
            return 2
        relaxation_values[field] = bool(resolved)
    allow_ambiguous = relaxation_values["allow_ambiguous"]
    allow_outside_base = relaxation_values["allow_outside_base"]
    allow_unverified = relaxation_values["allow_unverified"]
    allow_symlink_parents = relaxation_values["allow_symlink_parents"]
    max_path_length = args.max_path_length if args.max_path_length is not None else _infer_int(csv_path, "max_path_length", None)
    duplicate_policy = args.duplicate_policy or _infer_setting(csv_path, "duplicate_policy") or "report"
    quarantine_dir = args.quarantine_dir or _infer_setting(csv_path, "quarantine_dir") or "_Duplicates"
    return run_plan(
        csv_path=csv_path,
        base_dir=base_dir,
        apply_changes=args.apply,
        use_lineage=use_lineage,
        min_confidence=min_confidence,
        allow_ambiguous=allow_ambiguous,
        allow_outside_base=allow_outside_base,
        allow_unverified=allow_unverified,
        max_path_length=max_path_length,
        duplicate_policy=duplicate_policy,
        quarantine_dir=quarantine_dir,
        allow_legacy_plan=args.allow_legacy_plan,
        trust_recorded_hashes=args.trust_recorded_hashes,
        allow_symlink_parents=allow_symlink_parents,
        collision_detail=args.collision_detail,
        manifest_path=Path(args.manifest) if args.manifest else None,
        allow_workflow_path_breakage=args.allow_workflow_path_breakage,
    )
