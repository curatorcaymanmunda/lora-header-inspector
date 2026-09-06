"""Versioned sort-plan construction and collision reporting."""

from __future__ import annotations

import csv
import json
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set

from .duplicates import (
    DUPLICATE_POLICIES,
    resolve_primary_collisions,
    resolve_remaining_collisions,
)
from .fingerprints import HashCache, verify_record_fingerprint
from .paths import (
    absolute_without_resolving,
    build_path_safety_snapshot,
    check_destination_containment,
    default_subfolder,
    normal_path_key,
    safe_component,
)
from .schema import (
    PLAN_FIELDS,
    PLAN_SCHEMA_VERSION,
    collision_report_header,
    ensure_plan_fields,
    validate_inspection_schema,
    validate_plan_collection,
    validate_plan_row,
    write_csv_atomic,
    write_json_atomic,
)
from .sidecars import sidecar_snapshot_json, snapshot_sidecars

CONF_RANK = {"low": 0, "medium": 1, "high": 2}
TRUTHY = {"true", "1", "yes", "y", "on"}
FALSY = {"false", "0", "no", "n", "off"}
PLAN_RUNTIME_KEYS = {
    "_source", "_destination", "_sidecar_sources", "_transfers", "_duplicate_spec",
}


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUTHY


def explicitly_false(value: Any) -> bool:
    return str(value or "").strip().lower() in FALSY


def _bool_state(value: Any) -> Optional[bool]:
    if truthy(value):
        return True
    if explicitly_false(value):
        return False
    return None


def _user_move_override(record: Mapping[str, str]) -> Optional[bool]:
    explicit = _bool_state(record.get("move_override", ""))
    if explicit is not None:
        return explicit
    move_state = _bool_state(record.get("move", ""))
    generated_state = _bool_state(record.get("generated_move", ""))
    if generated_state is not None and move_state is not None and move_state != generated_state:
        return move_state
    prior_action = str(record.get("action", "") or "").strip()
    if generated_state is None and prior_action in {"", "move"} and move_state is False:
        return False
    return None


def _skip(record: MutableMapping[str, str], action: str, reason: str) -> None:
    record.update(move="False", generated_move="False", action=action, reason=reason)


def _candidate_destination(record: Mapping[str, str], base_dir: Path, use_lineage: bool) -> Path:
    raw = str(record.get("destination", "") or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else base_dir / path
    architecture = safe_component(record.get("arch", "Unknown"), "Unknown")
    bucket = Path(architecture)
    if use_lineage:
        subfolder = default_subfolder(record)
        if subfolder:
            bucket /= safe_component(subfolder)
    return base_dir / bucket / Path(record["path"]).name


def _path_length_error(destination: Path, sidecars, max_path_length: int) -> str:
    if not max_path_length:
        return ""
    targets = [destination] + [destination.parent / item.path.name for item in sidecars]
    too_long = [target.name for target in targets if len(str(target)) > max_path_length]
    if not too_long:
        return ""
    return (
        f"destination path exceeds {max_path_length} characters: "
        + ", ".join(too_long)
    )


def build_sort_plan(
    csv_path: Path,
    base_dir: Path,
    use_lineage: bool = True,
    min_conf: str = "high",
    allow_ambiguous: bool = False,
    allow_outside_base: bool = False,
    allow_unverified: bool = False,
    max_path_length: int = 0,
    duplicate_policy: str = "report",
    quarantine_dir: str = "_Duplicates",
    allow_legacy_plan: bool = False,
    allow_symlink_parents: bool = False,
    trust_recorded_hashes: bool = False,
) -> List[Dict[str, str]]:
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise ValueError(f"unsupported duplicate policy: {duplicate_policy}")

    base_dir = absolute_without_resolving(base_dir)
    base_resolved = base_dir.resolve(strict=False)
    min_rank = CONF_RANK.get(min_conf, CONF_RANK["high"])
    plan: List[Dict[str, str]] = []
    seen_sources: Set[str] = set()
    plan_id = str(uuid.uuid4())
    cache = HashCache()

    def seed_hash_cache(record: Mapping[str, str]) -> None:
        """Under --trust-recorded-hashes, reuse a recorded full hash.

        The digest is stored under the identity it was recorded at (device,
        inode, size, mtime, ctime); full_file_sha256 builds its lookup key
        from the file's *current* identity, so any change since inspection
        misses the key and forces a real re-hash. Content cannot change
        without moving ctime for an unprivileged writer, which is the same
        trust boundary the in-process HashCache already relies on. What this
        deliberately stops detecting is a hand-edited full_sha256 column on
        an untouched file; apply-time relink steps still verify digests
        against real bytes before linking.
        """
        if not trust_recorded_hashes:
            return
        digest = str(record.get("full_sha256", "") or "")
        if not digest:
            return
        try:
            key = (
                normal_path_key(Path(record["path"])),
                int(record["device"]),
                int(record["inode"]),
                int(record["file_size"]),
                int(record["mtime_ns"]),
                int(record["ctime_ns"]),
            )
        except (KeyError, TypeError, ValueError):
            return
        cache.setdefault(key, digest)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        valid, reason = validate_inspection_schema(reader.fieldnames, allow_legacy=allow_legacy_plan)
        if not valid:
            raise ValueError(reason)
        header_fields = {str(value or "").strip() for value in reader.fieldnames}
        header_is_legacy = not (
            {"schema_name", "schema_version"}.issubset(header_fields)
            or {"plan_schema_name", "plan_schema_version"}.issubset(header_fields)
        )
        source_rows = [
            {str(key): str(value or "") for key, value in original.items() if key is not None}
            for original in reader
        ]
        for row_number, source_row in enumerate(source_rows, start=2):
            schema_ok, schema_reason = validate_plan_row(
                source_row,
                allow_legacy=bool(allow_legacy_plan and header_is_legacy),
            )
            if not schema_ok:
                raise ValueError("row %d: %s" % (row_number, schema_reason))
        collection_ok, collection_reason = validate_plan_collection(
            source_rows, expected_base_resolved=str(base_resolved)
        )
        if not collection_ok:
            raise ValueError(collection_reason)

        for row_number, source_row in enumerate(source_rows, start=2):
            is_legacy_import = header_is_legacy
            reviewed_move_override = _user_move_override(source_row)
            seed_hash_cache(source_row)
            record = ensure_plan_fields(source_row)
            record.update(
                plan_schema_version=PLAN_SCHEMA_VERSION,
                plan_id=record.get("plan_id") or plan_id,
                plan_base_dir=str(base_dir),
                plan_base_resolved=str(base_resolved),
                use_lineage=str(bool(use_lineage)),
                min_confidence=min_conf,
                allow_ambiguous=str(bool(allow_ambiguous)),
                allow_outside_base=str(bool(allow_outside_base)),
                allow_unverified=str(bool(allow_unverified)),
                allow_symlink_parents=str(bool(allow_symlink_parents)),
                max_path_length=str(max_path_length),
                legacy_imported=str(bool(is_legacy_import)),
                bucket="",
                action="",
                reason="",
                verified="False",
                sidecars="[]",
                sidecar_fingerprints_json="[]",
                sidecar_count="0",
                path_safety_json="",
                collision_group="",
                collision_kind="",
                collision_target="",
                collision_sources="",
                collision_member_count="",
                canonical_path="",
                duplicate_of="",
                duplicate_role="",
                duplicate_destination="",
                distinct_inode_count="",
                logical_duplicate_bytes="",
                duplicate_policy=duplicate_policy,
                quarantine_dir=quarantine_dir,
                sidecar_notes="",
                collision_details="",
                generated_move="False",
            )

            path_text = str(record.get("path", "") or "").strip()
            if not path_text:
                _skip(record, "skip_no_path", "missing source path")
                plan.append(record)
                continue
            try:
                source = absolute_without_resolving(Path(path_text))
                source_key = normal_path_key(source)
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

            try:
                source_is_symlink = source.is_symlink()
                source_is_file = source.is_file()
            except OSError as exc:
                _skip(record, "skip_invalid_path", f"could not inspect source path: {exc}")
                plan.append(record)
                continue
            if source_is_symlink:
                _skip(record, "skip_symlink", "source path is a symlink")
                plan.append(record)
                continue
            if not source_is_file:
                _skip(record, "skip_missing", "source file was not found")
                plan.append(record)
                continue
            if str(record.get("error", "") or "").strip():
                _skip(record, "skip_error", record["error"][:200])
                plan.append(record)
                continue

            architecture = str(record.get("arch", "Unknown") or "Unknown").strip()
            confidence = str(record.get("confidence", "low") or "low").strip().lower()
            status = str(record.get("status", "ambiguous") or "ambiguous").strip().lower()
            if architecture == "Unknown":
                record["bucket"] = "Unknown"
                _skip(record, "skip_unknown", str(record.get("evidence", ""))[:200])
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
                _skip(record, "skip_low_confidence", f"confidence={confidence} is below {min_conf}")
                plan.append(record)
                continue
            if reviewed_move_override is False:
                _skip(record, "skip_user_false", "move=False in reviewed CSV")
                plan.append(record)
                continue

            try:
                destination = absolute_without_resolving(
                    _candidate_destination(record, base_dir, use_lineage)
                )
                contained, containment_reason, resolved = check_destination_containment(
                    destination, base_dir, allow_outside=allow_outside_base
                )
            except (OSError, ValueError, RuntimeError) as exc:
                _skip(record, "skip_invalid_destination", f"invalid destination: {exc}")
                plan.append(record)
                continue
            record["destination"] = str(destination)
            if not contained:
                _skip(record, "skip_outside_base", containment_reason)
                plan.append(record)
                continue
            try:
                record["bucket"] = str(resolved.parent.relative_to(base_resolved))
            except ValueError:
                record["bucket"] = str(resolved.parent)

            if normal_path_key(source) == normal_path_key(destination):
                _skip(record, "already_in_place", "source already equals destination")
                plan.append(record)
                continue

            verified, verify_reason = verify_record_fingerprint(
                record, source, allow_unverified=allow_unverified, cache=cache
            )
            record["verified"] = str(verified)
            if not verified:
                _skip(record, "skip_stale_or_unverified", verify_reason)
                plan.append(record)
                continue

            sidecars = snapshot_sidecars(source, include_hash=True, hash_cache=cache)
            record["sidecars"] = json.dumps([item.path.name for item in sidecars], ensure_ascii=False)
            record["sidecar_fingerprints_json"] = sidecar_snapshot_json(sidecars)
            record["sidecar_count"] = str(len(sidecars))
            path_error = _path_length_error(destination, sidecars, max_path_length)
            if path_error:
                _skip(record, "skip_path_too_long", path_error)
                plan.append(record)
                continue

            record.update(
                move="True",
                generated_move="True",
                action="candidate",
                reason=verify_reason,
                _source=str(source),
                _destination=str(destination),
                _sidecar_sources=json.dumps([str(item.path) for item in sidecars], ensure_ascii=False),
            )
            transfers = [(str(source), str(destination))] + [
                (str(item.path), str(destination.parent / item.path.name)) for item in sidecars
            ]
            record["_transfers"] = json.dumps(transfers, ensure_ascii=False)
            plan.append(record)

    resolve_primary_collisions(
        plan,
        duplicate_policy=duplicate_policy,
        base_dir=base_dir,
        quarantine_dir=quarantine_dir,
        cache=cache,
    )
    resolve_remaining_collisions(plan, cache)

    # Freeze the lexical and resolved path topology after all duplicate policies
    # have produced their final operations. Apply revalidates this exact snapshot.
    for record in plan:
        action = str(record.get("action", "") or "")
        if action not in {"move", "hardlink_duplicate_group", "quarantine_duplicate_group"}:
            continue
        descriptors: List[Dict[str, Any]] = []
        if action == "hardlink_duplicate_group":
            try:
                spec = json.loads(record.get("_duplicate_spec", "{}") or "{}")
            except json.JSONDecodeError as exc:
                _skip(record, "skip_path_safety", "invalid duplicate specification: %s" % exc)
                continue
            for item in spec.get("links", []):
                if isinstance(item, Mapping):
                    descriptors.append(
                        {
                            "source": str(item.get("source", "")),
                            "target": str(item.get("target", "")),
                            "managed_target": True,
                        }
                    )
            for item in spec.get("replacements", []):
                if isinstance(item, Mapping):
                    descriptors.append(
                        {
                            "source": str(spec.get("canonical_path", "")),
                            "target": str(item.get("path", "")),
                            "managed_target": False,
                        }
                    )
        else:
            try:
                raw_pairs = json.loads(record.get("_transfers", "[]") or "[]")
            except json.JSONDecodeError as exc:
                _skip(record, "skip_path_safety", "invalid transfer specification: %s" % exc)
                continue
            for item in raw_pairs:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    descriptors.append(
                        {
                            "source": str(item[0]),
                            "target": str(item[1]),
                            "managed_target": True,
                        }
                    )
        snapshot, safety_error = build_path_safety_snapshot(
            descriptors,
            base=base_dir,
            allow_outside=allow_outside_base,
            allow_symlink_parents=allow_symlink_parents,
        )
        if safety_error or snapshot is None:
            _skip(record, "skip_path_safety", safety_error or "could not build path-safety snapshot")
            continue
        record["path_safety_json"] = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return plan


def summarize_plan(plan: Sequence[Mapping[str, str]], label: str = "Sort plan") -> None:
    buckets = Counter(record.get("bucket") or "Unknown" for record in plan)
    actions = Counter(record.get("action") or "unknown" for record in plan)
    statuses = Counter(record.get("status") or "legacy/unknown" for record in plan)
    print(f"\n{label}: {len(plan)} entries")
    print("By status:")
    for status, count in sorted(statuses.items()):
        print(f"  {status:32s} {count}")
    print("By bucket:")
    for bucket, count in sorted(buckets.items()):
        print(f"  {bucket:32s} {count}")
    print("By action:")
    for action, count in sorted(actions.items()):
        print(f"  {action:32s} {count}")


def collision_groups(plan: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in plan:
        group_id = str(row.get("collision_group", "") or "")
        if group_id:
            grouped[group_id].append(row)
    output = []
    for group_id, rows in sorted(grouped.items()):
        first = rows[0]
        try:
            sources = json.loads(first.get("collision_sources", "[]") or "[]")
        except json.JSONDecodeError:
            sources = []
        try:
            details = json.loads(first.get("collision_details", "{}") or "{}")
        except json.JSONDecodeError:
            details = {}
        output.append(
            {
                "group_id": group_id,
                "kind": first.get("collision_kind", ""),
                "target": first.get("collision_target", ""),
                "policy": first.get("duplicate_policy", "report"),
                "canonical_path": first.get("canonical_path", ""),
                "content_sha256": first.get("content_sha256", ""),
                "distinct_inode_count": first.get("distinct_inode_count", ""),
                "logical_duplicate_bytes": first.get("logical_duplicate_bytes", ""),
                "sources": sources,
                "sidecar_notes": json.loads(first.get("sidecar_notes", "[]") or "[]") if first.get("sidecar_notes") else [],
                "details": details,
                "rows": [
                    {
                        "path": row.get("path", ""),
                        "action": row.get("action", ""),
                        "reason": row.get("reason", ""),
                        "canonical": row.get("canonical", ""),
                        "duplicate_role": row.get("duplicate_role", ""),
                        "duplicate_destination": row.get("duplicate_destination", ""),
                    }
                    for row in rows
                ],
            }
        )
    return output


def print_collision_report(plan: Sequence[Mapping[str, str]], detail: str = "summary") -> None:
    groups = collision_groups(plan)
    if not groups or detail == "none":
        return
    print(f"\nCollision groups: {len(groups)}")
    for group in groups:
        label = group["kind"].upper()
        print(f"[{label}] {group['group_id']}")
        print(f"  target: {group['target']}")
        print(f"  policy: {group['policy']}")
        if group["content_sha256"]:
            print(f"  sha256: {group['content_sha256']}")
        print(f"  sources: {len(group['sources'])}")
        if detail == "full":
            for source in group["sources"]:
                print(f"    - {source}")
            if group["canonical_path"]:
                print(f"  canonical: {group['canonical_path']}")
            for note in group["sidecar_notes"]:
                print(f"  sidecar: {note}")
            for row in group["rows"]:
                print(f"  action: {row['action']}  {row['path']}")


def write_plan_csv(plan: Sequence[Mapping[str, str]], output_path: Path) -> None:
    if not plan:
        print("No records to write.")
        return
    write_csv_atomic(plan, output_path, PLAN_FIELDS)
    print(f"Sort plan written to: {output_path}")


def plan_output_path(csv_path: Path) -> Path:
    if csv_path.stem.endswith("_sort_plan"):
        return csv_path
    return csv_path.with_name(csv_path.stem + "_sort_plan.csv")


def collision_report_path(csv_path: Path) -> Path:
    stem = csv_path.stem[:-10] if csv_path.stem.endswith("_sort_plan") else csv_path.stem
    return csv_path.with_name(stem + "_collision_report.json")


def write_collision_report(plan: Sequence[Mapping[str, str]], path: Path, source_csv: Path, base_dir: Path) -> None:
    payload = collision_report_header()
    payload.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_csv=str(source_csv),
        base_dir=str(base_dir),
        groups=collision_groups(plan),
    )
    write_json_atomic(payload, path)
    print(f"Collision report written to: {path}")


def run_plan(
    csv_path: Path,
    base_dir: Path,
    *,
    apply_changes: bool = False,
    use_lineage: bool = True,
    min_confidence: str = "high",
    allow_ambiguous: bool = False,
    allow_outside_base: bool = False,
    allow_unverified: bool = False,
    max_path_length: Optional[int] = None,
    duplicate_policy: str = "report",
    quarantine_dir: str = "_Duplicates",
    allow_legacy_plan: bool = False,
    allow_symlink_parents: bool = False,
    trust_recorded_hashes: bool = False,
    collision_detail: str = "summary",
    manifest_path: Optional[Path] = None,
    allow_workflow_path_breakage: bool = False,
) -> int:
    if max_path_length is None:
        max_path_length = 259 if os.name == "nt" else 0
    print(
        f"lora-sort: CSV={csv_path} base={base_dir} subfolders={'on' if use_lineage else 'off'} "
        f"min-confidence={min_confidence} duplicate-policy={duplicate_policy} "
        f"apply={'YES' if apply_changes else 'no (dry run)'}"
    )
    try:
        plan = build_sort_plan(
            csv_path=csv_path,
            trust_recorded_hashes=trust_recorded_hashes,
            base_dir=base_dir,
            use_lineage=use_lineage,
            min_conf=min_confidence,
            allow_ambiguous=allow_ambiguous,
            allow_outside_base=allow_outside_base,
            allow_unverified=allow_unverified,
            max_path_length=max_path_length,
            duplicate_policy=duplicate_policy,
            quarantine_dir=quarantine_dir,
            allow_legacy_plan=allow_legacy_plan,
            allow_symlink_parents=allow_symlink_parents,
        )
    except Exception as exc:
        print(f"ERROR: could not build sort plan: {exc}")
        return 1

    summarize_plan(plan)
    print_collision_report(plan, collision_detail)
    plan_path = plan_output_path(csv_path)
    write_plan_csv(plan, plan_path)
    report_path = collision_report_path(csv_path)
    write_collision_report(plan, report_path, csv_path, base_dir)

    if not apply_changes:
        print(f"Review the plan CSV, then apply it with an explicit --base: {plan_path}")

    from .transactions import apply_plan

    counts = apply_plan(
        plan=plan,
        csv_path=csv_path,
        base_dir=base_dir,
        dry_run=not apply_changes,
        allow_unverified=allow_unverified,
        manifest_path=manifest_path,
        allow_workflow_path_breakage=allow_workflow_path_breakage,
        allow_outside_base=allow_outside_base,
        allow_symlink_parents=allow_symlink_parents,
    )
    if apply_changes:
        write_plan_csv(plan, plan_path)
        if counts["errors"] or counts["rolled_back"]:
            return 1
    return 0
