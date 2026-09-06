"""Collision grouping, exact-duplicate detection, and duplicate policies."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .fingerprints import HashCache, file_identity, full_file_sha256, same_inode
from .paths import absolute_without_resolving, nearest_existing_parent, normal_path_key, safe_component

DUPLICATE_POLICIES = ("report", "hardlink", "quarantine")


def compare_file_group(paths: Sequence[Path], cache: Optional[HashCache] = None) -> Dict[str, Any]:
    """Classify regular files as same inode, exact duplicates, or different content."""
    unique: List[Path] = []
    seen: Set[str] = set()
    for raw in paths:
        path = absolute_without_resolving(raw)
        key = normal_path_key(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        return {
            "kind": "unverified_collision",
            "reason": "collision group is empty",
            "sha256": "",
            "size": "",
            "identities": {},
            "hashes": {},
        }

    identities: Dict[str, Dict[str, int]] = {}
    try:
        for path in unique:
            identities[str(path)] = file_identity(path).to_dict()
    except Exception as exc:
        return {
            "kind": "unverified_collision",
            "reason": f"could not inspect collision members: {exc}",
            "sha256": "",
            "size": "",
            "identities": identities,
            "hashes": {},
        }

    inode_pairs = {(item["device"], item["inode"]) for item in identities.values()}
    sizes = {item["size"] for item in identities.values()}
    if len(inode_pairs) == 1:
        try:
            digest = full_file_sha256(unique[0], cache)
        except Exception as exc:
            return {
                "kind": "unverified_collision",
                "reason": f"could not hash shared inode: {exc}",
                "sha256": "",
                "size": str(next(iter(sizes))) if sizes else "",
                "identities": identities,
                "hashes": {},
            }
        return {
            "kind": "same_inode",
            "reason": "all paths already reference the same device/inode",
            "sha256": digest,
            "size": str(next(iter(sizes))) if sizes else "",
            "identities": identities,
            "hashes": {str(path): digest for path in unique},
        }

    if len(sizes) != 1:
        rendered = ", ".join(f"{Path(path).name}={item['size']}" for path, item in identities.items())
        return {
            "kind": "different_content",
            "reason": f"same destination but different file sizes: {rendered}",
            "sha256": "",
            "size": "mixed",
            "identities": identities,
            "hashes": {},
        }

    hashes: Dict[str, str] = {}
    try:
        for path in unique:
            hashes[str(path)] = full_file_sha256(path, cache)
    except Exception as exc:
        return {
            "kind": "unverified_collision",
            "reason": f"could not complete full-file hash comparison: {exc}",
            "sha256": "",
            "size": str(next(iter(sizes))),
            "identities": identities,
            "hashes": hashes,
        }

    distinct_hashes = set(hashes.values())
    if len(distinct_hashes) == 1:
        digest = next(iter(distinct_hashes))
        return {
            "kind": "exact_duplicate",
            "reason": "all collision members have the same full SHA-256",
            "sha256": digest,
            "size": str(next(iter(sizes))),
            "identities": identities,
            "hashes": hashes,
        }
    rendered = ", ".join(f"{Path(path).name}={digest[:12]}" for path, digest in hashes.items())
    return {
        "kind": "different_content",
        "reason": f"same destination but different full SHA-256 values: {rendered}",
        "sha256": "",
        "size": str(next(iter(sizes))),
        "identities": identities,
        "hashes": hashes,
    }


def collision_id(destination: Path) -> str:
    digest = hashlib.sha256(normal_path_key(destination).encode("utf-8", "surrogatepass")).hexdigest()
    return f"collision-{digest[:12]}"


def hardlink_compatible(source: Path, target: Path) -> Tuple[bool, str]:
    try:
        source_device = source.stat().st_dev
        target_device = nearest_existing_parent(target.parent).stat().st_dev
    except OSError as exc:
        return False, f"could not determine filesystem devices: {exc}"
    if source_device != target_device:
        return False, "hard links require source and target to be on the same filesystem"
    return True, ""


def _json_paths(value: str) -> List[Path]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [Path(item) for item in text.split(";") if item]
    if not isinstance(parsed, list):
        return []
    return [Path(str(item)) for item in parsed]


def _transfer_pairs(record: Mapping[str, str]) -> List[Tuple[Path, Path]]:
    text = str(record.get("_transfers", "") or "")
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    output = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                output.append((Path(str(item[0])), Path(str(item[1]))))
    return output


def _skip(record: MutableMapping[str, str], action: str, reason: str) -> None:
    record.update(move="False", generated_move="False", action=action, reason=reason)


def _deterministic_source_index(plan: Sequence[Mapping[str, str]], indexes: Sequence[int]) -> int:
    def score(index: int):
        record = plan[index]
        try:
            sidecar_count = int(record.get("sidecar_count", "0") or 0)
        except ValueError:
            sidecar_count = 0
        rendered = str(record.get("path", ""))
        return (-sidecar_count, len(rendered), rendered.casefold())

    return min(indexes, key=score)


def choose_canonical_index(
    plan: Sequence[Mapping[str, str]], indexes: Sequence[int]
) -> Tuple[Optional[int], str]:
    explicit = [
        index
        for index in indexes
        if str(plan[index].get("canonical", "")).strip().lower() in {"1", "true", "yes", "on"}
        or str(plan[index].get("duplicate_preference", "")).strip().lower()
        in {"canonical", "prefer", "preferred"}
    ]
    if len(explicit) > 1:
        return None, "more than one collision member is marked canonical"
    if explicit:
        return explicit[0], "selected by the reviewed plan"
    return _deterministic_source_index(plan, indexes), "selected deterministically (most sidecars, then shortest path)"


def _annotate(
    plan: Sequence[MutableMapping[str, str]],
    indexes: Sequence[int],
    *,
    group_id: str,
    kind: str,
    destination: Path,
    all_paths: Sequence[Path],
    comparison: Mapping[str, Any],
    canonical_path: Optional[Path],
    policy: str,
    details: Mapping[str, Any],
) -> None:
    identities = comparison.get("identities", {})
    inode_pairs = {
        (int(item["device"]), int(item["inode"]))
        for item in identities.values()
        if isinstance(item, dict) and "device" in item and "inode" in item
    }
    try:
        size = int(comparison.get("size", ""))
    except (TypeError, ValueError):
        size = 0
    logical = max(0, len(inode_pairs) - 1) * size
    sources_json = json.dumps([str(path) for path in all_paths], ensure_ascii=False)
    detail_json = json.dumps(dict(details), ensure_ascii=False, sort_keys=True)
    for index in indexes:
        plan[index].update(
            collision_group=group_id,
            collision_kind=kind,
            collision_target=str(destination),
            collision_sources=sources_json,
            collision_member_count=str(len(all_paths)),
            canonical_path=str(canonical_path or ""),
            content_sha256=str(comparison.get("sha256", "")),
            distinct_inode_count=str(len(inode_pairs)),
            logical_duplicate_bytes=str(logical),
            duplicate_policy=policy,
            collision_details=detail_json,
        )


def _sidecar_notes(
    plan: Sequence[Mapping[str, str]], indexes: Sequence[int], cache: HashCache
) -> List[str]:
    by_name: Dict[str, List[Path]] = defaultdict(list)
    display: Dict[str, str] = {}
    for index in indexes:
        for sidecar in _json_paths(plan[index].get("_sidecar_sources", "")):
            key = sidecar.name.casefold()
            by_name[key].append(sidecar)
            display.setdefault(key, sidecar.name)
    notes = []
    for key in sorted(by_name):
        paths = by_name[key]
        name = display[key]
        if len(paths) == 1:
            notes.append(f"{name}: present only beside {paths[0].parent}")
            continue
        comparison = compare_file_group(paths, cache)
        if comparison["kind"] in {"same_inode", "exact_duplicate"}:
            notes.append(f"{name}: identical beside {len(paths)} duplicate copies")
        elif comparison["kind"] == "different_content":
            notes.append(f"{name}: different content beside {len(paths)} duplicate copies; all source versions remain preserved")
        else:
            notes.append(f"{name}: comparison unavailable ({comparison.get('reason', '')})")
    return notes


def _build_hardlink_spec(
    plan: Sequence[MutableMapping[str, str]],
    indexes: Sequence[int],
    *,
    destination: Path,
    destination_exists: bool,
    canonical_index: Optional[int],
    donor_index: int,
    digest: str,
    group_id: str,
    cache: HashCache,
) -> Tuple[Optional[Dict[str, Any]], str, List[str]]:
    if destination_exists:
        canonical_path = destination
    elif canonical_index is not None:
        canonical_path = Path(plan[canonical_index]["path"])
    else:
        return None, "no canonical source was selected", []

    links: List[Dict[str, Any]] = []
    notes: List[str] = []
    if not destination_exists:
        links.append(
            {
                "source": str(canonical_path),
                "target": str(destination),
                "kind": "primary",
                "source_identity": file_identity(canonical_path).to_dict(),
                "source_sha256": digest,
            }
        )

    # The normalized destination receives the selected donor's sidecars. Other
    # source sidecars remain at their existing paths and are inventoried in the report.
    for sidecar in _json_paths(plan[donor_index].get("_sidecar_sources", "")):
        target = destination.parent / sidecar.name
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                return None, f"sidecar destination is not a regular file: {target}", notes
            comparison = compare_file_group([sidecar, target], cache)
            if comparison["kind"] not in {"same_inode", "exact_duplicate"}:
                return None, f"sidecar destination contains different content: {target}", notes
            notes.append(f"destination already has identical sidecar {target.name}")
            continue
        links.append(
            {
                "source": str(sidecar),
                "target": str(target),
                "kind": "sidecar",
                "source_identity": file_identity(sidecar).to_dict(),
                "source_sha256": full_file_sha256(sidecar, cache),
            }
        )

    replacements: List[Dict[str, Any]] = []
    already_linked: List[str] = []
    for index in indexes:
        source = Path(plan[index]["path"])
        if normal_path_key(source) == normal_path_key(canonical_path):
            continue
        if same_inode(source, canonical_path):
            already_linked.append(str(source))
        else:
            replacements.append(
                {
                    "path": str(source),
                    "source_identity": file_identity(source).to_dict(),
                    "source_sha256": digest,
                }
            )

    for item in links:
        compatible, reason = hardlink_compatible(Path(item["source"]), Path(item["target"]))
        if not compatible:
            return None, f"hard-link compatibility failed: {reason}", notes
    for item in replacements:
        compatible, reason = hardlink_compatible(canonical_path, Path(item["path"]))
        if not compatible:
            return None, f"hard-link compatibility failed: {reason}", notes

    return (
        {
            "policy": "hardlink",
            "group_id": group_id,
            "canonical_path": str(canonical_path),
            "destination": str(destination),
            "content_sha256": digest,
            "links": links,
            "replacements": replacements,
            "already_linked": already_linked,
            "member_paths": [str(Path(plan[index]["path"])) for index in indexes],
        },
        "",
        notes,
    )


def _quarantine_destination(
    base_dir: Path, quarantine_dir: str, digest: str, source: Path, ordinal: int
) -> Path:
    group = digest[:16] if digest else "unverified"
    parent = safe_component(source.parent.name or "root", "root")
    slot = f"{ordinal:03d}_{parent}"
    return base_dir / safe_component(quarantine_dir, "_Duplicates") / group / slot / source.name


def _build_quarantine_spec(
    plan: Sequence[MutableMapping[str, str]],
    indexes: Sequence[int],
    *,
    destination: Path,
    destination_exists: bool,
    canonical_index: Optional[int],
    digest: str,
    group_id: str,
    base_dir: Path,
    quarantine_dir: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    transfers: List[Tuple[str, str]] = []
    canonical_path = destination if destination_exists else (
        Path(plan[canonical_index]["path"]) if canonical_index is not None else None
    )
    if canonical_path is None:
        return None, "no canonical source was selected"

    if not destination_exists and canonical_index is not None:
        canonical_source = Path(plan[canonical_index]["path"])
        for sidecar in _json_paths(plan[canonical_index].get("_sidecar_sources", "")):
            transfers.append((str(sidecar), str(destination.parent / sidecar.name)))
        transfers.append((str(canonical_source), str(destination)))

    ordinal = 1
    for index in indexes:
        source = Path(plan[index]["path"])
        if not destination_exists and canonical_index == index:
            continue
        quarantine = _quarantine_destination(base_dir, quarantine_dir, digest, source, ordinal)
        ordinal += 1
        plan[index]["duplicate_destination"] = str(quarantine)
        for sidecar in _json_paths(plan[index].get("_sidecar_sources", "")):
            transfers.append((str(sidecar), str(quarantine.parent / sidecar.name)))
        transfers.append((str(source), str(quarantine)))

    targets: Set[str] = set()
    for raw_source, raw_target in transfers:
        source = Path(raw_source)
        target = Path(raw_target)
        key = normal_path_key(target)
        if key in targets:
            return None, f"quarantine plan contains duplicate target: {target}"
        targets.add(key)
        if normal_path_key(source) == key:
            return None, f"quarantine target equals source: {source}"
        if target.exists() or target.is_symlink():
            return None, f"quarantine or canonical destination already exists: {target}"

    return (
        {
            "policy": "quarantine",
            "group_id": group_id,
            "canonical_path": str(canonical_path),
            "destination": str(destination),
            "content_sha256": digest,
            "transfers": transfers,
            "member_paths": [str(Path(plan[index]["path"])) for index in indexes],
        },
        "",
    )


def resolve_primary_collisions(
    plan: Sequence[MutableMapping[str, str]],
    *,
    duplicate_policy: str,
    base_dir: Path,
    quarantine_dir: str,
    cache: HashCache,
) -> None:
    by_target: Dict[str, List[int]] = defaultdict(list)
    target_display: Dict[str, Path] = {}
    for index, record in enumerate(plan):
        if record.get("action") != "candidate":
            continue
        destination = Path(record["destination"])
        key = normal_path_key(destination)
        by_target[key].append(index)
        target_display.setdefault(key, destination)

    for key, indexes in by_target.items():
        destination = target_display[key]
        destination_exists = destination.exists() or destination.is_symlink()
        if len(indexes) == 1 and not destination_exists:
            continue

        group_id = collision_id(destination)
        source_paths = [Path(plan[index]["path"]) for index in indexes]
        all_paths = list(source_paths)
        if destination_exists:
            if destination.is_symlink() or not destination.is_file():
                comparison = {
                    "kind": "unverified_collision",
                    "reason": "destination exists but is not a regular file",
                    "sha256": "",
                    "size": "",
                    "identities": {},
                    "hashes": {},
                }
                _annotate(
                    plan,
                    indexes,
                    group_id=group_id,
                    kind="destination_not_regular",
                    destination=destination,
                    all_paths=source_paths + [destination],
                    comparison=comparison,
                    canonical_path=None,
                    policy=duplicate_policy,
                    details={"comparison_reason": comparison["reason"]},
                )
                for index in indexes:
                    _skip(plan[index], "skip_collision", f"destination is not a regular file: {destination}")
                continue
            all_paths.append(destination)

        comparison = compare_file_group(all_paths, cache)
        base_kind = str(comparison["kind"])
        kind = f"existing_{base_kind}" if destination_exists else base_kind
        canonical_index, canonical_reason = choose_canonical_index(plan, indexes)
        canonical_path = destination if destination_exists else (
            Path(plan[canonical_index]["path"]) if canonical_index is not None else None
        )
        _annotate(
            plan,
            indexes,
            group_id=group_id,
            kind=kind,
            destination=destination,
            all_paths=all_paths,
            comparison=comparison,
            canonical_path=canonical_path,
            policy=duplicate_policy,
            details={
                "comparison_reason": comparison.get("reason", ""),
                "canonical_reason": canonical_reason,
                "hashes": comparison.get("hashes", {}),
                "identities": comparison.get("identities", {}),
            },
        )

        if base_kind == "different_content":
            for index in indexes:
                _skip(plan[index], "skip_content_collision", str(comparison.get("reason", "different content")))
            continue
        if base_kind == "unverified_collision":
            for index in indexes:
                _skip(plan[index], "skip_collision_unverified", str(comparison.get("reason", "unverified collision")))
            continue
        if base_kind not in {"same_inode", "exact_duplicate"}:
            for index in indexes:
                _skip(plan[index], "skip_plan_collision", "collision group was not safely classifiable")
            continue
        if canonical_index is None and not destination_exists:
            for index in indexes:
                _skip(plan[index], "skip_canonical_conflict", canonical_reason)
            continue

        donor_index = canonical_index if canonical_index is not None else _deterministic_source_index(plan, indexes)
        for index in indexes:
            source = Path(plan[index]["path"])
            is_canonical = canonical_path is not None and normal_path_key(source) == normal_path_key(canonical_path)
            plan[index]["canonical"] = "True" if is_canonical else "False"
            plan[index]["duplicate_role"] = "canonical" if is_canonical else "duplicate"
            plan[index]["duplicate_of"] = str(canonical_path or "")

        digest = str(comparison.get("sha256", ""))
        reason = (
            f"{len(all_paths)} paths contain identical bytes; sha256={digest}"
            if digest
            else f"{len(all_paths)} paths already share one inode"
        )
        notes = _sidecar_notes(plan, indexes, cache)
        for index in indexes:
            plan[index]["sidecar_notes"] = json.dumps(notes, ensure_ascii=False)

        if duplicate_policy == "report":
            action = (
                "skip_existing_exact_duplicate"
                if destination_exists
                else "skip_same_inode_group"
                if base_kind == "same_inode"
                else "skip_exact_duplicate_group"
            )
            for index in indexes:
                _skip(plan[index], action, reason)
            continue

        if duplicate_policy == "hardlink":
            spec, error, additional_notes = _build_hardlink_spec(
                plan,
                indexes,
                destination=destination,
                destination_exists=destination_exists,
                canonical_index=canonical_index,
                donor_index=donor_index,
                digest=digest,
                group_id=group_id,
                cache=cache,
            )
            merged_notes = notes + additional_notes
            for index in indexes:
                plan[index]["sidecar_notes"] = json.dumps(merged_notes, ensure_ascii=False)
            if error or spec is None:
                action = "skip_hardlink_incompatible" if error.startswith("hard-link compatibility") else "skip_sidecar_content_collision"
                for index in indexes:
                    _skip(plan[index], action, error or "could not build hard-link plan")
                continue
            if not spec["links"] and not spec["replacements"]:
                for index in indexes:
                    _skip(plan[index], "already_hardlinked_group", "all paths already share the same inode")
                continue
            leader_index = canonical_index if canonical_index is not None else min(indexes)
            leader = plan[leader_index]
            leader.update(
                move="True",
                generated_move="True",
                action="hardlink_duplicate_group",
                reason=reason,
                _duplicate_spec=json.dumps(spec, ensure_ascii=False),
                _transfers=json.dumps(
                    [(item["source"], item["target"]) for item in spec["links"]]
                    + [(spec["canonical_path"], item["path"]) for item in spec["replacements"]],
                    ensure_ascii=False,
                ),
            )
            for index in indexes:
                if index != leader_index:
                    plan[index].update(
                        move="False",
                        generated_move="False",
                        action="hardlink_duplicate_member",
                        reason=f"handled by collision-group leader {leader['path']}",
                    )
            continue

        if duplicate_policy == "quarantine":
            spec, error = _build_quarantine_spec(
                plan,
                indexes,
                destination=destination,
                destination_exists=destination_exists,
                canonical_index=canonical_index,
                digest=digest,
                group_id=group_id,
                base_dir=base_dir,
                quarantine_dir=quarantine_dir,
            )
            if error or spec is None:
                for index in indexes:
                    _skip(plan[index], "skip_quarantine_collision", error or "could not build quarantine plan")
                continue
            leader_index = canonical_index if canonical_index is not None else min(indexes)
            leader = plan[leader_index]
            leader.update(
                move="True",
                generated_move="True",
                action="quarantine_duplicate_group",
                reason=reason,
                _duplicate_spec=json.dumps(spec, ensure_ascii=False),
                _transfers=json.dumps(spec["transfers"], ensure_ascii=False),
            )
            for index in indexes:
                if index != leader_index:
                    plan[index].update(
                        move="False",
                        generated_move="False",
                        action="quarantine_duplicate_member",
                        reason=f"handled by collision-group leader {leader['path']}",
                    )


def resolve_remaining_collisions(
    plan: Sequence[MutableMapping[str, str]], cache: HashCache
) -> None:
    # Existing sidecar targets not already handled by a primary collision group.
    for index, record in enumerate(plan):
        if record.get("action") != "candidate":
            continue
        pairs = _transfer_pairs(record)
        for pair_index, (source, destination) in enumerate(pairs):
            if normal_path_key(source) == normal_path_key(destination):
                continue
            if not (destination.exists() or destination.is_symlink()):
                continue
            label = "primary" if pair_index == 0 else "sidecar"
            if destination.is_symlink() or not destination.is_file():
                comparison = {
                    "kind": "unverified_collision",
                    "reason": "target is not a regular file",
                    "sha256": "",
                    "size": "",
                    "identities": {},
                    "hashes": {},
                }
            else:
                comparison = compare_file_group([source, destination], cache)
            _annotate(
                plan,
                [index],
                group_id=collision_id(destination),
                kind=f"existing_{label}_{comparison['kind']}",
                destination=destination,
                all_paths=[source, destination],
                comparison=comparison,
                canonical_path=destination if comparison["kind"] in {"same_inode", "exact_duplicate"} else None,
                policy=record.get("duplicate_policy", "report"),
                details={"comparison_reason": comparison.get("reason", "")},
            )
            action = (
                "skip_existing_sidecar_duplicate"
                if label == "sidecar" and comparison["kind"] in {"same_inode", "exact_duplicate"}
                else "skip_existing_exact_duplicate"
                if label == "primary" and comparison["kind"] in {"same_inode", "exact_duplicate"}
                else "skip_sidecar_collision"
                if label == "sidecar"
                else "skip_collision"
            )
            _skip(record, action, str(comparison.get("reason", "target exists")))
            break

    target_map: Dict[str, List[Tuple[int, Path, Path]]] = defaultdict(list)
    for index, record in enumerate(plan):
        if record.get("action") != "candidate":
            continue
        for source, destination in _transfer_pairs(record):
            target_map[normal_path_key(destination)].append((index, source, destination))

    for entries in target_map.values():
        indexes = sorted({item[0] for item in entries})
        if len(indexes) < 2:
            continue
        destination = entries[0][2]
        sources = [item[1] for item in entries]
        comparison = compare_file_group(sources, cache)
        _annotate(
            plan,
            indexes,
            group_id=collision_id(destination),
            kind=f"sidecar_plan_{comparison['kind']}",
            destination=destination,
            all_paths=sources,
            comparison=comparison,
            canonical_path=None,
            policy=plan[indexes[0]].get("duplicate_policy", "report"),
            details={"comparison_reason": comparison.get("reason", "")},
        )
        for index in indexes:
            _skip(
                plan[index],
                "skip_sidecar_plan_collision",
                "two or more plan rows target the same sidecar path; "
                + str(comparison.get("reason", "comparison unavailable")),
            )

    for record in plan:
        if record.get("action") == "candidate":
            record.update(action="move", move="True", generated_move="True", reason=record.get("reason") or "fingerprint matches")
