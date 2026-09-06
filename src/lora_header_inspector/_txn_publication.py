"""Publication primitives: no-overwrite moves, links, backups."""

# This module is part of the transactions layer. Do not import it directly:
# use ``lora_header_inspector.transactions``, which assembles the public API.
# Cross-cutting primitives are called as ``_txn.<name>`` so that patching an
# attribute on the transactions facade behaves exactly as it did when the
# engine was a single module.

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .fingerprints import (
    HashCache,
    file_identity,
    full_file_sha256,
    identity_matches,
    same_inode,
    verify_record_fingerprint,
)
from .paths import (
    absolute_without_resolving,
    normal_path_key,
    verify_path_safety_snapshot,
)
from .sidecars import verify_sidecar_snapshot
from .version import (
    MANIFEST_SCHEMA_NAME,
    UNDO_SCHEMA_NAME,
    UNDO_SCHEMA_VERSION,
)
from .schema import MANIFEST_SCHEMA_VERSION

from . import transactions as _txn

COPY_CHUNK_BYTES = 8 * 1024 * 1024

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _copy_to_temp(
    source: Path,
    destination_parent: Path,
    on_staged: Optional[Callable[[Path], None]] = None,
) -> Path:
    """Copy ``source`` into a uniquely named, fsynced staging file."""
    destination_parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".lora-copy-", suffix=".tmp", dir=str(destination_parent)
    )
    temp_path = Path(temp_name)
    try:
        if on_staged is not None:
            on_staged(temp_path)
        with source.open("rb", buffering=0) as src, os.fdopen(
            descriptor, "wb", buffering=0
        ) as dst:
            while True:
                chunk = src.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(str(source), str(temp_path), follow_symlinks=False)
        return temp_path
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _hardlink_unavailable(exc: OSError) -> bool:
    unavailable = {
        errno.EXDEV,
        errno.EPERM,
        errno.EACCES,
        getattr(errno, "EOPNOTSUPP", errno.EPERM),
        getattr(errno, "ENOTSUP", errno.EPERM),
        getattr(errno, "ENOSYS", errno.EPERM),
    }
    return exc.errno in unavailable


def _matches_stable_identity(path: Path, expected: Mapping[str, Any]) -> bool:
    """Compare identity fields that are not changed merely by adding a hard link."""
    try:
        current = file_identity(path)
        return (
            current.device == int(expected["device"])
            and current.inode == int(expected["inode"])
            and current.size == int(expected["size"])
            and current.mtime_ns == int(expected["mtime_ns"])
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _inode_unchanged(path: Path, payload: Mapping[str, Any]) -> bool:
    """True when path still holds the journaled device, inode, and size.

    Unlike identity_matches this ignores mtime and ctime. Creating a hard link
    to a file bumps its ctime without touching the data, so a step interrupted
    between backup publication and old-name unlink legitimately fails the full
    identity check while the inode itself is untouched.
    """
    try:
        current = file_identity(path)
        return (
            current.device == int(payload["device"])
            and current.inode == int(payload["inode"])
            and current.size == int(payload["size"])
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _publish_staging_no_replace(
    staging: Path,
    destination: Path,
    on_destination_created: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> str:
    """Publish a verified staging file without ever replacing an existing path.

    Hard-link publication is atomic and invisible until complete. On a filesystem
    without hard links, the final path is created with ``O_EXCL`` and populated
    through the descriptor that won that exclusive creation. That fallback may make
    a partial final file visible during the second copy, but it cannot overwrite an
    unrelated file and the journal can safely remove it after an interruption.
    """
    try:
        os.link(str(staging), str(destination), follow_symlinks=False)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise FileExistsError("destination already exists: %s" % destination)
        if not _txn._hardlink_unavailable(exc):
            raise

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(str(destination), flags, 0o600)
        owned = os.fstat(descriptor)
        created_identity = {
            "device": str(owned.st_dev),
            "inode": str(owned.st_ino),
            "size": str(owned.st_size),
            "mtime_ns": str(owned.st_mtime_ns),
        }
        try:
            if on_destination_created is not None:
                on_destination_created(created_identity)
            with staging.open("rb", buffering=0) as src, os.fdopen(
                descriptor, "wb", buffering=0
            ) as dst:
                descriptor = -1
                while True:
                    chunk = src.read(COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            shutil.copystat(str(staging), str(destination), follow_symlinks=False)
            _txn._fsync_directory(destination.parent)
        except Exception:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                if destination.exists() and not destination.is_symlink():
                    current = destination.stat()
                    if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                        destination.unlink()
                        _txn._fsync_directory(destination.parent)
            except OSError:
                pass
            raise
        return "copy-verify-exclusive-finalize-unlink"
    else:
        staging.unlink()
        _txn._fsync_directory(destination.parent)
        return "copy-verify-link-publish-unlink"


def safe_move(
    source: Path,
    destination: Path,
    expected_sha256: str = "",
    *,
    on_staged: Optional[Callable[[Path], None]] = None,
    on_destination_created: Optional[Callable[[Mapping[str, Any]], None]] = None,
    on_published: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Move one regular file without overwriting and with recoverable phases.

    On a filesystem that supports hard links, a same-device move is published by
    linking the source to the destination and then unlinking the old name.  This
    gives us a genuine no-overwrite operation.  Cross-device moves are copied to
    a journalled staging file, fully hashed, then published without replacement
    before the source is removed.
    """
    source = absolute_without_resolving(source)
    destination = absolute_without_resolving(destination)
    if normal_path_key(source) == normal_path_key(destination):
        raise ValueError("source and destination are the same path")
    if source.is_symlink() or not source.is_file():
        raise ValueError("source missing or is a symlink: %s" % source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination already exists: %s" % destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    source_parent = source.parent
    destination_parent = destination.parent
    before = file_identity(source)

    # Fast, same-filesystem no-overwrite publication.
    try:
        os.link(str(source), str(destination), follow_symlinks=False)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise FileExistsError("destination already exists: %s" % destination)
        if not _txn._hardlink_unavailable(exc):
            raise
    else:
        destination_identity = file_identity(destination)
        if destination_identity.device != before.device or destination_identity.inode != before.inode:
            raise RuntimeError(
                "new destination link does not reference the inspected source inode"
            )
        if not _txn._matches_stable_identity(source, before.to_dict()):
            # Do not unlink either name when another process changed the source path
            # during publication. The extra destination link preserves the inspected
            # inode for manual recovery rather than risking data loss.
            raise RuntimeError("source identity changed during same-filesystem publication")
        result: Dict[str, Any] = {
            "method": "link-unlink",
            "sha256": expected_sha256,
            "size": str(before.size),
            "source_identity": before.to_dict(),
            "destination_identity": destination_identity.to_dict(),
        }
        try:
            if on_published is not None:
                on_published(result)
            source.unlink()
            _txn._fsync_directory(source_parent)
            if destination_parent != source_parent:
                _txn._fsync_directory(destination_parent)
        except Exception:
            try:
                if destination.exists() and not destination.is_symlink() and same_inode(
                    source, destination
                ):
                    destination.unlink()
                    _txn._fsync_directory(destination_parent)
            except OSError:
                pass
            raise
        return result

    # Cross-device or hard-link-incapable filesystem.
    source_digest = expected_sha256 or full_file_sha256(source)
    temp_path = _txn._copy_to_temp(source, destination_parent, on_staged=on_staged)
    published = False
    method = ""
    try:
        temp_identity = file_identity(temp_path)
        if temp_identity.size != before.size:
            raise IOError(
                "staged copy size mismatch: source=%d temporary=%d"
                % (before.size, temp_identity.size)
            )
        temp_digest = full_file_sha256(temp_path)
        if temp_digest != source_digest:
            raise IOError("staged copy SHA-256 mismatch")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("destination appeared during copy: %s" % destination)
        method = _txn._publish_staging_no_replace(
            temp_path,
            destination,
            on_destination_created=on_destination_created,
        )
        published = True
        destination_identity = file_identity(destination)
        if destination_identity.size != before.size:
            raise IOError(
                "published destination size mismatch: source=%d destination=%d"
                % (before.size, destination_identity.size)
            )
        destination_digest = full_file_sha256(destination)
        if destination_digest != source_digest:
            raise IOError("published destination SHA-256 mismatch")
        result = {
            "method": method,
            "sha256": source_digest,
            "size": str(before.size),
            "source_identity": before.to_dict(),
            "destination_identity": destination_identity.to_dict(),
            "staging_path": str(temp_path),
        }
        if on_published is not None:
            on_published(result)
        _txn._fsync_directory(destination_parent)
        try:
            source.unlink()
            _txn._fsync_directory(source_parent)
        except Exception:
            try:
                if destination.exists() and not destination.is_symlink():
                    destination.unlink()
                    _txn._fsync_directory(destination_parent)
            except Exception as rollback_exc:
                raise RuntimeError(
                    "source removal failed after verified copy; destination rollback also failed: %s"
                    % rollback_exc
                )
            raise
        return result
    except Exception:
        if published:
            try:
                if destination.exists() and not destination.is_symlink():
                    destination.unlink()
                    _txn._fsync_directory(destination_parent)
            except OSError:
                pass
        raise
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
                _txn._fsync_directory(temp_path.parent)
            except OSError:
                pass


def atomic_hardlink(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"hard-link source is not a regular file: {source}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"hard-link target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(str(source), str(target), follow_symlinks=False)
    _txn._fsync_directory(target.parent)


def _unique_sibling(path: Path, marker: str) -> Path:
    """Return a non-existing sibling path without creating it."""
    for _ in range(100):
        candidate = path.with_name(
            f".{path.name}.{marker}.{uuid.uuid4().hex}.tmp"
        )
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"could not allocate a private sibling path for {path}")


def _restore_backup(path: Path, backup: Path, digest: str = "") -> List[str]:
    """Restore ``backup`` to ``path`` without overwriting an unrelated file."""
    errors: List[str] = []
    try:
        if backup.is_symlink():
            return [f"backup path became a symlink: {backup}"]
        if not backup.exists():
            return []
        if path.exists() or path.is_symlink():
            return [f"cannot restore backup because destination is occupied: {path}"]
        _txn.safe_move(backup, path, expected_sha256=digest)
    except Exception as exc:
        errors.append(f"could not restore {backup} -> {path}: {exc}")
    return errors


def atomic_relink(
    path: Path,
    canonical: Path,
    *,
    expected_identity: Optional[Mapping[str, Any]] = None,
    expected_sha256: str = "",
) -> None:
    """Replace an exact duplicate by a hard link without overwriting a race.

    The old inode is first moved to a private sibling backup using ``safe_move``.
    The canonical link is then created at the now-vacant public path with
    ``O_EXCL``-equivalent hard-link semantics.  A concurrent creator therefore
    causes a clean failure rather than being overwritten, and the private backup
    is restored whenever the public path is still available.
    """
    path = absolute_without_resolving(path)
    canonical = absolute_without_resolving(canonical)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"relink target is not a regular file: {path}")
    if canonical.is_symlink() or not canonical.is_file():
        raise ValueError(f"canonical path is not a regular file: {canonical}")
    if expected_identity and not _txn._matches_stable_identity(path, expected_identity):
        raise ValueError(f"relink target changed before replacement: {path}")
    digest = expected_sha256 or full_file_sha256(path)
    if full_file_sha256(canonical) != digest:
        raise ValueError("relink target and canonical path no longer have identical content")
    if same_inode(path, canonical):
        return

    backup = _txn._unique_sibling(path, "relink-backup")
    _txn.safe_move(path, backup, expected_sha256=digest)
    try:
        _txn.atomic_hardlink(canonical, path)
        if not same_inode(path, canonical):
            raise RuntimeError("new relink path does not reference the canonical inode")
        backup.unlink()
        _txn._fsync_directory(backup.parent)
    except Exception as exc:
        restore_errors: List[str] = []
        try:
            if path.exists() and not path.is_symlink() and same_inode(path, canonical):
                path.unlink()
                _txn._fsync_directory(path.parent)
        except Exception as remove_exc:
            restore_errors.append(f"could not remove failed relink target {path}: {remove_exc}")
        restore_errors.extend(_txn._restore_backup(path, backup, digest))
        if restore_errors:
            raise RuntimeError(
                f"relink failed: {exc}; rollback errors: {' | '.join(restore_errors)}"
            ) from exc
        raise


def break_hardlink(
    path: Path,
    *,
    canonical: Optional[Path] = None,
    expected_sha256: str = "",
) -> None:
    """Replace a linked path by an independent byte-identical inode safely.

    The current link is moved to a private backup before a verified copy is
    published at the public name. No unrelated file is overwritten if another
    process races to claim the path.
    """
    path = absolute_without_resolving(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"cannot break link for non-regular file: {path}")
    if canonical is not None:
        canonical = absolute_without_resolving(canonical)
        if canonical.is_symlink() or not canonical.is_file():
            raise ValueError(f"canonical path is unavailable: {canonical}")
        if not same_inode(path, canonical):
            raise ValueError(f"path no longer shares the canonical inode: {path}")

    digest = expected_sha256 or full_file_sha256(path)
    staging = _txn._copy_to_temp(path, path.parent)
    if full_file_sha256(staging) != digest:
        staging.unlink(missing_ok=True)
        raise IOError("independent-copy staging hash mismatch")
    backup = _txn._unique_sibling(path, "breaklink-backup")
    _txn.safe_move(path, backup, expected_sha256=digest)
    published = False
    try:
        _txn._publish_staging_no_replace(staging, path)
        published = True
        if full_file_sha256(path) != digest:
            raise IOError("independent hard-link copy hash mismatch")
        if canonical is not None and same_inode(path, canonical):
            raise RuntimeError("hard link was not broken")
        backup.unlink()
        _txn._fsync_directory(backup.parent)
    except Exception as exc:
        errors: List[str] = []
        try:
            if published and path.exists() and not path.is_symlink():
                path.unlink()
                _txn._fsync_directory(path.parent)
        except Exception as remove_exc:
            errors.append(f"could not remove independent copy {path}: {remove_exc}")
        errors.extend(_txn._restore_backup(path, backup, digest))
        if errors:
            raise RuntimeError(
                f"could not break hard link: {exc}; rollback errors: {' | '.join(errors)}"
            ) from exc
        raise
    finally:
        try:
            staging.unlink()
        except OSError:
            pass
