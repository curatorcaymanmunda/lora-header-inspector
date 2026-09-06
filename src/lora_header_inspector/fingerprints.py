"""Header, stat, and full-content fingerprints.

Full model hashes are deliberately streamed.  They are calculated only when a
collision or an explicitly requested audit needs them, so ordinary header-only
inspection remains fast even on very large libraries.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Tuple

from .header import read_header_hash
from .models import FileFingerprint, FileIdentity
from .paths import normal_path_key

HASH_CHUNK_BYTES = 8 * 1024 * 1024
CacheKey = Tuple[str, int, int, int, int, int]


class HashCache(dict):
    """In-memory SHA-256 cache keyed by path and stable file identity.

    The cache is intentionally process-local.  Persisting hashes without also
    persisting and validating device, inode, size, mtime, and ctime would make a
    stale hash dangerously persuasive.
    """

    def sha256(self, path: Path) -> str:
        return full_file_sha256(path, self)


def file_identity(path: Path) -> FileIdentity:
    if path.is_symlink() or not path.is_file():
        raise ValueError("not a regular non-symlink file: %s" % path)
    return FileIdentity.from_path(path)


def _cache_key(path: Path, identity: FileIdentity) -> CacheKey:
    return (
        normal_path_key(path),
        identity.device,
        identity.inode,
        identity.size,
        identity.mtime_ns,
        identity.ctime_ns,
    )


def full_file_sha256(
    path: Path,
    cache: Optional[MutableMapping[CacheKey, str]] = None,
    *,
    chunk_bytes: int = HASH_CHUNK_BYTES,
) -> str:
    """Stream a SHA-256 and reject a file that changes during hashing."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    before = file_identity(path)
    key = _cache_key(path, before)
    if cache is not None and key in cache:
        return str(cache[key])

    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)

    after = file_identity(path)
    if after != before:
        raise RuntimeError("file changed while hashing: %s" % path)

    value = digest.hexdigest()
    if cache is not None:
        cache[key] = value
    return value


def fingerprint(
    path: Path,
    include_content: bool = False,
    cache: Optional[MutableMapping[CacheKey, str]] = None,
) -> FileFingerprint:
    identity = file_identity(path)
    return FileFingerprint(
        identity=identity,
        header_sha256=read_header_hash(path),
        content_sha256=full_file_sha256(path, cache) if include_content else "",
    )


def same_inode(first: Path, second: Path) -> bool:
    try:
        one = file_identity(first)
        two = file_identity(second)
    except (OSError, ValueError):
        return False
    return one.device == two.device and one.inode == two.inode


def verify_record_fingerprint(
    record: Mapping[str, str],
    source: Path,
    allow_unverified: bool = False,
    require_content_hash: bool = False,
    cache: Optional[MutableMapping[CacheKey, str]] = None,
) -> Tuple[bool, str]:
    """Verify an inspection record immediately before planning or mutation."""
    expected_size = str(record.get("file_size", "") or "").strip()
    expected_mtime = str(record.get("mtime_ns", "") or "").strip()
    expected_ctime = str(record.get("ctime_ns", "") or "").strip()
    expected_device = str(record.get("device", "") or "").strip()
    expected_inode = str(record.get("inode", "") or "").strip()
    expected_header = str(record.get("header_sha256", "") or "").strip().lower()
    expected_content = str(
        record.get("full_sha256", "")
        or record.get("content_sha256", "")
        or ""
    ).strip().lower()

    required = (expected_size, expected_mtime, expected_ctime, expected_header)
    if not all(required):
        if allow_unverified:
            return True, "fingerprint absent; explicitly allowed"
        return False, "inspection fingerprint is missing"

    try:
        current = file_identity(source)
        current_header = read_header_hash(source)
    except Exception as exc:
        return False, "could not verify source fingerprint: %s" % exc

    mismatches = []
    try:
        if current.size != int(expected_size):
            mismatches.append("size")
    except ValueError:
        mismatches.append("recorded size is invalid")
    try:
        if current.mtime_ns != int(expected_mtime):
            mismatches.append("mtime")
    except ValueError:
        mismatches.append("recorded mtime is invalid")
    try:
        if current.ctime_ns != int(expected_ctime):
            mismatches.append("ctime")
    except ValueError:
        mismatches.append("recorded ctime is invalid")

    if expected_device:
        try:
            if current.device != int(expected_device):
                mismatches.append("device")
        except ValueError:
            mismatches.append("recorded device is invalid")
    if expected_inode:
        try:
            if current.inode != int(expected_inode):
                mismatches.append("inode")
        except ValueError:
            mismatches.append("recorded inode is invalid")

    if current_header.lower() != expected_header:
        mismatches.append("header hash")

    if expected_content or require_content_hash:
        if not expected_content and require_content_hash:
            mismatches.append("content hash is missing")
        elif expected_content:
            try:
                current_content = full_file_sha256(source, cache)
            except Exception as exc:
                return False, "could not verify full content hash: %s" % exc
            if current_content.lower() != expected_content:
                mismatches.append("content hash")

    if mismatches:
        return False, "source changed after inspection: " + ", ".join(mismatches)
    return True, "fingerprint matches"


def identity_matches(path: Path, payload: Mapping[str, object]) -> bool:
    try:
        current = file_identity(path)
        expected = FileIdentity(
            device=int(payload["device"]),
            inode=int(payload["inode"]),
            size=int(payload["size"]),
            mtime_ns=int(payload["mtime_ns"]),
            ctime_ns=int(payload.get("ctime_ns", current.ctime_ns)),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return current == expected
