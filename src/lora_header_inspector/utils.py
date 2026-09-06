"""Small, dependency-free utilities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Set, TypeVar


T = TypeVar("T")


def unique(items: Iterable[T]) -> List[T]:
    seen: Set[T] = set()
    result: List[T] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def join_semicolon(items: Iterable[str]) -> str:
    return ";".join(value for value in unique(item for item in items if item))


def absolute_without_resolving(path: Path) -> Path:
    """Return an absolute path without dereferencing the final symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def atomic_json_write(data: Mapping[str, Any], path: Path) -> None:
    path = absolute_without_resolving(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def create_json_exclusive(data: Mapping[str, Any], path: Path) -> None:
    """Create a JSON file only when the public path is still absent.

    Initial operation journals use this instead of replace semantics.  A crash
    while writing can leave an incomplete journal, but it happens before any
    filesystem mutation and, crucially, can never overwrite a file that won a
    race for the requested path.
    """
    path = absolute_without_resolving(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o600)
    owned = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path.parent)
    except Exception:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            current = path.stat()
            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                path.unlink()
                fsync_directory(path.parent)
        except OSError:
            pass
        raise


def fsync_directory(path: Path) -> None:
    """Best-effort directory fsync on platforms that support it."""
    if os.name == "nt":
        return
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return "%s-%s" % (prefix, digest.hexdigest()[:length])


