"""Crash-aware move, hard-link, quarantine, recovery, and undo operations.

This facade assembles the transactions layer from six modules and is the only
supported import path. Every layer module calls cross-cutting primitives as
attributes of this module, so patching ``transactions.<name>`` intercepts all
internal call sites exactly as it did when the engine was one flat module.
"""

from __future__ import annotations

import os  # re-exported: tests patch transactions.os.link

from .schema import (
    manifest_header,
    undo_header,
    write_json_atomic,
    write_json_exclusive,
)

from ._txn_publication import (
    COPY_CHUNK_BYTES,
    _copy_to_temp,
    _fsync_directory,
    _hardlink_unavailable,
    _inode_unchanged,
    _matches_stable_identity,
    _publish_staging_no_replace,
    _restore_backup,
    _unique_sibling,
    _utc_stamp,
    atomic_hardlink,
    atomic_relink,
    break_hardlink,
    safe_move,
)
from ._txn_relink import (
    _journaled_break_link_step,
    _journaled_relink_step,
    _rollback_break_link_step,
    _rollback_relink_step,
)
from ._txn_journal import (
    _create_manifest,
    _default_manifest_path,
    _journal_paths,
    _load_manifest,
    _load_recoverable_journal,
    _load_undo_manifest,
    _new_manifest,
    _protected_manifest_paths,
    _save_manifest,
)
from ._txn_apply import (
    _apply_hardlink_operation,
    _apply_move_operation,
    _apply_quarantine_operation,
    _current_regular_transfers,
    _preflight_transfers,
    _records_for_group,
    _remove_staging,
    _rollback_hardlink_steps,
    _rollback_move_steps,
    _same_regular_content,
    _verify_group_records,
    apply_plan,
)
from ._txn_undo import (
    _execute_undo_hardlink_operation,
    _execute_undo_move_operation,
    _prepare_undo_operation,
    _rollback_undo_hardlink_steps,
    undo_manifest,
)
from ._txn_recover import (
    _analyze_break_link_rollback_step,
    _analyze_hardlink_rollback_steps,
    _analyze_move_rollback_steps,
    _analyze_relink_rollback_step,
    _analyze_undo_hardlink_rollback_steps,
    recover_manifest,
)

__all__ = [
    "COPY_CHUNK_BYTES",
    "apply_plan",
    "atomic_hardlink",
    "atomic_relink",
    "break_hardlink",
    "manifest_header",
    "recover_manifest",
    "safe_move",
    "undo_header",
    "undo_manifest",
    "write_json_atomic",
    "write_json_exclusive",
]
