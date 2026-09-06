# Proposed changelog entry: 0.6.2

### Fault injection

- Added `tests/test_fault_injection.py`: ten scenarios that run apply and undo in a subprocess and hard-kill it (`os._exit`) the moment the journal reaches a target phase, then assert payload survival, clean dry-run recovery, clean applied recovery, and no surviving staging temporaries.
- Covered phases: all-pending, same-filesystem published-before-unlink, cross-filesystem staging / publishing / published, completed-primary-with-pending-sidecar, duplicate backup-published, duplicate relinked, quarantine published, and a crash mid-undo recovered to applied state.
- The kill hook flushes subprocess coverage data before exiting so crashed children are measurable.

### Recovery fix (found by fault injection)

- A crash between backup publication and old-name unlink during a hard-link relink was unrecoverable: creating the backup link bumps the original's ctime, so the full identity check reported the path as occupied in both dry-run analysis and applied rollback.
- Added `_inode_unchanged` (device, inode, and size; times ignored) and taught `_rollback_relink_step` and `_analyze_relink_rollback_step` to recognize the state where both names still reference the original inode. Rollback now removes only the extra backup name.

### Architecture

- Split the transaction engine into six layer modules: `_txn_publication`, `_txn_relink`, `_txn_journal`, `_txn_apply`, `_txn_undo`, `_txn_recover`.
- `transactions.py` is now a facade assembling the public API. Layer modules call cross-cutting primitives as `_txn.<name>` attributes of the facade, so patching `transactions.<name>` intercepts every internal call site exactly as it did in the flat module. All existing mock-based tests pass unchanged.
- Layer modules are private; `lora_header_inspector.transactions` remains the only supported import path.

### Dead code

- Pruned fourteen unreferenced functions from `utils.py` (stale duplicates of canonical versions now living in `paths`, `planner`, `fingerprints`, and `schema`). Seven live functions remain. Statement count 167 -> 85; coverage 44% -> 69%.
- Note before merging: if the package `__init__.py` re-exports any pruned utils name, restore that single function or drop it from the prune.

### Verification

- Full compatibility suite: 192 tests green (182 existing + 10 fault injection).
- Transaction layer branch coverage 58% -> 64% with crashed subprocesses counted; recovery analyzers remain the thinnest area and are the suggested next target.

### Header parsing increments (from the external parsing review)

- Lowered the default header cap from 256 MB to the official safetensors limit of 100 MB; `read_header` keeps its override and `lora-inspect` gained `--max-header-bytes`.
- The first header byte must be `{`. A leading space that `json.loads` would forgive is now rejected before decoding.
- Bytes after the top-level JSON value must be `0x20` space padding only; trailing newlines or NULs fail closed. Alignment padding remains legal and remains part of the raw-byte header hash.
- Added the official dtype allowlist (including the F8 variants). An unrecognised dtype such as `fp16` is a warning by default, recorded in `header_warnings_json`, so sloppy but loadable files still classify.
- Added opt-in `--strict-layout` (`read_header(strict_layout=True)`): recognised dtypes only, `end - start` must equal element count times dtype size, dimensions bounded by the schema's 48-bit size, and non-empty tensors must tile the payload exactly with no holes and no overlaps. Off by default per the project's classify-sloppy-files principle.
- Thirteen new tests in `tests/test_header_security.py` cover first-byte, padding, cap, dtype, tiling, hole, overlap, size-lie, uncovered-tail, lenient-default, and CLI flag exposure.
- Deliberately not adopted from the review: Unicode-normalized key deduplication (homoglyph keys simply fail to match signatures and classify as Unknown, which is the honest result) and switching to mmap or streaming JSON (LoRA headers are tens to hundreds of KB; `read` after the cap is simpler and equally safe).

### Enumeration

- Added `lora-inspect --enumerate-only`: recursively counts matching safetensors files per bucket (first directory level below each target) and exits without reading a single header. Deduplication matches `collect_files`, so the total equals the record count a full inspection of the same targets would produce. Guarded by a test that fails if any header is read.

### Parallel inspection

- Added `lora-inspect --jobs N`: inspects and hashes with a thread pool. Effective because hashlib and file reads release the GIL; the shared HashCache needs no lock since collect_files deduplicates paths, so no two workers ever compute the same cache key. Row order always matches input order, making serial and parallel CSVs byte-identical (test-enforced).
- Default remains 1: no behavior change without the explicit flag, and parallel full-file hashing on spinning disks is a seek-thrash regression. Recommended: physical core count on SSD/NVMe with `--hash-content` or duplicate policies; not worth it for header-only passes.

### Plan-time hash reuse

- Plan-time record verification re-hashes the full content of every record that carries a full_sha256, which made `lora-sort` a second serial pass over the whole library after a parallel inspection. Added `--trust-recorded-hashes` (`build_sort_plan(trust_recorded_hashes=True)`): recorded hashes are seeded into the plan's HashCache under the identity they were recorded at (device, inode, size, mtime, ctime), so a lookup succeeds only when the file's current identity still matches. Identity changes force a real re-hash; an unprivileged writer cannot modify content without moving ctime, which is the same trust boundary the in-process HashCache already uses.
- The explicit trade-off, documented in the flag help: a hand-edited full_sha256 column on an untouched file is believed at plan time. Apply-time duplicate relinks still verify digests against real bytes before linking. Default remains off: the paranoid re-hash is pinned by a test that falsifies a CSV hash and asserts detection.
- The planner's cache is now also shared with record verification, so collision members are no longer hashed twice in default mode.
- Benchmark (40 x 48 MB, all records hashed): plan 2.00s -> 0.03s with the flag; on a multi-thousand-file library this removes the second full serial hashing pass entirely.

### Folder-safe versions

- Version-shaped metadata noise no longer mints one bucket per float. The anima version rule captured any dotted digit run verbatim, so kohya float formatting produced buckets like `Base__v10.000002` and `v1.1024.8` (observed on a real 4,257-file library). Added `folderable_version` in normalization: only one to three dot-separated components of at most three digits each may become a `__v` qualifier.
- The verbatim value is preserved in the `model_version` record field for review, and classification emits a warning naming the dropped version. Confirmed against all four bucket shapes observed in the field; sane versions (`1.0`, `v6`, `2.2`) are unaffected.

### Release hygiene

- Version bumped to 0.6.2 (tool and package).
- Added a repository `.gitignore` keeping census artifacts out of the repo: inspection CSVs, sort plans, manifests, undo and recovery journals, and any `*.safetensors`.
- Added `KNOWN_ISSUES.md` documenting the honest limitations: Unknown as a valid result (with the `--signature-report` contribution path), stricter-than-0.5 parsing, unresolved name collisions, and the deliberately serial apply.
