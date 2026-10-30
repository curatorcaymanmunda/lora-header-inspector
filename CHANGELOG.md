# Changelog

## Unreleased

- Add Linux GitHub Actions regression, explicit crash/recovery, build, metadata,
  and isolated wheel/sdist installation checks on Python 3.9–3.15.
- Fix fault-injection child-process imports in clean src-layout checkouts.
- Add conservative FLUX.3 Image Klein, MOSS-Image, and FIBO 2.0 structural rules,
  explicit metadata hints (including earlier FIBO), synthetic fixtures, and
  negative regression tests. Existing mutation authorization is unchanged.
- Reconcile historical test-count claims with current verified results in
  TEST_RESULTS.md.

## 0.6.0

### Architecture

- Replaced the three growing implementation monoliths with a Python package and thin compatibility launchers.
- Added a composable structural signature registry with stable rule IDs.
- Added typed evidence, typed conflicts, and independent reconciliation of architecture, structural family, ecosystem, model variant, model version, task, noise stage, and technique.
- Kept metadata fields independent so unrelated values cannot accidentally form one synthetic model name.

### On-disk contracts

- Added versioned inspection, plan, collision-report, apply-manifest, undo-journal, signature-report, and signature-fixture schemas.
- Unsupported versioned artifacts fail closed.
- Unversioned CSV import now requires `--allow-legacy-plan`.
- Unknown public CSV columns survive plan regeneration so reviewer notes are not discarded.

### Header and classification safety

- Added strict header-length, UTF-8, JSON-root, duplicate-key, tensor-record, shape, and data-offset validation.
- Added a bounded maximum header size before allocation.
- Added optional full-content hashing and stable streamed hashing with before/after identity checks.
- Added `ctime_ns` to inspection fingerprints so same-size payload rewrites with a restored mtime are rejected on filesystems where ctime records metadata change.
- Preserved conservative classification and explicit `ok`, `ambiguous`, `conflict`, and `error` states.

### Filesystem planning

- Added lexical and resolved path snapshots, parent-symlink topology checks, base containment, portable component sanitization, and path-length policy.
- Added versioned sidecar snapshots and apply-time sidecar revalidation, including preview and metadata forms based on either the model stem or the full `.safetensors` filename.
- Added content-aware collision groups: same inode, exact duplicate, existing exact destination, different content, unverified collision, and sidecar collision.
- Added deterministic canonical selection with reviewer overrides.

### Duplicate policies

- Added `report`, `hardlink`, and `quarantine` policies.
- Kept `report` as the default.
- Added explicit workflow-path-breakage acknowledgement for quarantine.
- Added sidecar reconciliation without pretending different previews or metadata are identical.
- Added no automatic deletion.

### Transactions

- Replaced overwrite-capable publication paths with no-overwrite publication.
- Added same-filesystem and cross-filesystem move paths with streamed copy verification.
- Added apply journals that persist every recoverable phase before mutation.
- Initial apply, undo, and recovery journals now use exclusive creation, so a path appearing after preflight is never replaced.
- Added backup-first hard-link replacement and hard-link break operations.
- Added whole-manifest undo preflight, undo journals, rollback-to-applied-state on partial undo, and interrupted-undo recovery.
- Added dry-run recovery analysis that blocks on an unrelated content race.

### Community maintenance

- Added payload-free signature reports.
- Added a versioned signature fixture format and validator.
- Added ten synthetic fixture specimens.
- Repaired the compatibility test runner so help, full discovery, selectors, filtering, and smoke mode coexist.
- Added contributor, schema, architecture, duplicate, recovery, fixture, migration, and security documentation.

## 0.5 baseline

The uploaded baseline already provided header-only inspection, structured architecture and qualifier output, conservative movement eligibility, source fingerprints, sidecar preflight, and ordinary move manifests. Version 0.6 preserves those user-facing principles while replacing the internal monoliths and strengthening collision and transaction semantics.
