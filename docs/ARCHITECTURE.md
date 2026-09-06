# Architecture

## Pipeline

```text
safetensors file
    -> bounded header parser
    -> normalized key index
    -> structural signature registry
    -> field-scoped metadata evidence
    -> filename evidence
    -> evidence reconciliation and conflicts
    -> versioned inspection row
    -> reviewed plan
    -> collision and duplicate analysis
    -> dry-run report
    -> journalled apply
    -> undo or interrupted-operation recovery
```

Inspection and mutation are intentionally separate trust domains.

## Header layer

`header.py` reads only the safetensors header for ordinary inspection. It validates the declared length against a configurable ceiling and the actual file size, rejects duplicate JSON keys, validates tensor records, and calculates the raw-header SHA-256. The inspection fingerprint also records device, inode, size, mtime, and ctime.

A full content hash is optional during inspection and lazy during collision analysis.

## Normalized key index

`normalization.py` builds a case-insensitive search index without destroying original tensor names. Detectors query this index for module spellings, observed block indices, and LoRA input dimensions.

Observed block indices describe LoRA coverage. They are never treated as proof of total model depth.

## Structural signature registry

Each detector is registered with a stable detector name and order. It can emit one or more `SignatureMatch` values containing:

```text
rule_id
structural family
exact architecture, when justified
confidence
priority
ambiguity flag
human-readable evidence
```

Detector exceptions become low-confidence diagnostic evidence rather than terminating the batch.

## Hints

`hints.py` parses each trusted metadata field independently. This avoids accidental cross-field matches. Filename rules are word-bounded and lower priority. Common words and surnames are demoted unless model context is explicit.

## Reconciliation

`classifier.py` combines structural and hint evidence. It independently resolves singleton dimensions such as version, task, noise stage, and model variant. Disagreement creates typed conflicts rather than a confidence downgrade hidden in prose.

The final status is one of:

```text
ok
ambiguous
conflict
error
```

## Schemas

`schema.py` defines current field sets and validates schema identities before normalizing rows. Inspection indexes and reviewed plans are different contracts. Unsupported versioned artifacts are rejected rather than imported as legacy.

## Planner

`planner.py` evaluates eligibility, freezes base and path settings, snapshots parent topology and sidecars, computes destinations, groups target collisions, and delegates content comparison to `duplicates.py`.

The planner writes both a reviewable CSV and a machine-readable collision report.

## Duplicate engine

`duplicates.py` compares device/inode identities, sizes, and lazily calculated full SHA-256 hashes. It distinguishes aliases, exact copies, occupied identical destinations, and different-content files. Canonical selection is deterministic and reviewer-controllable.

## Transaction engine

`transactions.py` persists each recoverable phase before mutation. It provides:

- no-overwrite safe move;
- cross-filesystem verified copy and source removal;
- backup-first hard-link replacement;
- quarantine;
- per-operation rollback;
- whole-manifest undo preflight;
- undo journals;
- interrupted apply and undo recovery.

A journal is evidence of what the tool attempted, not a declaration that the filesystem still matches that state. Every undo and recovery path rechecks current identities. Initial journal creation is exclusive; later updates are atomic rewrites of the journal owned by that run.

## Compatibility launchers

The root scripts preserve established commands and imported function names. New integrations should prefer the package APIs and installed console commands.
