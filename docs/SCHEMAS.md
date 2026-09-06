# On-disk schemas

## Current identities

| Artifact | Schema name | Version |
|---|---|---:|
| Inspection CSV | `lora-inspection` | 2.0 |
| Reviewed plan CSV | `lora-sort-plan` | 2.0 |
| Collision report JSON | `lora-collision-report` | 1.0 |
| Apply/recovery manifest JSON | `lora-operation-manifest` | 2.0 |
| Undo journal JSON | `lora-undo-manifest` | 2.0 |
| Signature report JSON | `lora-signature-report` | 1.0 |
| Signature fixture JSON | `lora-signature-fixture` | 1.0 |

Schema versions describe on-disk meaning. Tool version describes the implementation that wrote the artifact.

## Inspection CSV

Important groups:

### Source identity

```text
record_id
input_path
path
source_was_symlink
file_size
mtime_ns
device
inode
header_sha256
full_sha256
header_bytes
payload_bytes
```

### Classification

```text
arch
structural_family
model_version
task_variant
noise_stage
confidence
status
ecosystem
model_variant
unresolved_hint
techniques
sort_subfolder
format
```

### Evidence

```text
evidence
evidence_json
conflicts_json
warnings_json
```

The prose `evidence` field is for people. Programs should use the JSON fields.

### Review fields

```text
move
destination
canonical
duplicate_preference
```

These survive conversion into a plan.

## Plan CSV

A plan embeds the inspection schema fields and adds:

- plan identity and frozen settings;
- generated and reviewed move decisions;
- source verification result;
- sidecar snapshots;
- path topology snapshot;
- collision group and duplicate classification;
- canonical source and policy;
- quarantine and sidecar reconciliation information.

Unknown public columns are retained to preserve reviewer notes.

A reviewed plan is not portable to a different base directory unless it is regenerated. Apply verifies that the explicit `--base` agrees with the frozen plan.

## Collision report

The collision report is a JSON grouping of all plan rows that converge on a primary or sidecar destination. Full detail includes source identities, hashes, canonical choice, policy, and reasons for blocking.

## Operation manifests

Manifests record phases, not only final moves. A step may include private temporary or backup paths required for recovery. The engine writes the journal before each mutation and updates it after durable publication.

Possible top-level outcomes include completed, completed with rollbacks, and attention required.

## Legacy import

Unversioned CSV import requires `--allow-legacy-plan`. A file carrying a versioned schema name with an unsupported version is never treated as legacy. This prevents accidental downgrade attacks and semantic drift.

## Plan identity invariants

All rows in a reviewed plan must share one `plan_id`, one resolved base, and one set of frozen planning settings. Rows copied from two different plans are rejected. A reviewed plan supplied against a different resolved `--base` is rejected before row processing.

The plan stores high-risk relaxation fields for auditability, but `--apply` requires those relaxations to be explicitly repeated on the command line.

## Stale-plan identity

Current inspection rows include `file_size`, `mtime_ns`, `ctime_ns`, `device`, `inode`, and `header_sha256`. A missing `ctime_ns` makes a row incompatible with the current versioned inspection schema; regenerate the index rather than hand-filling it.
