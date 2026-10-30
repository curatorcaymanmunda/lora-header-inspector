# LoRA Header Inspector

[![CI](https://github.com/curatorcaymanmunda/lora-header-inspector/actions/workflows/ci.yml/badge.svg)](https://github.com/curatorcaymanmunda/lora-header-inspector/actions/workflows/ci.yml)

LoRA Header Inspector is a dependency-free, header-only toolkit for identifying LoRA base architectures, reviewing a proposed library layout, analysing duplicate files, and applying reversible filesystem changes.

It does **not** import PyTorch, load tensor payloads, or generate images. The inspector reads the safetensors JSON header, preserves uncertainty, writes a versioned CSV index, and produces a dry-run sort plan. The mover is a separate step and remains opt-in.

## Design promises

- **Classification and mutation are separate.** Inspection never moves anything.
- **Structure, architecture, ecosystem, variant, and technique are separate concepts.** Pony, Illustrious, Turbo, and similar labels do not compete with SDXL, Flux, or Z-Image as one flat category.
- **Unknown is a valid result.** Sparse or contradictory evidence is not converted into confident fiction.
- **Dry run is the default.** `--apply` is required for filesystem changes.
- **No overwrite and no automatic deletion.** Destination races and different-content collisions are blocked.
- **Plans and journals are versioned.** Unsupported schemas fail closed.
- **Apply, undo, and recovery are journalled.** Same-filesystem and cross-filesystem moves use different safe publication paths.
- **Legacy commands remain available.** `inspect_lora.py`, `inspect_sort_lora.py`, and `test_lora_tools.py` are thin launchers around the package.

## Requirements

- Python 3.9 or newer
- No runtime dependencies
- A filesystem that supports normal file operations
- Hard-link duplicate policy requires the participating files to be on the same filesystem

## Installation

Run directly from an extracted release:

```bash
python3 inspect_lora.py --help
python3 inspect_sort_lora.py --help
```

Or install the package into a virtual environment:

```bash
python3 -m pip install .
```

Installed commands:

```text
lora-inspect
lora-sort
lora-smoke
lora-fixtures
```

The unified module entry point is also available:

```bash
python3 -m lora_header_inspector inspect --help
python3 -m lora_header_inspector sort --help
python3 -m lora_header_inspector smoke --help
python3 -m lora_header_inspector fixtures --help
```

## Safe first run

```bash
ROOT="/home/user/ComfyUI/models/loras"

python3 inspect_lora.py "$ROOT" \
  --base "$ROOT"
```

This writes:

```text
lora_inspect_index.csv
lora_inspect_index_sort_plan.csv
lora_inspect_index_collision_report.json
```

No files are moved. Review the index, plan, collision report, and every row marked `ambiguous`, `conflict`, or `error`.

The plan CSV contains hand-editable review fields:

```text
move
destination
canonical
duplicate_preference
```

A reviewed plan retains ordinary planning settings. High-risk relaxations are different: an apply run must repeat `--allow-ambiguous`, `--allow-outside-base`, `--allow-unverified`, or `--allow-symlink-parents` when the reviewed plan used them. An editable CSV is evidence of review, not authorization to escape a safety boundary.

Apply the reviewed plan with an explicit base directory:

```bash
python3 inspect_sort_lora.py \
  lora_inspect_index_sort_plan.csv \
  --base "$ROOT" \
  --apply
```

The apply run creates a versioned manifest before mutation. Keep that manifest until the library and saved workflows have been verified.

## Read-only smoke inspection

Inspect one file, a folder, or a glob without writing any CSV:

```bash
python3 test_lora_tools.py --inspect \
  "$ROOT/Krea2/example.safetensors"
```

Machine-readable output:

```bash
python3 test_lora_tools.py --inspect "$ROOT" --json > smoke.json
```

The installed equivalent is:

```bash
lora-smoke "$ROOT"
```

## Duplicate policies

Collisions are grouped and classified as:

```text
same_inode
exact_duplicate
existing_exact_duplicate
different_content
unverified_collision
sidecar_collision
```

The default policy is `report`:

```bash
python3 inspect_sort_lora.py index.csv \
  --base "$ROOT" \
  --duplicate-policy report \
  --collision-detail full
```

### `report`

No duplicate is modified. This is the default and safest discovery mode.

### `hardlink`

Exact duplicate model paths are preserved, but independent duplicate files are replaced by hard links to one canonical payload. Existing ComfyUI workflow paths continue to resolve.

```bash
python3 inspect_sort_lora.py index.csv \
  --base "$ROOT" \
  --duplicate-policy hardlink
```

Review the generated plan before applying it. Hard-linked model files should be treated as immutable. In-place writes through one path change every path sharing that inode.

### `quarantine`

One canonical model is moved into the normalized layout and redundant copies are moved beneath `<base>/_Duplicates/...`. Because old relative paths change, apply requires an explicit acknowledgement:

```bash
python3 inspect_sort_lora.py index.csv \
  --base "$ROOT" \
  --duplicate-policy quarantine

python3 inspect_sort_lora.py index_sort_plan.csv \
  --base "$ROOT" \
  --apply \
  --allow-workflow-path-breakage
```

No duplicate policy automatically deletes a model.

## Undo and interrupted-operation recovery

Dry-run an undo:

```bash
python3 inspect_sort_lora.py --undo path/to/apply_manifest.json
```

Apply the undo:

```bash
python3 inspect_sort_lora.py \
  --undo path/to/apply_manifest.json \
  --apply
```

Inspect recovery options for an interrupted apply or undo journal:

```bash
python3 inspect_sort_lora.py --recover path/to/manifest.json
```

Apply recovery only after reading the dry-run analysis:

```bash
python3 inspect_sort_lora.py \
  --recover path/to/manifest.json \
  --apply
```

Recovery aims to restore the pre-operation state. It refuses to overwrite an unrelated file that appeared after the journal was written.

## Signature research and fixtures

Create a payload-free structural report from real LoRAs:

```bash
python3 inspect_lora.py "$ROOT" \
  --no-sort-plan \
  --signature-report signature_report.json
```

By default, filenames are omitted and identity metadata is sanitized. Tensor keys, dtypes, and shapes are included, but data offsets and tensor payloads are not.

Validate the bundled fixture corpus:

```bash
python3 -m lora_header_inspector fixtures tests/fixtures/signatures
```

Fixtures use a versioned JSON contract. They are reviewable test data, not executable code. The bundled examples are explicitly marked synthetic. See [docs/REAL_FIXTURES.md](docs/REAL_FIXTURES.md) before contributing a real-header-derived specimen.

## Current recognition scope

The classifier has exact or metadata-assisted support for:

```text
SD 1.5, SD 2.x, SDXL, SD3/3.5
AuraFlow
Flux.1, Flux.2, Chroma, LongCat Image, HunyuanVideo
Krea 2
Z-Image
Ideogram 4
Qwen-Image
Anima
ERNIE-Image
MiniMax H3
Wan and Wan-derived Bernini-R
LTX-family metadata
```

Compatibility ecosystems and qualifiers include Pony, Illustrious, NoobAI, Animagine, Kontext, Schnell, Klein, Base, Turbo, version, task, noise stage, and several acceleration techniques.

Support does not imply that every sparse LoRA can be identified structurally. A LoRA exposes only the modules it targets. Metadata can settle a shared layout, and filenames remain weak evidence. See [MODEL_MATRIX.md](MODEL_MATRIX.md).

## Schemas and migration

Version 0.6 uses independent schema identities for inspection indexes, reviewed plans, collision reports, operation manifests, undo journals, signature reports, and signature fixtures. Do not apply a v0.5 plan as though it were a v0.6 plan.

Generate a fresh index whenever the classification rules or schema version changes. An unversioned legacy CSV can be imported only with `--allow-legacy-plan`, and missing fingerprints still require `--allow-unverified`. Those switches are migration tools, not recommended defaults.

See [MIGRATION.md](MIGRATION.md) and [docs/SCHEMAS.md](docs/SCHEMAS.md). The complete option summary is in [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md).

## Testing

Run everything:

```bash
python3 test_lora_tools.py
```

Run a selected legacy-style test:

```bash
python3 test_lora_tools.py \
  ClassifierTests.test_sd15_by_cross_attention_width
```

Filter by test id:

```bash
python3 test_lora_tools.py -k duplicate
```

The suite covers header parsing, classification, schemas, path topology, sidecars, exact duplicates, different-content collisions, hard links, quarantine, no-overwrite publication, cross-filesystem fallback, apply journals, undo, and recovery.

The release validation matrix and exact acceptance results are recorded in [TEST_RESULTS.md](TEST_RESULTS.md).

## Project layout

```text
inspect_lora.py                 compatibility inspector launcher
inspect_sort_lora.py            compatibility planner/mover launcher
test_lora_tools.py              regression runner and smoke launcher
src/lora_header_inspector/
    header.py                   bounded safetensors header parser
    signatures/                 structural detector registry
    hints.py                    field-scoped metadata and filename evidence
    classifier.py               conflict-aware evidence reconciliation
    schema.py                   versioned CSV/JSON contracts
    planner.py                  eligibility, destinations, collisions
    duplicates.py               content-aware duplicate analysis
    sidecars.py                 allowlisted sidecar discovery and snapshots
    transactions.py             apply, rollback, undo, recovery
    fixtures.py                 payload-free fixture validation
tests/
    fixtures/signatures/        versioned synthetic fixtures
```

## Security and limitations

Read [SECURITY.md](SECURITY.md) before applying plans to an irreplaceable library.

Important limits:

- Header-only classification cannot prove compatibility when a sparse LoRA omits distinctive modules and metadata.
- Filename hints are fallible and never deserve automatic trust by themselves.
- The bundled signature fixtures are synthetic. A broader, sanitized corpus of real exports from kohya, OneTrainer, ai-toolkit, Musubi, Diffusers, LyCORIS, and ComfyUI is still needed.
- Transaction journals reduce risk but do not replace backups, filesystem snapshots, or tested restore procedures.
- Network filesystems may provide weaker durability or locking semantics than local filesystems.

## Licence status

This refactor does not invent a licence for code originally shared by another author. Before public redistribution, preserve the original licence if one exists or obtain an explicit licence choice from the rights holder. See [LICENSE-NOTICE.md](LICENSE-NOTICE.md).
