# CLI reference

## Inspector

```bash
python3 inspect_lora.py TARGET [TARGET ...] [OPTIONS]
```

Important options:

| Option | Meaning |
|---|---|
| `--output PATH` | Inspection CSV path |
| `--base PATH` | Root used for the automatic dry-run plan |
| `--follow-symlinks` | Inspect final symlink targets; default is to report an error |
| `--hash-content` | Stream a full SHA-256 for every model |
| `--no-sort-plan` | Stop after writing the inspection index |
| `--signature-report PATH` | Write a payload-free structural report |
| `--include-fixture-filenames` | Include basenames in the signature report |
| `--omit-fixture-metadata` | Exclude trusted identity metadata from the report |

## Planner and transaction engine

```bash
python3 inspect_sort_lora.py [CSV] [OPTIONS]
```

| Option | Meaning |
|---|---|
| `--base PATH` | Architecture-bucket root; mandatory with `--apply` |
| `--apply` | Execute the reviewed operations |
| `--lineage` / `--no-lineage` | Enable or disable secondary compatibility folders |
| `--min-confidence LEVEL` | `low`, `medium`, or `high` |
| `--allow-ambiguous` / `--reject-ambiguous` | Relax or tighten status eligibility |
| `--allow-outside-base` / `--confine-to-base` | Relax or enforce destination containment |
| `--allow-symlink-parents` / `--reject-symlink-parents` | Relax or enforce parent-symlink policy |
| `--allow-unverified` / `--require-verification` | Relax or enforce source fingerprints |
| `--allow-legacy-plan` | Explicitly import an unversioned CSV |
| `--duplicate-policy POLICY` | `report`, `hardlink`, or `quarantine` |
| `--quarantine-dir NAME` | Quarantine directory under the base |
| `--collision-detail LEVEL` | `none`, `summary`, or `full` |
| `--allow-workflow-path-breakage` | Required to apply quarantine |
| `--manifest PATH` | Custom apply, undo, or recovery journal path |
| `--max-path-length N` | Reject longer destination paths; `0` disables |
| `--undo MANIFEST` | Dry-run reversal; add `--apply` to execute |
| `--recover MANIFEST` | Dry-run interrupted-operation recovery; add `--apply` |

Relaxed plan settings are visible in the CSV. During apply, the four `--allow-*` safety relaxations must be repeated explicitly.

## Smoke inspector

```bash
python3 test_lora_tools.py --inspect TARGET [TARGET ...]
```

Options:

```text
--json
--hash-content
--follow-symlinks
```

Without a path, `test_lora_tools.py` runs the regression suite.

## Fixture validator

```bash
python3 -m lora_header_inspector fixtures FIXTURE_OR_DIRECTORY [...]
```

Options:

```text
--json
--output PATH
```
