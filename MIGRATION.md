# Migration to 0.6

## Generate a fresh index

Version 0.6 changes both classification semantics and on-disk schemas. Do not apply a plan produced by an earlier revision.

```bash
ROOT="/path/to/ComfyUI/models/loras"
python3 inspect_lora.py "$ROOT" --base "$ROOT"
```

Review the newly generated index, plan, and collision report.

## Legacy launchers remain

These commands remain valid:

```bash
python3 inspect_lora.py ...
python3 inspect_sort_lora.py ...
python3 test_lora_tools.py ...
```

They now import the package under `src/lora_header_inspector`.

## Schema behavior

A current inspection index contains:

```text
schema_name=lora-inspection
schema_version=2.0
```

A current reviewed plan additionally contains:

```text
plan_schema_name=lora-sort-plan
plan_schema_version=2.0
```

A versioned artifact with an unsupported version is rejected even when legacy import is enabled. This prevents a newer engine from silently changing the meaning of an old reviewed plan.

## Importing an unversioned CSV

A deliberately reviewed old CSV can be read with:

```bash
python3 inspect_sort_lora.py old_index.csv \
  --base "$ROOT" \
  --allow-legacy-plan
```

Rows without size, modification time, and header hash are still blocked. Bypassing that protection requires:

```bash
--allow-unverified
```

That combination forfeits the strongest stale-plan check. A fresh inspection is safer and normally faster than auditing a legacy CSV.

## Duplicate behavior changed

Earlier collision handling could only say that several plan rows wanted one destination. Version 0.6 groups those rows and hashes them lazily to distinguish exact duplicates from different-content files.

The default remains non-mutating:

```bash
--duplicate-policy report
```

Hard-link and quarantine policies must be selected during plan generation, reviewed in the resulting CSV and JSON report, and then applied from that reviewed plan.

## Evidence changed

The human-readable `evidence` column remains, but automation should consume:

```text
evidence_json
conflicts_json
warnings_json
```

Each evidence item has a stable rule ID, source, category, value, confidence, priority, field, and precision flag.

## Smoke mode

The previous positional smoke command still works:

```bash
python3 test_lora_tools.py /path/to/model.safetensors
```

The explicit spelling is clearer:

```bash
python3 test_lora_tools.py --inspect /path/to/model.safetensors
```

Neither form writes a CSV or invokes the sorter.

## Relaxed plans require apply-time acknowledgement

Dry-run settings are retained in the reviewed plan. During `--apply`, these relaxations must be repeated explicitly when enabled:

```text
--allow-ambiguous
--allow-outside-base
--allow-unverified
--allow-symlink-parents
```

Their rejecting counterparts can tighten a reviewed plan without regenerating the original index.

## Fingerprint change in 0.6

The current inspection schema records `ctime_ns` in addition to size, mtime, device, inode, and header hash. Regenerate any pre-release 0.6 index that lacks this column. Do not copy a ctime value from another file or another scan.
