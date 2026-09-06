# Real-header-derived fixtures

The bundled fixtures are synthetic. They prove regression behavior but do not represent the full variety of trainer exports. Community-grade recognition needs sanitized fixtures derived from real headers.

## Export a payload-free report

```bash
python3 inspect_lora.py /path/to/models \
  --no-sort-plan \
  --signature-report signature_report.json
```

The report contains tensor keys, dtypes, shapes, a module-prefix histogram, sanitized identity metadata, and the current classification. It contains no tensor payload or data offsets.

Filenames are omitted by default. Include them only when they carry essential compatibility information and have been reviewed:

```bash
--include-fixture-filenames
```

Omit identity metadata entirely with:

```bash
--omit-fixture-metadata
```

## Privacy review

Before sharing a report or fixture, inspect it for:

- usernames or home directories;
- absolute Windows, POSIX, UNC, or `file://` paths;
- signed URLs, query strings, fragments, or access tokens;
- prompts, captions, dataset names, project names, or private person names;
- filenames that reveal private content;
- trainer configuration blobs unrelated to architecture identity.

The exporter sanitizes known identity fields, but the person sharing the report remains responsible for review.

## Convert a record into a fixture

A fixture contains:

```json
{
  "schema_name": "lora-signature-fixture",
  "schema_version": "1.0",
  "fixture_id": "portable-unique-id",
  "provenance": {
    "source_type": "real-header-derived",
    "exporter": "kohya",
    "notes": "Sanitized and independently verified"
  },
  "filename": "",
  "metadata": {},
  "tensors": [
    {"key": "...", "dtype": "BF16", "shape": [8, 3072]}
  ],
  "expected": {
    "arch": "...",
    "structural_family": "...",
    "status": "..."
  }
}
```

Do not include `data_offsets`, payload bytes, hashes that are not needed, or absolute paths.

## Validate

```bash
python3 -m lora_header_inspector fixtures path/to/fixture.json
```

A real-header-derived fixture should be independently labelled from the model card, trainer configuration, or known base checkpoint. The current classifier result is not ground truth merely because it appears in the generated report.

## Acceptance criteria

A contributed fixture should state:

- source type;
- exporter or trainer family;
- whether key names were transformed by ComfyUI or another converter;
- independently verified expected architecture;
- why the included keys are sufficient or intentionally ambiguous;
- closest lookalike model considered.
