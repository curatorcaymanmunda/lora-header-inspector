# Contributing

## Principles

1. Prefer an honest `Unknown` over a confident false positive.
2. A structural rule must document both positive requirements and plausible lookalikes.
3. A filename is weak evidence. Common words, surnames, and syllables must be scoped.
4. Architecture, ecosystem, checkpoint variant, version, task, noise stage, and training technique are independent dimensions.
5. Every behavior change needs regression tests.
6. Filesystem mutation must be no-overwrite, journalled, and reversible where possible.
7. Do not add a runtime dependency merely to avoid a small amount of standard-library code.

## Development setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 test_lora_tools.py
```

## Adding a structural signature

1. Add or extend a detector under `src/lora_header_inspector/signatures/`.
2. Register the detector with a stable name and order.
3. Give every `SignatureMatch` a stable `rule_id`.
4. Return a broad structural family when the keys are shared.
5. Return an exact architecture only when the positive evidence is distinctive.
6. Add negative tests for the closest lookalike architecture.
7. Add at least one versioned fixture under `tests/fixtures/signatures/`.
8. Update `MODEL_MATRIX.md` and `docs/SIGNATURE_RULES.md`.

Do not infer total model depth from the highest block index present in a LoRA. A LoRA may target any sparse subset of blocks.

## Adding metadata hints

Metadata fields are parsed independently. Do not concatenate all metadata into one search string. A model name split across two unrelated fields must not become a synthetic match.

Identity fields may carry stronger evidence than titles or filenames. Common words such as `wan`, `anima`, `chroma`, `klein`, and `schnell` need explicit context.

## Adding a filesystem operation

Any new mutating operation needs:

- preflight of every source and destination;
- no-overwrite publication;
- a manifest step written before mutation;
- apply-time identity revalidation;
- rollback behavior;
- undo behavior;
- interrupted-operation recovery;
- tests for destination races and partial failure;
- cross-filesystem semantics where applicable.

## Tests

```bash
python3 test_lora_tools.py
python3 test_lora_tools.py -k duplicate
python3 -m lora_header_inspector fixtures tests/fixtures/signatures
```

Tests must use temporary directories. Do not include model payloads in the repository.

## Fixture privacy

Follow `docs/REAL_FIXTURES.md`. A fixture must not include tensor payloads, data offsets, absolute paths, usernames, signed URLs, prompts, captions, or private tokens. Mark synthetic fixtures as synthetic and real-header-derived fixtures as real-header-derived.

## Compatibility

The package supports Python 3.9 and newer. Avoid syntax introduced after Python 3.9 unless the minimum version is deliberately changed and documented.

## CI and package validation

GitHub Actions runs the complete runner and an explicit fault-injection/recovery
selector on Linux with Python 3.9–3.15. Tests run before any package installation
so subprocess imports must work from a clean source checkout. All tests that
modify files must stay inside temporary directories; never point CI at a real
LoRA library or relax authorization/recovery assertions to make a job pass.

Each job also builds an sdist and a wheel from it, checks metadata with Twine,
and installs each artifact into a separate virtual environment. The installed
check runs from outside the repository with isolated Python mode to prevent
source imports from masking missing package contents. Build tools are development
dependencies only; the package remains free of runtime dependencies.

For an installed distribution, run from outside the checkout:

```bash
/path/to/venv/bin/python -I /path/to/checkout/ci/check_installed.py /path/to/checkout/tests/fixtures/signatures
```

Update [TEST_RESULTS.md](TEST_RESULTS.md) using observed interpreter versions,
runner counts, skips, and artifact checks. The older files in `validation/`
are explicitly historical and are not proof of current validation.
