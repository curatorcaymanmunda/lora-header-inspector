# Current validation: post-0.6.2 working tree

Verified locally on 2026-10-30, Linux x86-64. The package version remains 0.6.2;
these results include the unreleased CI and image-signature changes. They are
not a retroactive claim about the published 0.6.2 archive.

## Automated regression results

| Python | Complete compatibility runner | Explicit fault-injection selector |
|---|---|---|
| 3.9.25 | 228 passed; no failures, errors, or skips | 10 passed |
| 3.10.22 | 228 passed; no failures, errors, or skips | 10 passed |
| 3.11.17 | 228 passed; no failures, errors, or skips | 10 passed |
| 3.12.14 | 228 passed; no failures, errors, or skips | 10 passed |
| 3.13.16 | 228 passed; no failures, errors, or skips | 10 passed |
| 3.14.8 | 228 passed; no failures, errors, or skips | 10 passed |
| 3.15.0 | 228 passed; no failures, errors, or skips | 10 passed |

Commands, run in a clean source checkout without an editable installation:

```bash
python test_lora_tools.py
python test_lora_tools.py tests.test_fault_injection
```

The ten fault tests are included in the full count, then run explicitly again.
They hard-exit child processes after journal writes and verify interrupted
apply/undo recovery, same-filesystem and simulated cross-filesystem moves,
duplicate hard links, quarantine, sidecars, and staging cleanup. All mutation
uses synthetic data in temporary directories. Existing authorization,
no-overwrite, rollback, undo, and recovery tests remain enabled.

The baseline checkout actually discovers 216 tests, rather than either historical
count (182 or 192). Its ten fault tests initially failed from an uninstalled
checkout because child processes could not import the src-layout package.
The harness now adds `src/` to those child processes' import path, matching the
compatibility runner. No transaction implementation changed. Twelve new test
methods cover FLUX.3, MOSS-Image, and FIBO recognition, including subtests for
negative, partial, conflicting, and exporter-spelling cases: 228 methods total.

## Packaging and installed behavior

- `python -m build` produced a source archive and built the wheel from that
  archive, using isolated build environments.
- `python -m twine check --strict dist/*` passed for both distributions.
- Both distributions installed separately in fresh virtual environments on
  each of the seven Python versions above; `pip check` passed in all fourteen.
- Each installed package was exercised from outside the repository with
  `python -I ci/check_installed.py ...`, using its virtual environment interpreter.
  The script verifies the import comes from site-packages, all four console
  entry points load, the module entry point loads, all 13 synthetic signature
  fixtures pass, and read-only inspection leaves a temporary model unchanged.
- Workflow YAML parsing and `git diff --check` passed.

## Continuous integration

[CI workflow](.github/workflows/ci.yml) runs on pushes, pull requests, and manual
dispatch. Each Linux job uses Python 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, or 3.15 and runs
the full suite, the explicit fault selector, isolated builds, strict metadata
checks, and separate wheel/sdist installations outside the checkout. The
minimum remains tested even after upstream end of life. Jobs fail independently
without cancelling the other Python versions. Tokens have read-only contents
permission, checkout does not persist credentials, and jobs have timeouts.

These are verified local results; hosted CI results are recorded by the linked
workflow runs after the branch is pushed. CI does not publish packages or modify
real libraries. See [validation evidence](validation/README.md) and
[model-recognition limits](docs/RECENT_IMAGE_SIGNATURES.md).

## Validation limits

The new architecture fixtures are synthetic, derived from published module
names and dimensions. They do not establish broad real-export coverage, load
models, or test image generation. Sparse exports can remain unresolved. Earlier
FIBO generations rely on identity metadata. Local checks used Linux; macOS,
Windows, actual cross-device mounts, network and clustered filesystem semantics
are not certified by this matrix. Historical fuzz, stress, and coverage figures
below were not re-measured and must not be presented as current results.

---

# Historical validation for 0.6.0 (retained audit record)

Validation was run on Linux x86-64 with Python 3.13.5. The package declares Python 3.9 or newer; every Python source file was also parsed with Python 3.9 grammar.

## Automated regression suite

```text
182 tests passed
0 failures
0 errors
```

The same suite passed through both direct `unittest` discovery and the compatibility runner. A legacy short selector was also exercised independently.

The suite covers:

- bounded safetensors header parsing and malformed-header rejection;
- structural signatures, metadata hints, filename traps, typed evidence, and typed conflicts;
- versioned inspection and plan schemas;
- preservation of reviewed `move=False` and canonical selections;
- path containment, parent-symlink topology, stale-plan identity, and destination races;
- stem-based and full-filename sidecars;
- exact duplicates, same-inode aliases, different-content collisions, and occupied destinations;
- report, hard-link, and quarantine duplicate policies;
- same-filesystem and simulated cross-filesystem publication;
- apply journals, rollback, undo, interrupted apply, interrupted undo, and recovery;
- exclusive initial creation of apply, undo, and recovery journals.

Branch-aware coverage reported **77% overall**. Classification and evidence reconciliation are above 90%; the lower aggregate is principally the large transaction engine's defensive and platform-specific failure branches.

## Signature fixtures

All 10 bundled payload-free fixtures passed:

```text
Anima
Flux.2
ambiguous generic transformer
Ideogram 4
Krea 2
MiniMax H3
Pony V7 / AuraFlow
SDXL
Wan 2.2 I2V high-noise
Z-Image
```

These fixtures are synthetic regression specimens, not a claim of broad real-export validation.

## Deterministic classifier fuzzing

```text
seed: 12648430
cases: 10,000
uncaught exceptions: 0
invalid result schemas: 0
```

The generator mixed sparse and malformed tensor descriptions, contradictory metadata, common-word filename traps, and multiple architecture hints.

## Duplicate stress probe

A 240-file temporary library contained:

```text
100 exact-duplicate pairs
20 same-name, different-content pairs
```

Results:

```text
100 hard-link group operations completed
100 duplicate-member rows handled by their group leaders
40 different-content rows blocked
0 apply errors
0 rollbacks
all linked paths verified as one inode per exact group
undo return code 0
all original duplicate copies restored as independent files
```

## Named `nobody_nathalie` replay

The reported `ZimageBase/nobody_nathalie.safetensors` and `Zit/nobody_nathalie.safetensors` arrangement was replayed with two byte-identical files on distinct inodes.

The hard-link policy:

1. classified the group as `exact_duplicate`;
2. created `ZImage/Base/nobody_nathalie.safetensors`;
3. preserved both old paths;
4. verified all three paths shared one inode;
5. undid the operation;
6. restored the two original paths as independent but byte-identical files.

## Command-line acceptance test

A source-tree acceptance run completed this entire workflow:

```text
inspect -> report-only plan -> reviewed hard-link plan -> apply -> verify -> undo
```

It included an SDXL LoRA with a `model.safetensors.preview.png` sidecar and an exact Z-Image duplicate pair. The sidecar moved with the model, the duplicate group became hard links, the apply manifest completed, and undo restored the original layout.

## Packaging and static checks

The following passed:

- Python compilation;
- `tabnanny`;
- Python 3.9 grammar parsing;
- import of all public package modules;
- Git whitespace checks;
- local Markdown-link validation;
- wheel build and installation in an isolated virtual environment;
- installed console-script smoke inspection.

## Remaining validation limits

- The bundled architecture fixtures are synthetic.
- The tool does not load base models or perform image-generation compatibility tests.
- Network and clustered filesystems can provide semantics weaker than the local overlay filesystem used for validation.
- Windows and macOS behavior is covered by conservative code paths and unit tests where practical, but the final acceptance run was performed on Linux.
