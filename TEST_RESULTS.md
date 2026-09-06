# Validation results for 0.6.0

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
