# Known issues and honest limitations

## Unknown is a valid result

On a real multi-thousand-file library, roughly 5-10% of adapters classify as
Unknown. This is by design: when the header carries neither a decisive
structural signature nor trustworthy metadata, refusing to guess is the
correct answer, and the planner will never move an Unknown file. To help
extend coverage, run `lora-inspect <files> --signature-report report.json`
on your Unknowns: the report is payload-free and path-free and contains only
structural shape information suitable for contributing new signature rules.

## Stricter parsing than 0.5

The 0.6 parser enforces the official safetensors rules: 100 MB header cap,
first byte `{`, space-only padding after the JSON value, unique keys, and
payload-bounded offsets. Files that older tools (including the 0.5 baseline
preserved in this repository) read without complaint may now report errors.
Each error message names the violated rule; the file itself is untouched.
`--strict-layout` adds dtype allowlisting and no-holes/no-overlap payload
tiling for well-formedness auditing and stays off by default so sloppy but
loadable adapters still classify.

## Name collisions are reported, never resolved

Files with the same name but different content (`skip_content_collision`)
are deliberately left in place. Common causes observed in the wild: platform
download names (`V1.safetensors`), trainer defaults
(`checkpoint-e19_s304.safetensors`), and the same concept retrained across
architectures under one name. Rename before re-planning if you want them
sorted; the tool will not pick a winner.

## Apply is single-threaded on purpose

Inspection parallelizes (`--jobs`, or GNU parallel with a CSV merge), and
planning is cheap with `--trust-recorded-hashes`. Apply remains one ordered,
journalled, fsync-punctuated process: the crash-recovery guarantees assume
exactly one log. Do not run multiple applies against one tree.
