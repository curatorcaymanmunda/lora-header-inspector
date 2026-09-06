# Validation evidence

These logs were generated from the v0.6.0 source release.

- `tests_final.log`: 182 unit and regression tests, including the compatibility runner.
- `static_validation.log`: compilation, tabnanny, Python 3.9 grammar parsing, import sweep, whitespace checks, and local documentation-link checks.
- `coverage_report.log`: branch-aware coverage report; 77% overall, with higher concentration around classification and evidence reconciliation.
- `coverage_tests.log`: coverage test command result.
- `fuzz_10000.log`: deterministic 10,000-case sparse and malformed-header classifier fuzz run with zero uncaught exceptions.
- `duplicate_stress_240_files.log`: 100 exact-duplicate groups and 20 same-name/different-content groups, followed by hard-link apply and undo.
- `nobody_nathalie_acceptance.log`: replay of the reported `ZimageBase` / `Zit` exact-duplicate case.
- `cli_end_to_end.log`: command-line inspect, report, reviewed hard-link plan, apply, manifest verification, and undo.
- `wheel_build.log`: wheel build output.
- `wheel_smoke.log`: isolated virtual-environment install, `pip check`, entry-point tests, and real smoke inspection.

The architecture fixtures are sanitized header-only specimens. They validate recognition rules and artifact contracts; they do not load base models or run image generation.
