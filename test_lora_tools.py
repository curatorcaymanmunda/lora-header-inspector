#!/usr/bin/env python3
"""Compatibility test runner and read-only smoke inspector.

Examples:
    python test_lora_tools.py
    python test_lora_tools.py ClassifierTests.test_sd15_by_cross_attention_width
    python test_lora_tools.py --inspect /path/to/model.safetensors
    python test_lora_tools.py /path/to/model.safetensors --json
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import unittest
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_header_inspector.smoke import main as smoke_main


def _looks_like_smoke(argv: Iterable[str]) -> bool:
    values = list(argv)
    if "--inspect" in values:
        return True
    for argument in values:
        if argument.startswith("-"):
            continue
        expanded = os.path.expanduser(argument)
        path = Path(expanded)
        if (
            path.exists()
            or path.is_symlink()
            or path.suffix.lower() == ".safetensors"
            or any(character in expanded for character in "*?[")
        ):
            return True
    return False


def _test_modules() -> List[object]:
    modules = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        modules.append(importlib.import_module("tests." + path.stem))
    return modules


def _contains_failed_test(suite: unittest.TestSuite) -> bool:
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            if _contains_failed_test(test):
                return True
        elif test.__class__.__name__ == "_FailedTest":
            return True
    return False


def _load_selected_tests(selectors: List[str]) -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    combined = unittest.TestSuite()
    modules = _test_modules()
    for selector in selectors:
        loaded = None
        # Fully-qualified unittest names continue to work normally.
        if selector.startswith("tests."):
            candidate = loader.loadTestsFromName(selector)
            if not _contains_failed_test(candidate):
                loaded = candidate
        else:
            for module in modules:
                candidate = loader.loadTestsFromName(selector, module)
                if not _contains_failed_test(candidate) and candidate.countTestCases():
                    loaded = candidate
                    break
        if loaded is None:
            raise ValueError(f"test selector not found: {selector}")
        combined.addTests(loaded)
    return combined


def _run_tests(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("selectors", nargs="*")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-k", dest="pattern")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise ValueError("unrecognised test-runner argument(s): " + " ".join(unknown))

    loader = unittest.defaultTestLoader
    if args.pattern:
        loader.testNamePatterns = ["*" + args.pattern + "*"]
    if args.selectors:
        suite = _load_selected_tests(args.selectors)
    else:
        suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))
    verbosity = 0 if args.quiet else max(2, 1 + args.verbose)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


def _print_help() -> None:
    print(__doc__.strip())
    print(
        "\nModes:\n"
        "  no path supplied       run the complete regression suite\n"
        "  unittest selector      run one or more selected tests\n"
        "  path / --inspect       read-only smoke inspection\n\n"
        "Test options:\n"
        "  -q, --quiet            quiet test output\n"
        "  -v, --verbose          increase test verbosity\n"
        "  -k PATTERN             run tests whose full id contains PATTERN\n\n"
        "Smoke options are shown by:\n"
        "  python test_lora_tools.py --inspect --help\n"
    )


def main() -> int:
    argv = sys.argv[1:]
    if argv in (["-h"], ["--help"]):
        _print_help()
        return 0
    if _looks_like_smoke(argv):
        filtered = [item for item in argv if item != "--inspect"]
        return smoke_main(filtered)
    try:
        return _run_tests(argv)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
