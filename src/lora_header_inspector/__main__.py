"""Unified ``python -m lora_header_inspector`` entry point."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from . import __version__
from .cli_sort import main as sort_main
from .inspection import main as inspect_main
from .fixtures import main as fixtures_main
from .smoke import main as smoke_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lora_header_inspector")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", add_help=False)
    sub.add_parser("sort", add_help=False)
    sub.add_parser("smoke", add_help=False)
    sub.add_parser("fixtures", add_help=False)
    args, remainder = parser.parse_known_args(list(argv) if argv is not None else None)
    if args.command == "inspect":
        return inspect_main(remainder)
    if args.command == "sort":
        return sort_main(remainder)
    if args.command == "fixtures":
        return fixtures_main(remainder)
    return smoke_main(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
