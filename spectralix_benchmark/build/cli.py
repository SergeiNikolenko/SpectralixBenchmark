from __future__ import annotations

import argparse
import sys
from typing import Sequence

from spectralix_benchmark.build import level_benchmark_files, paper_eval_subsets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectralix-build",
        description="Build benchmark_v3 pools and paper evaluation subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "levels",
        help="Build Level A/B/C large pools (benchmark construction stage).",
    )
    subparsers.add_parser(
        "paper-eval",
        help="Build paper-facing eval subsets from Level A/B/C pools.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    tokens = list(argv) if argv is not None else sys.argv[1:]
    if not tokens:
        _build_parser().print_help()
        return

    command = tokens[0]
    passthrough = tokens[1:]
    if command in {"-h", "--help"}:
        _build_parser().print_help()
        return
    if command == "levels":
        level_benchmark_files.main(passthrough)
        return
    if command == "paper-eval":
        paper_eval_subsets.main(passthrough)
        return
    raise SystemExit(f"[ERROR] Unknown command: {command}")


if __name__ == "__main__":
    main()
