"""Generate the exact two-input LIBERO schema-v5 synchronous comparison."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from openpi_client import libero_report_sync_v5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exactly baseline_sync and bsp_spline_sync schema-v5 10K run directories."
    )
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs=2,
        metavar="RUN_DIR",
        help="the two paired synchronous formal run directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="a new or empty destination for the synchronous v5 report files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    libero_report_sync_v5.write_sync_pair_report_v5(args.run_dirs, output_dir=args.output_dir)
    print(f"schema-v5 synchronous report written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
