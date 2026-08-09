"""Generate the fixed π0.5 LIBERO baseline/BSP phase-one comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from openpi_client import libero_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and compare exactly ten phase-one LIBERO runs: "
            "baseline/BSP at 0k, 5k, 10k, 20k, and 30k optimizer steps."
        )
    )
    parser.add_argument(
        "run_dirs",
        metavar="RUN_DIR",
        nargs=10,
        type=Path,
        help="Ten run directories; manifest contents, never directory names, identify each run.",
    )
    parser.add_argument(
        "--bsp-verification",
        required=True,
        type=Path,
        help="Task-6 BSP cache verification diagnostics JSON.",
    )
    parser.add_argument(
        "--norm-comparison",
        required=True,
        type=Path,
        help="Baseline/BSP normalization comparison diagnostics JSON.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty destination for the six fixed report artifacts.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        comparison = libero_report.compare_phase_one(
            args.run_dirs,
            bsp_diagnostics_path=args.bsp_verification,
            norm_comparison_path=args.norm_comparison,
            output_dir=args.output_dir,
        )
    except libero_report.ComparisonError as error:
        parser.error(str(error))
    print(
        "Validated {} paired rollouts and wrote fixed 0k/5k/10k/20k/30k comparison artifacts to {}".format(
            comparison["protocol"]["total_episodes"], args.output_dir.expanduser().resolve()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
