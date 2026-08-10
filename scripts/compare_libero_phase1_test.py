import contextlib
import io
from pathlib import Path

import pytest

from scripts import compare_libero_phase1


def test_parser_requires_exactly_ten_runs_and_three_explicit_artifact_paths():
    parser = compare_libero_phase1.build_parser()
    arguments = [
        "run-a",
        "run-b",
        "run-c",
        "run-d",
        "run-e",
        "run-f",
        "run-g",
        "run-h",
        "run-i",
        "run-j",
        "--bsp-verification",
        "verify.json",
        "--norm-comparison",
        "norm.json",
        "--output-dir",
        "report",
    ]

    parsed = parser.parse_args(arguments)

    assert parsed.run_dirs == [Path(value) for value in arguments[:10]]
    assert parsed.bsp_verification == Path("verify.json")
    assert parsed.norm_comparison == Path("norm.json")
    assert parsed.output_dir == Path("report")

    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parser.parse_args(arguments[1:])

    help_text = parser.format_help().lower()
    assert "ten" in help_text
    for milestone in ("0k", "1k", "2k", "5k", "10k"):
        assert milestone in help_text
    assert "20k" not in help_text
    assert "30k" not in help_text
