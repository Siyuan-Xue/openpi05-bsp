from pathlib import Path

import pytest

from scripts import compare_libero_latency_v5 as cli


def test_cli_requires_exactly_three_mode_directories_and_output():
    parsed = cli.parse_args(
        [
            "baseline-async",
            "baseline-rtc",
            "bsp-spline-async",
            "--output-dir",
            "report",
        ]
    )
    assert parsed.run_dirs == [
        Path("baseline-async"),
        Path("baseline-rtc"),
        Path("bsp-spline-async"),
    ]
    assert parsed.output_dir == Path("report")

    with pytest.raises(SystemExit):
        cli.parse_args(["one", "two", "--output-dir", "report"])
    with pytest.raises(SystemExit):
        cli.parse_args(["one", "two", "three", "four", "--output-dir", "report"])


def test_main_forwards_the_exact_three_inputs(monkeypatch, capsys):
    observed = {}

    def write_report(run_dirs, *, output_dir):
        observed["run_dirs"] = list(run_dirs)
        observed["output_dir"] = output_dir
        return {"schema_version": 5, "protocol": {"execution_modes": ["a", "b", "c"]}}

    monkeypatch.setattr(cli.libero_report_v5, "write_three_mode_report_v5", write_report)
    result = cli.main(
        [
            "baseline-async",
            "baseline-rtc",
            "bsp-spline-async",
            "--output-dir",
            "report",
        ]
    )

    assert result == 0
    assert observed == {
        "run_dirs": [Path("baseline-async"), Path("baseline-rtc"), Path("bsp-spline-async")],
        "output_dir": Path("report"),
    }
    assert "schema-v5 report written" in capsys.readouterr().out
