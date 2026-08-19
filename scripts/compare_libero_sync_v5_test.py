from pathlib import Path

import pytest

from scripts import compare_libero_sync_v5 as cli


def test_cli_requires_exactly_two_sync_directories_and_output():
    parsed = cli.parse_args(
        [
            "baseline-sync",
            "bsp-spline-sync",
            "--output-dir",
            "sync-report",
        ]
    )
    assert parsed.run_dirs == [Path("baseline-sync"), Path("bsp-spline-sync")]
    assert parsed.output_dir == Path("sync-report")

    with pytest.raises(SystemExit):
        cli.parse_args(["baseline-sync", "--output-dir", "sync-report"])
    with pytest.raises(SystemExit):
        cli.parse_args(["one", "two", "three", "--output-dir", "sync-report"])


def test_main_forwards_the_exact_two_inputs(monkeypatch, capsys):
    observed = {}

    def write_report(run_dirs, *, output_dir):
        observed["run_dirs"] = list(run_dirs)
        observed["output_dir"] = output_dir
        return {"schema_version": 5}

    monkeypatch.setattr(cli.libero_report_sync_v5, "write_sync_pair_report_v5", write_report)
    result = cli.main(
        [
            "baseline-sync",
            "bsp-spline-sync",
            "--output-dir",
            "sync-report",
        ]
    )

    assert result == 0
    assert observed == {
        "run_dirs": [Path("baseline-sync"), Path("bsp-spline-sync")],
        "output_dir": Path("sync-report"),
    }
    assert "synchronous report written" in capsys.readouterr().out
