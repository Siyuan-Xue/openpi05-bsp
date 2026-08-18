from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_schema5_replaces_the_v4_runtime_surface_in_place():
    pairs = (
        ("examples/libero/main_v4.py", "examples/libero/main_v5.py"),
        (
            "packages/openpi-client/src/openpi_client/libero_control_v4.py",
            "packages/openpi-client/src/openpi_client/libero_control_v5.py",
        ),
        (
            "packages/openpi-client/src/openpi_client/libero_eval_v4.py",
            "packages/openpi-client/src/openpi_client/libero_eval_v5.py",
        ),
        (
            "packages/openpi-client/src/openpi_client/libero_report_v4.py",
            "packages/openpi-client/src/openpi_client/libero_report_v5.py",
        ),
        (
            "packages/openpi-client/src/openpi_client/libero_video_timing_v4.py",
            "packages/openpi-client/src/openpi_client/libero_video_timing_v5.py",
        ),
    )
    for old_path, new_path in pairs:
        assert not (ROOT / old_path).exists()
        assert (ROOT / new_path).is_file()


def test_schema5_cli_and_report_are_exactly_three_mode_random_latency():
    evaluator = (ROOT / "examples/libero/main_v5.py").read_text()
    control = (ROOT / "packages/openpi-client/src/openpi_client/libero_control_v5.py").read_text()
    report = (ROOT / "packages/openpi-client/src/openpi_client/libero_report_v5.py").read_text()

    assert "synthetic_latency_target_ms" not in evaluator
    assert '"baseline_async"' in control
    assert '"baseline_rtc"' in control
    assert '"bsp_spline_async"' in control
    assert '"baseline_sync_n5"' not in control
    assert '"bsp_spline_sync"' not in control
    assert '"comparison_v5.json"' in report
    assert '"task_metrics_v5.csv"' in report
    assert '"report_v5.md"' in report
    assert "write_three_mode_report_v5" in report
    assert "write_five_checkpoint_report_v5" not in report


def test_runbook_stops_after_server_gate_until_explicit_approval():
    runbook = (ROOT / "docs/pi05_libero_latency_experiment_v5.md").read_text()

    assert "通过服务器门禁后先停止" in runbook
    assert "没有用户再次明确批准" in runbook
    assert "不启动正式 2000-episode" in runbook
    assert "baseline_async" in runbook
    assert "baseline_rtc" in runbook
    assert "bsp_spline_async" in runbook
