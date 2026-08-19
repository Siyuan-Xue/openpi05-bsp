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


def test_schema5_preserves_three_mode_report_and_adds_two_sync_runtime_modes():
    evaluator = (ROOT / "examples/libero/main_v5.py").read_text()
    control = (ROOT / "packages/openpi-client/src/openpi_client/libero_control_v5.py").read_text()
    report = (ROOT / "packages/openpi-client/src/openpi_client/libero_report_v5.py").read_text()
    sync_report = (ROOT / "packages/openpi-client/src/openpi_client/libero_report_sync_v5.py").read_text()

    assert "synthetic_latency_target_ms" not in evaluator
    assert '"baseline_async"' in control
    assert '"baseline_rtc"' in control
    assert '"bsp_spline_async"' in control
    assert '"baseline_sync"' in control
    assert '"bsp_spline_sync"' in control
    assert '"comparison_v5.json"' in report
    assert '"task_metrics_v5.csv"' in report
    assert '"report_v5.md"' in report
    assert "write_three_mode_report_v5" in report
    assert "write_five_checkpoint_report_v5" not in report
    assert '"sync_comparison_v5.json"' in sync_report
    assert '"sync_task_metrics_v5.csv"' in sync_report
    assert '"sync_report_v5.md"' in sync_report
    assert "write_sync_pair_report_v5" in sync_report


def test_runbook_freezes_sync_extension_without_restarting_existing_modes():
    runbook = (ROOT / "docs/pi05_libero_latency_experiment_v5.md").read_text()

    assert "同步扩展" in runbook
    assert "baseline_sync_n5_h16_full_v2" in runbook
    assert "bsp_spline_sync_speedup2_phase0_v2" in runbook
    assert "完整执行 16 个动作" in runbook
    assert "从 `t_min` 开始" in runbook
    assert "本轮只补跑 `baseline_sync`" in runbook
    assert "不得重启" in runbook
    assert "baseline_async" in runbook
    assert "baseline_rtc" in runbook
    assert "bsp_spline_async" in runbook
