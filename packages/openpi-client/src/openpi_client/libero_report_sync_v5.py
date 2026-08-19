"""Strict, separate comparison for the two schema-v5 synchronous modes.

The original three-mode random-latency report remains frozen in
``libero_report_v5``.  This module deliberately uses different entry points
and output names so synchronous runs cannot be mixed into that report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence

from openpi_client import libero_artifacts
from openpi_client import libero_eval_v5
from openpi_client import libero_report_v5 as report


SYNC_EXECUTION_MODE_ORDER_V5 = (
    "baseline_sync",
    "bsp_spline_sync",
)
SYNC_OUTPUT_FILENAMES_V5 = (
    "sync_comparison_v5.json",
    "sync_task_metrics_v5.csv",
    "sync_report_v5.md",
)

_ROLLOUT_RECORD_IDENTITY_FIELDS = (
    "episode_id",
    "paired_key",
    "suite",
    "task_id",
    "task_name",
    "init_state_index",
    "init_state_fingerprint",
    "eval_seed",
)


def classify_sync_manifests_v5(
    manifests: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    """Classify exactly the baseline and BSP synchronous 10K manifests."""
    if isinstance(manifests, (str, bytes)) or not isinstance(manifests, Sequence):
        raise report.ComparisonErrorV5("Synchronous manifests must be a sequence")
    if len(manifests) != len(SYNC_EXECUTION_MODE_ORDER_V5):
        raise report.ComparisonErrorV5("Synchronous comparison requires exactly two manifests")
    normalized = [report._manifest_v5(manifest) for manifest in manifests]  # noqa: SLF001
    by_mode = {}  # type: Dict[str, Mapping[str, Any]]
    for manifest in normalized:
        mode = manifest["execution_mode"]
        if mode in by_mode:
            raise report.ComparisonErrorV5("Synchronous comparison contains duplicate mode {}".format(mode))
        by_mode[mode] = manifest
    if set(by_mode) != set(SYNC_EXECUTION_MODE_ORDER_V5):
        raise report.ComparisonErrorV5("Comparison requires exactly the two sync execution modes")
    if len({manifest["checkpoint_step"] for manifest in normalized}) != 1:
        raise report.ComparisonErrorV5("Both synchronous manifests must share one checkpoint_step")
    reference = report._rollout_manifest_identity(by_mode[SYNC_EXECUTION_MODE_ORDER_V5[0]])  # noqa: SLF001
    if report._rollout_manifest_identity(by_mode[SYNC_EXECUTION_MODE_ORDER_V5[1]]) != reference:  # noqa: SLF001
        raise report.ComparisonErrorV5("Both synchronous manifests must share one rollout identity")
    return {mode: by_mode[mode] for mode in SYNC_EXECUTION_MODE_ORDER_V5}


def _records_by_key(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> Dict[str, Mapping[str, Any]]:
    result = {}  # type: Dict[str, Mapping[str, Any]]
    for record in records:
        paired_key = record.get("paired_key")
        if not isinstance(paired_key, str) or not paired_key:
            raise report.ComparisonErrorV5("{} contains a record without paired_key".format(label))
        if paired_key in result:
            raise report.ComparisonErrorV5("{} contains duplicate paired_key {}".format(label, paired_key))
        result[paired_key] = record
    return result


def _validate_pair_rollouts(runs_by_mode: Mapping[str, report.RunDataV5]) -> None:
    for mode in SYNC_EXECUTION_MODE_ORDER_V5:
        run = runs_by_mode[mode]
        report._validate_formal_episode_grid(run.records, path=run.path)  # noqa: SLF001
        if any(
            record.get("execution_mode") != mode or record.get("eval_seed") != run.manifest["eval_seed"]
            for record in run.records
        ):
            raise report.ComparisonErrorV5("Synchronous rollout identity does not match its run")
    baseline = _records_by_key(runs_by_mode["baseline_sync"].records, label="baseline_sync")
    bsp = _records_by_key(runs_by_mode["bsp_spline_sync"].records, label="bsp_spline_sync")
    if set(baseline) != set(bsp):
        raise report.ComparisonErrorV5("Synchronous paired rollout identity keys do not match")
    for key, baseline_record in baseline.items():
        bsp_record = bsp[key]
        mismatched = [
            field for field in _ROLLOUT_RECORD_IDENTITY_FIELDS if bsp_record.get(field) != baseline_record.get(field)
        ]
        if mismatched:
            raise report.ComparisonErrorV5("Synchronous rollout identity {} differs in {}".format(key, mismatched))
        baseline_samples = {
            request["request_id"]: (
                request["latency_sample_key"],
                request["sampled_target_latency_ns"],
            )
            for request in baseline_record["inference_requests"]
        }
        bsp_samples = {
            request["request_id"]: (
                request["latency_sample_key"],
                request["sampled_target_latency_ns"],
            )
            for request in bsp_record["inference_requests"]
        }
        for request_id in set(baseline_samples).intersection(bsp_samples):
            if baseline_samples[request_id] != bsp_samples[request_id]:
                raise report.ComparisonErrorV5(
                    "Synchronous paired latency sample differs for {} request {}".format(key, request_id)
                )


def compare_sync_pair_v5(
    runs_by_mode: Mapping[str, report.RunDataV5],
) -> Mapping[str, Any]:
    """Compare the exact two synchronous modes without widening the legacy reporter."""
    if not isinstance(runs_by_mode, Mapping) or set(runs_by_mode) != set(SYNC_EXECUTION_MODE_ORDER_V5):
        raise report.ComparisonErrorV5("Synchronous comparison requires exactly two runs by mode")
    ordered = {}  # type: Dict[str, report.RunDataV5]
    for mode in SYNC_EXECUTION_MODE_ORDER_V5:
        run = runs_by_mode[mode]
        if not isinstance(run, report.RunDataV5) or run.execution_mode != mode:
            raise report.ComparisonErrorV5("Synchronous run key does not match its execution_mode")
        ordered[mode] = run
    classified = classify_sync_manifests_v5([ordered[mode].manifest for mode in SYNC_EXECUTION_MODE_ORDER_V5])
    _validate_pair_rollouts(ordered)
    code_sha_by_mode = {mode: classified[mode]["code_sha"] for mode in SYNC_EXECUTION_MODE_ORDER_V5}
    return {
        "schema_version": 5,
        "checkpoint_step": classified["baseline_sync"]["checkpoint_step"],
        "latency_distribution": classified["baseline_sync"]["latency_distribution"],
        "success_rates": {
            mode: report._success_rate(ordered[mode].records)  # noqa: SLF001
            for mode in SYNC_EXECUTION_MODE_ORDER_V5
        },
        "paired_delta": report._paired_delta_report(  # noqa: SLF001
            ordered["baseline_sync"],
            ordered["bsp_spline_sync"],
        ),
        "diagnostics": {
            mode: report._latency_diagnostics(ordered[mode].records, classified[mode])  # noqa: SLF001
            for mode in SYNC_EXECUTION_MODE_ORDER_V5
        },
        "code_sha_by_mode": code_sha_by_mode,
        "same_binary_both_modes": len(set(code_sha_by_mode.values())) == 1,
        "inputs": {
            mode: {
                "run_directory": str(ordered[mode].path),
                "code_sha": classified[mode]["code_sha"],
                "policy_protocol": classified[mode]["policy_protocol"],
                "file_sha256": dict(sorted(ordered[mode].file_sha256.items())),
            }
            for mode in SYNC_EXECUTION_MODE_ORDER_V5
        },
    }


def _task_metrics_rows(runs_by_mode: Mapping[str, report.RunDataV5]) -> List[Mapping[str, Any]]:
    rows = []  # type: List[Mapping[str, Any]]
    for suite in libero_eval_v5.SUPPORTED_SUITES:
        for task_id in range(report.TASKS_PER_SUITE_V5):
            rates = {}  # type: Dict[str, float]
            task_name: Optional[str] = None
            for mode in SYNC_EXECUTION_MODE_ORDER_V5:
                records = [
                    record
                    for record in runs_by_mode[mode].records
                    if record["suite"] == suite and record["task_id"] == task_id
                ]
                if len(records) != report.TRIALS_PER_TASK_V5:
                    raise report.ComparisonErrorV5("Synchronous task metrics require 50 paired trials")
                rates[mode] = sum(record["success"] is True for record in records) / len(records)
                current_name = str(records[0]["task_name"])
                if task_name is None:
                    task_name = current_name
                elif task_name != current_name:
                    raise report.ComparisonErrorV5("Synchronous cross-mode task names do not match")
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task_name,
                    "baseline_sync_success_rate": rates["baseline_sync"],
                    "bsp_spline_sync_success_rate": rates["bsp_spline_sync"],
                    "bsp_spline_sync_minus_baseline_sync": (rates["bsp_spline_sync"] - rates["baseline_sync"]),
                }
            )
    return rows


def _render_report(comparison: Mapping[str, Any]) -> str:
    delta = comparison["paired_delta"]
    return "\n".join(
        [
            "# LIBERO schema-v5 synchronous comparison",
            "",
            "| mode | macro success rate |",
            "|---|---:|",
            "| `baseline_sync` | {:.6f} |".format(comparison["success_rates"]["baseline_sync"]),
            "| `bsp_spline_sync` | {:.6f} |".format(comparison["success_rates"]["bsp_spline_sync"]),
            "",
            "Paired BSP-minus-baseline delta: {:.6f}; bootstrap 95% CI [{:.6f}, {:.6f}].".format(
                delta["observed_delta"],
                delta["bootstrap_95_ci"][0],
                delta["bootstrap_95_ci"][1],
            ),
            "",
            "The original three-mode schema-v5 report remains a separate artifact.",
            "",
        ]
    )


def write_sync_pair_report_v5(
    run_dirs: Sequence[Any],
    *,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Write a report for exactly baseline_sync and bsp_spline_sync."""
    if isinstance(run_dirs, (str, bytes)) or not isinstance(run_dirs, Sequence):
        raise report.ComparisonErrorV5("run_dirs must be a sequence")
    if len(run_dirs) != len(SYNC_EXECUTION_MODE_ORDER_V5):
        raise report.ComparisonErrorV5("Synchronous report requires exactly two runs")
    if all(isinstance(value, report.RunDataV5) for value in run_dirs):
        runs = list(run_dirs)
    elif any(isinstance(value, report.RunDataV5) for value in run_dirs):
        raise report.ComparisonErrorV5("Do not mix RunDataV5 objects and run directories")
    else:
        resolved = [Path(value).expanduser().resolve() for value in run_dirs]
        if len(set(resolved)) != len(resolved):
            raise report.ComparisonErrorV5("Synchronous run directories must be unique")
        runs = [report.load_run_v5(path) for path in resolved]
    runs_by_mode = {}  # type: Dict[str, report.RunDataV5]
    for run in runs:
        if run.execution_mode in runs_by_mode:
            raise report.ComparisonErrorV5("Duplicate synchronous run mode {}".format(run.execution_mode))
        runs_by_mode[run.execution_mode] = run
    comparison = compare_sync_pair_v5(runs_by_mode)
    payload = {
        "schema_version": 5,
        "protocol": {
            "checkpoint_step": 10000,
            "execution_modes": list(SYNC_EXECUTION_MODE_ORDER_V5),
            "suites": list(libero_eval_v5.SUPPORTED_SUITES),
            "tasks_per_suite": report.TASKS_PER_SUITE_V5,
            "trials_per_task": report.TRIALS_PER_TASK_V5,
            "episodes_per_run": report.EPISODES_PER_RUN_V5,
        },
        "comparison": comparison,
    }
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise report.ComparisonErrorV5("Synchronous report output path must be a directory")
        if any(output.iterdir()):
            raise report.ComparisonErrorV5("Synchronous report output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    libero_artifacts.atomic_text(output / "sync_comparison_v5.json", libero_artifacts.json_text(payload))
    libero_artifacts.write_csv(output / "sync_task_metrics_v5.csv", _task_metrics_rows(runs_by_mode))
    libero_artifacts.atomic_text(output / "sync_report_v5.md", _render_report(comparison))
    actual = {path.name for path in output.iterdir()}
    if actual != set(SYNC_OUTPUT_FILENAMES_V5):
        raise report.ComparisonErrorV5("Unexpected synchronous report output set: {}".format(sorted(actual)))
    return payload
