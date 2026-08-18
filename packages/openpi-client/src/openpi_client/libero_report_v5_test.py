from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from openpi_client import latency_sampling
from openpi_client import libero_control_v5 as control
from openpi_client import libero_eval_v5 as evaluation
from openpi_client import libero_report_v5 as report


MODES = ("baseline_async", "baseline_rtc", "bsp_spline_async")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _checkpoint(mode: str, *, identity_suffix: str = "") -> control.CheckpointIdentityV1:
    bsp = mode == "bsp_spline_async"
    return control.CheckpointIdentityV1(
        code_sha="a" * 40,
        config_name=("pi05_libero_bsp_lora_h16" if bsp else "pi05_libero_baseline_lora_h16"),
        checkpoint_step=10000,
        checkpoint=("/checkpoints/bsp/10000" if bsp else "/checkpoints/baseline/10000"),
        container_digest="sha256:" + "b" * 64,
        norm_hash=_sha(("bsp-norm" if bsp else "baseline-norm") + identity_suffix),
        bsp_cache_hash=_sha("sidecar") if bsp else None,
        bsp_cache_manifest_fingerprint=_sha("sidecar-manifest") if bsp else None,
    )


def _calibration(mode: str, *, identity_suffix: str = "") -> control.LatencyCalibrationV2:
    checkpoint = _checkpoint(mode, identity_suffix=identity_suffix)
    raw_warmup = [100_000_000] * 5
    target_warmup = [300_000_000] * 5
    raw_measurement = [100_000_000] * 20
    target_measurement = [300_000_000] * 20
    return control.LatencyCalibrationV2.create(
        execution_mode=mode,
        checkpoint_identity_fingerprint=checkpoint.fingerprint,
        server_metadata_fingerprint=_sha("server"),
        canonical_observation_identity=control.CalibrationObservationIdentityV1(
            suite="libero_spatial",
            task_id=0,
            init_state_index=0,
            init_state_fingerprint=_sha("state"),
            request_fingerprint=_sha("observation"),
        ),
        seed_namespace="openpi-libero-calibration-v2/{}/{}".format(mode, checkpoint.fingerprint),
        bootstrap_request_fingerprint=_sha("bootstrap") if mode == "baseline_rtc" else None,
        warmup_request_fingerprints=[_sha("warmup-{}".format(index)) for index in range(5)],
        measurement_request_fingerprints=[_sha("measurement-{}".format(index)) for index in range(20)],
        warmup_raw_inference_latency_ns=raw_warmup,
        warmup_sampled_target_latency_ns=target_warmup,
        warmup_requested_synthetic_delay_ns=[200_000_000] * 5,
        warmup_observed_synthetic_delay_ns=[200_000_000] * 5,
        warmup_observed_effective_latency_ns=target_warmup,
        warmup_latency_overshoot_ns=[0] * 5,
        measurement_raw_inference_latency_ns=raw_measurement,
        measurement_sampled_target_latency_ns=target_measurement,
        measurement_requested_synthetic_delay_ns=[200_000_000] * 20,
        measurement_observed_synthetic_delay_ns=[200_000_000] * 20,
        measurement_observed_effective_latency_ns=target_measurement,
        measurement_latency_overshoot_ns=[0] * 20,
    )


def _manifest(mode: str, *, identity_suffix: str = "") -> dict:
    spec = control.EXECUTION_MODES[mode]
    checkpoint = _checkpoint(mode, identity_suffix=identity_suffix)
    return evaluation.EvaluationManifestV5(
        schema_version=5,
        dataset_fps=10,
        source_demo_control_hz=20,
        control_freq_hz=20,
        controller_period_ns=50_000_000,
        video_fps=40,
        video_show_inference_waits=True,
        latency_distribution={
            "distribution": "normal",
            "mean_ns": latency_sampling.DEFAULT_MEAN_NS,
            "stddev_ns": latency_sampling.DEFAULT_STDDEV_NS,
            "seed": latency_sampling.DEFAULT_SEED,
            "sampler_version": latency_sampling.SAMPLER_VERSION,
            "negative_policy": latency_sampling.NEGATIVE_POLICY,
        },
        theoretical_p95_latency_ns=control.THEORETICAL_P95_LATENCY_NS,
        scheduling_latency_budget_ns=control.SCHEDULING_LATENCY_BUDGET_NS,
        scheduling_delay_ticks=control.SCHEDULING_DELAY_TICKS,
        execution_mode=mode,
        execution_parameters=spec.to_parameters_dict(),
        latency_calibration=_calibration(mode, identity_suffix=identity_suffix),
        server_metadata_fingerprint=_sha("server"),
        code_sha=checkpoint.code_sha,
        dataset_revision="v2.0",
        config_name=checkpoint.config_name,
        checkpoint_step=checkpoint.checkpoint_step,
        bsp_cache_hash=checkpoint.bsp_cache_hash,
        bsp_cache_manifest_fingerprint=checkpoint.bsp_cache_manifest_fingerprint,
        norm_hash=checkpoint.norm_hash,
        checkpoint=checkpoint.checkpoint,
        container_digest=checkpoint.container_digest,
        train_seed=42,
        eval_seed=42,
        policy_variant=spec.policy_variant,
        bsp_parameters=evaluation.BSP_PARAMETERS,
        policy_protocol=spec.policy_protocol,
        expected_action_horizon=spec.expected_action_horizon,
        suites=evaluation.SUPPORTED_SUITES,
        task_ids=tuple(range(10)),
        trials_per_task=50,
        num_steps_wait=10,
        max_steps_by_suite=evaluation.MAX_STEPS_BY_SUITE,
        connection_timeout_s=30.0,
        inference_timeout_s=120.0,
        infrastructure_retries=2,
    ).to_dict()


def _records(mode: str) -> list[dict]:
    records = []
    sampler = latency_sampling.NormalLatencySamplerV1()
    success_cutoff = {
        "baseline_async": 40,
        "baseline_rtc": 45,
        "bsp_spline_async": 48,
    }[mode]
    for suite in evaluation.SUPPORTED_SUITES:
        for task_id in range(10):
            for trial_index in range(50):
                fingerprint = _sha("{}:{}:{}".format(suite, task_id, trial_index))
                paired_key = "{}/task-{:03d}/init-{:03d}/{}".format(suite, task_id, trial_index, fingerprint)
                sample_key = latency_sampling.LatencySampleKeyV1(
                    namespace="formal",
                    seed=42,
                    suite=suite,
                    task_id=task_id,
                    trial_index=trial_index,
                    request_ordinal=0,
                )
                sampled_target = sampler.sample_target_ns(sample_key)
                raw = 10_000_000
                underflow = 50_000_000 if mode == "baseline_async" else 0
                records.append(
                    {
                        "episode_id": "{}-task-{:03d}-init-{:03d}-{}".format(suite, task_id, trial_index, fingerprint),
                        "paired_key": paired_key,
                        "suite": suite,
                        "task_id": task_id,
                        "task_name": "{} task {}".format(suite, task_id),
                        "init_state_index": trial_index,
                        "init_state_fingerprint": fingerprint,
                        "eval_seed": 42,
                        "execution_mode": mode,
                        "status": "success" if trial_index < success_cutoff else "timeout_failure",
                        "success": trial_index < success_cutoff,
                        "include_in_success_rate": True,
                        "steps": 20,
                        "episode_duration_ns": 1_000_000_000 + underflow,
                        "inference_requests": [
                            {
                                "request_id": 0,
                                "latency_sample_key": sample_key.to_dict(),
                                "sampled_target_latency_ns": sampled_target,
                            }
                        ],
                        "inference_latencies": [
                            {
                                "raw_inference_latency_ns": raw,
                                "sampled_target_latency_ns": sampled_target,
                                "requested_synthetic_delay_ns": sampled_target - raw,
                                "observed_synthetic_delay_ns": sampled_target - raw,
                                "observed_effective_latency_ns": sampled_target,
                                "latency_overshoot_ns": 0,
                                "duration_ns": sampled_target,
                            }
                        ],
                        "control_stalls": ([{"duration_ns": underflow}] if underflow else []),
                        "action_underflows": ([{"duration_ns": underflow}] if underflow else []),
                        "action_seams": [
                            {
                                "arm_l2_jump": 0.2 if mode == "baseline_async" else 0.1,
                                "arm_max_abs_jump": 0.1,
                                "gripper_abs_jump": 0.05,
                            }
                        ],
                    }
                )
    return records


def _runs() -> dict[str, report.RunDataV5]:
    return {
        mode: report.RunDataV5(
            path=Path("/runs") / mode,
            manifest=_manifest(mode),
            records=_records(mode),
            summary={},
            video_audits=(),
            file_sha256={"manifest.json": _sha(mode)},
        )
        for mode in MODES
    }


def test_classifier_accepts_exactly_the_three_schema_v5_modes():
    classified = report.classify_checkpoint_manifests_v5([_manifest(mode) for mode in reversed(MODES)])
    assert tuple(classified) == MODES
    assert all(manifest["checkpoint_step"] == 10000 for manifest in classified.values())


@pytest.mark.parametrize(
    "manifests, message",
    [
        ([_manifest(mode) for mode in MODES[:-1]], "exactly three"),
        ([_manifest("baseline_async")] * 3, "duplicate execution mode"),
    ],
)
def test_classifier_rejects_missing_or_duplicate_modes(manifests, message):
    with pytest.raises(report.ComparisonErrorV5, match=message):
        report.classify_checkpoint_manifests_v5(manifests)


def test_classifier_rejects_cross_mode_latency_distribution_mismatch():
    manifests = [_manifest(mode) for mode in MODES]
    manifests[2]["latency_distribution"]["stddev_ns"] = 61_000_000
    with pytest.raises(report.ComparisonErrorV5, match="manifest"):
        report.classify_checkpoint_manifests_v5(manifests)


def test_classifier_rejects_baseline_checkpoint_identity_mismatch():
    manifests = [_manifest(mode) for mode in MODES]
    manifests[1] = _manifest("baseline_rtc", identity_suffix="-wrong")
    with pytest.raises(report.ComparisonErrorV5, match="Baseline family identity"):
        report.classify_checkpoint_manifests_v5(manifests)


def test_compare_reports_three_required_deltas_and_runtime_diagnostics():
    comparison = report.compare_checkpoint_v5(_runs())
    assert comparison["schema_version"] == 5
    assert comparison["success_rates"] == pytest.approx(
        {
            "baseline_async": 0.8,
            "baseline_rtc": 0.9,
            "bsp_spline_async": 0.96,
        }
    )
    assert set(comparison["primary_paired_deltas"]) == {
        "baseline_rtc_minus_baseline_async",
        "bsp_spline_async_minus_baseline_async",
        "bsp_spline_async_minus_baseline_rtc",
    }
    assert comparison["diagnostics"]["baseline_async"]["action_underflow_count"] == 2000
    assert comparison["diagnostics"]["baseline_async"]["action_seam_count"] == 2000
    assert comparison["diagnostics"]["baseline_rtc"]["latency_hidden_ratio"] == 1.0


def test_compare_reports_real_clock_overshoot_separately_from_requested_delay():
    runs = _runs()
    records = copy.deepcopy(list(runs["baseline_async"].records))
    event = records[0]["inference_latencies"][0]
    event["observed_synthetic_delay_ns"] += 500_000
    event["observed_effective_latency_ns"] += 500_000
    event["latency_overshoot_ns"] = 500_000
    event["duration_ns"] += 500_000
    runs["baseline_async"] = report.RunDataV5(
        path=runs["baseline_async"].path,
        manifest=runs["baseline_async"].manifest,
        records=records,
        summary={},
        video_audits=(),
        file_sha256=runs["baseline_async"].file_sha256,
    )

    diagnostics = report.compare_checkpoint_v5(runs)["diagnostics"]["baseline_async"]

    assert diagnostics["latency_overshoot_ns_total"] == 500_000
    assert diagnostics["latency_overshoot_ns_p95"] == 0
    assert diagnostics["requested_synthetic_delay_ns_total"] < diagnostics["observed_synthetic_delay_ns_total"]


def test_compare_rejects_unpaired_request_sample_identity():
    runs = _runs()
    mutated = copy.deepcopy(list(runs["baseline_rtc"].records))
    mutated[0]["inference_requests"][0]["sampled_target_latency_ns"] += 1
    runs["baseline_rtc"] = report.RunDataV5(
        path=runs["baseline_rtc"].path,
        manifest=runs["baseline_rtc"].manifest,
        records=mutated,
        summary={},
        video_audits=(),
        file_sha256=runs["baseline_rtc"].file_sha256,
    )
    with pytest.raises(report.ComparisonErrorV5, match="paired latency sample"):
        report.compare_checkpoint_v5(runs)


def test_writer_emits_only_the_three_frozen_output_files(tmp_path):
    output = tmp_path / "report"
    result = report.write_three_mode_report_v5(list(_runs().values()), output_dir=output)
    assert {path.name for path in output.iterdir()} == set(report.OUTPUT_FILENAMES_V5)
    assert result["schema_version"] == 5
    assert result["protocol"]["execution_modes"] == list(MODES)
    comparison = json.loads((output / "comparison_v5.json").read_text())
    assert comparison == result
    with (output / "task_metrics_v5.csv").open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 40
    assert float(rows[0]["baseline_rtc_minus_baseline_async"]) == pytest.approx(0.1)
    markdown = (output / "report_v5.md").read_text()
    assert "baseline_rtc_minus_baseline_async" in markdown
    assert "bsp_spline_async_minus_baseline_rtc" in markdown


def test_writer_rejects_any_input_count_other_than_three(tmp_path):
    runs = list(_runs().values())
    for values in (runs[:2], runs + runs[:1]):
        with pytest.raises(report.ComparisonErrorV5, match="exactly three"):
            report.write_three_mode_report_v5(values, output_dir=tmp_path / "output")


def test_writer_refuses_to_mix_with_existing_output(tmp_path):
    output = tmp_path / "report"
    output.mkdir()
    (output / "evidence.txt").write_text("preserve")
    with pytest.raises(report.ComparisonErrorV5, match="new or empty"):
        report.write_three_mode_report_v5(list(_runs().values()), output_dir=output)


def test_strict_json_reader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":5,"schema_version":5}')
    with pytest.raises(report.ComparisonErrorV5, match="duplicate JSON key"):
        report._load_strict_json(path)  # noqa: SLF001
