"""Strict, dependency-free comparison for the fixed LIBERO phase-one protocol."""

from __future__ import annotations

import dataclasses
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from openpi_client import libero_artifacts
from openpi_client import libero_eval


MILESTONES = (0, 1000, 2000, 5000, 10000)
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10000
TASKS_PER_SUITE = 10
TRIALS_PER_TASK = 50
EPISODES_PER_RUN = len(libero_eval.SUPPORTED_SUITES) * TASKS_PER_SUITE * TRIALS_PER_TASK
TOTAL_EPISODES = EPISODES_PER_RUN * 2 * len(MILESTONES)
_REPLAN_STEPS = libero_eval.BSP_PARAMETERS["executed_actions"]
OUTPUT_FILENAMES = (
    "task_comparison.csv",
    "suite_comparison.csv",
    "learning_curve.csv",
    "comparison.json",
    "report.md",
    "learning_curve.svg",
)
_MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
_BSP_VERIFICATION_FLAGS = (
    "strict_reconstruction_tolerance",
    "ground_truth_knots_nondecreasing",
    "tail_padding_valid",
    "future_segment_mapping_valid",
    "target_index_bounds_valid",
    "no_cross_episode_mapping",
    "all_frames_covered",
    "targets_match_rebuild",
    "mapping_matches_rebuild",
    "cache_contents_deterministic",
)
_STATE_STAT_FIELDS = ("mean", "std", "q01", "q99")
_IDENTITY_FIELDS = (
    "suite",
    "task_id",
    "task_name",
    "init_state_index",
    "init_state_fingerprint",
    "eval_seed",
)
_SHARED_MANIFEST_FIELDS = (
    "code_sha",
    "dataset_revision",
    "container_digest",
    "train_seed",
    "eval_seed",
    "suites",
    "task_ids",
    "trials_per_task",
    "num_steps_wait",
    "max_steps_by_suite",
    "connection_timeout_s",
    "inference_timeout_s",
)
_PHASE_ONE_CONFIGS = {
    ("baseline", "full"): "pi05_libero_baseline_h16",
    ("bsp", "full"): "pi05_libero_bsp_h16",
    ("baseline", "lora"): "pi05_libero_baseline_lora_h16",
    ("bsp", "lora"): "pi05_libero_bsp_lora_h16",
}


class ComparisonError(ValueError):
    """An input is not an auditable phase-one A/B result."""


@dataclasses.dataclass(frozen=True)
class RunData:
    path: Path
    manifest: Mapping[str, Any]
    records: Sequence[Mapping[str, Any]]
    summary: Mapping[str, Any]
    file_sha256: Mapping[str, str]

    @property
    def variant(self) -> str:
        return str(self.manifest["policy_variant"])

    @property
    def checkpoint_step(self) -> int:
        return int(self.manifest["checkpoint_step"])


def _reject_constant(value: str) -> None:
    raise ComparisonError("JSON contains a non-standard numeric constant: {}".format(value))


def load_strict_json(path: Path) -> Mapping[str, Any]:
    """Load one RFC-compatible JSON object and reject NaN/Infinity/truncation."""
    try:
        with Path(path).open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file, parse_constant=_reject_constant)
    except ComparisonError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ComparisonError("Invalid JSON file {}: {}".format(path, error)) from error
    if not isinstance(payload, dict):
        raise ComparisonError("JSON file {} must contain one object".format(path))
    return payload


def load_strict_jsonl(path: Path) -> List[Mapping[str, Any]]:
    """Load strict non-empty JSON objects from every JSONL line."""
    records: List[Mapping[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise ComparisonError("Blank JSONL line at {}:{}".format(path, line_number))
                try:
                    payload = json.loads(line, parse_constant=_reject_constant)
                except ComparisonError:
                    raise
                except json.JSONDecodeError as error:
                    raise ComparisonError(
                        "Invalid JSONL record at {}:{}: {}".format(path, line_number, error)
                    ) from error
                if not isinstance(payload, dict):
                    raise ComparisonError("JSONL record at {}:{} must be an object".format(path, line_number))
                records.append(payload)
    except ComparisonError:
        raise
    except (OSError, UnicodeError) as error:
        raise ComparisonError("Unable to read JSONL file {}: {}".format(path, error)) from error
    if not records:
        raise ComparisonError("JSONL file {} is empty".format(path))
    return records


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with Path(path).open("rb") as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as error:
        raise ComparisonError("Unable to hash input file {}: {}".format(path, error)) from error
    return hasher.hexdigest()


def _require_fields(payload: Mapping[str, Any], fields: Sequence[str], *, label: str) -> None:
    missing = sorted(field for field in fields if field not in payload)
    if missing:
        raise ComparisonError("{} is missing required fields: {}".format(label, missing))


def _training_family(manifest: Mapping[str, Any]) -> str:
    variant = manifest["policy_variant"]
    config_name = manifest["config_name"]
    matches = [
        family
        for (candidate_variant, family), candidate_config in _PHASE_ONE_CONFIGS.items()
        if candidate_variant == variant and candidate_config == config_name
    ]
    if len(matches) != 1:
        raise ComparisonError("{} manifest has unsupported phase-one config_name {}".format(variant, config_name))
    return matches[0]


def _validate_manifest(manifest: Mapping[str, Any]) -> Tuple[str, int]:
    required = (
        "schema_version",
        "native_control_hz",
        "replan_steps",
        "code_sha",
        "dataset_revision",
        "config_name",
        "checkpoint_step",
        "bsp_cache_hash",
        "bsp_cache_manifest_fingerprint",
        "norm_hash",
        "checkpoint",
        "container_digest",
        "train_seed",
        "eval_seed",
        "policy_variant",
        "bsp_parameters",
        "policy_protocol",
        "expected_action_horizon",
        "execution_horizon",
        "suites",
        "task_ids",
        "trials_per_task",
        "num_steps_wait",
        "max_steps_by_suite",
        "connection_timeout_s",
        "inference_timeout_s",
        "infrastructure_retries",
    )
    _require_fields(manifest, required, label="evaluation manifest")
    if manifest["schema_version"] != 2:
        raise ComparisonError("Phase-one comparison requires evaluation manifest schema_version 2")
    if manifest["native_control_hz"] != 10 or manifest["replan_steps"] != _REPLAN_STEPS:
        raise ComparisonError("Evaluation manifest has an incompatible native-control protocol")
    variant = manifest["policy_variant"]
    step = manifest["checkpoint_step"]
    if variant not in ("baseline", "bsp"):
        raise ComparisonError("Evaluation manifest policy_variant must be baseline or bsp")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ComparisonError("Evaluation manifest checkpoint_step must be an integer")
    if step not in MILESTONES:
        raise ComparisonError("Unexpected phase-one checkpoint_step {}".format(step))
    _training_family(manifest)
    expected = {
        "baseline": ("baseline_h16", 16),
        "bsp": ("bsp_decoded_h8", _REPLAN_STEPS),
    }[variant]
    actual = (manifest["policy_protocol"], manifest["expected_action_horizon"])
    if actual != expected:
        raise ComparisonError(
            "{}@{} has config/protocol/horizon {}, expected {}".format(variant, step, actual, expected)
        )
    if manifest["execution_horizon"] != _REPLAN_STEPS:
        raise ComparisonError("Phase-one evaluation must execute exactly eight actions per replan")
    if tuple(manifest["suites"]) != tuple(libero_eval.SUPPORTED_SUITES):
        raise ComparisonError("Phase-one evaluation requires all four suites in canonical order")
    if tuple(manifest["task_ids"]) != tuple(range(TASKS_PER_SUITE)):
        raise ComparisonError("Phase-one evaluation requires task ids 0..9")
    if manifest["trials_per_task"] != TRIALS_PER_TASK:
        raise ComparisonError("Phase-one evaluation requires exactly 50 trials per task")
    if manifest["num_steps_wait"] != 10 or manifest["max_steps_by_suite"] != _MAX_STEPS_BY_SUITE:
        raise ComparisonError("Evaluation wait/max-step protocol does not match phase one")
    if manifest["infrastructure_retries"] != 2:
        raise ComparisonError("Evaluation must retain exactly two infrastructure retries")
    if manifest["train_seed"] != 42 or manifest["eval_seed"] != 42:
        raise ComparisonError("Phase-one training and evaluation seeds must both be 42")
    for field in ("code_sha", "dataset_revision", "checkpoint", "container_digest"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ComparisonError("Evaluation manifest {} must be non-empty".format(field))
    if manifest["dataset_revision"] != "v2.0":
        raise ComparisonError("Phase-one evaluation requires dataset revision v2.0")
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", manifest["code_sha"]) is None:
        raise ComparisonError("Evaluation manifest code_sha must be lowercase 40- or 64-character hex")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", manifest["container_digest"]) is None:
        raise ComparisonError("Evaluation manifest container_digest must be sha256: followed by 64 lowercase hex")
    for field in ("connection_timeout_s", "inference_timeout_s"):
        _finite_number(manifest[field], label=field, nonnegative=True, strictly_positive=True)
    normalized_checkpoint = manifest["checkpoint"].rstrip("/")
    if not normalized_checkpoint or normalized_checkpoint.rsplit("/", 1)[-1] != str(step):
        raise ComparisonError("Evaluation checkpoint terminal component must equal checkpoint_step")
    if not libero_artifacts.is_sha256(manifest["norm_hash"]):
        raise ComparisonError("Evaluation manifest norm_hash must be a lowercase SHA256")
    if manifest["bsp_parameters"] != libero_eval.BSP_PARAMETERS:
        raise ComparisonError("Evaluation manifest BSP parameters do not match the fixed protocol")
    if variant == "baseline":
        if manifest["bsp_cache_hash"] is not None or manifest["bsp_cache_manifest_fingerprint"] is not None:
            raise ComparisonError("Baseline manifests must record null BSP cache identities")
    else:
        if not libero_artifacts.is_sha256(manifest["bsp_cache_hash"]):
            raise ComparisonError("BSP cache hash must be the actual NPZ lowercase SHA256")
        if not libero_artifacts.is_sha256(manifest["bsp_cache_manifest_fingerprint"]):
            raise ComparisonError("BSP cache manifest fingerprint must be a lowercase SHA256")
    return str(variant), int(step)


def classify_phase_one_manifests(
    manifests: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    """Identify baseline/BSP at all five fixed milestones using manifest contents only."""
    required_run_count = 2 * len(MILESTONES)
    if len(manifests) != required_run_count:
        raise ComparisonError("Phase-one comparison requires exactly {} run manifests".format(required_run_count))
    classified: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for manifest in manifests:
        key = _validate_manifest(manifest)
        if key in classified:
            raise ComparisonError("Duplicate phase-one run identity {}@{}".format(*key))
        classified[key] = manifest
    expected = {(variant, step) for variant in ("baseline", "bsp") for step in MILESTONES}
    if set(classified) != expected:
        missing = sorted(expected.difference(classified))
        extra = sorted(set(classified).difference(expected))
        raise ComparisonError("Phase-one run set mismatch; missing={}, extra={}".format(missing, extra))

    training_families = {_training_family(manifest) for manifest in manifests}
    if len(training_families) != 1:
        raise ComparisonError(
            "All phase-one runs must use one training family; found {}".format(sorted(training_families))
        )

    normalized_checkpoints = [manifest["checkpoint"].rstrip("/") for manifest in manifests]
    if len(set(normalized_checkpoints)) != len(normalized_checkpoints):
        raise ComparisonError("All normalized checkpoint identities must be globally unique")

    reference = next(iter(classified.values()))
    for key, manifest in classified.items():
        mismatched = [field for field in _SHARED_MANIFEST_FIELDS if manifest[field] != reference[field]]
        if mismatched:
            raise ComparisonError("Run {}@{} has mismatched shared identities: {}".format(*key, mismatched))
    for variant in ("baseline", "bsp"):
        variant_manifests = [classified[(variant, step)] for step in MILESTONES]
        norm_hashes = {manifest["norm_hash"] for manifest in variant_manifests}
        if len(norm_hashes) != 1:
            raise ComparisonError("{} norm_hash must remain identical across milestones".format(variant))
    bsp_manifests = [classified[("bsp", step)] for step in MILESTONES]
    for field in ("bsp_cache_hash", "bsp_cache_manifest_fingerprint"):
        if len({manifest[field] for manifest in bsp_manifests}) != 1:
            raise ComparisonError("BSP {} must remain identical across milestones".format(field))
    return classified


def _finite_number(value: Any, *, label: str, nonnegative: bool = False, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError("{} must be numeric".format(label))
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0) or (strictly_positive and number <= 0.0):
        qualifier = " and positive" if strictly_positive else " and non-negative" if nonnegative else ""
        raise ComparisonError("{} must be finite{}".format(label, qualifier))
    return number


def _validate_episode(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    required = (
        "episode_id",
        "paired_key",
        "suite",
        "task_id",
        "task_name",
        "init_state_index",
        "init_state_fingerprint",
        "eval_seed",
        "status",
        "success",
        "include_in_success_rate",
        "attempts",
        "failure_kind",
        "infrastructure_kind",
        "error",
        "steps",
        "replans",
        "inference_ms",
        "mean_inference_ms",
        "infrastructure_history",
    )
    _require_fields(record, required, label="episode record")
    if record["suite"] not in libero_eval.SUPPORTED_SUITES:
        raise ComparisonError("Episode has unsupported suite")
    for field, upper in (("task_id", 10), ("init_state_index", 50)):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= upper:
            raise ComparisonError("Episode {} is outside the fixed phase-one range".format(field))
    for field in ("episode_id", "paired_key", "task_name", "init_state_fingerprint"):
        if not isinstance(record[field], str) or not record[field]:
            raise ComparisonError("Episode {} must be non-empty".format(field))
    if not libero_artifacts.is_sha256(record["init_state_fingerprint"]):
        raise ComparisonError("Episode init_state_fingerprint must be a lowercase SHA256")
    try:
        canonical_identity = libero_eval.EpisodeIdentity(
            suite=record["suite"],
            task_id=record["task_id"],
            task_name=record["task_name"],
            init_state_index=record["init_state_index"],
            init_state_fingerprint=record["init_state_fingerprint"],
        )
    except (TypeError, ValueError) as error:
        raise ComparisonError("Episode identity is malformed") from error
    if record["paired_key"] != canonical_identity.paired_key:
        raise ComparisonError("Episode paired_key is not canonical for its identity")
    if record["episode_id"] != canonical_identity.episode_id:
        raise ComparisonError("Episode episode_id is not canonical for its identity")
    if record["eval_seed"] != manifest["eval_seed"]:
        raise ComparisonError("Episode eval_seed does not match its manifest")
    if record["include_in_success_rate"] is not True:
        raise ComparisonError("Every phase-one episode must be eligible; infrastructure run is incomplete")
    if not isinstance(record["success"], bool):
        raise ComparisonError("Every phase-one episode success value must be boolean")
    if record["success"]:
        if record["status"] != "success" or record["failure_kind"] is not None or record["error"] is not None:
            raise ComparisonError("Successful episode has inconsistent status/failure/error fields")
    else:
        failure_kind = record["failure_kind"]
        if failure_kind not in ("policy", "timeout"):
            raise ComparisonError("Failed episode must have policy or timeout failure_kind")
        if record["status"] != "{}_failure".format(failure_kind):
            raise ComparisonError("Failed episode status does not match failure_kind")
        if not isinstance(record["error"], str) or not record["error"].strip():
            raise ComparisonError("Failed episode must record a non-empty error")
    if record["infrastructure_kind"] is not None:
        raise ComparisonError("Eligible episode cannot retain an infrastructure failure kind")
    for field in ("attempts", "steps", "replans"):
        value = record[field]
        minimum = 1 if field == "attempts" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ComparisonError("Episode {} must be an integer >= {}".format(field, minimum))
    if record["attempts"] > 3:
        raise ComparisonError("Episode attempts cannot exceed the three-attempt retry protocol")
    max_steps = _MAX_STEPS_BY_SUITE[record["suite"]]
    if record["steps"] > max_steps:
        raise ComparisonError("Episode steps exceed the suite maximum")
    if record["replans"] > math.ceil(max_steps / _REPLAN_STEPS):
        raise ComparisonError("Episode replans exceed the suite maximum at an eight-step horizon")
    timings = record["inference_ms"]
    if not isinstance(timings, list):
        raise ComparisonError("Episode inference_ms must be a list")
    finite_timings = [_finite_number(timing, label="inference_ms", nonnegative=True) for timing in timings]
    if len(finite_timings) != record["replans"]:
        raise ComparisonError("Episode inference_ms length must equal replans")
    if finite_timings:
        recorded_mean = _finite_number(record["mean_inference_ms"], label="mean_inference_ms", nonnegative=True)
        expected_mean = sum(finite_timings) / len(finite_timings)
        if not math.isclose(recorded_mean, expected_mean, rel_tol=1e-12, abs_tol=1e-12):
            raise ComparisonError("Episode mean_inference_ms is inconsistent with inference_ms")
    elif record["mean_inference_ms"] is not None:
        raise ComparisonError("Episode mean_inference_ms must be null when inference_ms is empty")
    history = record["infrastructure_history"]
    if not isinstance(history, list):
        raise ComparisonError("Episode infrastructure_history must be a list")
    if len(history) != record["attempts"] - 1:
        raise ComparisonError("Episode infrastructure_history length must equal attempts - 1")
    for expected_attempt, entry in enumerate(history, start=1):
        if not isinstance(entry, dict):
            raise ComparisonError("Episode infrastructure history entries must be objects")
        _require_fields(entry, ("attempt", "kind", "error"), label="infrastructure history entry")
        if entry["attempt"] != expected_attempt:
            raise ComparisonError("Episode infrastructure history attempts must be consecutive")
        if entry["kind"] not in ("simulator", "container", "network"):
            raise ComparisonError("Episode infrastructure history has an invalid kind")
        if not isinstance(entry["error"], str) or not entry["error"].strip():
            raise ComparisonError("Episode infrastructure history error must be non-empty")


def _derive_summary(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault((str(record["suite"]), int(record["task_id"])), []).append(record)
    tasks: List[Mapping[str, Any]] = []
    for (suite, task_id), group in sorted(groups.items()):
        successes = sum(record["success"] is True for record in group)
        tasks.append(
            {
                "suite": suite,
                "task_id": task_id,
                "task_name": group[0]["task_name"],
                "requested_episodes": len(group),
                "eligible_episodes": len(group),
                "successes": successes,
                "failures": len(group) - successes,
                "incomplete_infrastructure_count": 0,
                "success_rate": successes / len(group),
            }
        )
    suites: List[Mapping[str, Any]] = []
    for suite in sorted({str(record["suite"]) for record in records}):
        suite_records = [record for record in records if record["suite"] == suite]
        suite_tasks = [row for row in tasks if row["suite"] == suite]
        successes = sum(record["success"] is True for record in suite_records)
        suites.append(
            {
                "suite": suite,
                "tasks": len(suite_tasks),
                "requested_episodes": len(suite_records),
                "eligible_episodes": len(suite_records),
                "successes": successes,
                "failures": len(suite_records) - successes,
                "incomplete_infrastructure_count": 0,
                "success_rate": successes / len(suite_records),
                "task_macro_success_rate": sum(row["success_rate"] for row in suite_tasks) / len(suite_tasks),
            }
        )
    suite_rates = [row["success_rate"] for row in suites]
    suite_macro = sum(suite_rates) / len(suite_rates)
    return {
        "tasks": tasks,
        "suites": suites,
        "suite_macro_success_rate": suite_macro,
        "four_suite_macro_success_rate": suite_macro,
        "evaluated_suite_count": len(suites),
        "all_four_suites_evaluated": True,
        "requested_episodes": len(records),
        "eligible_episodes": len(records),
        "successes": sum(record["success"] is True for record in records),
        "incomplete_infrastructure_count": 0,
        "artifact_error_count": 0,
        "acceptance_complete": True,
    }


def _validate_complete_run(run: RunData) -> None:
    if len(run.records) != EPISODES_PER_RUN:
        raise ComparisonError(
            "{} has {} episodes; expected exactly {}".format(run.path, len(run.records), EPISODES_PER_RUN)
        )
    seen = set()
    groups: Dict[Tuple[str, int], List[int]] = {}
    task_names: Dict[Tuple[str, int], str] = {}
    for record in run.records:
        _validate_episode(record, run.manifest)
        paired_key = record["paired_key"]
        if paired_key in seen:
            raise ComparisonError("Duplicate paired_key in {}: {}".format(run.path, paired_key))
        seen.add(paired_key)
        group_key = (str(record["suite"]), int(record["task_id"]))
        groups.setdefault(group_key, []).append(int(record["init_state_index"]))
        previous_name = task_names.setdefault(group_key, str(record["task_name"]))
        if record["task_name"] != previous_name:
            raise ComparisonError("Task name changes within {}".format(group_key))
    expected_groups = {(suite, task_id) for suite in libero_eval.SUPPORTED_SUITES for task_id in range(10)}
    if set(groups) != expected_groups:
        raise ComparisonError("Run does not contain the exact four-suite/task grid")
    for group_key, init_indices in groups.items():
        if sorted(init_indices) != list(range(TRIALS_PER_TASK)):
            raise ComparisonError("{} does not contain init states 0..49 exactly once".format(group_key))
    derived = _derive_summary(run.records)
    if run.summary != derived:
        raise ComparisonError("summary.json is inconsistent with derived episode results for {}".format(run.path))


def load_run(run_dir: Path) -> RunData:
    """Load and fully validate one immutable phase-one run directory."""
    root = Path(run_dir).expanduser().resolve()
    required_paths = {name: root / name for name in ("manifest.json", "episodes.jsonl", "summary.json")}
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise ComparisonError("Run {} is missing required artifacts: {}".format(root, missing))
    artifact_errors_path = root / "artifact_errors.jsonl"
    file_hashes = {name: _file_sha256(path) for name, path in required_paths.items()}
    if artifact_errors_path.exists():
        if not artifact_errors_path.is_file():
            raise ComparisonError("artifact_errors.jsonl is not a file in {}".format(root))
        file_hashes["artifact_errors.jsonl"] = _file_sha256(artifact_errors_path)
        try:
            artifact_contents = artifact_errors_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ComparisonError("Unable to read artifact errors in {}: {}".format(root, error)) from error
        if artifact_contents.strip():
            raise ComparisonError("Run {} contains artifact errors and is acceptance-incomplete".format(root))
    manifest = load_strict_json(required_paths["manifest.json"])
    _validate_manifest(manifest)
    records = load_strict_jsonl(required_paths["episodes.jsonl"])
    summary = load_strict_json(required_paths["summary.json"])
    run = RunData(root, manifest, records, summary, file_hashes)
    _validate_complete_run(run)
    return run


def _records_by_key(records: Sequence[Mapping[str, Any]], *, label: str) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        if "paired_key" not in record or not isinstance(record["paired_key"], str):
            raise ComparisonError("{} record is missing a string paired_key".format(label))
        key = record["paired_key"]
        if key in indexed:
            raise ComparisonError("{} contains duplicate paired_key {}".format(label, key))
        indexed[key] = record
    return indexed


def validate_paired_records(
    baseline_records: Sequence[Mapping[str, Any]], bsp_records: Sequence[Mapping[str, Any]]
) -> None:
    """Require exact paired identities, task names, state fingerprints, and eval seeds."""
    baseline = _records_by_key(baseline_records, label="baseline")
    bsp = _records_by_key(bsp_records, label="bsp")
    if set(baseline) != set(bsp):
        missing = sorted(set(baseline).difference(bsp))[:5]
        extra = sorted(set(bsp).difference(baseline))[:5]
        raise ComparisonError("Paired rollout keys mismatch; missing={}, extra={}".format(missing, extra))
    for key in baseline:
        mismatched = [field for field in _IDENTITY_FIELDS if baseline[key].get(field) != bsp[key].get(field)]
        if mismatched:
            raise ComparisonError("Paired identity {} differs in fields {}".format(key, mismatched))


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ComparisonError("Cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def hierarchical_delta(paired_task_deltas: Mapping[Tuple[str, int], Sequence[float]]) -> float:
    """Average paired deltas as rollout -> task -> suite -> four-suite macro."""
    expected = {(suite, task_id) for suite in libero_eval.SUPPORTED_SUITES for task_id in range(10)}
    if set(paired_task_deltas) != expected:
        raise ComparisonError("Hierarchical delta requires exactly 40 LIBERO task strata")
    suite_means = []
    for suite in libero_eval.SUPPORTED_SUITES:
        task_means = []
        for task_id in range(10):
            values = list(paired_task_deltas[(suite, task_id)])
            if not values:
                raise ComparisonError("Paired task delta stratum is empty")
            task_means.append(sum(values) / len(values))
        suite_means.append(sum(task_means) / len(task_means))
    return sum(suite_means) / len(suite_means)


def stratified_paired_bootstrap(
    paired_task_deltas: Mapping[Tuple[str, int], Sequence[float]],
) -> Tuple[float, float]:
    """Return the fixed 10,000-resample, seed-42 hierarchical percentile interval."""
    hierarchical_delta(paired_task_deltas)
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    ordered_keys = [(suite, task_id) for suite in libero_eval.SUPPORTED_SUITES for task_id in range(10)]
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_task_deltas = {}
        for key in ordered_keys:
            values = list(paired_task_deltas[key])
            sampled_task_deltas[key] = rng.choices(values, k=len(values))
        samples.append(hierarchical_delta(sampled_task_deltas))
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def _rollout_diagnostics(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    steps = [float(record["steps"]) for record in records]
    successful_steps = [float(record["steps"]) for record in records if record["success"]]
    timings = [float(value) for record in records for value in record["inference_ms"]]
    return {
        "eligible_rollout_steps_mean": _mean(steps),
        "eligible_rollout_steps_median": statistics.median(steps),
        "successful_rollout_steps_mean": _mean(successful_steps),
        "successful_rollout_steps_median": statistics.median(successful_steps) if successful_steps else None,
        "inference_ms_mean": _mean(timings),
        "inference_ms_p50": _percentile(timings, 0.50) if timings else None,
        "inference_ms_p95": _percentile(timings, 0.95) if timings else None,
    }


def _paired_task_deltas(
    baseline: Sequence[Mapping[str, Any]], bsp: Sequence[Mapping[str, Any]]
) -> Mapping[Tuple[str, int], Sequence[float]]:
    baseline_by_key = _records_by_key(baseline, label="baseline")
    bsp_by_key = _records_by_key(bsp, label="bsp")
    paired: Dict[Tuple[str, int], List[float]] = {
        (suite, task_id): [] for suite in libero_eval.SUPPORTED_SUITES for task_id in range(10)
    }
    for key in sorted(baseline_by_key):
        base = baseline_by_key[key]
        other = bsp_by_key[key]
        paired[(str(base["suite"]), int(base["task_id"]))].append(
            float(bool(other["success"])) - float(bool(base["success"]))
        )
    return paired


def _metric_rows(
    step: int, baseline: Sequence[Mapping[str, Any]], bsp: Sequence[Mapping[str, Any]]
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], float, float, float, float]:
    baseline_by_key = _records_by_key(baseline, label="baseline")
    bsp_by_key = _records_by_key(bsp, label="bsp")
    task_rows: List[Mapping[str, Any]] = []
    for suite in libero_eval.SUPPORTED_SUITES:
        for task_id in range(10):
            keys = sorted(
                key
                for key, record in baseline_by_key.items()
                if record["suite"] == suite and record["task_id"] == task_id
            )
            baseline_successes = sum(baseline_by_key[key]["success"] is True for key in keys)
            bsp_successes = sum(bsp_by_key[key]["success"] is True for key in keys)
            task_rows.append(
                {
                    "checkpoint_step": step,
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": baseline_by_key[keys[0]]["task_name"],
                    "paired_rollouts": len(keys),
                    "baseline_successes": baseline_successes,
                    "baseline_success_rate": baseline_successes / len(keys),
                    "bsp_successes": bsp_successes,
                    "bsp_success_rate": bsp_successes / len(keys),
                    "bsp_minus_baseline": (bsp_successes - baseline_successes) / len(keys),
                }
            )
    suite_rows: List[Mapping[str, Any]] = []
    for suite in libero_eval.SUPPORTED_SUITES:
        rows = [row for row in task_rows if row["suite"] == suite]
        baseline_rate = sum(row["baseline_success_rate"] for row in rows) / len(rows)
        bsp_rate = sum(row["bsp_success_rate"] for row in rows) / len(rows)
        suite_rows.append(
            {
                "checkpoint_step": step,
                "suite": suite,
                "tasks": len(rows),
                "paired_rollouts": sum(row["paired_rollouts"] for row in rows),
                "baseline_task_macro_success_rate": baseline_rate,
                "bsp_task_macro_success_rate": bsp_rate,
                "bsp_minus_baseline": bsp_rate - baseline_rate,
            }
        )
    baseline_global = sum(row["baseline_task_macro_success_rate"] for row in suite_rows) / len(suite_rows)
    bsp_global = sum(row["bsp_task_macro_success_rate"] for row in suite_rows) / len(suite_rows)
    paired = _paired_task_deltas(baseline, bsp)
    observed = hierarchical_delta(paired)
    ci_low, ci_high = stratified_paired_bootstrap(paired)
    if not math.isclose(observed, bsp_global - baseline_global, rel_tol=0.0, abs_tol=1e-12):
        raise ComparisonError("Paired and hierarchical observed deltas disagree")
    return task_rows, suite_rows, baseline_global, bsp_global, ci_low, ci_high


def validate_diagnostics(
    manifests: Sequence[Mapping[str, Any]],
    bsp_diagnostics: Mapping[str, Any],
    norm_diagnostics: Mapping[str, Any],
) -> Mapping[str, float]:
    """Validate cache and normalization gates against all fixed manifests."""
    classified = classify_phase_one_manifests(manifests)
    required_bsp = (
        *_BSP_VERIFICATION_FLAGS,
        "verification_passed",
        "strict_comparison",
        "scipy_version",
        "required_scipy_version",
        "scipy_version_matches_required",
        "episode_count",
        "frame_count",
        "strict_max_reconstruction_error",
        "mean_reconstruction_error",
        "p95_reconstruction_error",
        "max_error_threshold",
        "cache_sha256",
        "cache_manifest_fingerprint",
        "cache_contents_sha256",
        "rebuilt_contents_sha256",
        "code_sha",
    )
    _require_fields(bsp_diagnostics, required_bsp, label="BSP verification diagnostics")
    if bsp_diagnostics["verification_passed"] is not True or bsp_diagnostics["strict_comparison"] is not True:
        raise ComparisonError("BSP verification and strict-comparison gates must pass")
    failed_flags = [flag for flag in _BSP_VERIFICATION_FLAGS if bsp_diagnostics[flag] is not True]
    if failed_flags:
        raise ComparisonError("BSP verification flags did not pass: {}".format(failed_flags))
    if (
        bsp_diagnostics["scipy_version"] != "1.15.3"
        or bsp_diagnostics["required_scipy_version"] != "1.15.3"
        or bsp_diagnostics["scipy_version_matches_required"] is not True
    ):
        raise ComparisonError("BSP cache must be verified with SciPy 1.15.3")
    if bsp_diagnostics["episode_count"] != 1693 or bsp_diagnostics["frame_count"] != 273465:
        raise ComparisonError("BSP verification metadata must be exactly 1,693 episodes / 273,465 frames")
    maximum = _finite_number(
        bsp_diagnostics["strict_max_reconstruction_error"],
        label="strict max reconstruction error",
        nonnegative=True,
    )
    mean = _finite_number(
        bsp_diagnostics["mean_reconstruction_error"], label="mean reconstruction error", nonnegative=True
    )
    p95 = _finite_number(
        bsp_diagnostics["p95_reconstruction_error"], label="p95 reconstruction error", nonnegative=True
    )
    threshold = _finite_number(bsp_diagnostics["max_error_threshold"], label="max error threshold", nonnegative=True)
    if threshold != 0.002 or not maximum < 0.002 or mean > maximum or p95 > maximum:
        raise ComparisonError("BSP reconstruction requires the strict maximum error < 0.002")
    contents_sha = bsp_diagnostics["cache_contents_sha256"]
    rebuilt_sha = bsp_diagnostics["rebuilt_contents_sha256"]
    if (
        not libero_artifacts.is_sha256(contents_sha)
        or not libero_artifacts.is_sha256(rebuilt_sha)
        or contents_sha != rebuilt_sha
    ):
        raise ComparisonError("BSP cache and rebuilt canonical content SHA256 values must match")
    if any(bsp_diagnostics["code_sha"] != manifest["code_sha"] for manifest in classified.values()):
        raise ComparisonError("BSP diagnostics code SHA does not match all evaluation manifests")
    for step in MILESTONES:
        manifest = classified[("bsp", step)]
        if bsp_diagnostics["cache_sha256"] != manifest["bsp_cache_hash"]:
            raise ComparisonError("BSP diagnostics NPZ SHA256 does not match evaluation manifests")
        if bsp_diagnostics["cache_manifest_fingerprint"] != manifest["bsp_cache_manifest_fingerprint"]:
            raise ComparisonError("BSP diagnostics fingerprint does not match evaluation manifests")

    required_norm = (
        "state_stats_equal",
        "asset_directories_isolated",
        "action_stats_isolated",
        "baseline_norm_stats_sha256",
        "bsp_norm_stats_sha256",
        "baseline_action_stats_sha256",
        "bsp_action_stats_sha256",
        "baseline_asset_dir",
        "bsp_asset_dir",
        "state_fields",
        "rtol",
        "atol",
    )
    _require_fields(norm_diagnostics, required_norm, label="normalization diagnostics")
    for flag in ("state_stats_equal", "asset_directories_isolated", "action_stats_isolated"):
        if norm_diagnostics[flag] is not True:
            raise ComparisonError("Normalization gate {} must be true".format(flag))
    state_fields = norm_diagnostics["state_fields"]
    if not isinstance(state_fields, dict) or set(state_fields) != set(_STATE_STAT_FIELDS):
        raise ComparisonError("Normalization state_fields must be exactly mean/std/q01/q99")
    for field in _STATE_STAT_FIELDS:
        result = state_fields[field]
        if not isinstance(result, dict) or result.get("equal") is not True:
            raise ComparisonError("Normalization state field {} must be equal".format(field))
    baseline_action_sha = norm_diagnostics["baseline_action_stats_sha256"]
    bsp_action_sha = norm_diagnostics["bsp_action_stats_sha256"]
    if (
        not libero_artifacts.is_sha256(baseline_action_sha)
        or not libero_artifacts.is_sha256(bsp_action_sha)
        or baseline_action_sha == bsp_action_sha
    ):
        raise ComparisonError("Baseline/BSP action-stat SHA256 values must be valid and distinct")
    baseline_asset_dir = norm_diagnostics["baseline_asset_dir"]
    bsp_asset_dir = norm_diagnostics["bsp_asset_dir"]
    if (
        not isinstance(baseline_asset_dir, str)
        or not baseline_asset_dir.strip()
        or not isinstance(bsp_asset_dir, str)
        or not bsp_asset_dir.strip()
        or Path(baseline_asset_dir) == Path(bsp_asset_dir)
    ):
        raise ComparisonError("Baseline/BSP normalization asset directories must be non-empty and distinct")
    if norm_diagnostics["rtol"] != 1e-7 or norm_diagnostics["atol"] != 1e-8:
        raise ComparisonError("Normalization comparison must use rtol=1e-7 and atol=1e-8")
    for step in MILESTONES:
        if norm_diagnostics["baseline_norm_stats_sha256"] != classified[("baseline", step)]["norm_hash"]:
            raise ComparisonError("Baseline norm file SHA256 does not match evaluation manifests")
        if norm_diagnostics["bsp_norm_stats_sha256"] != classified[("bsp", step)]["norm_hash"]:
            raise ComparisonError("BSP norm file SHA256 does not match evaluation manifests")
    return {
        "strict_max_reconstruction_error": maximum,
        "mean_reconstruction_error": mean,
        "p95_reconstruction_error": p95,
    }


def _render_markdown(milestones: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# π0.5 + LIBERO BSP 第一阶段固定里程碑比较",
        "",
        ("该报告只比较 0k、1k、2k、5k、10k 五个预先固定的 checkpoint；主指标为四套件分层宏平均成功率。"),
        "",
        "| optimizer step | baseline | BSP | BSP-baseline | paired bootstrap 95% CI |",
        "|---:|---:|---:|---:|:---|",
    ]
    for row in milestones:
        lines.append(
            "| {checkpoint_step} | {baseline_four_suite_macro_success_rate:.6f} | "
            "{bsp_four_suite_macro_success_rate:.6f} | {bsp_minus_baseline:.6f} | "
            "[{low:.6f}, {high:.6f}] |".format(low=row["bootstrap_95_ci"][0], high=row["bootstrap_95_ci"][1], **row)
        )
    lines.extend(
        [
            "",
            (
                "Bootstrap 在每个 task 的 50 个配对初始状态内有放回采样，"
                "再按 task → suite → 四套件宏平均；固定 seed=42、10,000 次。"
            ),
            "",
            "完成步数、推理延迟和 spline 重建误差仅作为诊断指标，不改变成功率口径。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_svg(milestones: Sequence[Mapping[str, Any]]) -> str:
    width, height = 760, 440
    left, right, top, bottom = 80, 30, 35, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    denominator = max(1, len(milestones) - 1)
    x_values = [left + index * plot_width / denominator for index in range(len(milestones))]

    def y(value: float) -> float:
        return top + (1.0 - value) * plot_height

    baseline_points = " ".join(
        "{:.2f},{:.2f}".format(x_values[index], y(row["baseline_four_suite_macro_success_rate"]))
        for index, row in enumerate(milestones)
    )
    bsp_points = " ".join(
        "{:.2f},{:.2f}".format(x_values[index], y(row["bsp_four_suite_macro_success_rate"]))
        for index, row in enumerate(milestones)
    )
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{}" y="22" font-family="sans-serif" font-size="16">LIBERO phase-one fixed milestones</text>'.format(
            left
        ),
    ]
    for tick in range(6):
        value = tick / 5
        y_pos = y(value)
        elements.append(
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#dddddd"/>'.format(left, y_pos, width - right, y_pos)
        )
        elements.append(
            '<text x="{}" y="{:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{:.1f}</text>'.format(
                left - 10, y_pos + 4, value
            )
        )
    elements.extend(
        [
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#222"/>'.format(
                left, top + plot_height, width - right, top + plot_height
            ),
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#222"/>'.format(left, top, left, top + plot_height),
            '<polyline points="{}" fill="none" stroke="#3b82f6" stroke-width="3"/>'.format(
                html.escape(baseline_points)
            ),
            '<polyline points="{}" fill="none" stroke="#ef4444" stroke-width="3"/>'.format(html.escape(bsp_points)),
        ]
    )
    for index, row in enumerate(milestones):
        x_pos = x_values[index]
        elements.append(
            '<text x="{:.2f}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12">{}</text>'.format(
                x_pos, height - 35, row["checkpoint_step"]
            )
        )
        for value, color in (
            (row["baseline_four_suite_macro_success_rate"], "#3b82f6"),
            (row["bsp_four_suite_macro_success_rate"], "#ef4444"),
        ):
            elements.append('<circle cx="{:.2f}" cy="{:.2f}" r="4" fill="{}"/>'.format(x_pos, y(value), color))
    elements.extend(
        [
            '<text x="{}" y="{}" font-family="sans-serif" font-size="12" fill="#3b82f6">baseline</text>'.format(
                width - 190, 20
            ),
            '<text x="{}" y="{}" font-family="sans-serif" font-size="12" fill="#ef4444">BSP</text>'.format(
                width - 100, 20
            ),
            (
                '<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12">optimizer step</text>'
            ).format(left + plot_width / 2, height - 10),
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def compare_phase_one(
    run_dirs: Sequence[Path],
    *,
    bsp_diagnostics_path: Path,
    norm_comparison_path: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Validate ten fixed runs and emit the complete phase-one comparison."""
    required_run_count = 2 * len(MILESTONES)
    if len(run_dirs) != required_run_count:
        raise ComparisonError("Exactly {} run directories are required".format(required_run_count))
    resolved_runs = [Path(path).expanduser().resolve() for path in run_dirs]
    if len(set(resolved_runs)) != required_run_count:
        raise ComparisonError("Run directories must be unique")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ComparisonError("Comparison output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    runs = [load_run(path) for path in resolved_runs]
    classified_manifests = classify_phase_one_manifests([run.manifest for run in runs])
    runs_by_key = {(run.variant, run.checkpoint_step): run for run in runs}
    if set(runs_by_key) != set(classified_manifests):
        raise ComparisonError("Loaded run classification is inconsistent")
    reference_records = runs_by_key[("baseline", MILESTONES[0])].records
    for run in runs:
        validate_paired_records(reference_records, run.records)

    bsp_diagnostics_file = Path(bsp_diagnostics_path).expanduser().resolve()
    norm_comparison_file = Path(norm_comparison_path).expanduser().resolve()
    bsp_diagnostics = load_strict_json(bsp_diagnostics_file)
    norm_diagnostics = load_strict_json(norm_comparison_file)
    reconstruction = validate_diagnostics([run.manifest for run in runs], bsp_diagnostics, norm_diagnostics)

    all_task_rows: List[Mapping[str, Any]] = []
    all_suite_rows: List[Mapping[str, Any]] = []
    milestones: List[Mapping[str, Any]] = []
    learning_rows: List[Mapping[str, Any]] = []
    for step in MILESTONES:
        baseline_run = runs_by_key[("baseline", step)]
        bsp_run = runs_by_key[("bsp", step)]
        validate_paired_records(baseline_run.records, bsp_run.records)
        task_rows, suite_rows, baseline_global, bsp_global, ci_low, ci_high = _metric_rows(
            step, baseline_run.records, bsp_run.records
        )
        all_task_rows.extend(task_rows)
        all_suite_rows.extend(suite_rows)
        baseline_diagnostics = _rollout_diagnostics(baseline_run.records)
        bsp_rollout_diagnostics = _rollout_diagnostics(bsp_run.records)
        milestone = {
            "checkpoint_step": step,
            "baseline_four_suite_macro_success_rate": baseline_global,
            "bsp_four_suite_macro_success_rate": bsp_global,
            "bsp_minus_baseline": bsp_global - baseline_global,
            "bootstrap_95_ci": [ci_low, ci_high],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "baseline_diagnostics": baseline_diagnostics,
            "bsp_diagnostics": bsp_rollout_diagnostics,
        }
        milestones.append(milestone)
        learning_rows.append(
            {
                "checkpoint_step": step,
                "baseline_four_suite_macro_success_rate": baseline_global,
                "bsp_four_suite_macro_success_rate": bsp_global,
                "bsp_minus_baseline": bsp_global - baseline_global,
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "baseline_eligible_steps_mean": baseline_diagnostics["eligible_rollout_steps_mean"],
                "baseline_eligible_steps_median": baseline_diagnostics["eligible_rollout_steps_median"],
                "baseline_successful_steps_mean": baseline_diagnostics["successful_rollout_steps_mean"],
                "baseline_successful_steps_median": baseline_diagnostics["successful_rollout_steps_median"],
                "baseline_inference_ms_mean": baseline_diagnostics["inference_ms_mean"],
                "baseline_inference_ms_p50": baseline_diagnostics["inference_ms_p50"],
                "baseline_inference_ms_p95": baseline_diagnostics["inference_ms_p95"],
                "bsp_eligible_steps_mean": bsp_rollout_diagnostics["eligible_rollout_steps_mean"],
                "bsp_eligible_steps_median": bsp_rollout_diagnostics["eligible_rollout_steps_median"],
                "bsp_successful_steps_mean": bsp_rollout_diagnostics["successful_rollout_steps_mean"],
                "bsp_successful_steps_median": bsp_rollout_diagnostics["successful_rollout_steps_median"],
                "bsp_inference_ms_mean": bsp_rollout_diagnostics["inference_ms_mean"],
                "bsp_inference_ms_p50": bsp_rollout_diagnostics["inference_ms_p50"],
                "bsp_inference_ms_p95": bsp_rollout_diagnostics["inference_ms_p95"],
                **reconstruction,
            }
        )

    run_inputs = []
    for variant in ("baseline", "bsp"):
        for step in MILESTONES:
            run = runs_by_key[(variant, step)]
            run_inputs.append(
                {
                    "policy_variant": variant,
                    "checkpoint_step": step,
                    "run_directory": str(run.path),
                    "file_sha256": dict(sorted(run.file_sha256.items())),
                }
            )
    comparison = {
        "schema_version": 1,
        "protocol": {
            "milestones": list(MILESTONES),
            "training_family": _training_family(next(iter(classified_manifests.values()))),
            "suites": list(libero_eval.SUPPORTED_SUITES),
            "tasks_per_suite": TASKS_PER_SUITE,
            "trials_per_task": TRIALS_PER_TASK,
            "episodes_per_run": EPISODES_PER_RUN,
            "total_episodes": TOTAL_EPISODES,
            "hierarchy": "paired rollout -> task -> suite -> four-suite macro",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "inputs": {
            "runs": run_inputs,
            "bsp_verification_diagnostics": {
                "path": str(bsp_diagnostics_file),
                "sha256": _file_sha256(bsp_diagnostics_file),
            },
            "norm_comparison": {
                "path": str(norm_comparison_file),
                "sha256": _file_sha256(norm_comparison_file),
            },
        },
        "bsp_reconstruction": reconstruction,
        "milestones": milestones,
        "tasks": all_task_rows,
        "suites": all_suite_rows,
    }
    try:
        comparison_json = libero_artifacts.json_text(comparison)
    except (TypeError, ValueError) as error:
        raise ComparisonError("Comparison contains a non-JSON diagnostic value") from error
    libero_artifacts.write_csv(output / "task_comparison.csv", all_task_rows)
    libero_artifacts.write_csv(output / "suite_comparison.csv", all_suite_rows)
    libero_artifacts.write_csv(output / "learning_curve.csv", learning_rows)
    libero_artifacts.atomic_text(output / "comparison.json", comparison_json)
    libero_artifacts.atomic_text(output / "report.md", _render_markdown(milestones))
    libero_artifacts.atomic_text(output / "learning_curve.svg", _render_svg(milestones))
    actual_outputs = {path.name for path in output.iterdir()}
    if actual_outputs != set(OUTPUT_FILENAMES):
        raise ComparisonError("Unexpected comparison output set: {}".format(sorted(actual_outputs)))
    return comparison
