"""Strict schema-v5 LIBERO run loading and three-mode comparison.

This module deliberately does not dispatch schema-v5 artifacts through the
schema-v2/v3 reader.  The only legacy report logic reused here is the fixed
task-within-suite hierarchical paired bootstrap.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import statistics
from types import MappingProxyType
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

from openpi_client import libero_artifacts
from openpi_client import libero_eval_v5
from openpi_client import libero_report as _legacy_bootstrap


EXECUTION_MODE_ORDER_V5 = (
    "baseline_async",
    "baseline_rtc",
    "bsp_spline_async",
)
TASKS_PER_SUITE_V5 = 10
TRIALS_PER_TASK_V5 = 50
EPISODES_PER_RUN_V5 = len(libero_eval_v5.SUPPORTED_SUITES) * TASKS_PER_SUITE_V5 * TRIALS_PER_TASK_V5
OUTPUT_FILENAMES_V5 = (
    "comparison_v5.json",
    "task_metrics_v5.csv",
    "report_v5.md",
)

_REQUIRED_ARTIFACTS = (
    "manifest.json",
    "episodes.jsonl",
    "summary.json",
    "video_audit.jsonl",
)
_FORMAL_MANIFEST_FIELDS = {
    "suites": list(libero_eval_v5.SUPPORTED_SUITES),
    "task_ids": list(range(TASKS_PER_SUITE_V5)),
    "trials_per_task": TRIALS_PER_TASK_V5,
    "num_steps_wait": 10,
    "max_steps_by_suite": dict(libero_eval_v5.MAX_STEPS_BY_SUITE),
    "train_seed": 42,
    "eval_seed": 42,
    "checkpoint_step": 10000,
}
_ROLLOUT_MANIFEST_FIELDS = (
    "schema_version",
    "dataset_fps",
    "source_demo_control_hz",
    "control_freq_hz",
    "controller_period_ns",
    "video_fps",
    "video_show_inference_waits",
    "latency_distribution",
    "theoretical_p95_latency_ns",
    "scheduling_latency_budget_ns",
    "scheduling_delay_ticks",
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
    "infrastructure_retries",
)
_FAMILY_IDENTITY_FIELDS = (
    "code_sha",
    "config_name",
    "checkpoint_step",
    "checkpoint",
    "container_digest",
    "norm_hash",
    "bsp_cache_hash",
    "bsp_cache_manifest_fingerprint",
    "policy_variant",
    "expected_action_horizon",
    "server_metadata_fingerprint",
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


class ComparisonErrorV5(ValueError):
    """A schema-v5 run or comparison input is not formally auditable."""


@dataclasses.dataclass(frozen=True)
class RunDataV5:
    path: Path
    manifest: Mapping[str, Any]
    records: Sequence[Mapping[str, Any]]
    summary: Mapping[str, Any]
    video_audits: Sequence[Mapping[str, Any]]
    file_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))
        object.__setattr__(self, "video_audits", tuple(self.video_audits))
        object.__setattr__(
            self,
            "file_sha256",
            MappingProxyType(dict(self.file_sha256)),
        )

    @property
    def execution_mode(self) -> str:
        return str(self.manifest["execution_mode"])

    @property
    def checkpoint_step(self) -> int:
        return int(self.manifest["checkpoint_step"])


def _reject_constant(value: str) -> None:
    raise ComparisonErrorV5("JSON contains a non-standard numeric constant: {}".format(value))


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value = {}  # type: Dict[str, Any]
    for key, item in pairs:
        if key in value:
            raise ComparisonErrorV5("JSON contains duplicate JSON key {!r}".format(key))
        value[key] = item
    return value


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as input_file:
            payload = json.load(
                input_file,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
    except ComparisonErrorV5:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ComparisonErrorV5("Invalid JSON file {}: {}".format(path, error)) from error
    if type(payload) is not dict:
        raise ComparisonErrorV5("JSON file {} must contain one object".format(path))
    return payload


def _load_strict_jsonl(
    path: Path,
    *,
    allow_empty: bool = False,
) -> List[Mapping[str, Any]]:
    records = []  # type: List[Mapping[str, Any]]
    try:
        with Path(path).open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise ComparisonErrorV5("Blank JSONL line at {}:{}".format(path, line_number))
                try:
                    payload = json.loads(
                        line,
                        parse_constant=_reject_constant,
                        object_pairs_hook=_unique_object,
                    )
                except ComparisonErrorV5:
                    raise
                except json.JSONDecodeError as error:
                    raise ComparisonErrorV5(
                        "Invalid JSONL record at {}:{}: {}".format(
                            path,
                            line_number,
                            error,
                        )
                    ) from error
                if type(payload) is not dict:
                    raise ComparisonErrorV5(
                        "JSONL record at {}:{} must be an object".format(
                            path,
                            line_number,
                        )
                    )
                records.append(payload)
    except ComparisonErrorV5:
        raise
    except (OSError, UnicodeError) as error:
        raise ComparisonErrorV5("Unable to read JSONL file {}: {}".format(path, error)) from error
    if not records and not allow_empty:
        raise ComparisonErrorV5("JSONL file {} is empty".format(path))
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ComparisonErrorV5("Unable to hash input file {}: {}".format(path, error)) from error
    return digest.hexdigest()


def _manifest_v5(value: Mapping[str, Any], *, formal: bool = True) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonErrorV5("Invalid schema-v5 manifest: expected a mapping")
    try:
        manifest = libero_eval_v5.EvaluationManifestV5.from_dict(dict(value))
        normalized = manifest.to_dict()
    except (TypeError, ValueError) as error:
        raise ComparisonErrorV5("Invalid schema-v5 manifest: {}".format(error)) from error
    if formal:
        mismatched = [field for field, expected in _FORMAL_MANIFEST_FIELDS.items() if normalized[field] != expected]
        if mismatched:
            raise ComparisonErrorV5(
                "Formal schema-v5 manifest or rollout identity fields mismatch: {}".format(mismatched)
            )
    return normalized


def _family_identity(manifest: Mapping[str, Any]) -> Tuple[Any, ...]:
    return tuple(manifest[field] for field in _FAMILY_IDENTITY_FIELDS)


def _rollout_manifest_identity(manifest: Mapping[str, Any]) -> Tuple[Any, ...]:
    return tuple(_hashable_json(manifest[field]) for field in _ROLLOUT_MANIFEST_FIELDS)


def _hashable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((key, _hashable_json(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(_hashable_json(item) for item in value)
    return value


def classify_checkpoint_manifests_v5(
    manifests: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    """Classify exactly three formal manifests for the 10K experiment."""
    if isinstance(manifests, (str, bytes)) or not isinstance(manifests, Sequence):
        raise ComparisonErrorV5("Checkpoint manifests must be a sequence")
    if len(manifests) != len(EXECUTION_MODE_ORDER_V5):
        raise ComparisonErrorV5("A checkpoint requires exactly three schema-v5 manifests")
    normalized = [_manifest_v5(manifest) for manifest in manifests]
    by_mode = {}  # type: Dict[str, Mapping[str, Any]]
    for manifest in normalized:
        mode = manifest["execution_mode"]
        if mode in by_mode:
            raise ComparisonErrorV5("Checkpoint contains duplicate execution mode {}".format(mode))
        by_mode[mode] = manifest
    if set(by_mode) != set(EXECUTION_MODE_ORDER_V5):
        raise ComparisonErrorV5("Checkpoint must contain exactly one of each execution mode")
    steps = {manifest["checkpoint_step"] for manifest in normalized}
    if len(steps) != 1:
        raise ComparisonErrorV5("All three manifests must share one checkpoint_step")

    if _family_identity(by_mode["baseline_async"]) != _family_identity(by_mode["baseline_rtc"]):
        raise ComparisonErrorV5("Baseline family identity does not match")

    reference_rollout = _rollout_manifest_identity(by_mode[EXECUTION_MODE_ORDER_V5[0]])
    for mode in EXECUTION_MODE_ORDER_V5[1:]:
        if _rollout_manifest_identity(by_mode[mode]) != reference_rollout:
            raise ComparisonErrorV5("All three manifests must share one rollout identity")
    return {mode: by_mode[mode] for mode in EXECUTION_MODE_ORDER_V5}


def _expected_bsp_budget(manifest: Mapping[str, Any]) -> Optional[int]:
    if manifest["execution_mode"] != "bsp_spline_async":
        return None
    calibration = manifest["latency_calibration"]
    if type(calibration) is not dict:
        raise ComparisonErrorV5("BSP async manifest is missing calibration")
    budget = calibration["derived_prefetch_budget_ns"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ComparisonErrorV5("BSP async calibration has an invalid budget")
    return budget


def _episode_v5(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> Tuple[libero_eval_v5.EpisodeRecordV5, Dict[str, Any]]:
    try:
        record = libero_eval_v5.EpisodeRecordV5.from_dict(
            value,
            expected_bsp_prefetch_budget_ns=_expected_bsp_budget(manifest),
        )
        normalized = record.to_dict()
    except (TypeError, ValueError) as error:
        raise ComparisonErrorV5("Invalid schema-v5 episode: {}".format(error)) from error
    if normalized["execution_mode"] != manifest["execution_mode"]:
        raise ComparisonErrorV5("Episode execution_mode does not match its manifest")
    if normalized["eval_seed"] != manifest["eval_seed"]:
        raise ComparisonErrorV5("Episode eval_seed does not match its manifest")
    if normalized["suite"] not in manifest["suites"] or normalized["task_id"] not in manifest["task_ids"]:
        raise ComparisonErrorV5("Episode selection is outside its manifest")
    return record, normalized


def _validate_formal_episode_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    path: Path,
) -> None:
    if len(records) != EPISODES_PER_RUN_V5:
        raise ComparisonErrorV5(
            "{} has {} episodes; expected exactly {}".format(
                path,
                len(records),
                EPISODES_PER_RUN_V5,
            )
        )
    paired_keys = set()
    episode_ids = set()
    groups = {}  # type: Dict[Tuple[str, int], List[int]]
    task_names = {}  # type: Dict[Tuple[str, int], str]
    for record in records:
        if record["status"] == "infrastructure_incomplete":
            raise ComparisonErrorV5("Formal comparison rejects infrastructure-incomplete episodes")
        if record["include_in_success_rate"] is not True:
            raise ComparisonErrorV5("Every formal episode must be denominator-eligible")
        paired_key = record["paired_key"]
        episode_id = record["episode_id"]
        if paired_key in paired_keys:
            raise ComparisonErrorV5("Formal run contains duplicate paired_key {}".format(paired_key))
        if episode_id in episode_ids:
            raise ComparisonErrorV5("Formal run contains duplicate episode_id {}".format(episode_id))
        paired_keys.add(paired_key)
        episode_ids.add(episode_id)
        group = (str(record["suite"]), int(record["task_id"]))
        groups.setdefault(group, []).append(int(record["init_state_index"]))
        previous_name = task_names.setdefault(group, str(record["task_name"]))
        if record["task_name"] != previous_name:
            raise ComparisonErrorV5("Task name changes within {}".format(group))
    expected_groups = {
        (suite, task_id) for suite in libero_eval_v5.SUPPORTED_SUITES for task_id in range(TASKS_PER_SUITE_V5)
    }
    if set(groups) != expected_groups:
        raise ComparisonErrorV5("Formal run does not contain the exact four-suite/task grid")
    for group, init_state_indices in groups.items():
        if sorted(init_state_indices) != list(range(TRIALS_PER_TASK_V5)):
            raise ComparisonErrorV5("{} does not contain init states 0..49 exactly once".format(group))


def _video_audits_v5(
    values: Sequence[Mapping[str, Any]],
    *,
    episodes_by_id: Mapping[str, libero_eval_v5.EpisodeRecordV5],
    manifest: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    normalized = []  # type: List[Dict[str, Any]]
    seen = set()
    for value in values:
        try:
            audit = libero_eval_v5.VideoArtifactAuditV5.from_dict(value)
            payload = audit.to_dict()
        except (TypeError, ValueError) as error:
            raise ComparisonErrorV5("Invalid schema-v5 video audit: {}".format(error)) from error
        episode_id = payload["episode_id"]
        if episode_id in seen:
            raise ComparisonErrorV5("Formal run contains duplicate video episode {}".format(episode_id))
        episode = episodes_by_id.get(episode_id)
        if episode is None:
            raise ComparisonErrorV5("Video audit refers to unknown episode {}".format(episode_id))
        if payload["video_show_inference_waits"] != manifest["video_show_inference_waits"]:
            raise ComparisonErrorV5("Video audit overlay setting does not match manifest")
        try:
            audit.validate_episode(episode)
        except (TypeError, ValueError) as error:
            raise ComparisonErrorV5("Video planned timing does not match episode: {}".format(error)) from error
        seen.add(episode_id)
        normalized.append(payload)
    return tuple(normalized)


def load_run_v5(run_dir: Path) -> RunDataV5:
    """Load and fully validate one formal schema-v5 run directory."""
    root = Path(run_dir).expanduser().resolve()
    paths = {name: root / name for name in _REQUIRED_ARTIFACTS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ComparisonErrorV5("Run {} is missing required artifacts: {}".format(root, missing))
    file_hashes = {name: _file_sha256(path) for name, path in paths.items()}

    artifact_errors_path = root / "artifact_errors.jsonl"
    if artifact_errors_path.exists():
        if not artifact_errors_path.is_file():
            raise ComparisonErrorV5("artifact_errors.jsonl exists but is not a file")
        artifact_errors = _load_strict_jsonl(
            artifact_errors_path,
            allow_empty=True,
        )
        if artifact_errors:
            for value in artifact_errors:
                try:
                    libero_eval_v5.ArtifactErrorV5.from_dict(value)
                except (TypeError, ValueError) as error:
                    raise ComparisonErrorV5("Invalid artifact error record: {}".format(error)) from error
            raise ComparisonErrorV5("Formal run contains artifact errors")

    manifest = _manifest_v5(_load_strict_json(paths["manifest.json"]))
    raw_records = _load_strict_jsonl(paths["episodes.jsonl"])
    record_objects = []  # type: List[libero_eval_v5.EpisodeRecordV5]
    records = []  # type: List[Dict[str, Any]]
    for value in raw_records:
        record, normalized = _episode_v5(value, manifest=manifest)
        record_objects.append(record)
        records.append(normalized)
    _validate_formal_episode_grid(records, path=root)

    episodes_by_id = {record.episode_id: record for record in record_objects}
    if len(episodes_by_id) != len(record_objects):
        raise ComparisonErrorV5("Formal run contains duplicate episode ids")
    video_audits = _video_audits_v5(
        _load_strict_jsonl(paths["video_audit.jsonl"], allow_empty=True),
        episodes_by_id=episodes_by_id,
        manifest=manifest,
    )

    summary = _load_strict_json(paths["summary.json"])
    try:
        derived_summary = libero_eval_v5.aggregate_records_v5(record_objects)
    except (TypeError, ValueError) as error:
        raise ComparisonErrorV5("Unable to derive schema-v5 summary: {}".format(error)) from error
    if not _json_exact_equal(summary, derived_summary):
        raise ComparisonErrorV5("summary.json is inconsistent with derived episode results for {}".format(root))
    return RunDataV5(
        path=root,
        manifest=manifest,
        records=records,
        summary=summary,
        video_audits=video_audits,
        file_sha256=file_hashes,
    )


def _json_exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_json_exact_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_exact_equal(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    return left == right


def _records_by_key(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> Dict[str, Mapping[str, Any]]:
    result = {}  # type: Dict[str, Mapping[str, Any]]
    for record in records:
        paired_key = record.get("paired_key")
        if not isinstance(paired_key, str) or not paired_key:
            raise ComparisonErrorV5("{} has a record without a paired_key".format(label))
        if paired_key in result:
            raise ComparisonErrorV5("{} has duplicate paired_key {}".format(label, paired_key))
        result[paired_key] = record
    return result


def _validate_rollout_identities(runs_by_mode: Mapping[str, RunDataV5]) -> None:
    for mode in EXECUTION_MODE_ORDER_V5:
        run = runs_by_mode[mode]
        _validate_formal_episode_grid(run.records, path=run.path)
        if any(
            record.get("execution_mode") != mode or record.get("eval_seed") != run.manifest["eval_seed"]
            for record in run.records
        ):
            raise ComparisonErrorV5("Run rollout identity does not match its execution mode and eval seed")
    reference_mode = EXECUTION_MODE_ORDER_V5[0]
    reference = _records_by_key(
        runs_by_mode[reference_mode].records,
        label=reference_mode,
    )
    if len(reference) != EPISODES_PER_RUN_V5:
        raise ComparisonErrorV5("Comparison requires exactly 2000 records per mode")
    for mode in EXECUTION_MODE_ORDER_V5[1:]:
        candidate = _records_by_key(runs_by_mode[mode].records, label=mode)
        if set(candidate) != set(reference):
            raise ComparisonErrorV5("Cross-mode rollout identity keys do not match")
        for key, reference_record in reference.items():
            mismatched = [
                field
                for field in _ROLLOUT_RECORD_IDENTITY_FIELDS
                if candidate[key].get(field) != reference_record.get(field)
            ]
            if mismatched:
                raise ComparisonErrorV5(
                    "Cross-mode rollout identity {} differs in {}".format(
                        key,
                        mismatched,
                    )
                )
            reference_samples = {
                request["request_id"]: (
                    request["latency_sample_key"],
                    request["sampled_target_latency_ns"],
                )
                for request in reference_record["inference_requests"]
            }
            candidate_samples = {
                request["request_id"]: (
                    request["latency_sample_key"],
                    request["sampled_target_latency_ns"],
                )
                for request in candidate[key]["inference_requests"]
            }
            for request_id in set(reference_samples).intersection(candidate_samples):
                if candidate_samples[request_id] != reference_samples[request_id]:
                    raise ComparisonErrorV5(
                        "Cross-mode paired latency sample differs for {} request {}".format(
                            key,
                            request_id,
                        )
                    )


def _success_rate(records: Sequence[Mapping[str, Any]]) -> float:
    task_rates = {}  # type: Dict[Tuple[str, int], float]
    for suite in libero_eval_v5.SUPPORTED_SUITES:
        for task_id in range(TASKS_PER_SUITE_V5):
            group = [record for record in records if record.get("suite") == suite and record.get("task_id") == task_id]
            if len(group) != TRIALS_PER_TASK_V5:
                raise ComparisonErrorV5("Success rate requires 50 rollouts per task")
            if any(type(record.get("success")) is not bool for record in group):
                raise ComparisonErrorV5("Formal comparison requires boolean episode success")
            task_rates[(suite, task_id)] = sum(record["success"] is True for record in group) / len(group)
    suite_rates = []
    for suite in libero_eval_v5.SUPPORTED_SUITES:
        suite_rates.append(
            sum(task_rates[(suite, task_id)] for task_id in range(TASKS_PER_SUITE_V5)) / TASKS_PER_SUITE_V5
        )
    return sum(suite_rates) / len(suite_rates)


def _paired_task_deltas(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> Mapping[Tuple[str, int], Sequence[float]]:
    before_by_key = _records_by_key(before, label="paired before mode")
    after_by_key = _records_by_key(after, label="paired after mode")
    if set(before_by_key) != set(after_by_key):
        raise ComparisonErrorV5("Paired comparison rollout keys do not match")
    values = {
        (suite, task_id): [] for suite in libero_eval_v5.SUPPORTED_SUITES for task_id in range(TASKS_PER_SUITE_V5)
    }  # type: Dict[Tuple[str, int], List[float]]
    for key in sorted(before_by_key):
        base = before_by_key[key]
        candidate = after_by_key[key]
        values[(str(base["suite"]), int(base["task_id"]))].append(
            float(candidate["success"] is True) - float(base["success"] is True)
        )
    return values


def _latency_diagnostics(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw_durations = []  # type: List[int]
    synthetic_delays = []  # type: List[int]
    effective_durations = []  # type: List[int]
    stall_durations = []  # type: List[int]
    underflow_durations = []  # type: List[int]
    episode_durations = []  # type: List[int]
    control_steps = []  # type: List[int]
    seam_l2_jumps = []  # type: List[float]
    seam_max_abs_jumps = []  # type: List[float]
    seam_gripper_jumps = []  # type: List[float]

    def duration(value: Any, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ComparisonErrorV5("{} must be a nonnegative integer".format(label))
        return value

    for record in records:
        events = record.get("inference_latencies")
        if not isinstance(events, list):
            raise ComparisonErrorV5("inference_latencies must remain a JSON list")
        for event in events:
            if type(event) is not dict:
                raise ComparisonErrorV5("inference latency must remain a JSON object")
            effective = duration(
                event.get("effective_inference_latency_ns"),
                label="effective inference latency",
            )
            raw = duration(
                event.get("raw_inference_latency_ns"),
                label="raw inference latency",
            )
            synthetic = duration(
                event.get("synthetic_delay_ns"),
                label="synthetic inference delay",
            )
            recorded = duration(event.get("duration_ns"), label="inference latency")
            if raw + synthetic != effective or recorded != effective:
                raise ComparisonErrorV5("inference latency breakdown does not match effective latency")
            raw_durations.append(raw)
            synthetic_delays.append(synthetic)
            effective_durations.append(effective)
        stalls = record.get("control_stalls")
        underflows = record.get("action_underflows")
        if not isinstance(stalls, list) or not isinstance(underflows, list):
            raise ComparisonErrorV5("stall and underflow events must remain JSON lists")
        for stall in stalls:
            if type(stall) is not dict:
                raise ComparisonErrorV5("control stall must remain a JSON object")
            stall_durations.append(duration(stall.get("duration_ns"), label="control stall duration"))
        for underflow in underflows:
            if type(underflow) is not dict:
                raise ComparisonErrorV5("action underflow must remain a JSON object")
            underflow_durations.append(duration(underflow.get("duration_ns"), label="action underflow duration"))
        episode_durations.append(duration(record.get("episode_duration_ns"), label="episode wall time"))
        control_steps.append(duration(record.get("steps"), label="control steps"))
        seams = record.get("action_seams")
        if not isinstance(seams, list):
            raise ComparisonErrorV5("action_seams must remain a JSON list")
        for seam in seams:
            if type(seam) is not dict:
                raise ComparisonErrorV5("action seam must remain a JSON object")
            for field, destination in (
                ("arm_l2_jump", seam_l2_jumps),
                ("arm_max_abs_jump", seam_max_abs_jumps),
                ("gripper_abs_jump", seam_gripper_jumps),
            ):
                value = seam.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ComparisonErrorV5("{} must be numeric".format(field))
                destination.append(float(value))

    def p95(values: Sequence[int]) -> Optional[int]:
        if not values:
            return None
        ordered = sorted(values)
        rank = (95 * len(ordered) + 99) // 100
        return ordered[rank - 1]

    def p95_float(values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        rank = (95 * len(ordered) + 99) // 100
        return ordered[rank - 1]

    calibration = manifest["latency_calibration"]
    calibration_p95 = calibration["empirical_effective_p95_ns"]
    effective_total = sum(effective_durations)
    stall_total = sum(stall_durations)
    hidden_ns = max(0, effective_total - stall_total)
    episode_total = sum(episode_durations)
    return {
        "inference_latency_count": len(effective_durations),
        "inference_latency_ns_total": effective_total,
        "inference_latency_ns_mean": (effective_total / len(effective_durations) if effective_durations else None),
        "inference_latency_ns_median": (statistics.median(effective_durations) if effective_durations else None),
        "inference_latency_ns_min": min(effective_durations) if effective_durations else None,
        "inference_latency_ns_max": max(effective_durations) if effective_durations else None,
        "raw_inference_latency_ns_total": sum(raw_durations),
        "raw_inference_latency_ns_mean": (sum(raw_durations) / len(raw_durations) if raw_durations else None),
        "raw_inference_latency_ns_p95": p95(raw_durations),
        "synthetic_delay_ns_total": sum(synthetic_delays),
        "synthetic_delay_ns_mean": (sum(synthetic_delays) / len(synthetic_delays) if synthetic_delays else None),
        "effective_inference_latency_ns_total": effective_total,
        "effective_inference_latency_ns_mean": (
            effective_total / len(effective_durations) if effective_durations else None
        ),
        "effective_inference_latency_ns_p95": p95(effective_durations),
        "control_stall_count": len(stall_durations),
        "control_stall_ns_total": stall_total,
        "control_stall_ns_mean": (stall_total / len(stall_durations) if stall_durations else None),
        "action_underflow_count": len(underflow_durations),
        "action_underflow_ns_total": sum(underflow_durations),
        "action_underflow_ns_mean": (
            sum(underflow_durations) / len(underflow_durations) if underflow_durations else None
        ),
        "latency_hidden_ns": hidden_ns,
        "latency_hidden_ratio": (hidden_ns / effective_total if effective_total else None),
        "control_steps_total": sum(control_steps),
        "control_steps_mean": (sum(control_steps) / len(control_steps) if control_steps else None),
        "episode_wall_time_ns_total": episode_total,
        "episode_wall_time_ns_mean": (episode_total / len(episode_durations) if episode_durations else None),
        "episode_throughput_per_minute": (
            len(episode_durations) * 60_000_000_000 / episode_total if episode_total else None
        ),
        "calibration_p95_latency_ns": calibration_p95,
        "action_seam_count": len(seam_l2_jumps),
        "arm_l2_jump_mean": (sum(seam_l2_jumps) / len(seam_l2_jumps) if seam_l2_jumps else None),
        "arm_l2_jump_p95": p95_float(seam_l2_jumps),
        "arm_max_abs_jump_mean": (sum(seam_max_abs_jumps) / len(seam_max_abs_jumps) if seam_max_abs_jumps else None),
        "arm_max_abs_jump_p95": p95_float(seam_max_abs_jumps),
        "gripper_abs_jump_mean": (sum(seam_gripper_jumps) / len(seam_gripper_jumps) if seam_gripper_jumps else None),
        "gripper_abs_jump_p95": p95_float(seam_gripper_jumps),
    }


def _paired_delta_report(
    before: RunDataV5,
    after: RunDataV5,
) -> Mapping[str, Any]:
    paired = _paired_task_deltas(before.records, after.records)
    try:
        observed = _legacy_bootstrap.hierarchical_delta(paired)
        low, high = _legacy_bootstrap.stratified_paired_bootstrap(paired)
    except ValueError as error:
        raise ComparisonErrorV5("Hierarchical bootstrap failed: {}".format(error)) from error
    return {
        "before_mode": before.execution_mode,
        "after_mode": after.execution_mode,
        "observed_delta": observed,
        "bootstrap_95_ci": [low, high],
        "bootstrap_seed": _legacy_bootstrap.BOOTSTRAP_SEED,
        "bootstrap_resamples": _legacy_bootstrap.BOOTSTRAP_RESAMPLES,
        "hierarchy": "paired rollout -> task -> suite -> four-suite macro",
    }


def compare_checkpoint_v5(
    runs_by_mode: Mapping[str, RunDataV5],
) -> Mapping[str, Any]:
    """Compare the three paired schema-v5 modes at the fixed 10K checkpoint."""
    if not isinstance(runs_by_mode, Mapping) or set(runs_by_mode) != set(EXECUTION_MODE_ORDER_V5):
        raise ComparisonErrorV5("Checkpoint comparison requires exactly three runs by mode")
    ordered = {}  # type: Dict[str, RunDataV5]
    for mode in EXECUTION_MODE_ORDER_V5:
        run = runs_by_mode[mode]
        if not isinstance(run, RunDataV5):
            raise ComparisonErrorV5("runs_by_mode values must be RunDataV5")
        if run.execution_mode != mode:
            raise ComparisonErrorV5("Run key does not match its execution_mode")
        ordered[mode] = run
    classified = classify_checkpoint_manifests_v5([ordered[mode].manifest for mode in EXECUTION_MODE_ORDER_V5])
    if any(ordered[mode].manifest["checkpoint_step"] != classified[mode]["checkpoint_step"] for mode in ordered):
        raise ComparisonErrorV5("Run manifests changed during classification")
    _validate_rollout_identities(ordered)
    success_rates = {mode: _success_rate(ordered[mode].records) for mode in EXECUTION_MODE_ORDER_V5}
    primary_deltas = {
        "baseline_rtc_minus_baseline_async": _paired_delta_report(
            ordered["baseline_async"],
            ordered["baseline_rtc"],
        ),
        "bsp_spline_async_minus_baseline_async": _paired_delta_report(
            ordered["baseline_async"],
            ordered["bsp_spline_async"],
        ),
        "bsp_spline_async_minus_baseline_rtc": _paired_delta_report(
            ordered["baseline_rtc"],
            ordered["bsp_spline_async"],
        ),
    }
    return {
        "schema_version": 5,
        "checkpoint_step": next(iter(classified.values()))["checkpoint_step"],
        "latency_distribution": next(iter(classified.values()))["latency_distribution"],
        "theoretical_p95_latency_ns": next(iter(classified.values()))["theoretical_p95_latency_ns"],
        "scheduling_latency_budget_ns": next(iter(classified.values()))["scheduling_latency_budget_ns"],
        "scheduling_delay_ticks": next(iter(classified.values()))["scheduling_delay_ticks"],
        "success_rates": success_rates,
        "primary_paired_deltas": primary_deltas,
        "diagnostics": {
            mode: _latency_diagnostics(ordered[mode].records, classified[mode]) for mode in EXECUTION_MODE_ORDER_V5
        },
        "inputs": {
            mode: {
                "run_directory": str(ordered[mode].path),
                "file_sha256": dict(sorted(ordered[mode].file_sha256.items())),
            }
            for mode in EXECUTION_MODE_ORDER_V5
        },
    }


def _render_report_v5(comparison: Mapping[str, Any]) -> str:
    rates = comparison["success_rates"]
    deltas = comparison["primary_paired_deltas"]
    distribution = comparison["latency_distribution"]
    lines = [
        "# LIBERO schema-v5 random-latency comparison",
        "",
        "Paired latency: Normal({:.0f} ms, {:.0f} ms), seed {}, sampler `{}`.".format(
            distribution["mean_ns"] / 1_000_000,
            distribution["stddev_ns"] / 1_000_000,
            distribution["seed"],
            distribution["sampler_version"],
        ),
        "",
        "| mode | macro success rate |",
        "|---|---:|",
    ]
    for mode in EXECUTION_MODE_ORDER_V5:
        lines.append("| `{}` | {:.6f} |".format(mode, rates[mode]))
    lines.extend(
        [
            "",
            "| paired contrast | observed success-rate delta | bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for name in (
        "baseline_rtc_minus_baseline_async",
        "bsp_spline_async_minus_baseline_async",
        "bsp_spline_async_minus_baseline_rtc",
    ):
        delta = deltas[name]
        lines.append(
            "| `{}` | {:.6f} | [{:.6f}, {:.6f}] |".format(
                name,
                delta["observed_delta"],
                delta["bootstrap_95_ci"][0],
                delta["bootstrap_95_ci"][1],
            )
        )
    lines.extend(
        [
            "",
            (
                "Primary deltas use the fixed seed-42, 10,000-resample paired "
                "rollout → task → suite → four-suite hierarchical bootstrap."
            ),
            ("Calibration p95 values are copied from validated manifests and " "are not recomputed by this reporter."),
            "",
        ]
    )
    return "\n".join(lines)


def _task_metrics_rows_v5(
    runs_by_mode: Mapping[str, RunDataV5],
) -> List[Mapping[str, Any]]:
    rows = []  # type: List[Mapping[str, Any]]
    for suite in libero_eval_v5.SUPPORTED_SUITES:
        for task_id in range(TASKS_PER_SUITE_V5):
            rates = {}  # type: Dict[str, float]
            task_name = None  # type: Optional[str]
            for mode in EXECUTION_MODE_ORDER_V5:
                records = [
                    record
                    for record in runs_by_mode[mode].records
                    if record["suite"] == suite and record["task_id"] == task_id
                ]
                if len(records) != TRIALS_PER_TASK_V5:
                    raise ComparisonErrorV5("Task metrics require 50 paired trials")
                rates[mode] = sum(record["success"] is True for record in records) / len(records)
                current_name = str(records[0]["task_name"])
                if task_name is None:
                    task_name = current_name
                elif task_name != current_name:
                    raise ComparisonErrorV5("Cross-mode task names do not match")
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task_name,
                    "baseline_async_success_rate": rates["baseline_async"],
                    "baseline_rtc_success_rate": rates["baseline_rtc"],
                    "bsp_spline_async_success_rate": rates["bsp_spline_async"],
                    "baseline_rtc_minus_baseline_async": (rates["baseline_rtc"] - rates["baseline_async"]),
                    "bsp_spline_async_minus_baseline_async": (rates["bsp_spline_async"] - rates["baseline_async"]),
                    "bsp_spline_async_minus_baseline_rtc": (rates["bsp_spline_async"] - rates["baseline_rtc"]),
                }
            )
    return rows


def write_three_mode_report_v5(
    run_dirs: Sequence[Any],
    *,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Write the exact three-input schema-v5 10K comparison."""
    if isinstance(run_dirs, (str, bytes)) or not isinstance(run_dirs, Sequence):
        raise ComparisonErrorV5("run_dirs must be a sequence")
    if len(run_dirs) != len(EXECUTION_MODE_ORDER_V5):
        raise ComparisonErrorV5("Report requires exactly three runs")
    if all(isinstance(value, RunDataV5) for value in run_dirs):
        runs = list(run_dirs)
    elif any(isinstance(value, RunDataV5) for value in run_dirs):
        raise ComparisonErrorV5("Do not mix RunDataV5 objects and run directories")
    else:
        resolved = [Path(value).expanduser().resolve() for value in run_dirs]
        if len(set(resolved)) != len(resolved):
            raise ComparisonErrorV5("Run directories must be unique")
        runs = [load_run_v5(path) for path in resolved]

    classified = classify_checkpoint_manifests_v5([run.manifest for run in runs])
    runs_by_mode = {}  # type: Dict[str, RunDataV5]
    for run in runs:
        if run.execution_mode in runs_by_mode:
            raise ComparisonErrorV5("Duplicate run mode {}".format(run.execution_mode))
        runs_by_mode[run.execution_mode] = run
    if set(runs_by_mode) != set(classified):
        raise ComparisonErrorV5("Loaded runs do not match the three classified modes")
    comparison = compare_checkpoint_v5(runs_by_mode)
    report = {
        "schema_version": 5,
        "protocol": {
            "checkpoint_step": 10000,
            "execution_modes": list(EXECUTION_MODE_ORDER_V5),
            "suites": list(libero_eval_v5.SUPPORTED_SUITES),
            "tasks_per_suite": TASKS_PER_SUITE_V5,
            "trials_per_task": TRIALS_PER_TASK_V5,
            "episodes_per_run": EPISODES_PER_RUN_V5,
            "latency_distribution": comparison["latency_distribution"],
            "theoretical_p95_latency_ns": comparison["theoretical_p95_latency_ns"],
            "scheduling_latency_budget_ns": comparison["scheduling_latency_budget_ns"],
            "scheduling_delay_ticks": comparison["scheduling_delay_ticks"],
            "primary_deltas": [
                "baseline_rtc_minus_baseline_async",
                "bsp_spline_async_minus_baseline_async",
                "bsp_spline_async_minus_baseline_rtc",
            ],
            "bootstrap_seed": _legacy_bootstrap.BOOTSTRAP_SEED,
            "bootstrap_resamples": _legacy_bootstrap.BOOTSTRAP_RESAMPLES,
        },
        "comparison": comparison,
    }

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise ComparisonErrorV5("Comparison output path must be a directory")
        if any(output.iterdir()):
            raise ComparisonErrorV5("Comparison output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    try:
        comparison_text = libero_artifacts.json_text(report)
    except (TypeError, ValueError) as error:
        raise ComparisonErrorV5("Schema-v5 report contains a non-JSON value") from error
    libero_artifacts.atomic_text(output / "comparison_v5.json", comparison_text)
    libero_artifacts.write_csv(
        output / "task_metrics_v5.csv",
        _task_metrics_rows_v5(runs_by_mode),
    )
    libero_artifacts.atomic_text(
        output / "report_v5.md",
        _render_report_v5(comparison),
    )
    actual_outputs = {path.name for path in output.iterdir()}
    if actual_outputs != set(OUTPUT_FILENAMES_V5):
        raise ComparisonErrorV5("Unexpected schema-v5 report output set: {}".format(sorted(actual_outputs)))
    return report
