"""Strict schema-v4 LIBERO run loading and four-mode comparison.

This module deliberately does not dispatch schema-v4 artifacts through the
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
from openpi_client import libero_control_v4
from openpi_client import libero_eval_v4
from openpi_client import libero_report as _legacy_bootstrap


MILESTONES_V4 = (0, 1000, 2000, 5000, 10000)
EXECUTION_MODE_ORDER_V4 = (
    "baseline_sync_n5",
    "baseline_rtc",
    "bsp_spline_sync",
    "bsp_spline_async",
)
TASKS_PER_SUITE_V4 = 10
TRIALS_PER_TASK_V4 = 50
EPISODES_PER_RUN_V4 = (
    len(libero_eval_v4.SUPPORTED_SUITES)
    * TASKS_PER_SUITE_V4
    * TRIALS_PER_TASK_V4
)
OUTPUT_FILENAMES_V4 = (
    "comparison_v4.json",
    "learning_curve_v4.csv",
    "report_v4.md",
)

_REQUIRED_ARTIFACTS = (
    "manifest.json",
    "episodes.jsonl",
    "summary.json",
    "video_audit.jsonl",
)
_FORMAL_MANIFEST_FIELDS = {
    "suites": list(libero_eval_v4.SUPPORTED_SUITES),
    "task_ids": list(range(TASKS_PER_SUITE_V4)),
    "trials_per_task": TRIALS_PER_TASK_V4,
    "num_steps_wait": 10,
    "max_steps_by_suite": dict(libero_eval_v4.MAX_STEPS_BY_SUITE),
    "train_seed": 42,
    "eval_seed": 42,
}
_ROLLOUT_MANIFEST_FIELDS = (
    "schema_version",
    "dataset_fps",
    "source_demo_control_hz",
    "control_freq_hz",
    "controller_period_ns",
    "video_fps",
    "video_show_inference_waits",
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


class ComparisonErrorV4(ValueError):
    """A schema-v4 run or comparison input is not formally auditable."""


@dataclasses.dataclass(frozen=True)
class RunDataV4:
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
    raise ComparisonErrorV4(
        "JSON contains a non-standard numeric constant: {}".format(value)
    )


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value = {}  # type: Dict[str, Any]
    for key, item in pairs:
        if key in value:
            raise ComparisonErrorV4(
                "JSON contains duplicate JSON key {!r}".format(key)
            )
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
    except ComparisonErrorV4:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ComparisonErrorV4(
            "Invalid JSON file {}: {}".format(path, error)
        ) from error
    if type(payload) is not dict:
        raise ComparisonErrorV4("JSON file {} must contain one object".format(path))
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
                    raise ComparisonErrorV4(
                        "Blank JSONL line at {}:{}".format(path, line_number)
                    )
                try:
                    payload = json.loads(
                        line,
                        parse_constant=_reject_constant,
                        object_pairs_hook=_unique_object,
                    )
                except ComparisonErrorV4:
                    raise
                except json.JSONDecodeError as error:
                    raise ComparisonErrorV4(
                        "Invalid JSONL record at {}:{}: {}".format(
                            path,
                            line_number,
                            error,
                        )
                    ) from error
                if type(payload) is not dict:
                    raise ComparisonErrorV4(
                        "JSONL record at {}:{} must be an object".format(
                            path,
                            line_number,
                        )
                    )
                records.append(payload)
    except ComparisonErrorV4:
        raise
    except (OSError, UnicodeError) as error:
        raise ComparisonErrorV4(
            "Unable to read JSONL file {}: {}".format(path, error)
        ) from error
    if not records and not allow_empty:
        raise ComparisonErrorV4("JSONL file {} is empty".format(path))
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ComparisonErrorV4(
            "Unable to hash input file {}: {}".format(path, error)
        ) from error
    return digest.hexdigest()


def _manifest_v4(value: Mapping[str, Any], *, formal: bool = True) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonErrorV4("Invalid schema-v4 manifest: expected a mapping")
    try:
        manifest = libero_eval_v4.EvaluationManifestV4.from_dict(dict(value))
        normalized = manifest.to_dict()
    except (TypeError, ValueError) as error:
        raise ComparisonErrorV4(
            "Invalid schema-v4 manifest: {}".format(error)
        ) from error
    if formal:
        mismatched = [
            field
            for field, expected in _FORMAL_MANIFEST_FIELDS.items()
            if normalized[field] != expected
        ]
        if mismatched:
            raise ComparisonErrorV4(
                "Formal schema-v4 manifest fields mismatch: {}".format(mismatched)
            )
    return normalized


def _family_identity(manifest: Mapping[str, Any]) -> Tuple[Any, ...]:
    return tuple(manifest[field] for field in _FAMILY_IDENTITY_FIELDS)


def _rollout_manifest_identity(manifest: Mapping[str, Any]) -> Tuple[Any, ...]:
    return tuple(
        _hashable_json(manifest[field]) for field in _ROLLOUT_MANIFEST_FIELDS
    )


def _hashable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((key, _hashable_json(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(_hashable_json(item) for item in value)
    return value


def classify_checkpoint_manifests_v4(
    manifests: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    """Classify exactly four formal manifests for one optimizer step."""
    if isinstance(manifests, (str, bytes)) or not isinstance(manifests, Sequence):
        raise ComparisonErrorV4("Checkpoint manifests must be a sequence")
    if len(manifests) != len(EXECUTION_MODE_ORDER_V4):
        raise ComparisonErrorV4(
            "A checkpoint requires exactly four schema-v4 manifests"
        )
    normalized = [_manifest_v4(manifest) for manifest in manifests]
    by_mode = {}  # type: Dict[str, Mapping[str, Any]]
    for manifest in normalized:
        mode = manifest["execution_mode"]
        if mode in by_mode:
            raise ComparisonErrorV4(
                "Checkpoint contains duplicate execution mode {}".format(mode)
            )
        by_mode[mode] = manifest
    if set(by_mode) != set(EXECUTION_MODE_ORDER_V4):
        raise ComparisonErrorV4(
            "Checkpoint must contain exactly one of each execution mode"
        )
    steps = {manifest["checkpoint_step"] for manifest in normalized}
    if len(steps) != 1:
        raise ComparisonErrorV4("All four manifests must share one checkpoint_step")

    for family, modes in (
        ("baseline", ("baseline_sync_n5", "baseline_rtc")),
        ("BSP", ("bsp_spline_sync", "bsp_spline_async")),
    ):
        if _family_identity(by_mode[modes[0]]) != _family_identity(
            by_mode[modes[1]]
        ):
            raise ComparisonErrorV4("{} family identity does not match".format(family))

    reference_rollout = _rollout_manifest_identity(
        by_mode[EXECUTION_MODE_ORDER_V4[0]]
    )
    for mode in EXECUTION_MODE_ORDER_V4[1:]:
        if _rollout_manifest_identity(by_mode[mode]) != reference_rollout:
            raise ComparisonErrorV4(
                "All four manifests must share one rollout identity"
            )
    return {mode: by_mode[mode] for mode in EXECUTION_MODE_ORDER_V4}


def classify_five_checkpoint_manifests_v4(
    manifests: Sequence[Mapping[str, Any]],
) -> Dict[int, Dict[str, Mapping[str, Any]]]:
    """Classify exactly twenty manifests into the five fixed checkpoints."""
    if isinstance(manifests, (str, bytes)) or not isinstance(manifests, Sequence):
        raise ComparisonErrorV4("Five-checkpoint manifests must be a sequence")
    if len(manifests) != len(MILESTONES_V4) * len(EXECUTION_MODE_ORDER_V4):
        raise ComparisonErrorV4(
            "Five-checkpoint comparison requires exactly 20 manifests"
        )
    normalized = [_manifest_v4(manifest) for manifest in manifests]
    groups = {}  # type: Dict[int, List[Mapping[str, Any]]]
    for manifest in normalized:
        groups.setdefault(int(manifest["checkpoint_step"]), []).append(manifest)
    if set(groups) != set(MILESTONES_V4):
        raise ComparisonErrorV4(
            "Five-checkpoint comparison requires steps {}".format(MILESTONES_V4)
        )
    return {
        step: classify_checkpoint_manifests_v4(groups[step])
        for step in MILESTONES_V4
    }


def _expected_bsp_budget(manifest: Mapping[str, Any]) -> Optional[int]:
    if manifest["execution_mode"] != "bsp_spline_async":
        return None
    calibration = manifest["latency_calibration"]
    if type(calibration) is not dict:
        raise ComparisonErrorV4("BSP async manifest is missing calibration")
    budget = calibration["derived_prefetch_budget_ns"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ComparisonErrorV4("BSP async calibration has an invalid budget")
    return budget


def _episode_v4(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> Tuple[libero_eval_v4.EpisodeRecordV4, Dict[str, Any]]:
    try:
        record = libero_eval_v4.EpisodeRecordV4.from_dict(
            value,
            expected_bsp_prefetch_budget_ns=_expected_bsp_budget(manifest),
        )
        normalized = record.to_dict()
    except (TypeError, ValueError) as error:
        raise ComparisonErrorV4(
            "Invalid schema-v4 episode: {}".format(error)
        ) from error
    if normalized["execution_mode"] != manifest["execution_mode"]:
        raise ComparisonErrorV4("Episode execution_mode does not match its manifest")
    if normalized["eval_seed"] != manifest["eval_seed"]:
        raise ComparisonErrorV4("Episode eval_seed does not match its manifest")
    if (
        normalized["suite"] not in manifest["suites"]
        or normalized["task_id"] not in manifest["task_ids"]
    ):
        raise ComparisonErrorV4("Episode selection is outside its manifest")
    return record, normalized


def _validate_formal_episode_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    path: Path,
) -> None:
    if len(records) != EPISODES_PER_RUN_V4:
        raise ComparisonErrorV4(
            "{} has {} episodes; expected exactly {}".format(
                path,
                len(records),
                EPISODES_PER_RUN_V4,
            )
        )
    paired_keys = set()
    episode_ids = set()
    groups = {}  # type: Dict[Tuple[str, int], List[int]]
    task_names = {}  # type: Dict[Tuple[str, int], str]
    for record in records:
        if record["status"] == "infrastructure_incomplete":
            raise ComparisonErrorV4(
                "Formal comparison rejects infrastructure-incomplete episodes"
            )
        if record["include_in_success_rate"] is not True:
            raise ComparisonErrorV4("Every formal episode must be denominator-eligible")
        paired_key = record["paired_key"]
        episode_id = record["episode_id"]
        if paired_key in paired_keys:
            raise ComparisonErrorV4(
                "Formal run contains duplicate paired_key {}".format(paired_key)
            )
        if episode_id in episode_ids:
            raise ComparisonErrorV4(
                "Formal run contains duplicate episode_id {}".format(episode_id)
            )
        paired_keys.add(paired_key)
        episode_ids.add(episode_id)
        group = (str(record["suite"]), int(record["task_id"]))
        groups.setdefault(group, []).append(int(record["init_state_index"]))
        previous_name = task_names.setdefault(group, str(record["task_name"]))
        if record["task_name"] != previous_name:
            raise ComparisonErrorV4("Task name changes within {}".format(group))
    expected_groups = {
        (suite, task_id)
        for suite in libero_eval_v4.SUPPORTED_SUITES
        for task_id in range(TASKS_PER_SUITE_V4)
    }
    if set(groups) != expected_groups:
        raise ComparisonErrorV4(
            "Formal run does not contain the exact four-suite/task grid"
        )
    for group, init_state_indices in groups.items():
        if sorted(init_state_indices) != list(range(TRIALS_PER_TASK_V4)):
            raise ComparisonErrorV4(
                "{} does not contain init states 0..49 exactly once".format(group)
            )


def _video_audits_v4(
    values: Sequence[Mapping[str, Any]],
    *,
    episodes_by_id: Mapping[str, libero_eval_v4.EpisodeRecordV4],
    manifest: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    normalized = []  # type: List[Dict[str, Any]]
    seen = set()
    for value in values:
        try:
            audit = libero_eval_v4.VideoArtifactAuditV4.from_dict(value)
            payload = audit.to_dict()
        except (TypeError, ValueError) as error:
            raise ComparisonErrorV4(
                "Invalid schema-v4 video audit: {}".format(error)
            ) from error
        episode_id = payload["episode_id"]
        if episode_id in seen:
            raise ComparisonErrorV4(
                "Formal run contains duplicate video episode {}".format(episode_id)
            )
        episode = episodes_by_id.get(episode_id)
        if episode is None:
            raise ComparisonErrorV4(
                "Video audit refers to unknown episode {}".format(episode_id)
            )
        if (
            payload["video_show_inference_waits"]
            != manifest["video_show_inference_waits"]
        ):
            raise ComparisonErrorV4(
                "Video audit overlay setting does not match manifest"
            )
        try:
            audit.validate_episode(episode)
        except (TypeError, ValueError) as error:
            raise ComparisonErrorV4(
                "Video planned timing does not match episode: {}".format(error)
            ) from error
        seen.add(episode_id)
        normalized.append(payload)
    return tuple(normalized)


def load_run_v4(run_dir: Path) -> RunDataV4:
    """Load and fully validate one formal schema-v4 run directory."""
    root = Path(run_dir).expanduser().resolve()
    paths = {name: root / name for name in _REQUIRED_ARTIFACTS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ComparisonErrorV4(
            "Run {} is missing required artifacts: {}".format(root, missing)
        )
    file_hashes = {name: _file_sha256(path) for name, path in paths.items()}

    artifact_errors_path = root / "artifact_errors.jsonl"
    if artifact_errors_path.exists():
        if not artifact_errors_path.is_file():
            raise ComparisonErrorV4("artifact_errors.jsonl exists but is not a file")
        artifact_errors = _load_strict_jsonl(
            artifact_errors_path,
            allow_empty=True,
        )
        if artifact_errors:
            for value in artifact_errors:
                try:
                    libero_eval_v4.ArtifactErrorV4.from_dict(value)
                except (TypeError, ValueError) as error:
                    raise ComparisonErrorV4(
                        "Invalid artifact error record: {}".format(error)
                    ) from error
            raise ComparisonErrorV4("Formal run contains artifact errors")

    manifest = _manifest_v4(_load_strict_json(paths["manifest.json"]))
    raw_records = _load_strict_jsonl(paths["episodes.jsonl"])
    record_objects = []  # type: List[libero_eval_v4.EpisodeRecordV4]
    records = []  # type: List[Dict[str, Any]]
    for value in raw_records:
        record, normalized = _episode_v4(value, manifest=manifest)
        record_objects.append(record)
        records.append(normalized)
    _validate_formal_episode_grid(records, path=root)

    episodes_by_id = {record.episode_id: record for record in record_objects}
    if len(episodes_by_id) != len(record_objects):
        raise ComparisonErrorV4("Formal run contains duplicate episode ids")
    video_audits = _video_audits_v4(
        _load_strict_jsonl(paths["video_audit.jsonl"], allow_empty=True),
        episodes_by_id=episodes_by_id,
        manifest=manifest,
    )

    summary = _load_strict_json(paths["summary.json"])
    try:
        derived_summary = libero_eval_v4.aggregate_records_v4(record_objects)
    except (TypeError, ValueError) as error:
        raise ComparisonErrorV4(
            "Unable to derive schema-v4 summary: {}".format(error)
        ) from error
    if not _json_exact_equal(summary, derived_summary):
        raise ComparisonErrorV4(
            "summary.json is inconsistent with derived episode results for {}".format(
                root
            )
        )
    return RunDataV4(
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
        return set(left) == set(right) and all(
            _json_exact_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_exact_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
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
            raise ComparisonErrorV4(
                "{} has a record without a paired_key".format(label)
            )
        if paired_key in result:
            raise ComparisonErrorV4(
                "{} has duplicate paired_key {}".format(label, paired_key)
            )
        result[paired_key] = record
    return result


def _validate_rollout_identities(runs_by_mode: Mapping[str, RunDataV4]) -> None:
    for mode in EXECUTION_MODE_ORDER_V4:
        run = runs_by_mode[mode]
        _validate_formal_episode_grid(run.records, path=run.path)
        if any(
            record.get("execution_mode") != mode
            or record.get("eval_seed") != run.manifest["eval_seed"]
            for record in run.records
        ):
            raise ComparisonErrorV4(
                "Run records do not match their execution mode and eval seed"
            )
    reference_mode = EXECUTION_MODE_ORDER_V4[0]
    reference = _records_by_key(
        runs_by_mode[reference_mode].records,
        label=reference_mode,
    )
    if len(reference) != EPISODES_PER_RUN_V4:
        raise ComparisonErrorV4("Comparison requires exactly 2000 records per mode")
    for mode in EXECUTION_MODE_ORDER_V4[1:]:
        candidate = _records_by_key(runs_by_mode[mode].records, label=mode)
        if set(candidate) != set(reference):
            raise ComparisonErrorV4("Cross-mode rollout identity keys do not match")
        for key, reference_record in reference.items():
            mismatched = [
                field
                for field in _ROLLOUT_RECORD_IDENTITY_FIELDS
                if candidate[key].get(field) != reference_record.get(field)
            ]
            if mismatched:
                raise ComparisonErrorV4(
                    "Cross-mode rollout identity {} differs in {}".format(
                        key,
                        mismatched,
                    )
                )


def _success_rate(records: Sequence[Mapping[str, Any]]) -> float:
    task_rates = {}  # type: Dict[Tuple[str, int], float]
    for suite in libero_eval_v4.SUPPORTED_SUITES:
        for task_id in range(TASKS_PER_SUITE_V4):
            group = [
                record
                for record in records
                if record.get("suite") == suite and record.get("task_id") == task_id
            ]
            if len(group) != TRIALS_PER_TASK_V4:
                raise ComparisonErrorV4("Success rate requires 50 rollouts per task")
            if any(type(record.get("success")) is not bool for record in group):
                raise ComparisonErrorV4(
                    "Formal comparison requires boolean episode success"
                )
            task_rates[(suite, task_id)] = sum(
                record["success"] is True for record in group
            ) / len(group)
    suite_rates = []
    for suite in libero_eval_v4.SUPPORTED_SUITES:
        suite_rates.append(
            sum(task_rates[(suite, task_id)] for task_id in range(TASKS_PER_SUITE_V4))
            / TASKS_PER_SUITE_V4
        )
    return sum(suite_rates) / len(suite_rates)


def _paired_task_deltas(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> Mapping[Tuple[str, int], Sequence[float]]:
    before_by_key = _records_by_key(before, label="paired before mode")
    after_by_key = _records_by_key(after, label="paired after mode")
    if set(before_by_key) != set(after_by_key):
        raise ComparisonErrorV4("Paired comparison rollout keys do not match")
    values = {
        (suite, task_id): []
        for suite in libero_eval_v4.SUPPORTED_SUITES
        for task_id in range(TASKS_PER_SUITE_V4)
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
    durations = []  # type: List[int]
    for record in records:
        events = record.get("inference_latencies")
        if not isinstance(events, list):
            raise ComparisonErrorV4("inference_latencies must remain a JSON list")
        for event in events:
            if type(event) is not dict:
                raise ComparisonErrorV4("inference latency must remain a JSON object")
            duration = event.get("duration_ns")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, int)
                or duration < 0
            ):
                raise ComparisonErrorV4(
                    "inference latency duration_ns must be nonnegative"
                )
            durations.append(duration)
    calibration = manifest["latency_calibration"]
    calibration_p95 = None
    if calibration is not None:
        calibration_p95 = calibration["p95_latency_ns"]
    return {
        "inference_latency_count": len(durations),
        "inference_latency_ns_total": sum(durations),
        "inference_latency_ns_mean": (
            sum(durations) / len(durations) if durations else None
        ),
        "inference_latency_ns_median": (
            statistics.median(durations) if durations else None
        ),
        "inference_latency_ns_min": min(durations) if durations else None,
        "inference_latency_ns_max": max(durations) if durations else None,
        "calibration_p95_latency_ns": calibration_p95,
    }


def _paired_delta_report(
    before: RunDataV4,
    after: RunDataV4,
) -> Mapping[str, Any]:
    paired = _paired_task_deltas(before.records, after.records)
    try:
        observed = _legacy_bootstrap.hierarchical_delta(paired)
        low, high = _legacy_bootstrap.stratified_paired_bootstrap(paired)
    except ValueError as error:
        raise ComparisonErrorV4(
            "Hierarchical bootstrap failed: {}".format(error)
        ) from error
    return {
        "before_mode": before.execution_mode,
        "after_mode": after.execution_mode,
        "observed_delta": observed,
        "bootstrap_95_ci": [low, high],
        "bootstrap_seed": _legacy_bootstrap.BOOTSTRAP_SEED,
        "bootstrap_resamples": _legacy_bootstrap.BOOTSTRAP_RESAMPLES,
        "hierarchy": "paired rollout -> task -> suite -> four-suite macro",
    }


def compare_checkpoint_v4(
    runs_by_mode: Mapping[str, RunDataV4],
) -> Mapping[str, Any]:
    """Compare the two within-family async/sync pairs at one checkpoint."""
    if not isinstance(runs_by_mode, Mapping) or set(runs_by_mode) != set(
        EXECUTION_MODE_ORDER_V4
    ):
        raise ComparisonErrorV4(
            "Checkpoint comparison requires exactly four runs by mode"
        )
    ordered = {}  # type: Dict[str, RunDataV4]
    for mode in EXECUTION_MODE_ORDER_V4:
        run = runs_by_mode[mode]
        if not isinstance(run, RunDataV4):
            raise ComparisonErrorV4("runs_by_mode values must be RunDataV4")
        if run.execution_mode != mode:
            raise ComparisonErrorV4("Run key does not match its execution_mode")
        ordered[mode] = run
    classified = classify_checkpoint_manifests_v4(
        [ordered[mode].manifest for mode in EXECUTION_MODE_ORDER_V4]
    )
    if any(
        ordered[mode].manifest["checkpoint_step"]
        != classified[mode]["checkpoint_step"]
        for mode in ordered
    ):
        raise ComparisonErrorV4("Run manifests changed during classification")
    _validate_rollout_identities(ordered)
    success_rates = {
        mode: _success_rate(ordered[mode].records)
        for mode in EXECUTION_MODE_ORDER_V4
    }
    primary_deltas = {
        "baseline_rtc_minus_baseline_sync_n5": _paired_delta_report(
            ordered["baseline_sync_n5"],
            ordered["baseline_rtc"],
        ),
        "bsp_spline_async_minus_bsp_spline_sync": _paired_delta_report(
            ordered["bsp_spline_sync"],
            ordered["bsp_spline_async"],
        ),
    }
    return {
        "schema_version": 4,
        "checkpoint_step": next(iter(classified.values()))["checkpoint_step"],
        "success_rates": success_rates,
        "primary_paired_deltas": primary_deltas,
        "diagnostics": {
            mode: _latency_diagnostics(ordered[mode].records, classified[mode])
            for mode in EXECUTION_MODE_ORDER_V4
        },
        "inputs": {
            mode: {
                "run_directory": str(ordered[mode].path),
                "file_sha256": dict(sorted(ordered[mode].file_sha256.items())),
            }
            for mode in EXECUTION_MODE_ORDER_V4
        },
    }


def _render_report_v4(checkpoints: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# LIBERO schema-v4 four-mode comparison",
        "",
        (
            "| step | baseline sync | baseline RTC | BSP sync | BSP async | "
            "RTC-sync | BSP async-sync |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in checkpoints:
        rates = checkpoint["success_rates"]
        deltas = checkpoint["primary_paired_deltas"]
        lines.append(
            (
                "| {step} | {base_sync:.6f} | {rtc:.6f} | {bsp_sync:.6f} | "
                "{bsp_async:.6f} | {rtc_delta:.6f} | {bsp_delta:.6f} |"
            ).format(
                step=checkpoint["checkpoint_step"],
                base_sync=rates["baseline_sync_n5"],
                rtc=rates["baseline_rtc"],
                bsp_sync=rates["bsp_spline_sync"],
                bsp_async=rates["bsp_spline_async"],
                rtc_delta=deltas["baseline_rtc_minus_baseline_sync_n5"][
                    "observed_delta"
                ],
                bsp_delta=deltas["bsp_spline_async_minus_bsp_spline_sync"][
                    "observed_delta"
                ],
            )
        )
    lines.extend(
        [
            "",
            (
                "Primary deltas use the fixed seed-42, 10,000-resample paired "
                "rollout → task → suite → four-suite hierarchical bootstrap."
            ),
            (
                "Calibration p95 values are copied from validated manifests and "
                "are not recomputed by this reporter."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _learning_rows_v4(
    checkpoints: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    rows = []  # type: List[Mapping[str, Any]]
    for checkpoint in checkpoints:
        rates = checkpoint["success_rates"]
        rtc = checkpoint["primary_paired_deltas"][
            "baseline_rtc_minus_baseline_sync_n5"
        ]
        bsp = checkpoint["primary_paired_deltas"][
            "bsp_spline_async_minus_bsp_spline_sync"
        ]
        rows.append(
            {
                "checkpoint_step": checkpoint["checkpoint_step"],
                "baseline_sync_n5_success_rate": rates["baseline_sync_n5"],
                "baseline_rtc_success_rate": rates["baseline_rtc"],
                "bsp_spline_sync_success_rate": rates["bsp_spline_sync"],
                "bsp_spline_async_success_rate": rates["bsp_spline_async"],
                "baseline_rtc_minus_baseline_sync_n5": rtc["observed_delta"],
                "baseline_rtc_bootstrap_95_ci_low": rtc["bootstrap_95_ci"][0],
                "baseline_rtc_bootstrap_95_ci_high": rtc["bootstrap_95_ci"][1],
                "bsp_spline_async_minus_bsp_spline_sync": bsp["observed_delta"],
                "bsp_spline_async_bootstrap_95_ci_low": bsp["bootstrap_95_ci"][0],
                "bsp_spline_async_bootstrap_95_ci_high": bsp["bootstrap_95_ci"][1],
            }
        )
    return rows


def write_five_checkpoint_report_v4(
    run_dirs: Sequence[Any],
    *,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Write a single- or fixed-five-checkpoint report using only ``*_v4`` files."""
    if isinstance(run_dirs, (str, bytes)) or not isinstance(run_dirs, Sequence):
        raise ComparisonErrorV4("run_dirs must be a sequence")
    if len(run_dirs) not in (4, 20):
        raise ComparisonErrorV4("Report requires exactly four or exactly 20 runs")
    if all(isinstance(value, RunDataV4) for value in run_dirs):
        runs = list(run_dirs)
    elif any(isinstance(value, RunDataV4) for value in run_dirs):
        raise ComparisonErrorV4("Do not mix RunDataV4 objects and run directories")
    else:
        resolved = [Path(value).expanduser().resolve() for value in run_dirs]
        if len(set(resolved)) != len(resolved):
            raise ComparisonErrorV4("Run directories must be unique")
        runs = [load_run_v4(path) for path in resolved]

    manifests = [run.manifest for run in runs]
    if len(runs) == 20:
        classify_five_checkpoint_manifests_v4(manifests)
        steps = MILESTONES_V4
    else:
        one = classify_checkpoint_manifests_v4(manifests)
        step = int(next(iter(one.values()))["checkpoint_step"])
        steps = (step,)
    runs_by_key = {}  # type: Dict[Tuple[int, str], RunDataV4]
    for run in runs:
        key = (run.checkpoint_step, run.execution_mode)
        if key in runs_by_key:
            raise ComparisonErrorV4("Duplicate run for checkpoint/mode {}".format(key))
        runs_by_key[key] = run
    expected_keys = {
        (step, mode)
        for step in steps
        for mode in EXECUTION_MODE_ORDER_V4
    }
    if set(runs_by_key) != expected_keys:
        raise ComparisonErrorV4(
            "Loaded runs do not match the classified checkpoint/mode grid"
        )
    checkpoints = [
        compare_checkpoint_v4(
            {mode: runs_by_key[(step, mode)] for mode in EXECUTION_MODE_ORDER_V4}
        )
        for step in steps
    ]
    report = {
        "schema_version": 4,
        "protocol": {
            "checkpoint_steps": list(steps),
            "execution_modes": list(EXECUTION_MODE_ORDER_V4),
            "suites": list(libero_eval_v4.SUPPORTED_SUITES),
            "tasks_per_suite": TASKS_PER_SUITE_V4,
            "trials_per_task": TRIALS_PER_TASK_V4,
            "episodes_per_run": EPISODES_PER_RUN_V4,
            "primary_deltas": [
                "baseline_rtc_minus_baseline_sync_n5",
                "bsp_spline_async_minus_bsp_spline_sync",
            ],
            "bootstrap_seed": _legacy_bootstrap.BOOTSTRAP_SEED,
            "bootstrap_resamples": _legacy_bootstrap.BOOTSTRAP_RESAMPLES,
        },
        "checkpoints": checkpoints,
    }

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise ComparisonErrorV4("Comparison output path must be a directory")
        if any(output.iterdir()):
            raise ComparisonErrorV4(
                "Comparison output directory must be new or empty"
            )
    output.mkdir(parents=True, exist_ok=True)
    try:
        comparison_text = libero_artifacts.json_text(report)
    except (TypeError, ValueError) as error:
        raise ComparisonErrorV4("Schema-v4 report contains a non-JSON value") from error
    libero_artifacts.atomic_text(output / "comparison_v4.json", comparison_text)
    libero_artifacts.write_csv(
        output / "learning_curve_v4.csv",
        _learning_rows_v4(checkpoints),
    )
    libero_artifacts.atomic_text(
        output / "report_v4.md",
        _render_report_v4(checkpoints),
    )
    actual_outputs = {path.name for path in output.iterdir()}
    if actual_outputs != set(OUTPUT_FILENAMES_V4):
        raise ComparisonErrorV4(
            "Unexpected schema-v4 report output set: {}".format(sorted(actual_outputs))
        )
    return report
