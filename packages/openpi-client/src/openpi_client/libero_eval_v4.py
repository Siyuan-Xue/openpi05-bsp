"""Strict schema-v4 LIBERO evaluation records and artifact persistence.

Schema-v4 is intentionally independent of the schema-v2/v3 producer and
writer.  Only rollout identities, deterministic flow seeds, and failure
classification are shared where their behavior is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from openpi_client import libero_artifacts
from openpi_client import libero_control_v4 as _control
from openpi_client import libero_eval as _legacy_identity
from openpi_client import libero_video_timing_v4 as _timing


EpisodeIdentity = _legacy_identity.EpisodeIdentity
InfrastructureFailure = _legacy_identity.InfrastructureFailure
PolicyFailure = _legacy_identity.PolicyFailure
classify_exception = _legacy_identity.classify_exception
fingerprint_init_state = _legacy_identity.fingerprint_init_state
stable_replan_seed = _legacy_identity.stable_replan_seed

SUPPORTED_SUITES = _legacy_identity.SUPPORTED_SUITES
MAX_STEPS_BY_SUITE = MappingProxyType(
    {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }
)
BSP_PARAMETERS = MappingProxyType(dict(_legacy_identity.BSP_PARAMETERS))

SCHEMA_VERSION = 4
DATASET_FPS = 10
SOURCE_DEMO_CONTROL_HZ = 20
CONTROL_FREQ_HZ = 20
CONTROLLER_PERIOD_NS = 50_000_000
VIDEO_FPS = 40
DATASET_REVISION = "v2.0"
INFRASTRUCTURE_RETRIES = 2

_INFRASTRUCTURE_KINDS = frozenset(("simulator", "container", "network"))
_FAILURE_KINDS = frozenset(("policy", "timeout"))
_STATUSES = frozenset(
    ("success", "policy_failure", "timeout_failure", "infrastructure_incomplete")
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CODE_SHA_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_CONTAINER_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

_MANIFEST_FIELDS = frozenset(
    (
        "schema_version",
        "dataset_fps",
        "source_demo_control_hz",
        "control_freq_hz",
        "controller_period_ns",
        "video_fps",
        "video_show_inference_waits",
        "execution_mode",
        "execution_parameters",
        "latency_calibration",
        "server_metadata_fingerprint",
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
        "suites",
        "task_ids",
        "trials_per_task",
        "num_steps_wait",
        "max_steps_by_suite",
        "connection_timeout_s",
        "inference_timeout_s",
        "infrastructure_retries",
    )
)
_EPISODE_FIELDS = frozenset(
    (
        "schema_version",
        "episode_id",
        "paired_key",
        "suite",
        "task_id",
        "task_name",
        "init_state_index",
        "init_state_fingerprint",
        "eval_seed",
        "execution_mode",
        "status",
        "success",
        "include_in_success_rate",
        "attempts",
        "failure_kind",
        "infrastructure_kind",
        "error",
        "steps",
        "replans",
        "episode_duration_ns",
        "inference_requests",
        "inference_latencies",
        "plan_activations",
        "action_underflows",
        "control_stalls",
        "infrastructure_history",
    )
)
_INFRASTRUCTURE_HISTORY_FIELDS = frozenset(("attempt", "kind", "error"))
_ARTIFACT_ERROR_FIELDS = frozenset(("episode_id", "artifact_type", "path", "error"))
_VIDEO_ARTIFACT_FIELDS = frozenset(
    (
        "schema_version",
        "episode_id",
        "execution_mode",
        "path",
        "video_show_inference_waits",
        "planned",
        "encoded_fps",
        "encoded_frame_count",
        "encoded_duration_ns",
        "artifact_padding_frame_count",
        "timing_gate_pass",
        "warning",
    )
)


def _require_json_object(value: Any, *, label: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("{} must be a JSON object".format(label))
    if any(not isinstance(key, str) for key in value):
        raise ValueError("{} keys must be strings".format(label))
    return value


def _require_exact_fields(value: Any, fields: frozenset, *, label: str) -> Dict[str, Any]:
    payload = _require_json_object(value, label=label)
    if set(payload) != fields:
        missing = sorted(fields.difference(payload))
        extra = sorted(set(payload).difference(fields))
        raise ValueError("{} fields mismatch; missing={}, extra={}".format(label, missing, extra))
    return payload


def _require_integer(
    value: Any,
    *,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(name))
    if minimum is not None and value < minimum:
        raise ValueError("{} must be at least {}".format(name, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("{} must be at most {}".format(name, maximum))
    return value


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    return _require_integer(value, name=name, minimum=0)


def _require_nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be a non-empty string".format(name))
    return value


def _require_optional_text(value: Any, *, name: str) -> Optional[str]:
    if value is not None:
        _require_nonempty_text(value, name=name)
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA256".format(name))
    return value


def _require_timeout(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("{} must be positive and finite".format(name))
    return float(value)


def _freeze_json(value: Any, *, label: str) -> Any:
    if isinstance(value, Mapping):
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("{} keys must be strings".format(label))
            copied[key] = _freeze_json(item, label="{}.{}".format(label, key))
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, label=label) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("{} contains a nonfinite value".format(label))
        return value
    raise ValueError("{} contains an unsupported JSON value".format(label))


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _json_values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_json_values_match(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, tuple)
            and len(actual) == len(expected)
            and all(
                _json_values_match(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _json_wire_values_match(actual: Any, expected: Any) -> bool:
    """Require JSON container identity before constructor normalization."""
    if isinstance(expected, dict):
        return (
            type(actual) is dict
            and set(actual) == set(expected)
            and all(_json_wire_values_match(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _json_wire_values_match(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _record_tuple(values: Any, record_type: Any, *, label: str) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("{} must be a sequence".format(label))
    records = tuple(values)
    if any(not isinstance(value, record_type) for value in records):
        raise ValueError("{} must contain only {}".format(label, record_type.__name__))
    return records


def _freeze_infrastructure_history(values: Any) -> Tuple[Mapping[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("infrastructure_history must be a sequence")
    result = []  # type: List[Mapping[str, Any]]
    previous_attempt = 0
    for entry in values:
        if not isinstance(entry, Mapping) or set(entry) != _INFRASTRUCTURE_HISTORY_FIELDS:
            raise ValueError("infrastructure history entries must have exact fields")
        attempt = _require_integer(entry["attempt"], name="history attempt", minimum=1)
        if attempt <= previous_attempt:
            raise ValueError("infrastructure history attempts must be strictly increasing")
        kind = entry["kind"]
        if not isinstance(kind, str) or kind not in _INFRASTRUCTURE_KINDS:
            raise ValueError("unsupported infrastructure history kind")
        error = _require_nonempty_text(entry["error"], name="infrastructure history error")
        result.append(
            MappingProxyType({"attempt": attempt, "kind": kind, "error": error})
        )
        previous_attempt = attempt
    return tuple(result)


def _validate_stall_source_frames(values: Any, *, steps: int) -> Tuple[Tuple[int, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("stall_source_frames must be a sequence")
    result = []  # type: List[Tuple[int, Any]]
    previous_step = -1
    for value in values:
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("stall_source_frames must contain (control_step, frame) tuples")
        control_step = _require_nonnegative_integer(
            value[0], name="stall source control_step"
        )
        if control_step <= previous_step or control_step > steps:
            raise ValueError("stall source control steps must be ordered within 0..steps")
        result.append((control_step, value[1]))
        previous_step = control_step
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class EvaluationManifestV4:
    schema_version: int
    dataset_fps: int
    source_demo_control_hz: int
    control_freq_hz: int
    controller_period_ns: int
    video_fps: int
    video_show_inference_waits: bool
    execution_mode: str
    execution_parameters: Mapping[str, Any]
    latency_calibration: Optional[_control.LatencyCalibrationV1]
    server_metadata_fingerprint: str
    code_sha: str
    dataset_revision: str
    config_name: str
    checkpoint_step: int
    bsp_cache_hash: Optional[str]
    bsp_cache_manifest_fingerprint: Optional[str]
    norm_hash: str
    checkpoint: str
    container_digest: str
    train_seed: int
    eval_seed: int
    policy_variant: str
    bsp_parameters: Mapping[str, Any]
    policy_protocol: str
    expected_action_horizon: int
    suites: Sequence[str]
    task_ids: Sequence[int]
    trials_per_task: int
    num_steps_wait: int
    max_steps_by_suite: Mapping[str, int]
    connection_timeout_s: float
    inference_timeout_s: float
    infrastructure_retries: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_parameters",
            _freeze_json(self.execution_parameters, label="execution_parameters"),
        )
        object.__setattr__(
            self,
            "bsp_parameters",
            _freeze_json(self.bsp_parameters, label="bsp_parameters"),
        )
        if isinstance(self.suites, (str, bytes)) or not isinstance(self.suites, Sequence):
            raise ValueError("suites must be a sequence")
        if isinstance(self.task_ids, (str, bytes)) or not isinstance(self.task_ids, Sequence):
            raise ValueError("task_ids must be a sequence")
        object.__setattr__(self, "suites", tuple(self.suites))
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        if not isinstance(self.max_steps_by_suite, Mapping):
            raise ValueError("max_steps_by_suite must be a mapping")
        max_steps = dict(self.max_steps_by_suite)
        if any(not isinstance(key, str) for key in max_steps):
            raise ValueError("max_steps_by_suite keys must be strings")
        object.__setattr__(self, "max_steps_by_suite", MappingProxyType(max_steps))
        self._validate()

    def _validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("manifest schema_version must be integer 4")
        for name, actual, expected in (
            ("dataset_fps", self.dataset_fps, DATASET_FPS),
            ("source_demo_control_hz", self.source_demo_control_hz, SOURCE_DEMO_CONTROL_HZ),
            ("control_freq_hz", self.control_freq_hz, CONTROL_FREQ_HZ),
            ("controller_period_ns", self.controller_period_ns, CONTROLLER_PERIOD_NS),
            ("video_fps", self.video_fps, VIDEO_FPS),
            ("infrastructure_retries", self.infrastructure_retries, INFRASTRUCTURE_RETRIES),
        ):
            if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
                raise ValueError("{} must be exactly {}".format(name, expected))
        _timing.validate_video_frequencies_v4(
            control_hz=self.control_freq_hz,
            video_fps=self.video_fps,
        )
        if type(self.video_show_inference_waits) is not bool:
            raise ValueError("video_show_inference_waits must be boolean")
        try:
            mode = _control.EXECUTION_MODES[self.execution_mode]
        except (KeyError, TypeError) as error:
            raise ValueError("unsupported schema-v4 execution_mode") from error
        expected_parameters = mode.to_parameters_dict()
        if not _json_values_match(self.execution_parameters, expected_parameters):
            raise ValueError("execution_parameters do not match the frozen mode table")
        _require_sha256(
            self.server_metadata_fingerprint,
            name="server_metadata_fingerprint",
        )
        if not isinstance(self.code_sha, str) or _CODE_SHA_PATTERN.fullmatch(self.code_sha) is None:
            raise ValueError("code_sha must be a lowercase 40- or 64-character Git SHA")
        if self.dataset_revision != DATASET_REVISION:
            raise ValueError("dataset_revision must be v2.0")
        _require_nonempty_text(self.config_name, name="config_name")
        _require_nonnegative_integer(self.checkpoint_step, name="checkpoint_step")
        _require_sha256(self.norm_hash, name="norm_hash")
        _require_nonempty_text(self.checkpoint, name="checkpoint")
        if (
            not isinstance(self.container_digest, str)
            or _CONTAINER_DIGEST_PATTERN.fullmatch(self.container_digest) is None
        ):
            raise ValueError("container_digest must be sha256: plus 64 lowercase hex")
        _require_nonnegative_integer(self.train_seed, name="train_seed")
        _require_nonnegative_integer(self.eval_seed, name="eval_seed")
        checkpoint_identity = _control.CheckpointIdentityV1(
            code_sha=self.code_sha,
            config_name=self.config_name,
            checkpoint_step=self.checkpoint_step,
            checkpoint=self.checkpoint,
            container_digest=self.container_digest,
            norm_hash=self.norm_hash,
            bsp_cache_hash=self.bsp_cache_hash,
            bsp_cache_manifest_fingerprint=self.bsp_cache_manifest_fingerprint,
        )
        if checkpoint_identity.checkpoint != self.checkpoint:
            raise ValueError("checkpoint must use the normalized checkpoint identity path")
        if mode.policy_variant == "baseline":
            if self.bsp_cache_hash is not None or self.bsp_cache_manifest_fingerprint is not None:
                raise ValueError("baseline modes require null BSP cache identities")
        elif self.bsp_cache_hash is None or self.bsp_cache_manifest_fingerprint is None:
            raise ValueError("BSP modes require both BSP cache identities")
        if (
            self.policy_variant != mode.policy_variant
            or self.policy_protocol != mode.policy_protocol
            or self.expected_action_horizon != mode.expected_action_horizon
        ):
            raise ValueError("manifest policy fields do not match execution_mode")
        _require_integer(
            self.expected_action_horizon,
            name="expected_action_horizon",
            minimum=1,
        )
        if not _json_values_match(self.bsp_parameters, dict(BSP_PARAMETERS)):
            raise ValueError("bsp_parameters must match the frozen LIBERO BSP identity")
        calibration = self.latency_calibration
        if mode.asynchronous:
            if not isinstance(calibration, _control.LatencyCalibrationV1):
                raise ValueError("asynchronous modes require LatencyCalibrationV1")
            calibration.to_dict()
            if calibration.execution_mode != self.execution_mode:
                raise ValueError("calibration execution_mode does not match manifest")
            if calibration.server_metadata_fingerprint != self.server_metadata_fingerprint:
                raise ValueError("calibration server fingerprint does not match manifest")
            if calibration.checkpoint_identity_fingerprint != checkpoint_identity.fingerprint:
                raise ValueError("calibration checkpoint fingerprint does not match manifest")
        elif self.latency_calibration is not None:
            raise ValueError("synchronous modes require null latency_calibration")

        suites = tuple(self.suites)
        if suites not in tuple((suite,) for suite in SUPPORTED_SUITES) + (SUPPORTED_SUITES,):
            raise ValueError("suites must be one suite or all four in canonical order")
        task_ids = tuple(
            _require_integer(value, name="task_id", minimum=0, maximum=9)
            for value in self.task_ids
        )
        if not task_ids or task_ids != tuple(sorted(set(task_ids))):
            raise ValueError("task_ids must be non-empty, unique, and sorted")
        if mode.asynchronous:
            if not isinstance(calibration, _control.LatencyCalibrationV1):
                raise ValueError("asynchronous modes require LatencyCalibrationV1")
            canonical = calibration.canonical_observation_identity
            if canonical.suite != suites[0]:
                raise ValueError(
                    "calibration canonical observation suite must match suites[0]"
                )
            if canonical.task_id != task_ids[0]:
                raise ValueError(
                    "calibration canonical observation task_id must match task_ids[0]"
                )
            if canonical.init_state_index != 0:
                raise ValueError(
                    "calibration canonical observation init_state_index must be zero"
                )
        _require_integer(self.trials_per_task, name="trials_per_task", minimum=1)
        _require_nonnegative_integer(self.num_steps_wait, name="num_steps_wait")
        if set(self.max_steps_by_suite) != set(suites):
            raise ValueError("max_steps_by_suite must exactly match selected suites")
        for suite, value in self.max_steps_by_suite.items():
            expected = MAX_STEPS_BY_SUITE[suite]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected
            ):
                raise ValueError("max steps for {} must be {}".format(suite, expected))
        _require_timeout(self.connection_timeout_s, name="connection_timeout_s")
        _require_timeout(self.inference_timeout_s, name="inference_timeout_s")

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "schema_version": self.schema_version,
            "dataset_fps": self.dataset_fps,
            "source_demo_control_hz": self.source_demo_control_hz,
            "control_freq_hz": self.control_freq_hz,
            "controller_period_ns": self.controller_period_ns,
            "video_fps": self.video_fps,
            "video_show_inference_waits": self.video_show_inference_waits,
            "execution_mode": self.execution_mode,
            "execution_parameters": _thaw_json(self.execution_parameters),
            "latency_calibration": (
                self.latency_calibration.to_dict()
                if self.latency_calibration is not None
                else None
            ),
            "server_metadata_fingerprint": self.server_metadata_fingerprint,
            "code_sha": self.code_sha,
            "dataset_revision": self.dataset_revision,
            "config_name": self.config_name,
            "checkpoint_step": self.checkpoint_step,
            "bsp_cache_hash": self.bsp_cache_hash,
            "bsp_cache_manifest_fingerprint": self.bsp_cache_manifest_fingerprint,
            "norm_hash": self.norm_hash,
            "checkpoint": self.checkpoint,
            "container_digest": self.container_digest,
            "train_seed": self.train_seed,
            "eval_seed": self.eval_seed,
            "policy_variant": self.policy_variant,
            "bsp_parameters": _thaw_json(self.bsp_parameters),
            "policy_protocol": self.policy_protocol,
            "expected_action_horizon": self.expected_action_horizon,
            "suites": list(self.suites),
            "task_ids": list(self.task_ids),
            "trials_per_task": self.trials_per_task,
            "num_steps_wait": self.num_steps_wait,
            "max_steps_by_suite": dict(self.max_steps_by_suite),
            "connection_timeout_s": self.connection_timeout_s,
            "inference_timeout_s": self.inference_timeout_s,
            "infrastructure_retries": self.infrastructure_retries,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationManifestV4":
        payload = dict(_require_exact_fields(value, _MANIFEST_FIELDS, label="v4 manifest"))
        _require_json_object(payload["execution_parameters"], label="execution_parameters")
        _require_json_object(payload["bsp_parameters"], label="bsp_parameters")
        _require_json_object(payload["max_steps_by_suite"], label="max_steps_by_suite")
        if not isinstance(payload["suites"], list):
            raise ValueError("suites must be a JSON list")
        if not isinstance(payload["task_ids"], list):
            raise ValueError("task_ids must be a JSON list")
        mode = _control.EXECUTION_MODES.get(payload["execution_mode"])
        if mode is not None and not _json_wire_values_match(
            payload["execution_parameters"], mode.to_parameters_dict()
        ):
            raise ValueError("execution_parameters must use the exact JSON container types")
        calibration_payload = payload["latency_calibration"]
        if calibration_payload is not None:
            _require_json_object(calibration_payload, label="latency_calibration")
            payload["latency_calibration"] = _control.LatencyCalibrationV1.from_dict(
                calibration_payload
            )
        return cls(**payload)


@dataclasses.dataclass(frozen=True)
class AttemptResultV4:
    execution_mode: str
    success: bool
    steps: int
    replans: int
    episode_duration_ns: int
    failure_kind: Optional[str] = None
    error: Optional[str] = None
    inference_requests: Tuple[_timing.RequestEventV4, ...] = ()
    inference_latencies: Tuple[_timing.LatencyEventV4, ...] = ()
    plan_activations: Tuple[_timing.PlanActivationV4, ...] = ()
    action_underflows: Tuple[_timing.ActionUnderflowV4, ...] = ()
    control_stalls: Tuple[_timing.ControlStallV4, ...] = ()
    replay_frames: Tuple[Any, ...] = dataclasses.field(default=(), repr=False, compare=False)
    stall_source_frames: Tuple[Tuple[int, Any], ...] = dataclasses.field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inference_requests",
            _record_tuple(
                self.inference_requests,
                _timing.RequestEventV4,
                label="inference_requests",
            ),
        )
        object.__setattr__(
            self,
            "inference_latencies",
            _record_tuple(
                self.inference_latencies,
                _timing.LatencyEventV4,
                label="inference_latencies",
            ),
        )
        object.__setattr__(
            self,
            "plan_activations",
            _record_tuple(
                self.plan_activations,
                _timing.PlanActivationV4,
                label="plan_activations",
            ),
        )
        object.__setattr__(
            self,
            "action_underflows",
            _record_tuple(
                self.action_underflows,
                _timing.ActionUnderflowV4,
                label="action_underflows",
            ),
        )
        object.__setattr__(
            self,
            "control_stalls",
            _record_tuple(
                self.control_stalls,
                _timing.ControlStallV4,
                label="control_stalls",
            ),
        )
        if isinstance(self.replay_frames, (str, bytes)) or not isinstance(
            self.replay_frames, Sequence
        ):
            raise ValueError("replay_frames must be a sequence")
        object.__setattr__(self, "replay_frames", tuple(self.replay_frames))
        object.__setattr__(
            self,
            "stall_source_frames",
            _validate_stall_source_frames(self.stall_source_frames, steps=self.steps),
        )
        self._validate()

    def _validate(self) -> None:
        if self.execution_mode not in _control.EXECUTION_MODES:
            raise ValueError("unsupported schema-v4 execution_mode")
        if type(self.success) is not bool:
            raise ValueError("attempt success must be boolean")
        _require_nonnegative_integer(self.steps, name="steps")
        _require_nonnegative_integer(self.replans, name="replans")
        _require_nonnegative_integer(self.episode_duration_ns, name="episode_duration_ns")
        if self.success:
            if self.failure_kind is not None or self.error is not None:
                raise ValueError("successful attempts cannot carry failure fields")
        elif self.failure_kind not in _FAILURE_KINDS:
            raise ValueError("failed attempts must be policy or timeout failures")
        _require_optional_text(self.error, name="attempt error")
        if self.replans != len(self.plan_activations):
            raise ValueError("replans must equal plan activation count")
        if not self.inference_requests:
            if any(
                (
                    self.inference_latencies,
                    self.plan_activations,
                    self.action_underflows,
                    self.control_stalls,
                )
            ):
                raise ValueError("timing events cannot exist without a request")
            if self.success:
                raise ValueError("a successful attempt must contain an initial request")


@dataclasses.dataclass(frozen=True)
class EpisodeRecordV4:
    schema_version: int
    episode_id: str
    paired_key: str
    suite: str
    task_id: int
    task_name: str
    init_state_index: int
    init_state_fingerprint: str
    eval_seed: int
    execution_mode: str
    status: str
    success: Optional[bool]
    include_in_success_rate: bool
    attempts: int
    failure_kind: Optional[str]
    infrastructure_kind: Optional[str]
    error: Optional[str]
    steps: int
    replans: int
    episode_duration_ns: int
    inference_requests: Tuple[_timing.RequestEventV4, ...]
    inference_latencies: Tuple[_timing.LatencyEventV4, ...]
    plan_activations: Tuple[_timing.PlanActivationV4, ...]
    action_underflows: Tuple[_timing.ActionUnderflowV4, ...]
    control_stalls: Tuple[_timing.ControlStallV4, ...]
    infrastructure_history: Tuple[Mapping[str, Any], ...]
    replay_frames: Tuple[Any, ...] = dataclasses.field(default=(), repr=False, compare=False)
    stall_source_frames: Tuple[Tuple[int, Any], ...] = dataclasses.field(
        default=(), repr=False, compare=False
    )
    expected_bsp_prefetch_budget_ns: Optional[int] = dataclasses.field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inference_requests",
            _record_tuple(
                self.inference_requests,
                _timing.RequestEventV4,
                label="inference_requests",
            ),
        )
        object.__setattr__(
            self,
            "inference_latencies",
            _record_tuple(
                self.inference_latencies,
                _timing.LatencyEventV4,
                label="inference_latencies",
            ),
        )
        object.__setattr__(
            self,
            "plan_activations",
            _record_tuple(
                self.plan_activations,
                _timing.PlanActivationV4,
                label="plan_activations",
            ),
        )
        object.__setattr__(
            self,
            "action_underflows",
            _record_tuple(
                self.action_underflows,
                _timing.ActionUnderflowV4,
                label="action_underflows",
            ),
        )
        object.__setattr__(
            self,
            "control_stalls",
            _record_tuple(
                self.control_stalls,
                _timing.ControlStallV4,
                label="control_stalls",
            ),
        )
        object.__setattr__(
            self,
            "infrastructure_history",
            _freeze_infrastructure_history(self.infrastructure_history),
        )
        if isinstance(self.replay_frames, (str, bytes)) or not isinstance(
            self.replay_frames, Sequence
        ):
            raise ValueError("replay_frames must be a sequence")
        object.__setattr__(self, "replay_frames", tuple(self.replay_frames))
        object.__setattr__(
            self,
            "stall_source_frames",
            _validate_stall_source_frames(self.stall_source_frames, steps=self.steps),
        )
        self._validate()

    @property
    def identity(self) -> EpisodeIdentity:
        return EpisodeIdentity(
            suite=self.suite,
            task_id=self.task_id,
            task_name=self.task_name,
            init_state_index=self.init_state_index,
            init_state_fingerprint=self.init_state_fingerprint,
        )

    def _validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("episode schema_version must be integer 4")
        _require_nonempty_text(self.episode_id, name="episode_id")
        _require_nonempty_text(self.paired_key, name="paired_key")
        _require_nonempty_text(self.task_name, name="task_name")
        _require_nonnegative_integer(self.task_id, name="task_id")
        _require_nonnegative_integer(self.init_state_index, name="init_state_index")
        _require_sha256(self.init_state_fingerprint, name="init_state_fingerprint")
        identity = self.identity
        if self.episode_id != identity.episode_id or self.paired_key != identity.paired_key:
            raise ValueError("episode_id and paired_key must match the rollout identity")
        _require_nonnegative_integer(self.eval_seed, name="eval_seed")
        if self.execution_mode not in _control.EXECUTION_MODES:
            raise ValueError("unsupported schema-v4 execution_mode")
        if self.execution_mode == "bsp_spline_async":
            if self.expected_bsp_prefetch_budget_ns is not None:
                _require_nonnegative_integer(
                    self.expected_bsp_prefetch_budget_ns,
                    name="expected_bsp_prefetch_budget_ns",
                )
        elif self.expected_bsp_prefetch_budget_ns is not None:
            raise ValueError("only bsp_spline_async accepts a prefetch budget")
        if self.status not in _STATUSES:
            raise ValueError("unsupported episode status")
        if self.success is not None and type(self.success) is not bool:
            raise ValueError("success must be boolean or null")
        if type(self.include_in_success_rate) is not bool:
            raise ValueError("include_in_success_rate must be boolean")
        _require_integer(self.attempts, name="attempts", minimum=1, maximum=3)
        _require_nonnegative_integer(self.steps, name="steps")
        _require_nonnegative_integer(self.replans, name="replans")
        _require_nonnegative_integer(self.episode_duration_ns, name="episode_duration_ns")
        _require_optional_text(self.error, name="episode error")
        if self.status == "success":
            expected = (True, True, None, None)
        elif self.status == "policy_failure":
            expected = (False, True, "policy", None)
        elif self.status == "timeout_failure":
            expected = (False, True, "timeout", None)
        else:
            expected = (None, False, None, self.infrastructure_kind)
        actual = (
            self.success,
            self.include_in_success_rate,
            self.failure_kind,
            self.infrastructure_kind,
        )
        if actual != expected:
            raise ValueError("episode status fields are inconsistent")
        if (
            self.infrastructure_kind is not None
            and self.infrastructure_kind not in _INFRASTRUCTURE_KINDS
        ):
            raise ValueError("unsupported infrastructure_kind")
        if self.status == "infrastructure_incomplete":
            if self.error is None or not self.infrastructure_history:
                raise ValueError("infrastructure-incomplete episodes require history and error")
            if tuple(entry["attempt"] for entry in self.infrastructure_history) != tuple(
                range(1, self.attempts + 1)
            ):
                raise ValueError("infrastructure-incomplete history must cover every attempt")
            final_history = self.infrastructure_history[-1]
            if (
                final_history["attempt"] != self.attempts
                or final_history["kind"] != self.infrastructure_kind
                or final_history["error"] != self.error
            ):
                raise ValueError("final infrastructure history must match episode failure")
            if any(
                (
                    self.steps,
                    self.replans,
                    self.episode_duration_ns,
                    self.inference_requests,
                    self.inference_latencies,
                    self.plan_activations,
                    self.action_underflows,
                    self.control_stalls,
                    self.replay_frames,
                    self.stall_source_frames,
                )
            ):
                raise ValueError("infrastructure-incomplete episodes must have empty metrics")
            return
        if tuple(entry["attempt"] for entry in self.infrastructure_history) != tuple(
            range(1, self.attempts)
        ):
            raise ValueError("retry history must cover every preceding attempt")
        if self.status == "success" and self.error is not None:
            raise ValueError("successful episodes cannot carry an error")
        if self.status == "policy_failure" and self.error is None:
            raise ValueError("policy failures must carry an error")
        if self.replans != len(self.plan_activations):
            raise ValueError("replans must equal plan activation count")
        if self.inference_requests:
            validation_args = {
                "requests": self.inference_requests,
                "latencies": self.inference_latencies,
                "activations": self.plan_activations,
                "underflows": self.action_underflows,
                "stalls": self.control_stalls,
                "steps": self.steps,
                "episode_duration_ns": self.episode_duration_ns,
                "execution_mode": self.execution_mode,
                "eval_seed": self.eval_seed,
                "identity": identity,
            }
            if self.execution_mode == "bsp_spline_async":
                validation_args["expected_bsp_prefetch_budget_ns"] = (
                    self.expected_bsp_prefetch_budget_ns
                )
            _timing.validate_timing_events_v4(**validation_args)
        elif any(
            (
                self.inference_latencies,
                self.plan_activations,
                self.action_underflows,
                self.control_stalls,
            )
        ):
            raise ValueError("timing events cannot exist without a request")
        elif self.status == "success":
            raise ValueError("successful episodes must contain an initial request")

    @classmethod
    def from_attempt(
        cls,
        identity: EpisodeIdentity,
        eval_seed: int,
        attempts: int,
        *,
        execution_mode: str,
        result: AttemptResultV4,
        infrastructure_history: Sequence[Mapping[str, Any]] = (),
        expected_bsp_prefetch_budget_ns: Optional[int] = None,
    ) -> "EpisodeRecordV4":
        if not isinstance(identity, EpisodeIdentity):
            raise ValueError("identity must be an EpisodeIdentity")
        if not isinstance(result, AttemptResultV4):
            raise ValueError("result must be AttemptResultV4")
        if result.execution_mode != execution_mode:
            raise ValueError("attempt execution_mode does not match episode")
        failure_kind = None if result.success else result.failure_kind
        status = "success" if result.success else "{}_failure".format(failure_kind)
        return cls(
            schema_version=SCHEMA_VERSION,
            episode_id=identity.episode_id,
            paired_key=identity.paired_key,
            suite=identity.suite,
            task_id=identity.task_id,
            task_name=identity.task_name,
            init_state_index=identity.init_state_index,
            init_state_fingerprint=identity.init_state_fingerprint,
            eval_seed=eval_seed,
            execution_mode=execution_mode,
            status=status,
            success=result.success,
            include_in_success_rate=True,
            attempts=attempts,
            failure_kind=failure_kind,
            infrastructure_kind=None,
            error=result.error,
            steps=result.steps,
            replans=result.replans,
            episode_duration_ns=result.episode_duration_ns,
            inference_requests=result.inference_requests,
            inference_latencies=result.inference_latencies,
            plan_activations=result.plan_activations,
            action_underflows=result.action_underflows,
            control_stalls=result.control_stalls,
            infrastructure_history=tuple(infrastructure_history),
            replay_frames=result.replay_frames,
            stall_source_frames=result.stall_source_frames,
            expected_bsp_prefetch_budget_ns=expected_bsp_prefetch_budget_ns,
        )

    @classmethod
    def infrastructure_incomplete(
        cls,
        identity: EpisodeIdentity,
        eval_seed: int,
        attempts: int,
        *,
        execution_mode: str,
        kind: str,
        error: str,
        infrastructure_history: Sequence[Mapping[str, Any]],
    ) -> "EpisodeRecordV4":
        return cls(
            schema_version=SCHEMA_VERSION,
            episode_id=identity.episode_id,
            paired_key=identity.paired_key,
            suite=identity.suite,
            task_id=identity.task_id,
            task_name=identity.task_name,
            init_state_index=identity.init_state_index,
            init_state_fingerprint=identity.init_state_fingerprint,
            eval_seed=eval_seed,
            execution_mode=execution_mode,
            status="infrastructure_incomplete",
            success=None,
            include_in_success_rate=False,
            attempts=attempts,
            failure_kind=None,
            infrastructure_kind=kind,
            error=error,
            steps=0,
            replans=0,
            episode_duration_ns=0,
            inference_requests=(),
            inference_latencies=(),
            plan_activations=(),
            action_underflows=(),
            control_stalls=(),
            infrastructure_history=tuple(infrastructure_history),
        )

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "paired_key": self.paired_key,
            "suite": self.suite,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "init_state_index": self.init_state_index,
            "init_state_fingerprint": self.init_state_fingerprint,
            "eval_seed": self.eval_seed,
            "execution_mode": self.execution_mode,
            "status": self.status,
            "success": self.success,
            "include_in_success_rate": self.include_in_success_rate,
            "attempts": self.attempts,
            "failure_kind": self.failure_kind,
            "infrastructure_kind": self.infrastructure_kind,
            "error": self.error,
            "steps": self.steps,
            "replans": self.replans,
            "episode_duration_ns": self.episode_duration_ns,
            "inference_requests": [event.to_dict() for event in self.inference_requests],
            "inference_latencies": [event.to_dict() for event in self.inference_latencies],
            "plan_activations": [event.to_dict() for event in self.plan_activations],
            "action_underflows": [event.to_dict() for event in self.action_underflows],
            "control_stalls": [event.to_dict() for event in self.control_stalls],
            "infrastructure_history": [dict(entry) for entry in self.infrastructure_history],
        }

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        expected_bsp_prefetch_budget_ns: Optional[int] = None,
    ) -> "EpisodeRecordV4":
        payload = dict(_require_exact_fields(value, _EPISODE_FIELDS, label="v4 episode"))
        event_fields = (
            ("inference_requests", _timing.RequestEventV4),
            ("inference_latencies", _timing.LatencyEventV4),
            ("plan_activations", _timing.PlanActivationV4),
            ("action_underflows", _timing.ActionUnderflowV4),
            ("control_stalls", _timing.ControlStallV4),
        )
        for field, record_type in event_fields:
            if not isinstance(payload[field], list):
                raise ValueError("{} must be a JSON list".format(field))
            payload[field] = tuple(record_type.from_dict(item) for item in payload[field])
        if not isinstance(payload["infrastructure_history"], list):
            raise ValueError("infrastructure_history must be a JSON list")
        if any(type(entry) is not dict for entry in payload["infrastructure_history"]):
            raise ValueError("infrastructure_history entries must be JSON objects")
        payload["expected_bsp_prefetch_budget_ns"] = expected_bsp_prefetch_budget_ns
        return cls(**payload)


def run_episode_with_retries_v4(
    identity: EpisodeIdentity,
    attempt: Callable[[int], AttemptResultV4],
    *,
    eval_seed: int,
    execution_mode: str,
    infrastructure_retries: int = INFRASTRUCTURE_RETRIES,
    expected_bsp_prefetch_budget_ns: Optional[int] = None,
) -> EpisodeRecordV4:
    """Retry only infrastructure and retain metrics from the final attempt."""
    if (
        isinstance(infrastructure_retries, bool)
        or not isinstance(infrastructure_retries, int)
        or infrastructure_retries != INFRASTRUCTURE_RETRIES
    ):
        raise ValueError("schema-v4 evaluation requires exactly two infrastructure retries")
    if execution_mode not in _control.EXECUTION_MODES:
        raise ValueError("unsupported schema-v4 execution_mode")
    history = []  # type: List[Dict[str, Any]]
    for attempt_number in range(1, infrastructure_retries + 2):
        try:
            result = attempt(attempt_number)
        except InfrastructureFailure as error:
            history.append(
                {"attempt": attempt_number, "kind": error.kind, "error": str(error)}
            )
            if attempt_number <= infrastructure_retries:
                continue
            return EpisodeRecordV4.infrastructure_incomplete(
                identity,
                eval_seed,
                attempt_number,
                execution_mode=execution_mode,
                kind=error.kind,
                error=str(error),
                infrastructure_history=history,
            )
        if not isinstance(result, AttemptResultV4):
            raise TypeError("attempt callback must return AttemptResultV4")
        return EpisodeRecordV4.from_attempt(
            identity,
            eval_seed,
            attempt_number,
            execution_mode=execution_mode,
            result=result,
            infrastructure_history=history,
            expected_bsp_prefetch_budget_ns=expected_bsp_prefetch_budget_ns,
        )
    raise AssertionError("unreachable retry state")


def _rate(successes: int, eligible: int) -> Optional[float]:
    return successes / eligible if eligible else None


def aggregate_records_v4(
    records: Sequence[EpisodeRecordV4],
    *,
    artifact_errors: Sequence["ArtifactErrorV4"] = (),
) -> Dict[str, Any]:
    """Aggregate final episode records without counting infrastructure gaps."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    episode_records = tuple(records)
    if any(not isinstance(record, EpisodeRecordV4) for record in episode_records):
        raise ValueError("records must contain only EpisodeRecordV4")
    for record in episode_records:
        record._validate()
    modes = {record.execution_mode for record in episode_records}
    if len(modes) > 1:
        raise ValueError("one summary cannot mix execution modes")
    if isinstance(artifact_errors, (str, bytes)) or not isinstance(artifact_errors, Sequence):
        raise ValueError("artifact_errors must be a sequence")
    errors = tuple(artifact_errors)
    if any(not isinstance(error, ArtifactErrorV4) for error in errors):
        raise ValueError("artifact_errors must contain only ArtifactErrorV4")
    for error in errors:
        error._validate()

    task_groups = {}  # type: Dict[Tuple[str, int], List[EpisodeRecordV4]]
    for record in episode_records:
        task_groups.setdefault((record.suite, record.task_id), []).append(record)
    tasks = []  # type: List[Dict[str, Any]]
    for (suite, task_id), group in sorted(task_groups.items()):
        eligible = sum(record.include_in_success_rate for record in group)
        successes = sum(record.success is True for record in group)
        tasks.append(
            {
                "suite": suite,
                "task_id": task_id,
                "task_name": group[0].task_name,
                "requested_episodes": len(group),
                "eligible_episodes": eligible,
                "successes": successes,
                "failures": eligible - successes,
                "incomplete_infrastructure_count": sum(
                    not record.include_in_success_rate for record in group
                ),
                "success_rate": _rate(successes, eligible),
            }
        )
    suites = []  # type: List[Dict[str, Any]]
    for suite in sorted({record.suite for record in episode_records}):
        suite_records = [record for record in episode_records if record.suite == suite]
        suite_tasks = [row for row in tasks if row["suite"] == suite]
        eligible = sum(record.include_in_success_rate for record in suite_records)
        successes = sum(record.success is True for record in suite_records)
        task_rates = [row["success_rate"] for row in suite_tasks if row["success_rate"] is not None]
        suites.append(
            {
                "suite": suite,
                "tasks": len(suite_tasks),
                "requested_episodes": len(suite_records),
                "eligible_episodes": eligible,
                "successes": successes,
                "failures": eligible - successes,
                "incomplete_infrastructure_count": sum(
                    not record.include_in_success_rate for record in suite_records
                ),
                "success_rate": _rate(successes, eligible),
                "task_macro_success_rate": (
                    sum(task_rates) / len(task_rates) if task_rates else None
                ),
            }
        )
    suite_rates = [row["success_rate"] for row in suites if row["success_rate"] is not None]
    all_four = {row["suite"] for row in suites} == set(SUPPORTED_SUITES)
    suite_macro = sum(suite_rates) / len(suite_rates) if suite_rates else None
    incomplete = sum(not record.include_in_success_rate for record in episode_records)
    return {
        "tasks": tasks,
        "suites": suites,
        "suite_macro_success_rate": suite_macro,
        "four_suite_macro_success_rate": (
            suite_macro if all_four and len(suite_rates) == len(SUPPORTED_SUITES) else None
        ),
        "evaluated_suite_count": len(suites),
        "all_four_suites_evaluated": all_four,
        "requested_episodes": len(episode_records),
        "eligible_episodes": sum(
            record.include_in_success_rate for record in episode_records
        ),
        "successes": sum(record.success is True for record in episode_records),
        "incomplete_infrastructure_count": incomplete,
        "artifact_error_count": len(errors),
        "acceptance_complete": incomplete == 0 and not errors,
    }


@dataclasses.dataclass(frozen=True)
class ArtifactErrorV4:
    episode_id: str
    artifact_type: str
    path: str
    error: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_nonempty_text(self.episode_id, name="artifact error episode_id")
        _require_nonempty_text(self.artifact_type, name="artifact error type")
        _require_nonempty_text(self.path, name="artifact error path")
        _require_nonempty_text(self.error, name="artifact error message")

    def to_dict(self) -> Dict[str, str]:
        self._validate()
        return {
            "episode_id": self.episode_id,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactErrorV4":
        return cls(
            **_require_exact_fields(
                value,
                _ARTIFACT_ERROR_FIELDS,
                label="v4 artifact error",
            )
        )


@dataclasses.dataclass(frozen=True)
class VideoArtifactAuditV4:
    schema_version: int
    episode_id: str
    execution_mode: str
    path: str
    video_show_inference_waits: bool
    planned: _timing.VideoTimingAuditV4
    encoded_fps: float
    encoded_frame_count: int
    encoded_duration_ns: int
    artifact_padding_frame_count: int
    timing_gate_pass: bool
    warning: Optional[str]

    def __post_init__(self) -> None:
        self._validate()

    @property
    def expected_encoded_frame_count(self) -> int:
        return self.planned.video_frame_count + self.artifact_padding_frame_count

    @property
    def timing_tolerance_ns(self) -> int:
        return (
            _timing.NANOSECONDS_PER_SECOND + self.planned.video_fps - 1
        ) // self.planned.video_fps

    def _validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("video artifact schema_version must be integer 4")
        _require_nonempty_text(self.episode_id, name="video episode_id")
        if self.execution_mode not in _control.EXECUTION_MODES:
            raise ValueError("unsupported video execution_mode")
        _require_nonempty_text(self.path, name="video path")
        if type(self.video_show_inference_waits) is not bool:
            raise ValueError("video_show_inference_waits must be boolean")
        if not isinstance(self.planned, _timing.VideoTimingAuditV4):
            raise ValueError("planned must be VideoTimingAuditV4")
        self.planned._validate()
        if (
            self.planned.control_hz != CONTROL_FREQ_HZ
            or self.planned.video_fps != VIDEO_FPS
        ):
            raise ValueError("planned video timing must use the schema-v4 20/40 Hz rates")
        if self.video_show_inference_waits:
            if self.planned.included_stall_count != self.planned.measured_stall_count:
                raise ValueError("enabled wait overlays must include every measured stall")
        elif self.planned.included_stall_count != 0:
            raise ValueError("disabled wait overlays cannot include control stalls")
        if (
            isinstance(self.encoded_fps, bool)
            or not isinstance(self.encoded_fps, (int, float))
            or not math.isfinite(self.encoded_fps)
            or self.encoded_fps <= 0
        ):
            raise ValueError("encoded_fps must be positive and finite")
        if not math.isclose(
            float(self.encoded_fps),
            float(self.planned.video_fps),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("encoded_fps does not match planned video_fps")
        _require_nonnegative_integer(self.encoded_frame_count, name="encoded_frame_count")
        _require_nonnegative_integer(self.encoded_duration_ns, name="encoded_duration_ns")
        _require_nonnegative_integer(
            self.artifact_padding_frame_count,
            name="artifact_padding_frame_count",
        )
        if self.artifact_padding_frame_count not in (0, 1):
            raise ValueError("artifact padding permits at most one frame")
        if self.planned.video_frame_count == 0:
            if self.artifact_padding_frame_count != 1:
                raise ValueError("an empty planned timeline requires one padding frame")
        elif self.artifact_padding_frame_count:
            raise ValueError("a non-empty planned timeline cannot use padding")
        if self.encoded_frame_count != self.expected_encoded_frame_count:
            raise ValueError("encoded frame count does not match planned timing")
        if type(self.timing_gate_pass) is not bool:
            raise ValueError("timing_gate_pass must be boolean")
        expected_gate = (
            abs(self.encoded_duration_ns - self.planned.expected_duration_ns)
            <= self.timing_tolerance_ns
        )
        if self.timing_gate_pass != expected_gate:
            raise ValueError("timing_gate_pass does not match encoded duration")
        if expected_gate:
            if self.warning is not None:
                raise ValueError("passing video timing cannot carry a warning")
        else:
            _require_nonempty_text(self.warning, name="video warning")

    def validate_episode(self, episode: EpisodeRecordV4) -> None:
        if not isinstance(episode, EpisodeRecordV4):
            raise ValueError("episode must be EpisodeRecordV4")
        episode._validate()
        if self.episode_id != episode.episode_id:
            raise ValueError("video episode_id does not match episode record")
        if self.execution_mode != episode.execution_mode:
            raise ValueError("video execution_mode does not match episode record")
        expected = _timing.build_video_timing_audit_v4(
            control_frame_count=episode.steps,
            requests=episode.inference_requests,
            latencies=episode.inference_latencies,
            activations=episode.plan_activations,
            underflows=episode.action_underflows,
            stalls=episode.control_stalls,
            include_stalls=self.video_show_inference_waits,
            control_hz=CONTROL_FREQ_HZ,
            video_fps=VIDEO_FPS,
        )
        if self.planned != expected:
            raise ValueError("video planned timing does not match episode events")

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "execution_mode": self.execution_mode,
            "path": self.path,
            "video_show_inference_waits": self.video_show_inference_waits,
            "planned": self.planned.to_dict(),
            "encoded_fps": self.encoded_fps,
            "encoded_frame_count": self.encoded_frame_count,
            "encoded_duration_ns": self.encoded_duration_ns,
            "artifact_padding_frame_count": self.artifact_padding_frame_count,
            "timing_gate_pass": self.timing_gate_pass,
            "warning": self.warning,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "VideoArtifactAuditV4":
        payload = dict(
            _require_exact_fields(
                value,
                _VIDEO_ARTIFACT_FIELDS,
                label="v4 video artifact audit",
            )
        )
        planned = _require_json_object(payload["planned"], label="planned")
        payload["planned"] = _timing.VideoTimingAuditV4.from_dict(planned)
        return cls(**payload)


def build_video_artifact_audit_v4(
    *,
    episode: EpisodeRecordV4,
    path: str,
    planned: _timing.VideoTimingAuditV4,
    video_show_inference_waits: bool,
    encoded_fps: float,
    encoded_frame_count: int,
    encoded_duration_s: float,
    artifact_padding_frame_count: int = 0,
) -> VideoArtifactAuditV4:
    """Cross-check an encoded video against its exact episode event timeline."""
    if not isinstance(episode, EpisodeRecordV4):
        raise ValueError("episode must be EpisodeRecordV4")
    if (
        isinstance(encoded_duration_s, bool)
        or not isinstance(encoded_duration_s, (int, float))
        or not math.isfinite(encoded_duration_s)
        or encoded_duration_s < 0
    ):
        raise ValueError("encoded_duration_s must be non-negative and finite")
    encoded_duration_ns = round(
        float(encoded_duration_s) * _timing.NANOSECONDS_PER_SECOND
    )
    if not isinstance(planned, _timing.VideoTimingAuditV4):
        raise ValueError("planned must be VideoTimingAuditV4")
    tolerance_ns = (
        _timing.NANOSECONDS_PER_SECOND + planned.video_fps - 1
    ) // planned.video_fps
    deviation_ns = encoded_duration_ns - planned.expected_duration_ns
    timing_gate_pass = abs(deviation_ns) <= tolerance_ns
    warning = None
    if not timing_gate_pass:
        warning = (
            "encoded duration deviates from expected duration by {:.6f} s "
            "(tolerance {:.6f} s)"
        ).format(
            abs(deviation_ns) / _timing.NANOSECONDS_PER_SECOND,
            tolerance_ns / _timing.NANOSECONDS_PER_SECOND,
        )
    audit = VideoArtifactAuditV4(
        schema_version=SCHEMA_VERSION,
        episode_id=episode.episode_id,
        execution_mode=episode.execution_mode,
        path=path,
        video_show_inference_waits=video_show_inference_waits,
        planned=planned,
        encoded_fps=encoded_fps,
        encoded_frame_count=encoded_frame_count,
        encoded_duration_ns=encoded_duration_ns,
        artifact_padding_frame_count=artifact_padding_frame_count,
        timing_gate_pass=timing_gate_pass,
        warning=warning,
    )
    audit.validate_episode(episode)
    return audit


def _safe_component(value: str, *, fallback: str) -> str:
    component = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return component or fallback


class VideoSelectorV4:
    """Reserve the first success and counted failure video per task."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._claimed = set()  # type: set

    def claim(self, record: EpisodeRecordV4) -> Optional[Path]:
        if not isinstance(record, EpisodeRecordV4):
            raise ValueError("record must be EpisodeRecordV4")
        record._validate()
        if record.success is True:
            outcome = "success"
        elif record.include_in_success_rate and record.failure_kind in _FAILURE_KINDS:
            outcome = "failure"
        else:
            return None
        key = (record.suite, record.task_id, outcome)
        if key in self._claimed:
            return None
        self._claimed.add(key)
        task_name = _safe_component(record.task_name, fallback="task")
        directory = (
            self._root
            / record.suite
            / "task-{:03d}-{}".format(record.task_id, task_name)
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "{}-init-{:03d}-{}.mp4".format(
            outcome,
            record.init_state_index,
            record.episode_id,
        )


def _atomic_append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded_line = (
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    existing = path.read_bytes() if path.exists() else b""
    if existing and not existing.endswith(b"\n"):
        raise ValueError("existing JSONL artifact must end with a complete newline")
    combined = (existing + encoded_line).decode("utf-8")
    libero_artifacts.atomic_text(path, combined)


class ArtifactWriterV4:
    """Persist schema-v4 artifacts without invoking any schema-v3 producer."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.video_audit_path = self.output_dir / "video_audit.jsonl"
        self.artifact_errors_path = self.output_dir / "artifact_errors.jsonl"
        for path in (
            self.episodes_path,
            self.video_audit_path,
            self.artifact_errors_path,
        ):
            if not path.exists():
                libero_artifacts.atomic_text(path, "")

    def write_manifest(self, manifest: EvaluationManifestV4) -> None:
        if not isinstance(manifest, EvaluationManifestV4):
            raise ValueError("manifest must be EvaluationManifestV4")
        libero_artifacts.atomic_text(
            self.manifest_path,
            libero_artifacts.json_text(manifest.to_dict()),
        )

    def append_episode(self, record: EpisodeRecordV4) -> None:
        if not isinstance(record, EpisodeRecordV4):
            raise ValueError("record must be EpisodeRecordV4")
        _atomic_append_jsonl(self.episodes_path, record.to_dict())

    def append_video_audit(self, audit: VideoArtifactAuditV4) -> None:
        if not isinstance(audit, VideoArtifactAuditV4):
            raise ValueError("audit must be VideoArtifactAuditV4")
        _atomic_append_jsonl(self.video_audit_path, audit.to_dict())

    def append_artifact_error(self, error: ArtifactErrorV4) -> None:
        if not isinstance(error, ArtifactErrorV4):
            raise ValueError("error must be ArtifactErrorV4")
        _atomic_append_jsonl(self.artifact_errors_path, error.to_dict())

    def write_summary(
        self,
        records: Sequence[EpisodeRecordV4],
        *,
        artifact_errors: Sequence[ArtifactErrorV4] = (),
    ) -> Dict[str, Any]:
        summary = aggregate_records_v4(records, artifact_errors=artifact_errors)
        libero_artifacts.atomic_text(
            self.summary_path,
            libero_artifacts.json_text(summary),
        )
        return summary
