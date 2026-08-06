"""Auditable paired evaluation for π0.5 baseline and BSP on LIBERO."""

from __future__ import annotations

import collections
from collections.abc import Mapping
import dataclasses
import logging
import math
from pathlib import Path
import subprocess
import time
from typing import Optional

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import inference as _inference
from openpi_client import libero_eval as _eval
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_NATIVE_HZ = 10
REPLAN_STEPS = 8
EXPECTED_TASKS_PER_SUITE = 10
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


@dataclasses.dataclass
class Args:
    # Policy server.
    host: str = "0.0.0.0"
    port: int = 8000
    connection_timeout_s: float = 30.0
    inference_timeout_s: float = 120.0
    resize_size: int = 224

    # Benchmark protocol. `task_suite_name` retains the official example's CLI name.
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    eval_seed: int = 42

    # Output directory must identify one policy/checkpoint evaluation run.
    output_dir: str = "data/libero/eval"

    # Audit manifest identities. `code_sha=auto` reads the current checkout.
    policy_variant: str = "baseline"
    expected_action_horizon: Optional[int] = None  # noqa: UP045 -- simulator client runs Python 3.8.
    code_sha: str = "auto"
    dataset_revision: str = "v2.1"
    bsp_cache_hash: Optional[str] = None  # noqa: UP045 -- simulator client runs Python 3.8.
    norm_hash: str = ""
    checkpoint: str = ""
    container_digest: str = ""
    train_seed: int = 42


class _ClientHolder:
    def __init__(self, args: Args):
        self._args = args
        self._client = None

    def get(self):
        if self._client is None:
            try:
                self._client = _websocket_client_policy.WebsocketClientPolicy(
                    self._args.host,
                    self._args.port,
                    connection_timeout=self._args.connection_timeout_s,
                    inference_timeout=self._args.inference_timeout_s,
                )
            except Exception as error:
                raise _eval.classify_exception(error, phase="server_connect") from error
        return self._client

    def invalidate(self) -> None:
        if self._client is None:
            return
        try:
            self._client.close()
        except Exception as error:
            logging.warning("Error while closing failed policy connection: %s", error)
        finally:
            self._client = None

    def close(self) -> None:
        self.invalidate()


class _TaskEnvironment:
    def __init__(self, task, resolution: int, seed: int):
        self._task = task
        self._resolution = resolution
        self._seed = seed
        self._env = None

    def _get(self):
        if self._env is None:
            try:
                self._env = _get_libero_env(self._task, self._resolution, self._seed)
            except Exception as error:
                raise _eval.classify_exception(error, phase="environment_create") from error
        return self._env

    def reset_to(self, initial_state):
        env = self._get()
        try:
            # Re-seeding every retry makes simulator randomness identical for the
            # same fixed initial state in baseline and BSP runs.
            env.seed(self._seed)
            env.reset()
            return env.set_init_state(initial_state)
        except Exception as error:
            self.invalidate()
            raise _eval.classify_exception(error, phase="environment_reset") from error

    def step(self, action):
        env = self._get()
        try:
            return env.step(action)
        except Exception as error:
            self.invalidate()
            raise _eval.classify_exception(error, phase="environment_step") from error

    def invalidate(self) -> None:
        if self._env is None:
            return
        try:
            self._env.close()
        except Exception as error:
            logging.warning("Error while closing failed LIBERO environment: %s", error)
        finally:
            self._env = None

    def close(self) -> None:
        self.invalidate()


def _resolve_code_sha(value: str) -> str:
    if value != "auto":
        return value
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_args(args: Args) -> tuple[tuple[str, ...], _eval.PolicyProtocol]:
    suites = _eval.resolve_suites(args.task_suite_name)
    protocol = _eval.resolve_policy_protocol(args.policy_variant, args.expected_action_horizon)
    if args.num_trials_per_task < 1 or args.num_steps_wait < 0:
        raise ValueError("Episode counts must be positive and wait steps non-negative")
    if args.eval_seed < 0 or args.train_seed < 0:
        raise ValueError("Training and evaluation seeds must be non-negative")
    if args.resize_size < 1 or args.connection_timeout_s <= 0 or args.inference_timeout_s <= 0:
        raise ValueError("Image size and connection/inference timeouts must be positive")
    return suites, protocol


def _make_manifest(
    args: Args, suites: tuple[str, ...], protocol: _eval.PolicyProtocol
) -> _eval.EvaluationManifest:
    bsp_cache_hash = args.bsp_cache_hash or None
    return _eval.EvaluationManifest(
        code_sha=_resolve_code_sha(args.code_sha),
        dataset_revision=args.dataset_revision,
        bsp_cache_hash=bsp_cache_hash,
        norm_hash=args.norm_hash,
        checkpoint=args.checkpoint,
        container_digest=args.container_digest,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        policy_variant=args.policy_variant,
        bsp_parameters=_eval.BSP_PARAMETERS,
        policy_protocol=protocol.name,
        expected_action_horizon=protocol.expected_action_horizon,
        execution_horizon=REPLAN_STEPS,
        suites=suites,
        trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait,
        max_steps_by_suite={suite: MAX_STEPS_BY_SUITE[suite] for suite in suites},
        connection_timeout_s=args.connection_timeout_s,
        inference_timeout_s=args.inference_timeout_s,
        infrastructure_retries=2,
    )


def _ensure_new_run_directory(output_dir: Path) -> None:
    collisions = sorted(path.name for path in output_dir.iterdir()) if output_dir.is_dir() else []
    if collisions:
        raise FileExistsError(
            f"Evaluation output directory is not empty ({collisions}); use a unique output_dir"
        )


def _initial_state_fingerprint(initial_state) -> str:
    state = np.ascontiguousarray(initial_state)
    return _eval.fingerprint_init_state(dtype=state.dtype.str, shape=state.shape, payload=state.tobytes())


def _prepare_observation(obs, task_description: str, resize_size: int) -> tuple[dict, np.ndarray]:
    # LIBERO camera arrays must be rotated 180 degrees to match training.
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, resize_size, resize_size)
    )
    request = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }
    return request, image


def _infer_action_plan(
    client_holder: _ClientHolder,
    request: dict,
    identity: _eval.EpisodeIdentity,
    eval_seed: int,
    replan_index: int,
    expected_action_horizon: int,
) -> tuple[tuple[tuple[float, ...], ...], float]:
    request[_inference.INFERENCE_SEED_KEY] = _eval.stable_replan_seed(eval_seed, identity, replan_index)
    start_time = time.monotonic()
    try:
        result = client_holder.get().infer(request)
    except _eval.InfrastructureFailure:
        client_holder.invalidate()
        raise
    except Exception as error:
        client_holder.invalidate()
        raise _eval.classify_exception(error, phase="policy_infer") from error
    elapsed_ms = (time.monotonic() - start_time) * 1_000
    if not isinstance(result, Mapping) or "actions" not in result:
        client_holder.invalidate()
        raise _eval.PolicyFailure("Policy response is missing the actions field")
    return (
        _eval.select_replan_actions(result["actions"], expected_horizon=expected_action_horizon),
        elapsed_ms,
    )


def _run_attempt(
    *,
    environment: _TaskEnvironment,
    client_holder: _ClientHolder,
    initial_state,
    identity: _eval.EpisodeIdentity,
    task_description: str,
    args: Args,
    max_steps: int,
) -> _eval.AttemptResult:
    protocol = _eval.resolve_policy_protocol(args.policy_variant, args.expected_action_horizon)
    obs = environment.reset_to(initial_state)
    action_plan = collections.deque()
    replay_images = []
    inference_ms = []
    replan_index = 0
    control_steps = 0

    for timestep in range(max_steps + args.num_steps_wait):
        if timestep < args.num_steps_wait:
            obs, _, _, _ = environment.step(LIBERO_DUMMY_ACTION)
            continue

        try:
            request, image = _prepare_observation(obs, task_description, args.resize_size)
        except Exception as error:
            environment.invalidate()
            raise _eval.classify_exception(error, phase="environment_step") from error
        replay_images.append(image)

        if not action_plan:
            try:
                chunk, elapsed_ms = _infer_action_plan(
                    client_holder,
                    request,
                    identity,
                    args.eval_seed,
                    replan_index,
                    protocol.expected_action_horizon,
                )
            except _eval.PolicyFailure as error:
                return _eval.AttemptResult(
                    success=False,
                    steps=control_steps,
                    replans=replan_index,
                    failure_kind="policy",
                    error=str(error),
                    inference_ms=tuple(inference_ms),
                    replay_frames=tuple(replay_images),
                )
            action_plan.extend(chunk)
            inference_ms.append(elapsed_ms)
            replan_index += 1

        action = action_plan.popleft()
        obs, _, done, _ = environment.step(list(action))
        control_steps += 1
        if bool(done):
            return _eval.AttemptResult(
                success=True,
                steps=control_steps,
                replans=replan_index,
                inference_ms=tuple(inference_ms),
                replay_frames=tuple(replay_images),
            )

    return _eval.AttemptResult(
        success=False,
        steps=control_steps,
        replans=replan_index,
        failure_kind="timeout",
        error="maximum rollout steps reached",
        inference_ms=tuple(inference_ms),
        replay_frames=tuple(replay_images),
    )


def _get_benchmark_suite(suite_name: str):
    # Lazy import keeps evaluation bookkeeping importable without LIBERO.
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite_name]()


def _persist_episode_artifacts(
    record: _eval.EpisodeRecord,
    writer: _eval.ArtifactWriter,
    video_selector: _eval.VideoSelector,
    *,
    video_encoder=None,
) -> tuple[_eval.EpisodeRecord, _eval.ArtifactError | None]:
    """Persist the rollout first, then encode or separately audit its video."""
    replay_frames = record.replay_frames
    persisted_record = dataclasses.replace(record, replay_frames=())
    writer.append_episode(persisted_record)

    video_path = video_selector.claim(persisted_record)
    if video_path is None:
        return persisted_record, None

    artifact_error = None
    if not replay_frames:
        artifact_error = _eval.ArtifactError(
            episode_id=persisted_record.identity.episode_id,
            artifact_type="video",
            path=str(video_path),
            error="selected rollout has no replay frames",
        )
    else:
        encoder = imageio.mimwrite if video_encoder is None else video_encoder
        try:
            encoder(
                video_path,
                [np.asarray(frame) for frame in replay_frames],
                fps=LIBERO_NATIVE_HZ,
            )
        except Exception as error:
            artifact_error = _eval.ArtifactError(
                episode_id=persisted_record.identity.episode_id,
                artifact_type="video",
                path=str(video_path),
                error=f"{type(error).__name__}: {error}",
            )

    if artifact_error is not None:
        writer.append_artifact_error(artifact_error)
    return persisted_record, artifact_error


def eval_libero(args: Args) -> dict:
    suites, protocol = _validate_args(args)
    output_dir = Path(args.output_dir)
    _ensure_new_run_directory(output_dir)
    writer = _eval.ArtifactWriter(output_dir)
    writer.write_manifest(_make_manifest(args, suites, protocol))
    video_selector = _eval.VideoSelector(output_dir / "videos")
    client_holder = _ClientHolder(args)
    records = []
    artifact_errors = []
    np.random.seed(args.eval_seed)

    try:
        for suite_name in suites:
            task_suite = _get_benchmark_suite(suite_name)
            if task_suite.n_tasks != EXPECTED_TASKS_PER_SUITE:
                raise ValueError(
                    f"Expected {EXPECTED_TASKS_PER_SUITE} tasks in {suite_name}, got {task_suite.n_tasks}"
                )
            logging.info("Evaluating suite %s", suite_name)

            for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc=suite_name):
                task = task_suite.get_task(task_id)
                task_description = str(task.language)
                initial_states = task_suite.get_task_init_states(task_id)
                if len(initial_states) < args.num_trials_per_task:
                    raise ValueError(
                        f"Task {suite_name}/{task_id} has {len(initial_states)} initial states, "
                        f"but {args.num_trials_per_task} were requested"
                    )
                environment = _TaskEnvironment(task, LIBERO_ENV_RESOLUTION, args.eval_seed)
                try:
                    for init_state_index in tqdm.tqdm(
                        range(args.num_trials_per_task),
                        desc=f"task-{task_id:03d}",
                        leave=False,
                    ):
                        initial_state = initial_states[init_state_index]
                        identity = _eval.EpisodeIdentity(
                            suite=suite_name,
                            task_id=task_id,
                            task_name=task_description,
                            init_state_index=init_state_index,
                            init_state_fingerprint=_initial_state_fingerprint(initial_state),
                        )

                        def attempt(_attempt_number):
                            return _run_attempt(
                                environment=environment,
                                client_holder=client_holder,
                                initial_state=initial_state,
                                identity=identity,
                                task_description=task_description,
                                args=args,
                                max_steps=MAX_STEPS_BY_SUITE[suite_name],
                            )

                        record = _eval.run_episode_with_retries(
                            identity,
                            attempt,
                            eval_seed=args.eval_seed,
                            infrastructure_retries=2,
                        )
                        record, artifact_error = _persist_episode_artifacts(record, writer, video_selector)
                        records.append(record)
                        if artifact_error is not None:
                            artifact_errors.append(artifact_error)
                        logging.info(
                            "%s status=%s attempts=%d steps=%d",
                            identity.episode_id,
                            record.status,
                            record.attempts,
                            record.steps,
                        )
                finally:
                    environment.close()
    finally:
        client_holder.close()

    summary = writer.write_summary(records, artifact_errors=artifact_errors)
    logging.info("Suite macro success rate: %s", summary["suite_macro_success_rate"])
    if not summary["acceptance_complete"]:
        raise RuntimeError(
            "Evaluation acceptance is incomplete: "
            f"{summary['incomplete_infrastructure_count']} exhausted infrastructure episodes, "
            f"{summary['artifact_error_count']} artifact errors"
        )
    return summary


def _get_libero_env(task, resolution, seed):
    """Initialize a task environment without importing LIBERO at module load."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle (adapted from robosuite)."""
    quat = np.asarray(quat).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
