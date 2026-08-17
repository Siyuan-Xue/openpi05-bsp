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
from typing import Optional, Tuple

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import inference as _inference
from openpi_client import libero_eval as _eval
from openpi_client import libero_video_timing as _video_timing
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
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
    # Omit for all ten tasks. A singleton such as `(0,)` enables the real EGL smoke run.
    task_ids: Optional[Tuple[int, ...]] = None  # noqa: UP045 -- simulator client runs Python 3.8.
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    eval_seed: int = 42
    control_freq: int = 20
    video_fps: int = 40
    video_show_inference_waits: bool = False

    # Output directory must identify one policy/checkpoint evaluation run.
    output_dir: str = "data/libero/eval"

    # Audit manifest identities. The evaluator always resolves a clean Git HEAD.
    policy_variant: str = "baseline"
    expected_action_horizon: Optional[int] = None  # noqa: UP045 -- simulator client runs Python 3.8.
    config_name: str = ""
    checkpoint_step: int = 0
    dataset_revision: str = "v2.0"
    bsp_cache_hash: Optional[str] = None  # noqa: UP045 -- simulator client runs Python 3.8.
    bsp_cache_manifest_fingerprint: Optional[str] = None  # noqa: UP045 -- Python 3.8.
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
    def __init__(self, task, resolution: int, seed: int, control_freq: int):
        self._task = task
        self._resolution = resolution
        self._seed = seed
        self._control_freq = control_freq
        self._env = None

    def _get(self):
        if self._env is None:
            try:
                self._env = _get_libero_env(
                    self._task,
                    self._resolution,
                    self._seed,
                    control_freq=self._control_freq,
                )
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


def _resolve_code_sha() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to resolve evaluator code identity from its Git checkout") from error
    if status.stdout.strip():
        raise RuntimeError("Evaluator Git checkout must be clean before writing a manifest")
    return result.stdout.strip()


def _validate_args(args: Args) -> tuple[tuple[str, ...], tuple[int, ...], _eval.PolicyProtocol]:
    suites = _eval.resolve_suites(args.task_suite_name)
    task_ids = _eval.resolve_task_ids(args.task_ids)
    protocol = _eval.resolve_policy_protocol(args.policy_variant, args.expected_action_horizon)
    if args.num_trials_per_task < 1 or args.num_steps_wait < 0:
        raise ValueError("Episode counts must be positive and wait steps non-negative")
    if args.eval_seed < 0 or args.train_seed < 0:
        raise ValueError("Training and evaluation seeds must be non-negative")
    if args.resize_size < 1 or args.connection_timeout_s <= 0 or args.inference_timeout_s <= 0:
        raise ValueError("Image size and connection/inference timeouts must be positive")
    _video_timing.validate_video_frequencies(
        control_hz=args.control_freq,
        video_fps=args.video_fps,
    )
    return suites, task_ids, protocol


def _make_manifest(
    args: Args,
    suites: tuple[str, ...],
    task_ids: tuple[int, ...],
    protocol: _eval.PolicyProtocol,
) -> _eval.EvaluationManifest:
    bsp_cache_hash = args.bsp_cache_hash or None
    bsp_cache_manifest_fingerprint = args.bsp_cache_manifest_fingerprint or None
    return _eval.EvaluationManifest(
        code_sha=_resolve_code_sha(),
        dataset_revision=args.dataset_revision,
        config_name=args.config_name,
        checkpoint_step=args.checkpoint_step,
        bsp_cache_hash=bsp_cache_hash,
        bsp_cache_manifest_fingerprint=bsp_cache_manifest_fingerprint,
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
        task_ids=task_ids,
        trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait,
        max_steps_by_suite={suite: MAX_STEPS_BY_SUITE[suite] for suite in suites},
        connection_timeout_s=args.connection_timeout_s,
        inference_timeout_s=args.inference_timeout_s,
        infrastructure_retries=2,
        dataset_fps=10,
        source_demo_control_hz=20,
        control_freq_hz=args.control_freq,
        video_fps=args.video_fps,
        video_show_inference_waits=args.video_show_inference_waits,
        inference_schedule=_video_timing.SYNCHRONOUS_INFERENCE_SCHEDULE,
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
) -> tuple[tuple[float, ...], ...]:
    request[_inference.INFERENCE_SEED_KEY] = _eval.stable_replan_seed(eval_seed, identity, replan_index)
    try:
        result = client_holder.get().infer(request)
    except _eval.InfrastructureFailure:
        client_holder.invalidate()
        raise
    except Exception as error:
        client_holder.invalidate()
        raise _eval.classify_exception(error, phase="policy_infer") from error
    if not isinstance(result, Mapping) or "actions" not in result:
        client_holder.invalidate()
        raise _eval.PolicyFailure("Policy response is missing the actions field")
    return _eval.select_replan_actions(
        result["actions"],
        expected_horizon=expected_action_horizon,
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
    episode_started_ns = time.monotonic_ns()
    action_plan = collections.deque()
    replay_images = []
    stall_source_frames = []
    inference_ms = []
    inference_requests = []
    control_stalls = []
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

        if not action_plan:
            request_started_ns = time.monotonic_ns()
            try:
                chunk = _infer_action_plan(
                    client_holder,
                    request,
                    identity,
                    args.eval_seed,
                    replan_index,
                    protocol.expected_action_horizon,
                )
            except _eval.PolicyFailure as error:
                request_completed_ns = time.monotonic_ns()
                request_duration_ns = request_completed_ns - request_started_ns
                request_started_offset_ns = request_started_ns - episode_started_ns
                inference_requests.append(
                    _video_timing.InferenceRequest(
                        replan_index=replan_index,
                        started_offset_ns=request_started_offset_ns,
                        duration_ns=request_duration_ns,
                    )
                )
                control_stalls.append(
                    _video_timing.ControlStall(
                        control_step=control_steps,
                        replan_index=replan_index,
                        started_offset_ns=request_started_offset_ns,
                        duration_ns=request_duration_ns,
                        reason=_video_timing.STALL_REASON_SYNCHRONOUS_INFERENCE,
                    )
                )
                stall_source_frames.append((control_steps, image))
                return _eval.AttemptResult(
                    success=False,
                    steps=control_steps,
                    replans=replan_index,
                    failure_kind="policy",
                    error=str(error),
                    inference_ms=tuple(inference_ms),
                    inference_requests=tuple(inference_requests),
                    control_stalls=tuple(control_stalls),
                    replay_frames=tuple(replay_images),
                    stall_source_frames=tuple(stall_source_frames),
                )
            request_completed_ns = time.monotonic_ns()
            request_duration_ns = request_completed_ns - request_started_ns
            request_started_offset_ns = request_started_ns - episode_started_ns
            inference_requests.append(
                _video_timing.InferenceRequest(
                    replan_index=replan_index,
                    started_offset_ns=request_started_offset_ns,
                    duration_ns=request_duration_ns,
                )
            )
            control_stalls.append(
                _video_timing.ControlStall(
                    control_step=control_steps,
                    replan_index=replan_index,
                    started_offset_ns=request_started_offset_ns,
                    duration_ns=request_duration_ns,
                    reason=_video_timing.STALL_REASON_SYNCHRONOUS_INFERENCE,
                )
            )
            action_plan.extend(chunk)
            inference_ms.append(request_duration_ns / 1_000_000)
            replan_index += 1

        action = action_plan.popleft()
        obs, _, done, _ = environment.step(list(action))
        replay_images.append(image)
        control_steps += 1
        if bool(done):
            return _eval.AttemptResult(
                success=True,
                steps=control_steps,
                replans=replan_index,
                inference_ms=tuple(inference_ms),
                inference_requests=tuple(inference_requests),
                control_stalls=tuple(control_stalls),
                replay_frames=tuple(replay_images),
            )

    return _eval.AttemptResult(
        success=False,
        steps=control_steps,
        replans=replan_index,
        failure_kind="timeout",
        error="maximum rollout steps reached",
        inference_ms=tuple(inference_ms),
        inference_requests=tuple(inference_requests),
        control_stalls=tuple(control_stalls),
        replay_frames=tuple(replay_images),
    )


def _get_benchmark_suite(suite_name: str):
    # Lazy import keeps evaluation bookkeeping importable without LIBERO.
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite_name]()


def _synchronous_stall_overlay_lines(
    stall: _video_timing.ControlStall,
) -> tuple[str, str]:
    return _video_timing.stall_overlay_lines(
        stall,
        inference_schedule=_video_timing.SYNCHRONOUS_INFERENCE_SCHEDULE,
    )


def _draw_video_overlay(frame, lines: tuple[str, ...]):
    """Draw timing text with Pillow on the copy supplied by render_overlay."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame).copy())
    draw = ImageDraw.Draw(image)
    overlay_height = min(image.height, 10 + 18 * len(lines))
    draw.rectangle((0, 0, image.width, overlay_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((6, 4 + index * 18), line, fill=(255, 255, 255))
    return np.asarray(image).copy()


def _build_video_frames(
    control_frames,
    stalls: tuple[_video_timing.ControlStall, ...],
    *,
    stall_source_frames=(),
    control_hz: int,
    video_fps: int,
    inference_schedule: str,
    overlay_renderer=None,
) -> tuple:
    """Expand a selected rollout and insert labeled, measured stall holds."""
    _video_timing.validate_inference_schedule(inference_schedule)
    renderer = _draw_video_overlay if overlay_renderer is None else overlay_renderer
    held_frames = _video_timing.expand_control_frames(
        control_frames,
        control_hz=control_hz,
        video_fps=video_fps,
    )
    hold_count = video_fps // control_hz
    frame_count = len(control_frames)
    stall_source_by_step = {}
    for entry in stall_source_frames:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError("stall_source_frames must contain (control_step, frame) tuples")
        control_step, source_frame = entry
        if isinstance(control_step, bool) or not isinstance(control_step, int):
            raise TypeError("stall source control_step must be an integer")
        if control_step < 0 or control_step > frame_count or control_step in stall_source_by_step:
            raise ValueError("stall source control_step must be unique and within the replay timeline")
        stall_source_by_step[control_step] = source_frame
    stall_frames_by_step = {}
    for stall, stall_frame_count in zip(  # noqa: B905 -- LIBERO client runs on Python 3.8.
        stalls,
        _video_timing.quantize_stall_frames(stalls, video_fps=video_fps),
    ):
        if stall.control_step > frame_count:
            raise ValueError(
                f"Control stall step {stall.control_step} exceeds {frame_count} replay frames"
            )
        stall_frames_by_step[stall.control_step] = (
            stall,
            stall_frame_count,
            _video_timing.stall_overlay_lines(
                stall,
                inference_schedule=inference_schedule,
            ),
        )
    if set(stall_source_by_step) - set(stall_frames_by_step):
        raise ValueError("Every transient stall source must correspond to a measured control stall")

    video_frames = []
    for control_step, frame in enumerate(control_frames):
        stall_event = stall_frames_by_step.get(control_step)
        if stall_event is not None:
            _, stall_frame_count, overlay_lines = stall_event
            if stall_frame_count:
                rendered_stall = _video_timing.render_overlay(
                    stall_source_by_step.get(control_step, frame),
                    overlay_lines,
                    renderer=renderer,
                )
                video_frames.extend(rendered_stall for _ in range(stall_frame_count))
        held_start = control_step * hold_count
        video_frames.extend(held_frames[held_start : held_start + hold_count])

    trailing_stall = stall_frames_by_step.get(frame_count)
    if trailing_stall is not None:
        _, stall_frame_count, overlay_lines = trailing_stall
        frame = stall_source_by_step.get(frame_count)
        if stall_frame_count:
            if frame is None:
                raise ValueError("Cannot render a trailing control stall without its request-time source frame")
            rendered_stall = _video_timing.render_overlay(
                frame,
                overlay_lines,
                renderer=renderer,
            )
            video_frames.extend(rendered_stall for _ in range(stall_frame_count))
    return tuple(video_frames)


def _read_encoded_video(video_path: Path) -> tuple[float, int, float]:
    reader = imageio.get_reader(video_path)
    try:
        metadata = reader.get_meta_data()
        encoded_fps = metadata["fps"]
        encoded_duration_s = metadata["duration"]
        encoded_frame_count = reader.count_frames()
    finally:
        reader.close()
    return float(encoded_fps), int(encoded_frame_count), float(encoded_duration_s)


def _persist_episode_artifacts(
    record: _eval.EpisodeRecord,
    writer: _eval.ArtifactWriter,
    video_selector: _eval.VideoSelector,
    *,
    control_hz: int = _video_timing.CONTROL_HZ,
    video_fps: int = _video_timing.DEFAULT_VIDEO_FPS,
    video_show_inference_waits: bool = False,
    inference_schedule: str = _video_timing.SYNCHRONOUS_INFERENCE_SCHEDULE,
    video_encoder=None,
) -> tuple[_eval.EpisodeRecord, _eval.ArtifactError | None]:
    """Persist the rollout first, then encode or separately audit its video."""
    replay_frames = record.replay_frames
    stall_source_frames = record.stall_source_frames
    persisted_record = dataclasses.replace(record, replay_frames=(), stall_source_frames=())
    writer.append_episode(persisted_record)

    video_path = video_selector.claim(persisted_record)
    if video_path is None:
        return persisted_record, None

    artifact_error = None
    encoder = imageio.mimwrite if video_encoder is None else video_encoder
    try:
        included_stalls = (
            persisted_record.control_stalls if video_show_inference_waits else ()
        )
        planned_audit = _video_timing.build_video_audit(
            control_frame_count=len(replay_frames),
            requests=persisted_record.inference_requests,
            stalls=included_stalls,
            control_hz=control_hz,
            video_fps=video_fps,
        )
        video_frames = _build_video_frames(
            replay_frames,
            included_stalls,
            stall_source_frames=stall_source_frames if video_show_inference_waits else (),
            control_hz=control_hz,
            video_fps=video_fps,
            inference_schedule=inference_schedule,
        )
        artifact_padding_frame_count = 0
        if not video_frames:
            if planned_audit.video_frame_count != 0:
                raise ValueError("Non-empty planned video timeline expanded to zero frames")
            source_by_step = dict(stall_source_frames)
            source_frame = source_by_step.get(0)
            if source_frame is None:
                raise ValueError("Zero-step selected failure has no request-time video source")
            if video_show_inference_waits and included_stalls:
                stall = included_stalls[0]
                source_frame = _video_timing.render_overlay(
                    source_frame,
                    _video_timing.stall_overlay_lines(
                        stall,
                        inference_schedule=inference_schedule,
                    ),
                    renderer=_draw_video_overlay,
                )
            video_frames = (source_frame,)
            artifact_padding_frame_count = 1
        expected_encoded_frame_count = (
            planned_audit.video_frame_count + artifact_padding_frame_count
        )
        if len(video_frames) != expected_encoded_frame_count:
            raise ValueError("Expanded video frame count does not match planned audit")
        encoder(
            video_path,
            [np.asarray(frame) for frame in video_frames],
            fps=video_fps,
        )
        encoded_fps, encoded_frame_count, encoded_duration_s = _read_encoded_video(video_path)
        video_audit = _eval.build_video_artifact_audit(
            episode_id=persisted_record.identity.episode_id,
            path=str(video_path),
            planned=planned_audit,
            measured_stalls=persisted_record.control_stalls,
            included_stalls=included_stalls,
            video_show_inference_waits=video_show_inference_waits,
            inference_schedule=inference_schedule,
            artifact_padding_frame_count=artifact_padding_frame_count,
            encoded_fps=encoded_fps,
            encoded_frame_count=encoded_frame_count,
            encoded_duration_s=encoded_duration_s,
        )
        if video_audit.warning is not None:
            logging.warning(
                "Video timing warning for %s: %s",
                persisted_record.identity.episode_id,
                video_audit.warning,
            )
        writer.append_video_audit(video_audit)
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
    suites, task_ids, protocol = _validate_args(args)
    output_dir = Path(args.output_dir)
    _ensure_new_run_directory(output_dir)
    writer = _eval.ArtifactWriter(output_dir)
    writer.write_manifest(_make_manifest(args, suites, task_ids, protocol))
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

            for task_id in tqdm.tqdm(task_ids, desc=suite_name):
                task = task_suite.get_task(task_id)
                task_description = str(task.language)
                initial_states = task_suite.get_task_init_states(task_id)
                if len(initial_states) < args.num_trials_per_task:
                    raise ValueError(
                        f"Task {suite_name}/{task_id} has {len(initial_states)} initial states, "
                        f"but {args.num_trials_per_task} were requested"
                    )
                environment = _TaskEnvironment(
                    task,
                    LIBERO_ENV_RESOLUTION,
                    args.eval_seed,
                    args.control_freq,
                )
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
                        record, artifact_error = _persist_episode_artifacts(
                            record,
                            writer,
                            video_selector,
                            control_hz=args.control_freq,
                            video_fps=args.video_fps,
                            video_show_inference_waits=args.video_show_inference_waits,
                            inference_schedule=_video_timing.SYNCHRONOUS_INFERENCE_SCHEDULE,
                        )
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


def _get_libero_env(task, resolution, seed, *, control_freq: int):
    """Initialize a task environment without importing LIBERO at module load."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=control_freq,
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
