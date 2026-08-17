import dataclasses
import sys
import types

import pytest

from openpi_client import inference
from openpi_client import libero_eval
from openpi_client import libero_video_timing as timing

from examples.libero import main as libero_main


def _identity():
    return libero_eval.EpisodeIdentity(
        suite="libero_spatial",
        task_id=0,
        task_name="pick up the block",
        init_state_index=2,
        init_state_fingerprint="state-fingerprint",
    )


class _Environment:
    def __init__(self):
        self.reset_states = []
        self.actions = []

    def reset_to(self, initial_state):
        self.reset_states.append(initial_state)
        return {"raw": "observation"}

    def step(self, action):
        self.actions.append(action)
        return {"raw": "next"}, 0.0, True, {}

    def invalidate(self):
        pass


class _ContinuingEnvironment(_Environment):
    def step(self, action):
        self.actions.append(action)
        return {"raw": "observation-{}".format(len(self.actions))}, 0.0, False, {}


class _Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def infer(self, request):
        self.requests.append(request.copy())
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {"actions": response}


class _ClientHolder:
    def __init__(self, responses):
        self.client = _Client(responses)
        self.invalidations = 0

    def get(self):
        return self.client

    def invalidate(self):
        self.invalidations += 1


def _args():
    return dataclasses.replace(
        libero_main.Args(),
        num_steps_wait=0,
        expected_action_horizon=16,
        config_name="pi05_libero_baseline_h16",
        checkpoint_step=10000,
        norm_hash="b" * 64,
        checkpoint="checkpoint/10000",
        container_digest="sha256:" + "d" * 64,
        code_sha="a" * 40,
    )


def _actions(horizon=16):
    return [[float(row + column) for column in range(7)] for row in range(horizon)]


def test_evaluator_defaults_to_the_real_official_dataset_revision():
    assert libero_main.Args().dataset_revision == "v2.0"


def test_evaluator_defaults_to_20_hz_control_and_40_fps_video():
    args = libero_main.Args()

    assert args.control_freq == 20
    assert args.video_fps == 40
    assert args.video_show_inference_waits is False


@pytest.mark.parametrize(
    ("control_freq", "video_fps"),
    [(10, 40), (20, 0), (20, 30), (True, 40)],
)
def test_evaluator_rejects_non_protocol_video_frequencies(control_freq, video_fps):
    with pytest.raises(ValueError):
        libero_main._validate_args(
            dataclasses.replace(_args(), control_freq=control_freq, video_fps=video_fps)
        )


def test_offscreen_environment_receives_explicit_20_hz_control_frequency(monkeypatch):
    captured = {}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def seed(self, seed):
            captured["seed"] = seed

    libero_package = types.ModuleType("libero")
    libero_module = types.ModuleType("libero.libero")
    envs_module = types.ModuleType("libero.libero.envs")
    libero_module.get_libero_path = lambda _name: "/benchmark"
    envs_module.OffScreenRenderEnv = FakeEnvironment
    libero_package.libero = libero_module
    libero_module.envs = envs_module
    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_module)

    task = types.SimpleNamespace(problem_folder="suite", bddl_file="task.bddl")
    libero_main._get_libero_env(task, 256, 42, control_freq=20)

    assert captured["control_freq"] == 20
    assert captured["camera_heights"] == 256
    assert captured["camera_widths"] == 256
    assert captured["seed"] == 42


def test_client_holder_passes_finite_inference_deadline(monkeypatch):
    captured = {}

    def fake_client(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(libero_main._websocket_client_policy, "WebsocketClientPolicy", fake_client)

    libero_main._ClientHolder(_args()).get()

    assert captured["inference_timeout"] == 120.0


def test_official_calibration_resolves_strict_horizon_10_protocol():
    args = dataclasses.replace(_args(), expected_action_horizon=10)

    _, _, protocol = libero_main._validate_args(args)

    assert protocol.name == "baseline_h10_calibration"
    assert protocol.expected_action_horizon == 10


def test_single_task_smoke_filter_is_canonical_and_recorded_in_manifest():
    args = dataclasses.replace(
        _args(),
        task_suite_name="libero_spatial",
        task_ids=(0,),
        num_trials_per_task=1,
    )

    suites, task_ids, protocol = libero_main._validate_args(args)
    manifest = libero_main._make_manifest(args, suites, task_ids, protocol).to_dict()

    assert suites == ("libero_spatial",)
    assert task_ids == (0,)
    assert manifest["task_ids"] == [0]
    assert manifest["trials_per_task"] == 1
    assert manifest["schema_version"] == 3
    assert manifest["dataset_fps"] == 10
    assert manifest["source_demo_control_hz"] == 20
    assert manifest["control_freq_hz"] == 20
    assert manifest["video_fps"] == 40
    assert manifest["video_show_inference_waits"] is False
    assert manifest["inference_schedule"] == "synchronous"


def test_automatic_code_sha_requires_clean_evaluator_checkout(monkeypatch):
    calls = []

    def clean_run(command, **kwargs):
        calls.append(command)
        output = "a" * 40 + "\n" if "rev-parse" in command else ""
        return types.SimpleNamespace(stdout=output)

    monkeypatch.setattr(libero_main.subprocess, "run", clean_run)

    assert libero_main._resolve_code_sha("auto") == "a" * 40
    assert calls[0][1:3] == ["-C", str(libero_main.Path(libero_main.__file__).resolve().parents[2])]
    assert "--untracked-files=all" in calls[1]

    def dirty_run(command, **kwargs):
        output = "a" * 40 + "\n" if "rev-parse" in command else "?? untracked.txt\n"
        return types.SimpleNamespace(stdout=output)

    monkeypatch.setattr(libero_main.subprocess, "run", dirty_run)
    with pytest.raises(RuntimeError, match="clean"):
        libero_main._resolve_code_sha("auto")


def test_manual_code_sha_compatibility_override_does_not_run_git(monkeypatch):
    monkeypatch.setattr(
        libero_main.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("manual compatibility override unexpectedly ran git"),
    )

    assert libero_main._resolve_code_sha("b" * 40) == "b" * 40


@pytest.mark.parametrize("task_ids", [(), (0, 0), (-1,), (10,)])
def test_invalid_task_filters_are_rejected(task_ids):
    with pytest.raises(ValueError):
        libero_main._validate_args(dataclasses.replace(_args(), task_ids=task_ids))


def test_run_attempt_sends_reserved_seed_and_uses_exact_initial_state(monkeypatch):
    environment = _Environment()
    holder = _ClientHolder([_actions()])
    initial_state = object()
    identity = _identity()
    monkeypatch.setattr(libero_main, "_prepare_observation", lambda obs, prompt, size: ({}, "frame"))

    result = libero_main._run_attempt(
        environment=environment,
        client_holder=holder,
        initial_state=initial_state,
        identity=identity,
        task_description="pick up the block",
        args=_args(),
        max_steps=1,
    )

    assert result.success
    assert environment.reset_states == [initial_state]
    assert holder.client.requests[0][inference.INFERENCE_SEED_KEY] == libero_eval.stable_replan_seed(
        42, identity, 0
    )
    assert environment.actions == [_actions()[0]]


def test_run_attempt_records_synchronous_request_and_stall_without_extra_steps(monkeypatch):
    environment = _Environment()
    holder = _ClientHolder([_actions()])
    identity = _identity()
    clock = iter((1_000, 1_100, 3_100))
    monkeypatch.setattr(libero_main.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setattr(libero_main, "_prepare_observation", lambda obs, prompt, size: ({}, "frame"))

    result = libero_main._run_attempt(
        environment=environment,
        client_holder=holder,
        initial_state=object(),
        identity=identity,
        task_description="pick up the block",
        args=_args(),
        max_steps=1,
    )

    assert result.inference_requests == (
        timing.InferenceRequest(replan_index=0, started_offset_ns=100, duration_ns=2_000),
    )
    assert result.control_stalls == (
        timing.ControlStall(
            control_step=0,
            replan_index=0,
            started_offset_ns=100,
            duration_ns=2_000,
        ),
    )
    assert result.inference_ms == (0.002,)
    assert result.replay_frames == ("frame",)
    assert len(environment.actions) == result.steps == 1


def test_first_request_policy_failure_keeps_current_frame_without_counting_a_step(monkeypatch):
    environment = _Environment()
    holder = _ClientHolder([])
    monkeypatch.setattr(holder.client, "infer", lambda request: {})
    clock = iter((1_000, 2_000, 50_002_000))
    monkeypatch.setattr(libero_main.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setattr(
        libero_main,
        "_prepare_observation",
        lambda obs, prompt, size: ({}, "current-request-frame"),
    )

    result = libero_main._run_attempt(
        environment=environment,
        client_holder=holder,
        initial_state=object(),
        identity=_identity(),
        task_description="pick up the block",
        args=_args(),
        max_steps=1,
    )

    assert not result.success
    assert result.failure_kind == "policy"
    assert result.steps == result.replans == 0
    assert environment.actions == []
    assert result.replay_frames == ()
    assert result.stall_source_frames == ((0, "current-request-frame"),)
    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(), 42, 1, success=False, failure_kind="policy", result=result
    )
    video_frames = libero_main._build_video_frames(
        record.replay_frames,
        record.control_stalls,
        stall_source_frames=record.stall_source_frames,
        control_hz=20,
        video_fps=40,
        inference_schedule="synchronous",
        overlay_renderer=lambda frame, lines: "first-failure-overlay",
    )
    assert video_frames == ("first-failure-overlay", "first-failure-overlay")
    assert "stall_source_frames" not in record.to_dict()


def test_later_replan_policy_failure_uses_latest_observation_without_extra_action(monkeypatch):
    environment = _ContinuingEnvironment()
    holder = _ClientHolder([_actions(), [[float("nan")] * 7 for _ in range(16)]])
    clock = iter((1_000, 2_000, 3_000, 4_000, 50_004_000))
    monkeypatch.setattr(libero_main.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setattr(
        libero_main,
        "_prepare_observation",
        lambda obs, prompt, size: ({}, "frame-{}".format(obs["raw"])),
    )

    result = libero_main._run_attempt(
        environment=environment,
        client_holder=holder,
        initial_state=object(),
        identity=_identity(),
        task_description="pick up the block",
        args=_args(),
        max_steps=9,
    )

    assert not result.success
    assert result.failure_kind == "policy"
    assert result.steps == len(environment.actions) == 8
    assert result.replans == 1
    assert result.replay_frames[-1] == "frame-observation-7"
    assert result.stall_source_frames == ((8, "frame-observation-8"),)
    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(), 42, 1, success=False, failure_kind="policy", result=result
    )
    video_frames = libero_main._build_video_frames(
        record.replay_frames,
        record.control_stalls,
        stall_source_frames=record.stall_source_frames,
        control_hz=20,
        video_fps=40,
        inference_schedule="synchronous",
        overlay_renderer=lambda frame, lines: "later-failure-overlay"
        if frame == "frame-observation-8"
        else pytest.fail("later failure used the wrong source frame"),
    )
    assert video_frames[-2:] == ("later-failure-overlay", "later-failure-overlay")
    assert "stall_source_frames" not in record.to_dict()


def test_synchronous_stall_overlay_uses_exact_current_mode_text():
    stall = timing.ControlStall(
        control_step=0,
        replan_index=0,
        started_offset_ns=100,
        duration_ns=125_000_000,
    )

    assert libero_main._synchronous_stall_overlay_lines(stall) == (
        "Synchronous inference",
        "Control stalled: 0.12 s",
    )


def test_network_retry_invalidates_client_and_reuses_init_state_and_seed(monkeypatch):
    environment = _Environment()
    holder = _ClientHolder([TimeoutError("stalled"), _actions()])
    initial_state = object()
    identity = _identity()
    monkeypatch.setattr(libero_main, "_prepare_observation", lambda obs, prompt, size: ({}, "frame"))

    def attempt(_attempt_number):
        return libero_main._run_attempt(
            environment=environment,
            client_holder=holder,
            initial_state=initial_state,
            identity=identity,
            task_description="pick up the block",
            args=_args(),
            max_steps=1,
        )

    record = libero_eval.run_episode_with_retries(identity, attempt, eval_seed=42)

    assert record.success
    assert record.attempts == 2
    assert holder.invalidations == 1
    assert environment.reset_states == [initial_state, initial_state]
    assert [request[inference.INFERENCE_SEED_KEY] for request in holder.client.requests] == [
        libero_eval.stable_replan_seed(42, identity, 0),
        libero_eval.stable_replan_seed(42, identity, 0),
    ]


def test_episode_is_persisted_before_video_error_is_audited(tmp_path):
    calls = []

    class Writer:
        def append_episode(self, record):
            calls.append(("episode", record.identity.episode_id))

        def append_artifact_error(self, error):
            calls.append(("artifact_error", error.episode_id))

    class Selector:
        def claim(self, record):
            return tmp_path / "video.mp4"

    def failed_encoder(*args, **kwargs):
        calls.append(("video", str(args[0])))
        raise RuntimeError("ffmpeg exited 1")

    attempt = libero_eval.AttemptResult(
        success=True,
        steps=1,
        replans=1,
        replay_frames=("frame",),
    )
    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(), 42, 1, success=True, result=attempt
    )

    persisted, artifact_error = libero_main._persist_episode_artifacts(
        record,
        Writer(),
        Selector(),
        video_encoder=failed_encoder,
    )

    assert persisted.replay_frames == ()
    assert artifact_error is not None
    assert [call[0] for call in calls] == ["episode", "video", "artifact_error"]


def test_unselected_video_is_not_expanded_or_encoded(monkeypatch):
    calls = []

    class Writer:
        def append_episode(self, record):
            calls.append("episode")

    class Selector:
        def claim(self, record):
            return None

    monkeypatch.setattr(
        libero_main,
        "_build_video_frames",
        lambda *args, **kwargs: pytest.fail("unselected video timeline was expanded"),
    )
    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(),
        42,
        1,
        success=True,
        result=libero_eval.AttemptResult(
            success=True,
            steps=1,
            replans=0,
            replay_frames=("frame",),
        ),
    )

    persisted, artifact_error = libero_main._persist_episode_artifacts(
        record,
        Writer(),
        Selector(),
        video_encoder=lambda *args, **kwargs: pytest.fail("unselected video was encoded"),
    )

    assert persisted.replay_frames == ()
    assert artifact_error is None
    assert calls == ["episode"]


def test_selected_video_disabled_enabled_pair_changes_only_artifact_timing(monkeypatch, tmp_path):
    encoded = {}
    audits = {}
    rollout_results = {}
    rollout_actions = {}

    monkeypatch.setattr(libero_main, "_prepare_observation", lambda obs, prompt, size: ({}, "frame"))
    for mode, show_waits in (("disabled", False), ("enabled", True)):
        environment = _Environment()
        rollout_results[mode] = libero_main._run_attempt(
            environment=environment,
            client_holder=_ClientHolder([_actions()]),
            initial_state=object(),
            identity=_identity(),
            task_description="pick up the block",
            args=dataclasses.replace(_args(), video_show_inference_waits=show_waits),
            max_steps=1,
        )
        rollout_actions[mode] = tuple(tuple(action) for action in environment.actions)

    assert rollout_actions["disabled"] == rollout_actions["enabled"] == (tuple(_actions()[0]),)
    assert (
        rollout_results["disabled"].steps,
        rollout_results["disabled"].success,
    ) == (
        rollout_results["enabled"].steps,
        rollout_results["enabled"].success,
    ) == (1, True)

    class Writer:
        def __init__(self, mode):
            self.mode = mode

        def append_episode(self, record):
            pass

        def append_video_audit(self, audit):
            audits[self.mode] = audit

        def append_artifact_error(self, error):
            pytest.fail(f"unexpected artifact error: {error}")

    class Selector:
        def __init__(self, mode):
            self.mode = mode

        def claim(self, record):
            return tmp_path / f"{self.mode}.mp4"

    class Reader:
        def __init__(self, path):
            self.path = path

        def get_meta_data(self):
            frames = encoded[self.path]
            return {"fps": 40.0, "duration": len(frames) / 40.0}

        def count_frames(self):
            return len(encoded[self.path])

        def close(self):
            pass

    def encoder(path, frames, *, fps):
        assert fps == 40
        encoded[path] = tuple(frames)

    monkeypatch.setattr(libero_main.imageio, "get_reader", Reader)
    monkeypatch.setattr(libero_main, "_draw_video_overlay", lambda frame, lines: frame)
    attempt = libero_eval.AttemptResult(
        success=True,
        steps=2,
        replans=1,
        inference_requests=(
            timing.InferenceRequest(0, started_offset_ns=0, duration_ns=25_000_000),
        ),
        control_stalls=(
            timing.ControlStall(0, 0, started_offset_ns=0, duration_ns=25_000_000),
        ),
        replay_frames=("frame-0", "frame-1"),
    )
    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(), 42, 1, success=True, result=attempt
    )

    persisted = {}
    for mode, show_waits in (("disabled", False), ("enabled", True)):
        persisted[mode], artifact_error = libero_main._persist_episode_artifacts(
            record,
            Writer(mode),
            Selector(mode),
            video_fps=40,
            video_show_inference_waits=show_waits,
            video_encoder=encoder,
        )
        assert artifact_error is None

    policy_fields = ("status", "success", "steps", "replans", "inference_requests", "control_stalls")
    assert tuple(getattr(persisted["disabled"], field) for field in policy_fields) == tuple(
        getattr(persisted["enabled"], field) for field in policy_fields
    )
    assert persisted["disabled"].steps == record.steps == len(record.replay_frames) == 2
    assert len(encoded[tmp_path / "disabled.mp4"]) == 4
    assert len(encoded[tmp_path / "enabled.mp4"]) == 5
    assert audits["disabled"].measured_stall_count == 1
    assert audits["disabled"].included_stall_count == 0
    assert audits["disabled"].expected_duration_ns == 100_000_000
    assert audits["enabled"].measured_stall_count == 1
    assert audits["enabled"].included_stall_count == 1
    assert audits["enabled"].expected_duration_ns == 125_000_000


def test_multi_frame_stall_renders_one_overlay_and_reuses_the_same_array_reference():
    rendered_overlay = object()
    renderer_calls = []

    def renderer(frame, lines):
        renderer_calls.append((frame, lines))
        return rendered_overlay

    stall = timing.ControlStall(0, 0, 0, 100_000_000)

    frames = libero_main._build_video_frames(
        ("frame",),
        (stall,),
        control_hz=20,
        video_fps=40,
        inference_schedule="synchronous",
        overlay_renderer=renderer,
    )

    assert renderer_calls == [
        (
            "frame",
            ("Synchronous inference", "Control stalled: 0.10 s"),
        )
    ]
    assert all(frame is rendered_overlay for frame in frames[:4])
    assert frames[4:] == ("frame", "frame")


def test_trailing_stall_uses_transient_request_frame_instead_of_last_control_frame():
    renderer_calls = []
    stall = timing.ControlStall(1, 1, 0, 50_000_000)

    frames = libero_main._build_video_frames(
        ("previous-control-frame",),
        (stall,),
        stall_source_frames=((1, "current-request-frame"),),
        control_hz=20,
        video_fps=40,
        inference_schedule="synchronous",
        overlay_renderer=lambda frame, lines: renderer_calls.append((frame, lines)) or "overlay",
    )

    assert frames == ("previous-control-frame", "previous-control-frame", "overlay", "overlay")
    assert renderer_calls == [
        (
            "current-request-frame",
            ("Synchronous inference", "Control stalled: 0.05 s"),
        )
    ]


def test_zero_step_policy_failure_encodes_quantized_wait_or_single_raw_padding_frame(
    monkeypatch, tmp_path
):
    encoded = []
    audits = []

    class Writer:
        def append_episode(self, record):
            pass

        def append_video_audit(self, audit):
            audits.append(audit)

        def append_artifact_error(self, error):
            pytest.fail("valid zero-step policy failure became an artifact error")

    class Selector:
        def claim(self, record):
            return tmp_path / "failure.mp4"

    class Reader:
        def get_meta_data(self):
            return {"fps": 40.0, "duration": len(encoded[-1]) / 40.0}

        def count_frames(self):
            return len(encoded[-1])

        def close(self):
            pass

    def encoder(path, frames, *, fps):
        encoded.append(tuple(frames))

    monkeypatch.setattr(libero_main.imageio, "get_reader", lambda path: Reader())
    monkeypatch.setattr(libero_main, "_draw_video_overlay", lambda frame, lines: "wait-overlay")
    result = libero_eval.AttemptResult(
        success=False,
        steps=0,
        replans=0,
        failure_kind="policy",
        inference_requests=(timing.InferenceRequest(0, 0, 50_000_000),),
        control_stalls=(timing.ControlStall(0, 0, 0, 50_000_000),),
        stall_source_frames=((0, "current-request-frame"),),
    )
    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(), 42, 1, success=False, failure_kind="policy", result=result
    )

    enabled, enabled_error = libero_main._persist_episode_artifacts(
        record,
        Writer(),
        Selector(),
        video_show_inference_waits=True,
        video_encoder=encoder,
    )
    disabled, disabled_error = libero_main._persist_episode_artifacts(
        record,
        Writer(),
        Selector(),
        video_show_inference_waits=False,
        video_encoder=encoder,
    )

    assert enabled_error is disabled_error is None
    assert enabled.steps == disabled.steps == record.steps == 0
    assert enabled.replans == disabled.replans == record.replans == 0
    assert enabled.success == disabled.success == record.success is False
    assert encoded == [
        ("wait-overlay", "wait-overlay"),
        ("current-request-frame",),
    ]
    assert audits[0].encoded_frame_count == 2
    assert audits[0].artifact_padding_frame_count == 0
    assert audits[1].artifact_padding_frame_count == 1
    assert audits[1].expected_encoded_frame_count == 1
    assert audits[1].encoded_frame_count == 1


def test_zero_quantized_first_wait_encodes_overlay_padding_without_fabricating_control_step(
    monkeypatch, tmp_path
):
    audits = []
    encoded = []

    class Writer:
        def append_episode(self, record):
            pass

        def append_video_audit(self, audit):
            audits.append(audit)

        def append_artifact_error(self, error):
            pytest.fail("sub-frame policy failure became an artifact error")

    class Selector:
        def claim(self, record):
            return tmp_path / "failure.mp4"

    class Reader:
        def get_meta_data(self):
            return {"fps": 40.0, "duration": 0.025}

        def count_frames(self):
            return 1

        def close(self):
            pass

    monkeypatch.setattr(libero_main.imageio, "get_reader", lambda path: Reader())
    monkeypatch.setattr(libero_main, "_draw_video_overlay", lambda frame, lines: "wait-overlay")

    result = libero_eval.AttemptResult(
        success=False,
        steps=0,
        replans=0,
        failure_kind="policy",
        inference_requests=(timing.InferenceRequest(0, 0, 1),),
        control_stalls=(timing.ControlStall(0, 0, 0, 1),),
        stall_source_frames=((0, "current-request-frame"),),
    )
    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(), 42, 1, success=False, failure_kind="policy", result=result
    )

    _, artifact_error = libero_main._persist_episode_artifacts(
        record,
        Writer(),
        Selector(),
        video_show_inference_waits=True,
        video_encoder=lambda path, frames, *, fps: encoded.append(tuple(frames)),
    )

    assert artifact_error is None
    assert audits[0].control_frame_count == 0
    assert audits[0].video_frame_count == 0
    assert audits[0].artifact_padding_frame_count == 1
    assert audits[0].encoded_frame_count == 1
    assert encoded == [("wait-overlay",)]


def test_wait_display_off_ignores_transient_trailing_source_and_keeps_control_video(
    monkeypatch, tmp_path
):
    encoded = []

    class Writer:
        def append_episode(self, record):
            pass

        def append_video_audit(self, audit):
            assert audit.included_stall_count == 0

        def append_artifact_error(self, error):
            pytest.fail("disabled wait display consumed a transient stall source")

    class Selector:
        def claim(self, record):
            return tmp_path / "failure.mp4"

    class Reader:
        def get_meta_data(self):
            return {"fps": 40.0, "duration": 0.05}

        def count_frames(self):
            return 2

        def close(self):
            pass

    monkeypatch.setattr(libero_main.imageio, "get_reader", lambda path: Reader())
    result = libero_eval.AttemptResult(
        success=False,
        steps=1,
        replans=1,
        failure_kind="policy",
        inference_requests=(
            timing.InferenceRequest(0, 0, 1),
            timing.InferenceRequest(1, 2, 50_000_000),
        ),
        control_stalls=(
            timing.ControlStall(0, 0, 0, 1),
            timing.ControlStall(1, 1, 2, 50_000_000),
        ),
        replay_frames=("control-frame",),
        stall_source_frames=((1, "current-request-frame"),),
    )
    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(), 42, 1, success=False, failure_kind="policy", result=result
    )

    _, artifact_error = libero_main._persist_episode_artifacts(
        record,
        Writer(),
        Selector(),
        video_show_inference_waits=False,
        video_encoder=lambda path, frames, *, fps: encoded.append(tuple(frames)),
    )

    assert artifact_error is None
    assert encoded == [("control-frame", "control-frame")]


def test_real_selector_keeps_first_zero_step_failure_mp4_and_summary_complete(
    monkeypatch, tmp_path
):
    writer = libero_eval.ArtifactWriter(tmp_path / "run")
    selector = libero_eval.VideoSelector(tmp_path / "run" / "videos")
    encoded_paths = []

    class Reader:
        def get_meta_data(self):
            return {"fps": 40.0, "duration": 0.025}

        def count_frames(self):
            return 1

        def close(self):
            pass

    def encoder(path, frames, *, fps):
        assert tuple(frames) == ("first-request-frame",)
        path.write_bytes(b"real-selector-test-mp4")
        encoded_paths.append(path)

    monkeypatch.setattr(libero_main.imageio, "get_reader", lambda path: Reader())
    records = []
    errors = []
    for init_index, frame in ((2, "first-request-frame"), (3, "later-failure-frame")):
        identity = dataclasses.replace(_identity(), init_state_index=init_index)
        result = libero_eval.AttemptResult(
            success=False,
            steps=0,
            replans=0,
            failure_kind="policy",
            error="missing actions",
            inference_requests=(timing.InferenceRequest(0, 0, 1),),
            control_stalls=(timing.ControlStall(0, 0, 0, 1),),
            stall_source_frames=((0, frame),),
        )
        record = libero_eval.EpisodeRecord.from_attempt(
            identity, 42, 1, success=False, failure_kind="policy", result=result
        )
        persisted, artifact_error = libero_main._persist_episode_artifacts(
            record,
            writer,
            selector,
            video_show_inference_waits=False,
            video_encoder=encoder,
        )
        records.append(persisted)
        if artifact_error is not None:
            errors.append(artifact_error)

    summary = libero_eval.aggregate_records(records, artifact_errors=errors)
    episode_lines = (tmp_path / "run" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    audit_lines = (tmp_path / "run" / "video_audit.jsonl").read_text(encoding="utf-8").splitlines()
    video_files = tuple((tmp_path / "run" / "videos").rglob("*.mp4"))

    assert len(episode_lines) == 2
    assert len(audit_lines) == 1
    assert len(encoded_paths) == len(video_files) == 1
    assert video_files[0].is_file()
    assert summary["eligible_episodes"] == 2
    assert summary["successes"] == 0
    assert summary["artifact_error_count"] == 0
    assert summary["acceptance_complete"] is True


def test_selected_zero_step_failure_without_request_frame_is_an_artifact_error(tmp_path):
    errors = []

    class Writer:
        def append_episode(self, record):
            pass

        def append_artifact_error(self, error):
            errors.append(error)

    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(),
        42,
        1,
        success=False,
        failure_kind="policy",
        result=libero_eval.AttemptResult(
            success=False,
            steps=0,
            replans=0,
            failure_kind="policy",
            error="missing actions",
        ),
    )

    _, artifact_error = libero_main._persist_episode_artifacts(
        record,
        Writer(),
        libero_eval.VideoSelector(tmp_path / "videos"),
        video_show_inference_waits=False,
        video_encoder=lambda *args, **kwargs: pytest.fail("missing source reached encoder"),
    )

    assert artifact_error is not None
    assert "request-time video source" in artifact_error.error
    assert errors == [artifact_error]


def test_discarded_infrastructure_attempt_cannot_leak_transient_frames_into_retry_record():
    discarded = libero_eval.AttemptResult(
        success=False,
        steps=0,
        replans=0,
        failure_kind="policy",
        stall_source_frames=((0, "discarded-frame"),),
    )

    def attempt(attempt_number):
        if attempt_number == 1:
            assert discarded.stall_source_frames == ((0, "discarded-frame"),)
            raise libero_eval.InfrastructureFailure("network", "retry this attempt")
        return libero_eval.AttemptResult(success=True, steps=1, replans=1)

    record = libero_eval.run_episode_with_retries(_identity(), attempt, eval_seed=42)

    assert record.success
    assert record.attempts == 2
    assert record.stall_source_frames == ()
    assert "stall_source_frames" not in record.to_dict()


def test_async_latency_without_underflow_adds_no_frames_but_partial_stall_uses_async_label():
    request = timing.InferenceRequest(0, 0, 300_000_000)
    renderer_calls = []

    no_stall_frames = libero_main._build_video_frames(
        ("frame",),
        (),
        control_hz=20,
        video_fps=40,
        inference_schedule="asynchronous",
        overlay_renderer=lambda frame, lines: renderer_calls.append(lines),
    )
    partial_stall = timing.ControlStall(
        0,
        0,
        250_000_000,
        50_000_000,
        reason="async_action_underflow",
    )
    partial_frames = libero_main._build_video_frames(
        ("frame",),
        (partial_stall,),
        control_hz=20,
        video_fps=40,
        inference_schedule="asynchronous",
        overlay_renderer=lambda frame, lines: renderer_calls.append(lines) or "async-overlay",
    )

    assert request.duration_ns == 300_000_000
    assert no_stall_frames == ("frame", "frame")
    assert partial_frames == ("async-overlay", "async-overlay", "frame", "frame")
    assert renderer_calls == [
        ("Waiting for policy actions", "Control stalled: 0.05 s"),
    ]


def test_duration_only_mismatch_logs_warning_without_artifact_error(monkeypatch, caplog, tmp_path):
    audits = []

    class Writer:
        def append_episode(self, record):
            pass

        def append_video_audit(self, audit):
            audits.append(audit)

        def append_artifact_error(self, error):
            pytest.fail("duration-only mismatch became an artifact error")

    class Selector:
        def claim(self, record):
            return tmp_path / "video.mp4"

    class Reader:
        def get_meta_data(self):
            return {"fps": 40.0, "duration": 0.2}

        def count_frames(self):
            return 5

        def close(self):
            pass

    monkeypatch.setattr(libero_main.imageio, "get_reader", lambda path: Reader())
    attempt = libero_eval.AttemptResult(
        success=True,
        steps=2,
        replans=1,
        inference_requests=(timing.InferenceRequest(0, 0, 25_000_000),),
        control_stalls=(timing.ControlStall(0, 0, 0, 25_000_000),),
        replay_frames=("frame-0", "frame-1"),
    )
    record = libero_eval.EpisodeRecord.from_attempt(
        _identity(), 42, 1, success=True, result=attempt
    )

    _, artifact_error = libero_main._persist_episode_artifacts(
        record,
        Writer(),
        Selector(),
        video_fps=40,
        video_show_inference_waits=True,
        video_encoder=lambda *args, **kwargs: None,
    )

    assert artifact_error is None
    assert not audits[0].timing_gate_pass
    assert "encoded duration deviates" in caplog.text
