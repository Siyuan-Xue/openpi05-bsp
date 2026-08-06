import dataclasses

import pytest

from openpi_client import inference
from openpi_client import libero_eval

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
