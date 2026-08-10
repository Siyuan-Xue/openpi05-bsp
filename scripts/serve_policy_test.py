import http
import types

import pytest
import tyro

from openpi.serving import websocket_policy_server
from scripts import serve_policy


def test_cli_help_preserves_serving_arguments_and_success_exit(capsys):
    with pytest.raises(SystemExit) as exit_info:
        tyro.cli(serve_policy.Args, args=["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for argument in ("--env", "--default-prompt", "--port", "--record"):
        assert argument in help_text
    assert "checkpoint" in help_text


def test_checkpoint_arguments_flow_to_policy_creation(monkeypatch):
    captured = {}

    def create_trained_policy(config, directory, *, default_prompt):
        captured.update(config=config, directory=directory, default_prompt=default_prompt)
        return "policy"

    monkeypatch.setattr(serve_policy._config, "get_config", lambda name: f"config:{name}")
    monkeypatch.setattr(serve_policy._policy_config, "create_trained_policy", create_trained_policy)
    args = serve_policy.Args(
        default_prompt="pick up the block",
        policy=serve_policy.Checkpoint(config="pi05_libero_bsp_h16", dir="checkpoint/10000"),
    )

    assert serve_policy.create_policy(args) == "policy"
    assert captured == {
        "config": "config:pi05_libero_bsp_h16",
        "directory": "checkpoint/10000",
        "default_prompt": "pick up the block",
    }


def test_health_endpoint_returns_ok_and_other_paths_continue_websocket_handshake():
    responses = []
    connection = types.SimpleNamespace(
        respond=lambda status, body: responses.append((status, body)) or "response",
    )

    assert websocket_policy_server._health_check(connection, types.SimpleNamespace(path="/healthz")) == "response"
    assert responses == [(http.HTTPStatus.OK, "OK\n")]
    assert websocket_policy_server._health_check(connection, types.SimpleNamespace(path="/infer")) is None
