from openpi_client import websocket_client_policy
import pytest


class _FakePacker:
    def pack(self, observation):
        return observation


class _FakeWebsocket:
    def __init__(self, response=b"response"):
        self.response = response
        self.sent = []
        self.recv_calls = []

    def send(self, data):
        self.sent.append(data)

    def recv(self, *args, **kwargs):
        self.recv_calls.append((args, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _client(timeout):
    client = object.__new__(websocket_client_policy.WebsocketClientPolicy)
    client._packer = _FakePacker()
    client._ws = _FakeWebsocket()
    client._inference_timeout = timeout
    return client


def test_infer_uses_configured_response_timeout(monkeypatch):
    client = _client(17.0)
    monkeypatch.setattr(websocket_client_policy.msgpack_numpy, "unpackb", lambda response: {"raw": response})

    assert client.infer({"observation": 1}) == {"raw": b"response"}

    assert client._ws.sent == [{"observation": 1}]
    assert client._ws.recv_calls == [((), {"timeout": 17.0})]


def test_default_inference_timeout_retains_unbounded_recv(monkeypatch):
    client = _client(None)
    monkeypatch.setattr(websocket_client_policy.msgpack_numpy, "unpackb", lambda response: {"raw": response})

    client.infer({"observation": 1})

    assert client._ws.recv_calls == [((), {})]


def test_inference_timeout_propagates_for_evaluator_network_classification():
    client = _client(17.0)
    client._ws.response = TimeoutError("policy response stalled")

    with pytest.raises(TimeoutError, match="stalled"):
        client.infer({"observation": 1})
