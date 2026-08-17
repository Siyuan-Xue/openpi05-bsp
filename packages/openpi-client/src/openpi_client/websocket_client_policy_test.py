import threading

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy
import pytest
import websockets.sync.server


class _FakePacker:
    def pack(self, observation):
        return observation


class _FakeWebsocket:
    def __init__(self, response=b"response"):
        self.response = response
        self.sent = []
        self.recv_calls = []
        self.close_calls = 0

    def send(self, data):
        self.sent.append(data)

    def recv(self, *args, **kwargs):
        self.recv_calls.append((args, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self):
        self.close_calls += 1


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


def test_infer_packed_sends_exact_bytes_and_decodes_response_once(monkeypatch):
    client = _client(None)
    decoded = []

    def unpack(response):
        decoded.append(response)
        return {"decoded": response}

    monkeypatch.setattr(websocket_client_policy.msgpack_numpy, "unpackb", unpack)

    result = client.infer_packed(b"immutable request")

    assert result == {"decoded": b"response"}
    assert client._ws.sent == [b"immutable request"]
    assert decoded == [b"response"]


def test_infer_packed_polls_receive_until_cancellation():
    cancel_event = threading.Event()
    client = _client(None)

    def recv(*args, **kwargs):
        client._ws.recv_calls.append((args, kwargs))
        cancel_event.set()
        raise TimeoutError("poll expired")

    client._ws.recv = recv

    with pytest.raises(RuntimeError, match="cancelled"):
        client.infer_packed(
            b"request",
            cancel_event=cancel_event,
            recv_poll_interval_s=0.025,
        )

    assert client._ws.recv_calls == [((), {"timeout": 0.025})]


def test_connection_retry_wait_is_interrupted_by_cancel_event(monkeypatch):
    class _CancelDuringWait:
        def __init__(self):
            self.wait_calls = []
            self.cancelled = False

        def is_set(self):
            return self.cancelled

        def wait(self, timeout):
            self.wait_calls.append(timeout)
            self.cancelled = True
            return True

    cancel_event = _CancelDuringWait()
    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionRefusedError("not ready")),
    )
    monkeypatch.setattr(
        websocket_client_policy.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(AssertionError("time.sleep must not be called")),
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1",
            8000,
            cancel_event=cancel_event,
        )

    assert cancel_event.wait_calls == [5]


def test_connect_uses_bounded_close_timeout_and_common_sync_arguments(monkeypatch):
    connection = _FakeWebsocket(response=msgpack_numpy.packb({"metadata": True}))
    connect_calls = []

    def connect(uri, **kwargs):
        connect_calls.append((uri, kwargs))
        return connection

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect)

    client = websocket_client_policy.WebsocketClientPolicy(
        "127.0.0.1",
        8000,
        api_key="secret",
        connection_timeout=2.0,
        close_timeout=0.25,
    )

    assert connect_calls[0][0] == "ws://127.0.0.1:8000"
    assert connect_calls[0][1]["additional_headers"] == {"Authorization": "Api-Key secret"}
    assert connect_calls[0][1]["compression"] is None
    assert connect_calls[0][1]["max_size"] is None
    assert connect_calls[0][1]["close_timeout"] == 0.25
    assert 0 < connect_calls[0][1]["open_timeout"] <= 2.0
    client.close()


def test_default_connection_preserves_websockets_close_timeout(monkeypatch):
    connection = _FakeWebsocket(response=msgpack_numpy.packb({"metadata": True}))
    connect_kwargs = []

    def connect(uri, **kwargs):
        connect_kwargs.append(kwargs)
        return connection

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect)

    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)

    assert "close_timeout" not in connect_kwargs[0]
    client.close()


def test_metadata_receive_poll_is_interrupted_by_cancel_event(monkeypatch):
    cancel_event = threading.Event()
    connection = _FakeWebsocket()

    def recv(*args, **kwargs):
        connection.recv_calls.append((args, kwargs))
        cancel_event.set()
        raise TimeoutError("metadata poll expired")

    connection.recv = recv
    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client,
        "connect",
        lambda *args, **kwargs: connection,
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1",
            8000,
            cancel_event=cancel_event,
            close_timeout=0.25,
        )

    assert connection.recv_calls == [((), {"timeout": 0.05})]
    assert connection.close_calls == 1


def test_localhost_binary_protocol_uses_shared_websockets_sync_surface(monkeypatch):
    received_requests = []
    received_headers = []
    handler_errors = []

    def handler(connection):
        try:
            received_headers.append(connection.request.headers["Authorization"])
            connection.send(msgpack_numpy.packb({"robot": "libero"}))
            request = connection.recv()
            received_requests.append(msgpack_numpy.unpackb(request))
            connection.send(msgpack_numpy.packb({"actions": [[0.5, -0.5]]}))
        except BaseException as error:
            handler_errors.append(error)

    server = websockets.sync.server.serve(handler, "127.0.0.1", 0)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    port = server.socket.getsockname()[1]
    real_connect = websocket_client_policy.websockets.sync.client.connect
    connect_kwargs = []
    close_calls = []

    class _RecordingConnection:
        def __init__(self, connection):
            self._connection = connection

        def send(self, payload):
            return self._connection.send(payload)

        def recv(self, *args, **kwargs):
            return self._connection.recv(*args, **kwargs)

        def close(self):
            close_calls.append(True)
            return self._connection.close()

    def recording_connect(uri, **kwargs):
        connect_kwargs.append(kwargs)
        return _RecordingConnection(real_connect(uri, **kwargs))

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", recording_connect)
    client = None
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1",
            port,
            api_key="loopback",
            connection_timeout=1.0,
            inference_timeout=1.0,
            close_timeout=0.25,
        )
        assert client.get_server_metadata() == {"robot": "libero"}
        result = client.infer_packed(msgpack_numpy.packb({"observation": [1, 2]}))
        assert result == {"actions": [[0.5, -0.5]]}
    finally:
        if client is not None:
            client.close()
        server.shutdown()
        server_thread.join(1.0)

    assert not server_thread.is_alive()
    assert handler_errors == []
    assert received_headers == ["Api-Key loopback"]
    assert received_requests == [{"observation": [1, 2]}]
    assert connect_kwargs[0]["additional_headers"] == {"Authorization": "Api-Key loopback"}
    assert connect_kwargs[0]["compression"] is None
    assert connect_kwargs[0]["max_size"] is None
    assert connect_kwargs[0]["close_timeout"] == 0.25
    assert close_calls == [True]
