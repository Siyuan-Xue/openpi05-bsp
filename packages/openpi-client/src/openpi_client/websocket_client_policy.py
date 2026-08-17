import logging
import threading
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class _OperationCancelledError(RuntimeError):
    pass


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        connection_timeout: Optional[float] = None,
        inference_timeout: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
        close_timeout: Optional[float] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        if connection_timeout is not None and connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if inference_timeout is not None and inference_timeout <= 0:
            raise ValueError("inference_timeout must be positive")
        if close_timeout is not None and close_timeout <= 0:
            raise ValueError("close_timeout must be positive")
        self._connection_timeout = connection_timeout
        self._inference_timeout = inference_timeout
        self._cancel_event = cancel_event
        self._close_timeout = close_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = time.monotonic() + self._connection_timeout if self._connection_timeout is not None else None
        while True:
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise _OperationCancelledError("Policy connection cancelled")
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                timeout_kwargs = {} if deadline is None else {"open_timeout": max(0.001, deadline - time.monotonic())}
                close_timeout_kwargs = {} if self._close_timeout is None else {"close_timeout": self._close_timeout}
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    **close_timeout_kwargs,
                    **timeout_kwargs,
                )
                try:
                    metadata_response = self._recv_server_metadata(conn, deadline)
                    metadata = msgpack_numpy.unpackb(metadata_response)
                except Exception:
                    conn.close()
                    raise
                return conn, metadata
            except ConnectionRefusedError as error:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for policy server at {self._uri}") from error
                logging.info("Still waiting for server...")
                sleep_seconds = 5 if deadline is None else min(5, max(0, deadline - time.monotonic()))
                if self._cancel_event is None:
                    time.sleep(sleep_seconds)
                elif self._cancel_event.wait(sleep_seconds):
                    raise _OperationCancelledError("Policy connection cancelled") from error

    def _recv_server_metadata(
        self,
        conn: websockets.sync.client.ClientConnection,
        deadline: Optional[float],
    ):
        if self._cancel_event is None:
            return conn.recv() if deadline is None else conn.recv(timeout=max(0.001, deadline - time.monotonic()))

        while True:
            if self._cancel_event.is_set():
                raise _OperationCancelledError("Policy connection cancelled")
            if deadline is None:
                recv_timeout = 0.05
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for policy server at {self._uri}")
                recv_timeout = min(0.05, remaining)
            try:
                return conn.recv(timeout=recv_timeout)
            except TimeoutError as error:
                if self._cancel_event.is_set():
                    raise _OperationCancelledError("Policy connection cancelled") from error
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for policy server at {self._uri}") from error

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        return self.infer_packed(self._packer.pack(obs))

    def infer_packed(
        self,
        payload: bytes,
        *,
        cancel_event: Optional[threading.Event] = None,
        recv_poll_interval_s: Optional[float] = None,
    ) -> Dict:  # noqa: UP006
        if recv_poll_interval_s is not None and recv_poll_interval_s <= 0:
            raise ValueError("recv_poll_interval_s must be positive")
        if cancel_event is not None and cancel_event.is_set():
            raise _OperationCancelledError("Policy inference cancelled")

        self._ws.send(payload)
        if cancel_event is None:
            response = (
                self._ws.recv()
                if self._inference_timeout is None
                else self._ws.recv(timeout=self._inference_timeout)
            )
        else:
            poll_interval = 0.05 if recv_poll_interval_s is None else recv_poll_interval_s
            deadline = (
                None
                if self._inference_timeout is None
                else time.monotonic() + self._inference_timeout
            )
            while True:
                if cancel_event.is_set():
                    raise _OperationCancelledError("Policy inference cancelled")
                if deadline is None:
                    recv_timeout = poll_interval
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for policy inference response")
                    recv_timeout = min(poll_interval, remaining)
                try:
                    response = self._ws.recv(timeout=recv_timeout)
                    break
                except TimeoutError:
                    if cancel_event.is_set():
                        raise _OperationCancelledError("Policy inference cancelled")
                    if deadline is not None and time.monotonic() >= deadline:
                        raise
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass

    def close(self) -> None:
        """Close the current websocket so an evaluator can reconnect after infrastructure failure."""
        self._ws.close()
