import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


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
        self._connection_timeout = connection_timeout
        self._inference_timeout = inference_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = time.monotonic() + self._connection_timeout if self._connection_timeout is not None else None
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                timeout_kwargs = {} if deadline is None else {"open_timeout": max(0.001, deadline - time.monotonic())}
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    **timeout_kwargs,
                )
                try:
                    metadata_response = (
                        conn.recv() if deadline is None else conn.recv(timeout=max(0.001, deadline - time.monotonic()))
                    )
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
                time.sleep(sleep_seconds)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = (
            self._ws.recv() if self._inference_timeout is None else self._ws.recv(timeout=self._inference_timeout)
        )
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
