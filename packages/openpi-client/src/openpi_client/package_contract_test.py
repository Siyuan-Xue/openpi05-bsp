"""Public packaging compatibility checks for the standalone client."""

from importlib import metadata


def test_client_metadata_supports_python_3_8_and_newer():
    assert metadata.metadata("openpi-client")["Requires-Python"] == ">=3.8"
