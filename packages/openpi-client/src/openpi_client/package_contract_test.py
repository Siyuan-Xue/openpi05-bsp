"""Public packaging compatibility checks for the standalone client."""

from __future__ import annotations

from pathlib import Path
import re


_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


class TestPackageContract:
    def test_client_metadata_supports_python_3_8_and_newer(self):
        """Catch a client release that declares compatibility below the host evaluator floor."""
        metadata = _PYPROJECT.read_text(encoding="utf-8")
        match = re.search(r'^requires-python\s*=\s*"(?P<specifier>[^"]+)"\s*$', metadata, re.MULTILINE)

        assert match is not None
        assert match.group("specifier") == ">=3.8"
