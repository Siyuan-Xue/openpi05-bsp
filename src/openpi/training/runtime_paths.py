"""Dependency-free runtime path resolution used before JAX initialization."""

from collections.abc import Mapping
import os
from pathlib import Path


def jax_compilation_cache_dir(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the configured JAX cache, preserving OpenPI's existing fallback."""
    environment = os.environ if environ is None else environ
    if configured := environment.get("JAX_COMPILATION_CACHE_DIR"):
        return str(Path(configured).expanduser())
    return str(Path.home() / ".cache" / "jax")
