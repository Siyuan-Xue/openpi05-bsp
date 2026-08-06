"""Fail-fast validation for LIBERO Compose host bind mounts."""

from collections.abc import Mapping
import itertools
import os
from pathlib import Path
import sys


_REQUIRED_MOUNT_VARIABLES = (
    "BSP_REPO_DIR",
    "BSP_EXPERIMENTS_DIR",
    "BSP_OPENPI_CACHE_DIR",
    "BSP_JAX_CACHE_DIR",
)


class PreflightError(ValueError):
    """A host mount cannot safely satisfy the Compose contract."""


def _resolve_directory(variable: str, environ: Mapping[str, str]) -> Path:
    raw_path = environ.get(variable, "").strip()
    if not raw_path:
        raise PreflightError(f"{variable} is required")

    path = Path(raw_path)
    if not path.is_absolute():
        raise PreflightError(f"{variable} must be an absolute path: {raw_path!r}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise PreflightError(f"{variable} does not exist: {path}") from error
    if resolved == Path(resolved.anchor):
        raise PreflightError(f"{variable} must not be the filesystem root: {resolved}")
    if not resolved.is_dir():
        raise PreflightError(f"{variable} must be a directory: {resolved}")
    return resolved


def validate_mount_roots(environ: Mapping[str, str] | None = None) -> dict[str, Path]:
    """Validate every required mount without creating or modifying host paths."""
    environment = os.environ if environ is None else environ
    resolved = {variable: _resolve_directory(variable, environment) for variable in _REQUIRED_MOUNT_VARIABLES}
    project_marker = resolved["BSP_REPO_DIR"] / "pyproject.toml"
    if not project_marker.is_file():
        raise PreflightError(f"BSP_REPO_DIR must contain the OpenPI pyproject.toml: {project_marker}")
    for (left_name, left_path), (right_name, right_path) in itertools.combinations(resolved.items(), 2):
        if left_path == right_path or left_path in right_path.parents or right_path in left_path.parents:
            raise PreflightError(
                f"Mount roots overlap: {left_name}={left_path} and {right_name}={right_path}"
            )
    return resolved


def main() -> int:
    try:
        resolved = validate_mount_roots()
    except PreflightError as error:
        print(f"LIBERO Compose mount preflight failed: {error}", file=sys.stderr)
        return 2

    print("LIBERO Compose mount preflight passed:")
    for variable in _REQUIRED_MOUNT_VARIABLES:
        print(f"{variable}={resolved[variable]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
