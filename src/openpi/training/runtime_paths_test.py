import pathlib

from openpi.training import runtime_paths


def test_explicit_environment_path_takes_precedence():
    result = runtime_paths.jax_compilation_cache_dir(
        {"JAX_COMPILATION_CACHE_DIR": "/mnt/workspace/openpi-bsp/cache/jax"}
    )

    assert result == "/mnt/workspace/openpi-bsp/cache/jax"


def test_missing_environment_path_preserves_existing_default():
    result = runtime_paths.jax_compilation_cache_dir({})

    assert result == str(pathlib.Path.home() / ".cache" / "jax")
