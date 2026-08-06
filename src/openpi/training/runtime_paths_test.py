import pathlib
import unittest

from openpi.training import runtime_paths


class JaxCompilationCachePathTest(unittest.TestCase):
    def test_explicit_environment_path_takes_precedence(self):
        result = runtime_paths.jax_compilation_cache_dir(
            {"JAX_COMPILATION_CACHE_DIR": "/mnt/workspace/openpi-bsp/cache/jax"}
        )

        self.assertEqual(result, "/mnt/workspace/openpi-bsp/cache/jax")

    def test_missing_environment_path_preserves_existing_default(self):
        result = runtime_paths.jax_compilation_cache_dir({})

        self.assertEqual(result, str(pathlib.Path.home() / ".cache" / "jax"))


if __name__ == "__main__":
    unittest.main()
