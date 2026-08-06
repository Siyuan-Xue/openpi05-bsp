import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import libero_compose_preflight as preflight


_MODULE_PATH = Path(__file__).with_name("libero_compose_preflight.py")
_VARIABLES = (
    "BSP_REPO_DIR",
    "BSP_EXPERIMENTS_DIR",
    "BSP_OPENPI_CACHE_DIR",
    "BSP_JAX_CACHE_DIR",
)


class LiberoComposePreflightTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        repo = self.root / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("[project]\nname = 'openpi'\n")
        for directory in ("experiments", "openpi", "jax"):
            (self.root / directory).mkdir()
        self.environment = {
            "BSP_REPO_DIR": str(repo),
            "BSP_EXPERIMENTS_DIR": str(self.root / "experiments"),
            "BSP_OPENPI_CACHE_DIR": str(self.root / "openpi"),
            "BSP_JAX_CACHE_DIR": str(self.root / "jax"),
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_valid_existing_absolute_directories_are_resolved(self):
        result = preflight.validate_mount_roots(self.environment)

        self.assertEqual(result, {name: Path(value).resolve() for name, value in self.environment.items()})

    def test_missing_variable_is_rejected(self):
        del self.environment["BSP_JAX_CACHE_DIR"]

        with self.assertRaisesRegex(preflight.PreflightError, "BSP_JAX_CACHE_DIR.*required"):
            preflight.validate_mount_roots(self.environment)

    def test_relative_path_is_rejected(self):
        self.environment["BSP_EXPERIMENTS_DIR"] = "relative/experiments"

        with self.assertRaisesRegex(preflight.PreflightError, "BSP_EXPERIMENTS_DIR.*absolute"):
            preflight.validate_mount_roots(self.environment)

    def test_tilde_path_is_rejected_before_compose_can_misinterpret_it(self):
        self.environment["BSP_JAX_CACHE_DIR"] = "~/jax-cache"

        with self.assertRaisesRegex(preflight.PreflightError, "BSP_JAX_CACHE_DIR.*absolute"):
            preflight.validate_mount_roots(self.environment)

    def test_nonexistent_path_is_rejected(self):
        self.environment["BSP_OPENPI_CACHE_DIR"] = str(self.root / "missing")

        with self.assertRaisesRegex(preflight.PreflightError, "BSP_OPENPI_CACHE_DIR.*does not exist"):
            preflight.validate_mount_roots(self.environment)

    def test_filesystem_root_is_rejected(self):
        self.environment["BSP_JAX_CACHE_DIR"] = Path(self.root.anchor).as_posix()

        with self.assertRaisesRegex(preflight.PreflightError, "BSP_JAX_CACHE_DIR.*filesystem root"):
            preflight.validate_mount_roots(self.environment)

    def test_regular_file_is_rejected(self):
        file_path = self.root / "cache-file"
        file_path.write_text("not a directory")
        self.environment["BSP_JAX_CACHE_DIR"] = str(file_path)

        with self.assertRaisesRegex(preflight.PreflightError, "BSP_JAX_CACHE_DIR.*directory"):
            preflight.validate_mount_roots(self.environment)

    def test_repo_without_project_marker_is_rejected(self):
        Path(self.environment["BSP_REPO_DIR"], "pyproject.toml").unlink()

        with self.assertRaisesRegex(preflight.PreflightError, "BSP_REPO_DIR.*pyproject.toml"):
            preflight.validate_mount_roots(self.environment)

    def test_command_validates_the_real_process_environment(self):
        environment = dict(os.environ)
        environment.update(self.environment)

        result = subprocess.run(
            [sys.executable, str(_MODULE_PATH)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for variable in _VARIABLES:
            self.assertIn(f"{variable}=", result.stdout)


if __name__ == "__main__":
    unittest.main()
