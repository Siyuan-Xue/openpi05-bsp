import ast
import pathlib
import re
import tomllib
import unittest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _service_block(compose: str, service_name: str) -> str:
    lines = compose.splitlines()
    start = lines.index(f"  {service_name}:")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("  ") and not line.startswith("    ") and line.rstrip().endswith(":"):
            end = index
            break
    return "\n".join(lines[start:end])


def _required_mounts(service: str) -> dict[str, tuple[str, str | None]]:
    mounts = {}
    pattern = re.compile(
        r"^\s*-\s+\$\{(?P<variable>[A-Z0-9_]+):\?[^}]+\}:(?P<target>/[^:\s]+)(?::(?P<mode>[^\s]+))?\s*$"
    )
    for line in service.splitlines():
        if match := pattern.match(line):
            mounts[match.group("variable")] = (match.group("target"), match.group("mode"))
    return mounts


class ServerRuntimeContractTest(unittest.TestCase):
    def test_server_python_and_scipy_are_reproducibly_pinned(self):
        project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
        lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())
        openpi = next(package for package in lock["package"] if package["name"] == "openpi")
        scipy = next(package for package in lock["package"] if package["name"] == "scipy")

        self.assertIn("scipy==1.15.3", project["project"]["dependencies"])
        self.assertEqual((_REPO_ROOT / ".python-version").read_text().strip(), "3.11.9")
        self.assertIn({"name": "scipy"}, openpi["dependencies"])
        self.assertIn({"name": "scipy", "specifier": "==1.15.3"}, openpi["metadata"]["requires-dist"])
        self.assertEqual(scipy["version"], "1.15.3")

    def test_train_uses_dependency_free_jax_cache_path_resolver(self):
        tree = ast.parse((_REPO_ROOT / "scripts/train.py").read_text())
        cache_updates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "jax_compilation_cache_dir"
        ]

        self.assertEqual(len(cache_updates), 1)
        cache_path = cache_updates[0].args[1]
        self.assertIsInstance(cache_path, ast.Call)
        self.assertIsInstance(cache_path.func, ast.Attribute)
        self.assertEqual(cache_path.func.attr, "jax_compilation_cache_dir")

    def test_policy_image_pins_uv_and_runs_against_the_frozen_lock(self):
        dockerfile = (_REPO_ROOT / "scripts/docker/serve_policy.Dockerfile").read_text()

        self.assertIn("COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/", dockerfile)
        command = next(line for line in dockerfile.splitlines() if line.startswith("CMD "))
        self.assertRegex(command, r"uv run\s+--frozen\s+--no-dev\s+scripts/serve_policy\.py")

    def test_compose_requires_external_absolute_mount_roots(self):
        compose = (_REPO_ROOT / "examples/libero/compose.yml").read_text()
        runtime_mounts = _required_mounts(_service_block(compose, "runtime"))
        server_mounts = _required_mounts(_service_block(compose, "openpi_server"))

        self.assertEqual(
            runtime_mounts,
            {
                "BSP_REPO_DIR": ("/app", None),
                "BSP_EXPERIMENTS_DIR": ("/experiments", None),
            },
        )
        self.assertEqual(
            server_mounts,
            {
                "BSP_REPO_DIR": ("/app", None),
                "BSP_EXPERIMENTS_DIR": ("/experiments", "ro"),
                "BSP_OPENPI_CACHE_DIR": ("/openpi_assets", None),
                "BSP_JAX_CACHE_DIR": ("/jax_cache", None),
            },
        )
        self.assertNotIn("$PWD", compose)
        self.assertNotIn("../../data", compose)
        self.assertNotIn("~", compose)

    def test_compose_uses_headless_egl_by_default(self):
        compose = (_REPO_ROOT / "examples/libero/compose.yml").read_text()
        runtime = _service_block(compose, "runtime")
        server = _service_block(compose, "openpi_server")

        self.assertIn("MUJOCO_GL=${MUJOCO_GL:-egl}", runtime)
        self.assertIn("PYOPENGL_PLATFORM=egl", runtime)
        self.assertNotIn("DISPLAY=", runtime)
        self.assertNotIn("/tmp/.X11-unix", runtime)
        self.assertIn("OPENPI_DATA_HOME=/openpi_assets", server)
        self.assertIn("JAX_COMPILATION_CACHE_DIR=/jax_cache", server)


if __name__ == "__main__":
    unittest.main()
