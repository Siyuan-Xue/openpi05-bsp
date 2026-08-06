import ast
import pathlib
import re
import shlex
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


def _long_bind_mounts(service: str) -> dict[str, dict[str, object]]:
    mounts = []
    current = None
    for line in service.splitlines():
        stripped = line.strip()
        if stripped == "- type: bind":
            if current is not None:
                mounts.append(current)
            current = {"type": "bind"}
        elif current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            if key in {"source", "target", "read_only", "create_host_path"}:
                current[key] = value.strip()
    if current is not None:
        mounts.append(current)

    by_variable = {}
    source_pattern = re.compile(r"^\$\{(?P<variable>[A-Z0-9_]+):\?[^}]+\}$")
    for mount in mounts:
        match = source_pattern.match(str(mount.get("source", "")))
        if match:
            by_variable[match.group("variable")] = {
                "target": mount.get("target"),
                "read_only": mount.get("read_only") == "true",
                "create_host_path": mount.get("create_host_path") == "true",
            }
    return by_variable


def _readme_client_args(readme: str) -> list[list[str]]:
    assignments = re.findall(r'export CLIENT_ARGS="(.*?)"', readme, flags=re.DOTALL)
    return [shlex.split(assignment.replace("\\\n", " ")) for assignment in assignments]


def _option_value(tokens: list[str], option: str) -> str:
    try:
        return tokens[tokens.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise AssertionError(f"Missing value for {option} in {tokens}") from error


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

    def test_policy_image_pins_uv_and_runs_without_runtime_sync(self):
        dockerfile = (_REPO_ROOT / "scripts/docker/serve_policy.Dockerfile").read_text()

        self.assertIn("COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/", dockerfile)
        self.assertIn("ENV PYTHONPATH=/app/src:/app/packages/openpi-client/src", dockerfile)
        command = next(line for line in dockerfile.splitlines() if line.startswith("CMD "))
        self.assertNotIn("uv ", command)
        self.assertIn("exec /.venv/bin/python scripts/serve_policy.py $SERVER_ARGS", command)

    def test_compose_requires_external_absolute_mount_roots(self):
        compose = (_REPO_ROOT / "examples/libero/compose.yml").read_text()
        runtime_mounts = _long_bind_mounts(_service_block(compose, "runtime"))
        server_mounts = _long_bind_mounts(_service_block(compose, "openpi_server"))

        self.assertEqual(
            runtime_mounts,
            {
                "BSP_REPO_DIR": {"target": "/app", "read_only": True, "create_host_path": False},
                "BSP_EXPERIMENTS_DIR": {
                    "target": "/experiments",
                    "read_only": False,
                    "create_host_path": False,
                },
            },
        )
        self.assertEqual(
            server_mounts,
            {
                "BSP_REPO_DIR": {"target": "/app", "read_only": True, "create_host_path": False},
                "BSP_EXPERIMENTS_DIR": {
                    "target": "/experiments",
                    "read_only": True,
                    "create_host_path": False,
                },
                "BSP_OPENPI_CACHE_DIR": {
                    "target": "/openpi_assets",
                    "read_only": False,
                    "create_host_path": False,
                },
                "BSP_JAX_CACHE_DIR": {
                    "target": "/jax_cache",
                    "read_only": False,
                    "create_host_path": False,
                },
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

    def test_readme_runs_mount_preflight_before_compose(self):
        readme = (_REPO_ROOT / "examples/libero/README.md").read_text()

        self.assertIn("python3 scripts/libero_compose_preflight.py", readme)
        preflight = readme.index("python3 scripts/libero_compose_preflight.py")
        compose_config = readme.index("docker compose -f examples/libero/compose.yml config")

        self.assertLess(preflight, compose_config)

    def test_readme_uses_only_real_nested_evaluator_flags(self):
        readme = (_REPO_ROOT / "examples/libero/README.md").read_text()
        main_tree = ast.parse((_REPO_ROOT / "examples/libero/main.py").read_text())
        args_class = next(node for node in main_tree.body if isinstance(node, ast.ClassDef) and node.name == "Args")
        allowed = {
            f"--args.{node.target.id.replace('_', '-')}"
            for node in args_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assignments = _readme_client_args(readme)

        self.assertGreaterEqual(len(assignments), 2)
        for tokens in assignments:
            flags = [token for token in tokens if token.startswith("--")]
            self.assertTrue(flags)
            self.assertTrue(all(flag.startswith("--args.") for flag in flags))
            self.assertTrue(set(flags).issubset(allowed))

    def test_readme_calibration_and_h16_examples_are_auditable(self):
        readme = (_REPO_ROOT / "examples/libero/README.md").read_text()
        calibration, baseline = _readme_client_args(readme)[:2]
        required_identity_options = {
            "--args.config-name",
            "--args.checkpoint-step",
            "--args.code-sha",
            "--args.dataset-revision",
            "--args.norm-hash",
            "--args.checkpoint",
            "--args.container-digest",
        }

        self.assertTrue(required_identity_options.issubset(calibration))
        self.assertEqual(_option_value(calibration, "--args.policy-variant"), "baseline")
        self.assertEqual(_option_value(calibration, "--args.task-suite-name"), "libero_spatial")
        self.assertEqual(_option_value(calibration, "--args.task-ids"), "0")
        self.assertEqual(_option_value(calibration, "--args.expected-action-horizon"), "10")
        self.assertEqual(_option_value(calibration, "--args.num-trials-per-task"), "1")
        self.assertEqual(_option_value(calibration, "--args.config-name"), "pi05_libero")
        self.assertEqual(_option_value(calibration, "--args.checkpoint-step"), "30000")

        self.assertTrue(required_identity_options.issubset(baseline))
        self.assertEqual(_option_value(baseline, "--args.policy-variant"), "baseline")
        self.assertEqual(_option_value(baseline, "--args.expected-action-horizon"), "16")
        self.assertEqual(_option_value(baseline, "--args.config-name"), "pi05_libero_baseline_h16")
        self.assertEqual(_option_value(baseline, "--args.checkpoint-step"), "30000")


if __name__ == "__main__":
    unittest.main()
