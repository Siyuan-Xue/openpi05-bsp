"""Dependency-free contracts for the specialized repository documentation."""

from __future__ import annotations

import fnmatch
from pathlib import Path
import re
import subprocess
import unittest
from urllib.parse import unquote

_ROOT = Path(__file__).resolve().parents[1]
_ARCHITECTURE = _ROOT / "docs/repository_architecture.md"
_CANONICAL_DOCUMENTS = (
    _ROOT / "README.md",
    _ROOT / "CONTRIBUTING.md",
    _ARCHITECTURE,
    _ROOT / "docs/pi05_libero_bsp_phase1_server.md",
    _ROOT / "docs/norm_stats.md",
    _ROOT / "docs/remote_inference.md",
    _ROOT / "examples/libero/README.md",
)
_OBSOLETE_PATHS = (
    "docs/docker.md",
    "docs/pi05_libero_bsp_server_state.md",
    "docs/superpowers",
    ".superpowers/sdd/pi05-libero-bsp-sdd-plan",
)
_LOCAL_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_DELETION_PATTERN = re.compile(r"<!--\s*deletion-pattern:\s*([^\s]+)\s*-->")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class RepositoryDocumentationContractTest(unittest.TestCase):
    def test_canonical_documents_replace_obsolete_process_artifacts(self):
        self.assertTrue(all(path.is_file() for path in _CANONICAL_DOCUMENTS))
        tracked = set(
            subprocess.run(
                ["git", "ls-files"],
                cwd=_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
        for obsolete in _OBSOLETE_PATHS:
            matches = [path for path in tracked if path == obsolete or path.startswith(f"{obsolete}/")]
            for relative_path in matches:
                self.assertFalse((_ROOT / relative_path).exists())
            obsolete_path = _ROOT / obsolete
            if obsolete_path.is_dir():
                self.assertFalse(any(path.is_file() for path in obsolete_path.rglob("*")))
            else:
                self.assertFalse(obsolete_path.exists())

    def test_readme_exposes_only_the_specialized_supported_surface(self):
        readme = _read(_ROOT / "README.md")
        for config_name in (
            "pi05_libero",
            "pi05_libero_baseline_h16",
            "pi05_libero_bsp_h16",
            "pi05_libero_baseline_lora_h16",
            "pi05_libero_bsp_lora_h16",
        ):
            self.assertIn(config_name, readme)
        for milestone in ("0", "1000", "2000", "5000", "10000"):
            self.assertRegex(readme, rf"(?<!\d){milestone}(?!\d)")
        for link in (
            "docs/repository_architecture.md",
            "docs/pi05_libero_bsp_phase1_server.md",
            "docs/norm_stats.md",
            "docs/remote_inference.md",
        ):
            self.assertIn(link, readme)
        for removed_link in (
            "docs/docker.md",
            "examples/aloha_real",
            "examples/aloha_sim",
            "examples/droid",
            "examples/simple_client",
            "examples/ur5",
            "examples/inference.ipynb",
            "scripts/train_pytorch.py",
        ):
            self.assertNotIn(f"]({removed_link}", readme)

    def test_contributing_guide_enforces_the_specialized_fork_boundary(self):
        guide = _read(_ROOT / "CONTRIBUTING.md")
        self.assertIn("phase1-runtime-2c09840", guide)
        self.assertIn("23 dependency-free", guide)
        self.assertIn("server-only", guide.lower())
        self.assertIn("/mnt/data/siyuanxue", guide)
        self.assertNotIn("Physical-Intelligence/openpi/issues", guide)
        self.assertNotIn("Physical-Intelligence/openpi/discussions", guide)

    def test_host_runbook_has_no_removed_container_route(self):
        runbook = _read(_ROOT / "docs/pi05_libero_bsp_phase1_server.md")
        for forbidden in (
            "docker",
            "compose",
            "nvidia-container",
            "libero_compose_preflight",
            "install_docker",
        ):
            self.assertNotIn(forbidden, runbook.lower())
        for required in (
            "/mnt/data/siyuanxue",
            "phase1-runtime-2c09840",
            "2c098404a3cce0c86f0b863dcd8d3aeb18a55d94",
            "Python 3.11.9",
            "Python 3.8.20",
            "__EGL_VENDOR_LIBRARY_FILENAMES",
            "physical-intelligence/libero",
            "v2.0",
            "WebSocket",
        ):
            self.assertIn(required, runbook)

    def test_host_runbook_bootstraps_pinned_uv_without_remote_script_execution(self):
        runbook = _read(_ROOT / "docs/pi05_libero_bsp_phase1_server.md")
        for required in (
            "UV_VERSION=0.11.32",
            "https://releases.astral.sh/github/uv/releases/download/$UV_VERSION",
            "uv-x86_64-unknown-linux-gnu.tar.gz",
            "aab924fd522efd06f1c5f3b93a243864fc453132c94b2dc49f1371b528a4b967",
            '"$UV_RELEASE_BASE/$UV_ARCHIVE_NAME.sha256"',
            'sha256sum -c "$(basename "$UV_CHECKSUM_FILE")"',
            "--strip-components=1",
            'test -x "$UV_BIN"',
        ):
            self.assertIn(required, runbook)
        self.assertNotRegex(runbook, r"curl[^\n]*\|\s*sh\b")

    def test_architecture_deletion_patterns_cover_every_removed_path(self):
        architecture = _read(_ARCHITECTURE)
        patterns = _DELETION_PATTERN.findall(architecture)
        self.assertTrue(patterns, "architecture document must declare deletion-pattern markers")
        result = subprocess.run(
            ["git", "diff", "--name-status", "phase1-pre-slim-1b976fc...HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        deleted = [line.split("\t", 1)[1] for line in result.stdout.splitlines() if line.startswith("D\t")]
        uncovered = [path for path in deleted if not any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)]
        self.assertEqual(uncovered, [])

    def test_canonical_markdown_links_resolve(self):
        broken: list[str] = []
        for document in _CANONICAL_DOCUMENTS:
            for raw_target in _LOCAL_LINK.findall(_read(document)):
                target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                relative = unquote(target.split("#", 1)[0])
                if relative and not (document.parent / relative).resolve().exists():
                    broken.append(f"{document.relative_to(_ROOT)} -> {target}")
        self.assertEqual(broken, [])

    def test_documented_data_paths_respect_the_only_writable_namespace(self):
        unsafe: list[str] = []
        for document in _CANONICAL_DOCUMENTS:
            for match in re.finditer(r"/mnt/data[^\s`'\"<>)]*", _read(document)):
                path = match.group(0).rstrip(".,;:")
                if path != "/mnt/data" and not path.startswith("/mnt/data/siyuanxue"):
                    unsafe.append(f"{document.relative_to(_ROOT)}: {path}")
        self.assertEqual(unsafe, [])

    def test_canonical_documents_contain_no_machine_identity_or_secret(self):
        machine_identity = re.compile(r"(?:dsw-[a-z0-9-]{8,}|GPU-[0-9a-f-]{16,})", re.IGNORECASE)
        secret = re.compile(r"(?:hf_|sk-)[A-Za-z0-9_-]{12,}")
        findings = []
        for document in _CANONICAL_DOCUMENTS:
            text = _read(document)
            for pattern in (machine_identity, secret):
                findings.extend(f"{document.relative_to(_ROOT)}: {match.group(0)}" for match in pattern.finditer(text))
        self.assertEqual(findings, [])

    def test_lightweight_ci_runs_documentation_contract(self):
        workflow = _read(_ROOT / ".github/workflows/test.yml")
        self.assertIn("scripts.repository_documentation_contract_test", workflow)
        self.assertIn("fetch-depth: 0", workflow)


if __name__ == "__main__":
    unittest.main()
