"""Stdlib-only pytest contracts for the specialized repository documentation."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

_ROOT = Path(__file__).resolve().parents[2]
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


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_canonical_documents_replace_obsolete_process_artifacts():
    assert all(path.is_file() for path in _CANONICAL_DOCUMENTS)
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
            assert not ((_ROOT / relative_path).exists())
        obsolete_path = _ROOT / obsolete
        if obsolete_path.is_dir():
            assert not (any(path.is_file() for path in obsolete_path.rglob("*")))
        else:
            assert not (obsolete_path.exists())


def test_readme_exposes_only_the_specialized_supported_surface():
    readme = _read(_ROOT / "README.md")
    for config_name in (
        "pi05_libero",
        "pi05_libero_baseline_h16",
        "pi05_libero_bsp_h16",
        "pi05_libero_baseline_lora_h16",
        "pi05_libero_bsp_lora_h16",
    ):
        assert config_name in readme
    for milestone in ("0", "1000", "2000", "5000", "10000"):
        assert re.search(f"(?<!\\d){milestone}(?!\\d)", readme) is not None
    for link in (
        "docs/repository_architecture.md",
        "docs/pi05_libero_bsp_phase1_server.md",
        "docs/norm_stats.md",
        "docs/remote_inference.md",
    ):
        assert link in readme
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
        assert f"]({removed_link}" not in readme


def test_contributing_guide_enforces_the_specialized_fork_boundary():
    guide = _read(_ROOT / "CONTRIBUTING.md")
    assert "phase1-runtime-2c09840" in guide
    assert "one test style and one runner: pytest" in guide
    assert "stdlib-only" in guide
    assert "server-only" in guide.lower()
    assert "/mnt/data/siyuanxue" in guide
    assert "Physical-Intelligence/openpi/issues" not in guide
    assert "Physical-Intelligence/openpi/discussions" not in guide


def test_host_runbook_has_no_removed_container_route():
    runbook = _read(_ROOT / "docs/pi05_libero_bsp_phase1_server.md")
    for forbidden in (
        "docker",
        "compose",
        "nvidia-container",
        "libero_compose_preflight",
        "install_docker",
    ):
        assert forbidden not in runbook.lower()
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
        assert required in runbook


def test_host_runbook_bootstraps_pinned_uv_without_remote_script_execution():
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
        assert required in runbook
    assert re.search("curl[^\\n]*\\|\\s*sh\\b", runbook) is None


def test_canonical_markdown_links_resolve():
    broken: list[str] = []
    for document in _CANONICAL_DOCUMENTS:
        for raw_target in _LOCAL_LINK.findall(_read(document)):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = unquote(target.split("#", 1)[0])
            if relative and not (document.parent / relative).resolve().exists():
                broken.append(f"{document.relative_to(_ROOT)} -> {target}")
    assert broken == []


def test_documented_data_paths_respect_the_only_writable_namespace():
    unsafe: list[str] = []
    for document in _CANONICAL_DOCUMENTS:
        for match in re.finditer(r"/mnt/data[^\s`'\"<>)]*", _read(document)):
            path = match.group(0).rstrip(".,;:")
            if path != "/mnt/data" and not path.startswith("/mnt/data/siyuanxue"):
                unsafe.append(f"{document.relative_to(_ROOT)}: {path}")
    assert unsafe == []


def test_canonical_documents_contain_no_machine_identity_or_secret():
    machine_identity = re.compile(r"(?:dsw-[a-z0-9-]{8,}|GPU-[0-9a-f-]{16,})", re.IGNORECASE)
    secret = re.compile(r"(?:hf_|sk-)[A-Za-z0-9_-]{12,}")
    findings = []
    for document in _CANONICAL_DOCUMENTS:
        text = _read(document)
        for pattern in (machine_identity, secret):
            findings.extend(f"{document.relative_to(_ROOT)}: {match.group(0)}" for match in pattern.finditer(text))
    assert findings == []
