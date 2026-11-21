"""Repository hygiene: tracked steering/spec files, license, lockfile and pinned deps.

Requirements 2.1, 2.2 and 12.2.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@requires_git
def test_git_tracks_kiro_license_and_lockfile() -> None:
    result = _git("ls-files")
    assert result.returncode == 0, result.stderr
    tracked = set(result.stdout.splitlines())
    for steering in ("delivery.md", "product.md", "structure.md", "tech.md"):
        assert f".kiro/steering/{steering}" in tracked
    spec_dir = ".kiro/specs/m0-walking-go1"
    for spec_file in ("requirements.md", "design.md", "tasks.md"):
        assert f"{spec_dir}/{spec_file}" in tracked
    assert "LICENSE" in tracked
    assert "uv.lock" in tracked


@requires_git
@pytest.mark.parametrize("path", [".kiro/", ".kiro/steering/delivery.md", "LICENSE"])
def test_gitignore_does_not_match_required_files(path: str) -> None:
    # --no-index evaluates the ignore rules even for already-tracked paths.
    # Exit status: 0 = ignored, 1 = not ignored, anything else = error.
    result = _git("check-ignore", "--no-index", "--verbose", path)
    assert result.returncode == 1, f"{path} is ignored: {result.stdout.strip()} {result.stderr}"


def test_gitignore_lists_runs() -> None:
    lines = [line.strip() for line in (REPO_ROOT / ".gitignore").read_text().splitlines()]
    assert "runs/" in lines


def test_runtime_dependencies_are_exactly_numpy_and_pinned_mujoco() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    deps = [re.sub(r"\s+", "", dep) for dep in project["dependencies"]]
    names = sorted(re.match(r"[A-Za-z0-9._-]+", dep).group(0).lower() for dep in deps)
    assert names == ["mujoco", "numpy"], deps
    assert "mujoco==3.15.0" in deps, deps
