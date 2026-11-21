"""Smoke tests for the `robustlab` console entry point (Requirement 2.5)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _robustlab_command() -> list[str]:
    """Return the installed console script if present, else the module entry point."""
    script = Path(sys.executable).with_name("robustlab")
    if script.exists():
        return [str(script)]
    return [sys.executable, "-m", "robustlab.cli"]


def test_doctor_exits_zero_and_reports_versions() -> None:
    result = subprocess.run(
        [*_robustlab_command(), "doctor"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    labels = {line.split(maxsplit=1)[0] for line in lines if line.strip()}
    assert {"python", "numpy", "mujoco", "platform", "blas"} <= labels, result.stdout
    mujoco_line = next(line for line in lines if line.startswith("mujoco "))
    assert mujoco_line.split() == ["mujoco", "3.15.0"], mujoco_line
