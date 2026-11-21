"""`robustlab walk` scene and push flags (Requirements 5.4, 7.1-7.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import POLICY_PATH, REPO_ROOT
from robustlab.cli import EXIT_INVALID, main
from robustlab.experiments.trace import MANIFEST_REQUIRED_KEYS, read_run

BASE = ["walk", "--policy", str(POLICY_PATH), "--command", "0.5,0,0", "--root", str(REPO_ROOT)]


def _walk(out: Path, *extra: str) -> int:
    return main([*BASE, "--seconds", "0.4", "--out", str(out), *extra])


@pytest.mark.parametrize(
    "extra",
    [
        ("--push-force", "50"),  # partial flag set
        ("--push-force", "200", "--push-direction", "right", "--push-start", "0.1",
         "--push-duration", "0.1"),  # over 150 N
        ("--push-force", "50", "--push-direction", "sideways", "--push-start", "0.1",
         "--push-duration", "0.1"),
        ("--push-force", "50", "--push-direction", "right", "--push-start", "0.101",
         "--push-duration", "0.1"),  # not a multiple of 0.004 s
        ("--push-force", "50", "--push-direction", "right", "--push-start", "0.36",
         "--push-duration", "0.08"),  # ends after the 0.4 s horizon
    ],
)  # fmt: skip
def test_invalid_push_flags_exit_2_and_write_nothing(tmp_path: Path, extra) -> None:
    out = tmp_path / "run"
    assert _walk(out, *extra) == EXIT_INVALID
    assert not (out / "manifest.json").exists()


def test_push_run_manifest_and_trajectory(tmp_path: Path, capsys) -> None:
    out = tmp_path / "pushed"
    flags = ("--push-force", "40", "--push-direction", "left", "--push-start", "0.1")
    assert _walk(out, *flags, "--push-duration", "0.1") == 0
    printed = capsys.readouterr().out
    assert "FELL" in printed and "push world force / impulse" in printed
    trajectory, manifest = read_run(out)
    assert [k for k in MANIFEST_REQUIRED_KEYS if k not in manifest] == []
    assert manifest["schema_version"] == 2
    assert (
        manifest["scene"]["name"] == "fall-contact" and len(manifest["scene"]["fall_geoms"]) == 29
    )
    push = manifest["push"]
    assert push["world_force_n"] == [0.0, 40.0, 0.0]
    assert push["active_substeps"] == [25, 50] and push["impulse_ns"] == pytest.approx(4.0)
    assert sum(row[1] for row in trajectory["push_impulse"]) == pytest.approx(4.0)
    assert len(trajectory["nonfoot_contact"]) == 20
    fall = manifest["outcome"]["fall"]
    assert fall["nonfoot_contact_detection"] is True
    assert fall["kind"] in ("none", "physical", "numerical")


def test_feet_only_scene_without_push(tmp_path: Path) -> None:
    out = tmp_path / "m0"
    assert _walk(out, "--scene", "feet-only") == 0
    _, manifest = read_run(out)
    assert manifest["scene"]["name"] == "feet-only" and manifest["push"] is None
    assert manifest["outcome"]["fall"]["nonfoot_contact_detection"] is False
