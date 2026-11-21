"""`robustlab experiment` CLI (Requirement 7)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import REPO_ROOT
from robustlab.cli import EXIT_INVALID, EXIT_WRITE, main
from test_suite_config import BASE


def _config(tmp_path: Path, **changes) -> Path:
    cfg = json.loads(json.dumps(BASE))
    cfg.update(changes)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(cfg))
    return path


def _run(config: Path, out: Path, *extra: str) -> int:
    return main(["experiment", "--config", str(config), "--out", str(out), "--root", str(REPO_ROOT),
                 *extra])  # fmt: skip


def test_experiment_runs_and_prints_summary(tmp_path: Path, capsys) -> None:
    out = tmp_path / "exp"
    assert _run(_config(tmp_path), out, "--no-save-trajectories") == 0
    printed = capsys.readouterr().out
    for text in ("trial  start_s", "nominal   falls", "attacked  falls", "results_sha256"):
        assert text in printed
    report = json.loads((out / "experiment.json").read_text())
    assert report["trajectories_saved"] is False and not (out / "trials").exists()
    assert _run(_config(tmp_path), out) == EXIT_WRITE  # refuses to overwrite


def test_dry_run_simulates_nothing(tmp_path: Path, capsys) -> None:
    out = tmp_path / "dry"
    assert _run(_config(tmp_path), out, "--dry-run", "--trials", "2") == 0
    printed = capsys.readouterr().out
    assert "rollouts: 4" in printed and printed.count("start substep") == 2
    assert not out.exists()


def test_invalid_config_exits_2(tmp_path: Path, capsys) -> None:
    assert _run(_config(tmp_path, trials=3), tmp_path / "bad") == EXIT_INVALID
    assert "trials" in capsys.readouterr().err
    assert _run(_config(tmp_path, command=[3.0, 0.0, 0.0]), tmp_path / "bad2") == EXIT_INVALID
    assert not (tmp_path / "bad2" / "experiment.json").exists()
