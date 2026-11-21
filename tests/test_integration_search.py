"""Physics discovery: reproducible, and the final failure re-simulates to the recorded trajectory
(M4 Requirements 5.2, 7.5, 8.2)."""

from __future__ import annotations

import json

import pytest

from conftest import POLICY_PATH, REPO_ROOT
from robustlab.cli import main
from robustlab.experiments.stack import load_stack
from robustlab.experiments.suite import load_suite_config, run_suite
from robustlab.search.engine import PhysicsEvaluator, SearchConfig, run_discovery
from robustlab.search.space import Candidate

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def stack():
    return load_stack(REPO_ROOT, POLICY_PATH, "fall-contact")


def test_random_discovery_reproduces_and_replays(stack, tmp_path) -> None:
    config = SearchConfig("random", budget=32, seed=0, shrink_budget=8)
    first = run_discovery(config, stack)
    second = run_discovery(config, stack)
    assert first["results_sha256"] == second["results_sha256"]
    assert first["evaluations"] == second["evaluations"]
    assert first["accounting"]["search_rollouts"] == 32
    final = first["final"]
    assert final is not None and first["search"]["found_failure"]
    c = Candidate(*(final["candidate"][k] for k in
                    ("force_n", "direction_deg", "start_substep", "duration_substeps")))  # fmt: skip
    # Final replay: authoritative paired re-simulation matches the discovered scenario.
    trial, _, _ = PhysicsEvaluator(stack).paired_trial(c)
    assert trial["attacked"]["outcome"] == "fell"
    assert trial["attacked"]["digest"] == final["digest"]
    assert trial["paired"]["pairing_ok"] and trial["stance_at_start"] is not None
    # The exported scenario reruns through the M2 experiment runner with the same outcome.
    scenario = load_suite_config(final["scenario_config"])
    report = run_suite(scenario, stack, tmp_path / "rerun", save_trajectories=False)
    assert report["trials"][0]["attacked"]["digest"] == final["digest"]


def test_discover_cli_writes_record(tmp_path, capsys) -> None:
    out = tmp_path / "cli"
    args = ["discover", "--algorithm", "cmaes", "--budget", "16", "--seed", "3",
            "--shrink-budget", "4", "--out", str(out), "--root", str(REPO_ROOT)]  # fmt: skip
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "results_sha256" in printed and "rollouts" in printed
    record = json.loads((out / "discovery.json").read_text())
    assert (
        record["config"]["algorithm"] == "cmaes" and record["accounting"]["search_rollouts"] == 16
    )
    assert record["search"]["library"].startswith("pycma")
    assert main(args) == 3  # refuses to overwrite
