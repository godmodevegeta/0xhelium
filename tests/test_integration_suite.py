"""The committed 16-trial example suite: pairing, determinism and report consistency
(Requirements 3, 7.3)."""

from __future__ import annotations

import pytest

from conftest import POLICY_PATH, REPO_ROOT
from robustlab.experiments.stack import load_stack
from robustlab.experiments.suite import load_suite_config, run_suite

pytestmark = pytest.mark.integration

CONFIG = REPO_ROOT / "examples" / "m2" / "sideways-shove-stratified-16.json"


def test_example_suite_is_paired_and_deterministic(tmp_path) -> None:
    config = load_suite_config(CONFIG)
    stack = load_stack(REPO_ROOT, POLICY_PATH, config.scene)
    first = run_suite(config, stack, tmp_path / "a", save_trajectories=False)
    second = run_suite(config, stack, tmp_path / "b", save_trajectories=False)
    assert first["results_sha256"] == second["results_sha256"]
    assert first["sampling"]["start_substeps"] == second["sampling"]["start_substeps"]
    agg = first["aggregates"]
    assert agg["paired"]["pairing_ok_all"] and agg["paired"]["max_pre_attack_abs_diff"] == 0.0
    assert agg["nominal"]["distinct_trajectories"] == 1
    assert agg["nominal"]["physical_falls"] == 0 and agg["nominal"]["numerical_failures"] == 0
    assert agg["attacked"]["distinct_trajectories"] == config.trials
    starts = first["sampling"]["start_substeps"]
    assert len(set(starts)) == 16 and all(500 <= k < 756 for k in starts)
    att = agg["attacked"]
    print(
        f"\nexample suite: attacked falls {att['physical_falls']}/{att['valid_trials']}, "
        f"numerical {att['numerical_failures']}, results_sha256 {first['results_sha256']}"
    )
