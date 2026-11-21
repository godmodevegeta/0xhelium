"""Closed-loop parity with the oracle and run-to-run determinism (Requirements 7.7, 9.5)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import MODEL_XML, POLICY_PATH, REPO_ROOT
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner
from robustlab.physics.model import apply_model_parameters, load_model, resolve_ids
from robustlab.policy.actor import Actor

pytestmark = pytest.mark.integration
ATOL = 1e-6


def _runner(policy) -> EpisodeRunner:
    model = load_model(MODEL_XML)
    apply_model_parameters(model, policy.model_parameters)
    return EpisodeRunner(model, resolve_ids(model), policy, Actor(policy))


@pytest.mark.parametrize("command", [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.3, 0.6)])
def test_closed_loop_matches_oracle_500_ticks(policy, oracle, command) -> None:
    ours = _runner(policy)
    reference = oracle.Simulator(oracle.Policy.load(POLICY_PATH), root=REPO_ROOT)
    obs_ours = ours.reset(command).copy()
    obs_ref = reference.reset(command=command)
    max_obs = float(np.max(np.abs(obs_ours - obs_ref)))
    max_action = max_qpos = 0.0
    for tick in range(500):
        action = ours.tick().copy()
        step = reference.step()
        max_action = max(max_action, float(np.max(np.abs(action - step["action"]))))
        max_qpos = max(max_qpos, float(np.max(np.abs(ours.data.qpos - step["qpos"]))))
        max_obs = max(max_obs, float(np.max(np.abs(ours.obs - step["observation"]))))
        assert max_action <= ATOL and max_qpos <= ATOL, f"command={command} tick={tick}"
    print(
        f"\nclosed-loop oracle parity command={command}: 500 ticks, max abs error "
        f"action {max_action:.3e}, qpos {max_qpos:.3e}, observation {max_obs:.3e}"
    )


def test_same_configuration_gives_identical_trajectories(policy) -> None:
    config = EpisodeConfig(command=(1.0, 0.0, 0.0), seconds=4.0, seed=0)
    first = _runner(policy).run(config)
    second = _runner(policy).run(config)
    for name in ("time", "qpos", "qvel", "action", "command", "contact", "up_z", "trunk_z"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
