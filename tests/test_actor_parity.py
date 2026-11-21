"""Actor parity with the weeble oracle (Requirement 6.5)."""

from __future__ import annotations

import numpy as np

from conftest import MODEL_XML, POLICY_PATH, REPO_ROOT
from robustlab.policy.actor import Actor

SEED = 6
ATOL = 1e-7


# Feature: m0-walking-go1, Property 5: Actor matches the Oracle
def test_property_actor_matches_oracle(policy, oracle) -> None:
    reference = oracle.Policy.load(POLICY_PATH)
    actor = Actor(policy)
    rng = np.random.default_rng(SEED)
    unit = rng.uniform(-3.0, 3.0, size=(1000, policy.obs_dim))
    observations = (policy.mean + unit * policy.std).astype(np.float32)
    errors = np.empty(len(observations))
    for i, obs in enumerate(observations):
        ours = actor.act(obs)
        theirs = reference.action(obs)
        assert theirs.dtype == np.float32
        errors[i] = np.max(np.abs(ours - theirs))
        assert errors[i] <= ATOL, f"seed={SEED} case={i} max_abs_err={errors[i]:.3e}"
    print(f"\nactor-oracle parity: 1000 observations, max abs error {errors.max():.3e}")


def test_reset_parity(policy, oracle) -> None:
    """Demo: action at the nominal reset observation (oracle-built observation)."""
    reference = oracle.Policy.load(POLICY_PATH)
    sim = oracle.Simulator(reference, root=REPO_ROOT)
    assert sim.model.opt.timestep == 0.004 and MODEL_XML.exists()
    obs = sim.reset(command=(1.0, 0.0, 0.0))
    ours = Actor(policy).act(np.asarray(obs, np.float32)).copy()
    theirs = reference.action(obs)
    error = float(np.max(np.abs(ours - theirs)))
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print(f"\nreset observation command (1, 0, 0)\n  ours   {ours}\n  oracle {theirs}")
    print(f"  max abs error {error:.3e}")
    assert error <= ATOL
