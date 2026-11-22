"""Closed-loop reflex checks with MuJoCo rollouts (M6). Marked integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from robustlab.adapt.evaluate import Context, Evaluator, calibrate, run_episode
from robustlab.adapt.harden import HardenConfig, decode, load_calibration, run_hardening
from robustlab.adapt.reflex import N_PARAMS, ReflexActor
from robustlab.adapt.sets import load_sets
from robustlab.experiments.suite import trajectory_digest
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner

pytestmark = pytest.mark.integration

SETS = REPO_ROOT / "examples/m6/hardening-sets-v1.json"
CALIBRATION = REPO_ROOT / "examples/m6/reflex-calibration-v1.json"


@pytest.fixture(scope="module")
def stack():
    from robustlab.experiments.stack import load_stack

    return load_stack(REPO_ROOT, Path("policies/go1-walk-s0.json"), "fall-contact")


@pytest.fixture(scope="module")
def calibration(stack):
    return load_calibration(CALIBRATION, stack.policy.sha256)


@pytest.fixture(scope="module")
def sets():
    return load_sets(SETS, REPO_ROOT)


@pytest.fixture(scope="module")
def ctx(stack, calibration):
    return Context(stack, calibration)


def _by_id(sets, split, id_):
    return next(e for e in sets.split(split) if e.id == id_)


def test_zero_residual_episode_is_bit_identical(stack, calibration, sets) -> None:
    spec = _by_id(sets, "train", "m2-100N-500")
    push = spec.push.scenario()
    base = EpisodeRunner(stack.model, stack.ids, stack.policy, stack.actor).run(
        EpisodeConfig(spec.command, spec.horizon_s, 0, push))  # fmt: skip
    zero = ReflexActor(stack.actor, stack.policy, calibration, np.zeros(N_PARAMS))
    wrapped = EpisodeRunner(stack.model, stack.ids, stack.policy, zero).run(
        EpisodeConfig(spec.command, spec.horizon_s, 0, push))  # fmt: skip
    assert trajectory_digest(base) == trajectory_digest(wrapped)
    np.testing.assert_array_equal(base.qpos, wrapped.qpos)
    assert len(zero.activations) == base.n_ticks and not any(zero.activations)


def test_stop_on_fall_does_not_change_metrics(ctx, sets) -> None:
    spec = _by_id(sets, "train", "m4-cmaes-s3")
    full, _ = run_episode(ctx, spec, None, stop_on_fall=False)
    short, _ = run_episode(ctx, spec, None, stop_on_fall=True)
    assert full["outcome"] == "fell" and short["stopped_at_fall"]
    keys = ("outcome", "reason", "fall_time_s", "time_after_push_s", "max_tilt_after_push_deg",
            "progress_m", "progress_until_s", "min_trunk_z")  # fmt: skip
    assert {k: full[k] for k in keys} == {k: short[k] for k in keys}
    assert short["ticks_simulated"] < full["ticks_simulated"]


def test_confirmed_failures_fall_under_the_frozen_baseline(ctx, sets) -> None:
    confirmed = [e for s in ("train", "validation_preview") for e in sets.split(s)
                 if e.confirmed_failure]  # fmt: skip
    assert len(confirmed) == 17
    for spec in confirmed:
        metrics, _ = run_episode(ctx, spec, None)
        assert metrics["outcome"] == "fell", spec.id


def test_push_is_applied_with_the_reflex(ctx, sets) -> None:
    spec = _by_id(sets, "train", "m3-150N-610")
    metrics, record = run_episode(ctx, spec, decode(np.full(N_PARAMS, 0.4)), stop_on_fall=False,
                                  keep=True)  # fmt: skip
    impulse = record.push_impulse.sum(axis=0)
    assert np.allclose(impulse, [0.0, -150.0 * 0.1, 0.0], atol=1e-9)
    assert metrics["reflex"]["active_ticks"] > 0


def test_committed_calibration_reproduces(stack, calibration) -> None:
    again = calibrate(stack)
    assert again.to_dict() == calibration.to_dict()
    assert json.loads(CALIBRATION.read_text())["center"] == list(again.center)


def test_parallel_equals_sequential(stack, calibration, sets) -> None:
    params = tuple(decode(np.full(N_PARAMS, 0.2)))
    tasks = [(None, e) for e in sets.split("train")[:3]] + [
        (params, e) for e in sets.split("validation_preview")[:3]
    ]
    sequential = Evaluator(Context(stack, calibration), 1).map(tasks)
    pool = Evaluator(Context(stack, calibration), 2)
    try:
        parallel = pool.map(tasks)
    finally:
        pool.close()
    assert parallel == sequential


def test_small_hardening_run_reproduces(stack, calibration, sets) -> None:
    config = HardenConfig(seed=1, max_generations=2, popsize=4, focus="m2-100N-500", workers=1)
    a = run_hardening(stack, calibration, sets, config)
    b = run_hardening(stack, calibration, sets, config)
    assert a["reproducibility_sha256"] == b["reproducibility_sha256"]
    assert a["accounting"]["optimization_rollouts"] == 2 * 4 * 4
    assert a["status"] == "candidate"
    # baseline measurements are the frozen policy's: every confirmed failure falls
    train = a["measurements"]["train"]["baseline"]
    assert all(m["outcome"] == "fell" for m in train if m["confirmed_failure"])
