"""Observation, control loop, trace and upright-predicate tests (Requirements 2.3, 7, 9, 10.1-10.2)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from conftest import MODEL_XML, POLICY_PATH, REPO_ROOT
from robustlab.experiments.trace import (
    MANIFEST_REQUIRED_KEYS,
    read_run,
    trajectory_dict,
    write_run,
)
from robustlab.experiments.walk import (
    EpisodeConfig,
    EpisodeRecord,
    EpisodeRunner,
    WalkCriteria,
    upright_checks,
)
from robustlab.physics.model import apply_model_parameters, load_model, resolve_ids
from robustlab.policy.actor import Actor

SEED = 7


@pytest.fixture(scope="module")
def model(policy):
    loaded = load_model(MODEL_XML)
    apply_model_parameters(loaded, policy.model_parameters)
    return loaded


class StubActor:
    """Returns a scripted action sequence and records the observations it receives."""

    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, np.float32)
        self.seen: list[np.ndarray] = []

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.seen.append(obs.copy())
        return self.actions[len(self.seen) - 1].copy()


def _slices(policy):
    return {name: slice(off, off + size) for name, off, size in policy.fields}


def test_reset_observation(model, policy) -> None:
    runner = EpisodeRunner(model, resolve_ids(model), policy, Actor(policy))
    obs = runner.reset((1.0, 0.0, 0.0))
    f = _slices(policy)
    assert obs.dtype == np.float32 and obs.shape == (48,)
    np.testing.assert_allclose(obs[f["gravity"]], [0, 0, -1], atol=1e-6)
    np.testing.assert_array_equal(obs[f["joint_position"]], np.zeros(12, np.float32))
    np.testing.assert_array_equal(obs[f["joint_velocity"]], np.zeros(12, np.float32))
    np.testing.assert_array_equal(obs[f["last_action"]], np.zeros(12, np.float32))
    np.testing.assert_array_equal(obs[f["command"]], np.float32([1, 0, 0]))
    np.testing.assert_array_equal(runner.data.ctrl, runner.data.qpos[7:])


def test_run_rejects_non_multiple_horizon(model, policy) -> None:
    runner = EpisodeRunner(model, resolve_ids(model), policy, Actor(policy))
    with pytest.raises(ValueError):
        runner.run(EpisodeConfig((0, 0, 0), 0.03))


# Feature: m0-walking-go1, Property 6: Observation carries the delayed action
def test_property_observation_carries_delayed_action(model, policy) -> None:
    rng = np.random.default_rng(SEED)
    ids = resolve_ids(model)
    action_field = _slices(policy)["last_action"]
    for case in range(100):
        n = int(rng.integers(2, 12))
        actions = rng.uniform(-0.3, 0.3, size=(n + 1, 12)).astype(np.float32)
        stub = StubActor(actions)
        runner = EpisodeRunner(model, ids, policy, stub)
        runner.reset(tuple(rng.uniform(-0.5, 0.5, 3)))
        for _ in range(n):
            runner.tick()
        stub.act(runner.obs)  # capture o[n] without stepping
        context = f"seed={SEED} case={case} n={n}"
        # stub.seen[k] is o[k]; o[0] and o[1] carry zeros, o[t+1] carries a[t-1].
        np.testing.assert_array_equal(stub.seen[0][action_field], 0, err_msg=context)
        np.testing.assert_array_equal(stub.seen[1][action_field], 0, err_msg=context)
        for t in range(1, n):
            np.testing.assert_array_equal(
                stub.seen[t + 1][action_field], actions[t - 1], err_msg=f"{context} t={t}"
            )


# Feature: m0-walking-go1, Property 7: Tick control invariant
def test_property_tick_control_invariant(model, policy) -> None:
    rng = np.random.default_rng(SEED)
    ids = resolve_ids(model)
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def on_control(_model, data):  # MuJoCo calls this once per mj_step substep
        calls.append((data.ctrl.copy(), data.xfrc_applied.copy()))

    boundary = [np.full(12, -1.0), np.full(12, 1.0), np.zeros(12)]
    actions = boundary + [rng.uniform(-1, 1, 12) for _ in range(100)]
    stub = StubActor(np.asarray(actions))
    runner = EpisodeRunner(model, ids, policy, stub)
    runner.reset((0.5, 0.0, 0.0))
    mujoco.set_mjcb_control(on_control)
    try:
        for case, action in enumerate(np.asarray(actions, np.float32)):
            runner.data.xfrc_applied[:] = rng.normal(0, 50, runner.data.xfrc_applied.shape)
            calls.clear()
            time_before = runner.data.time
            runner.tick()
            expected = (policy.default_pose + np.float32(0.5) * action).astype(np.float64)
            context = f"seed={SEED} case={case}"
            assert len(calls) == 5, context
            assert math.isclose(runner.data.time - time_before, 0.02, abs_tol=1e-12), context
            for ctrl, xfrc in calls:
                np.testing.assert_array_equal(ctrl, expected, err_msg=context)
                assert not np.any(xfrc), context
    finally:
        mujoco.set_mjcb_control(None)


def _synthetic_record(rng: np.random.Generator, n: int) -> EpisodeRecord:
    return EpisodeRecord(
        ctrl_dt=0.02,
        sim_dt=0.004,
        n_substeps=5,
        initial_qpos=rng.normal(size=19),
        initial_qvel=rng.normal(size=18),
        time=np.arange(1, n + 1) * 0.02,
        qpos=rng.normal(size=(n, 19)) * 10 ** rng.uniform(-8, 3),
        qvel=rng.normal(size=(n, 18)),
        action=rng.uniform(-1, 1, size=(n, 12)).astype(np.float32),
        command=rng.uniform(-1, 1, size=(n, 3)).astype(np.float32),
        contact=rng.random((n, 4)) < 0.5,
        up_z=rng.uniform(-1, 1, n),
        trunk_z=rng.uniform(0, 0.4, n),
        finite=np.ones(n, bool),
        warning_total=np.zeros(n, np.int64),
        tick_ns=rng.integers(1, 10**6, n),
    )


# Feature: m0-walking-go1, Property 8: Trace serialization round trip
def test_property_trace_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    for case in range(100):
        n = int(rng.integers(1, 40))
        record = _synthetic_record(rng, n)
        manifest = {
            "schema_version": 1,
            "run_id": f"case{case}",
            "nested": {"x": float(rng.normal()), "list": rng.normal(size=5).tolist()},
            "flag": bool(rng.random() < 0.5),
            "none": None,
            "tiny": float(rng.normal() * 1e-300),
        }
        run_dir = tmp_path / f"run{case}"
        write_run(run_dir, record, manifest)
        trajectory, manifest_back = read_run(run_dir)
        context = f"seed={SEED} case={case}"
        assert trajectory == trajectory_dict(record), context
        assert manifest_back == manifest, context
        for name in ("time", "qpos", "qvel", "action", "command", "contact", "up_z", "trunk_z"):
            np.testing.assert_array_equal(
                np.asarray(trajectory[name], getattr(record, name).dtype),
                getattr(record, name),
                err_msg=f"{context} {name}",
            )
        leftovers = sorted(p.name for p in run_dir.iterdir())
        assert leftovers == ["manifest.json", "trajectory.json"], context


def test_write_run_refuses_to_overwrite(tmp_path: Path) -> None:
    record = _synthetic_record(np.random.default_rng(0), 3)
    write_run(tmp_path, record, {"schema_version": 1})
    with pytest.raises(FileExistsError):
        write_run(tmp_path, record, {"schema_version": 1})
    write_run(tmp_path, record, {"schema_version": 1, "again": True}, overwrite=True)
    assert read_run(tmp_path)[1]["again"] is True


def _reference_upright(up_z, trunk_z, finite, warnings, criteria):
    c45 = math.cos(math.radians(criteria.tilt_deg))
    c60 = math.cos(math.radians(criteria.tilt_compare_deg))
    ok45 = all(
        u >= c45 and z >= criteria.min_trunk_z and f and w == 0
        for u, z, f, w in zip(up_z, trunk_z, finite, warnings, strict=True)
    )
    ok60 = all(u >= c60 for u in up_z)
    return ok45, ok60


# Feature: m0-walking-go1, Property 9: Upright predicate
def test_property_upright_predicate() -> None:
    rng = np.random.default_rng(SEED)
    criteria = WalkCriteria()
    c45 = math.cos(math.radians(45))
    c60 = math.cos(math.radians(60))
    special_up = [c45, np.nextafter(c45, -1), c60, np.nextafter(c60, -1), 1.0, np.nan, -np.inf]
    special_z = [0.18, np.nextafter(0.18, 0), 0.3, np.nan, np.inf]
    for case in range(300):
        n = int(rng.integers(1, 30))
        up_z = rng.uniform(0.3, 1.0, n)
        trunk_z = rng.uniform(0.15, 0.35, n)
        finite = rng.random(n) > 0.03
        warnings = (rng.random(n) < 0.03).astype(np.int64)
        if case % 2 == 0:  # plant boundary values
            for _ in range(int(rng.integers(1, 4))):
                up_z[int(rng.integers(n))] = special_up[int(rng.integers(len(special_up)))]
                trunk_z[int(rng.integers(n))] = special_z[int(rng.integers(len(special_z)))]
        times = np.arange(1, n + 1) * 0.02
        got = upright_checks(up_z, trunk_z, finite, warnings, times, criteria)
        ok45, ok60 = _reference_upright(up_z, trunk_z, finite, warnings, criteria)
        context = f"seed={SEED} case={case}"
        assert got["upright_45"] == ok45, context
        assert got["upright_60"] == ok60, context
        assert (got["first_violation_time"] is None) == ok45, context


def test_exact_boundaries() -> None:
    criteria = WalkCriteria()
    c45 = math.cos(math.radians(45))
    c60 = math.cos(math.radians(60))
    ok, finite, warn, t = np.array([True]), np.array([True]), np.array([0]), np.array([0.02])
    at_45 = upright_checks(np.array([c45]), np.array([0.18]), finite, warn, t, criteria)
    assert at_45["upright_45"] and at_45["upright_60"]
    at_60 = upright_checks(np.array([c60]), np.array([0.3]), ok, warn, t, criteria)
    assert not at_60["upright_45"] and at_60["upright_60"]
    below = upright_checks(
        np.array([1.0]), np.array([np.nextafter(0.18, 0)]), ok, warn, t, criteria
    )
    assert not below["upright_45"] and below["first_violation_time"] == 0.02


_WALK_PROBE = """
import json, sys
from robustlab.cli import main
code = main(["walk", "--policy", {policy!r}, "--command", "0.5,0,0", "--seconds", "0.4",
             "--out", {out!r}, "--root", {root!r}])
loaded = sorted(n for n in sys.modules if n.split(".")[0] in ("jax", "brax", "flax", "torch"))
print("PROBE=" + json.dumps({{"code": code, "loaded": loaded}}))
"""


def test_short_walk_subprocess_manifest_and_import_hygiene(tmp_path: Path) -> None:
    out = tmp_path / "short"
    probe = _WALK_PROBE.format(policy=str(POLICY_PATH), out=str(out), root=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    line = next(x for x in result.stdout.splitlines() if x.startswith("PROBE="))
    probe_result = json.loads(line.removeprefix("PROBE="))
    assert probe_result == {"code": 0, "loaded": []}, result.stdout + result.stderr
    trajectory, manifest = read_run(out)
    missing = [key for key in MANIFEST_REQUIRED_KEYS if key not in manifest]
    assert missing == []
    assert manifest["n_ticks"] == 20 and len(trajectory["qpos"]) == 20
    assert manifest["sources"]["weeble"]["commit"] == "60878341072256ccc68f6118ec23a298c55e1bb9"
    assert manifest["versions"]["mujoco"] == "3.15.0"
    assert manifest["policy"]["sha256"].startswith("42138b2b")
    assert manifest["assets"]["assets/go1/walk.xml"].startswith("5f9e9cba")
    for key in ("ctrl_dt", "sim_dt", "n_substeps"):
        assert manifest[key] == {"ctrl_dt": 0.02, "sim_dt": 0.004, "n_substeps": 5}[key]
    # 20 Ticks are all inside the 50-Tick warm-up, so no latency statistics are reported.
    assert manifest["latency"]["measured_ticks"] == 0
