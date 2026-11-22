"""Run multi-axis fixed scenarios (M5): single paired runs and push-timing suites.

Cases recorded for a single run:
- `scenario`: the scenario as specified (friction, payload and push if any)
- `matched`: same friction, payload, command, horizon and reset; push removed (only with a push)
- `original`: the trained nominal condition (nominal friction, no payload, no push)

The primary comparison ("ghost") is `matched` when the scenario has a push, otherwise `original`.
All physics goes through the M1 EpisodeRunner and the M2 `run_trial` / `run_suite`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from robustlab.experiments.suite import (
    episode_summary,
    load_suite_config,
    run_suite,
    run_trial,
)
from robustlab.experiments.trace import _atomic_write_json, git_revision, write_run
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner, WalkCriteria, evaluate
from robustlab.physics.model import FOOT_GEOMS
from robustlab.physics.scenario_model import (
    MODEL_CACHE,
    ScenarioModel,
    ScenarioModelCache,
    friction_report,
    mass_report,
)
from robustlab.physics.scene import Payload
from robustlab.scenarios.schema import NOMINAL_FRICTION_MU, Scenario

RESULT_NAME = "scenario_result.json"
RESULT_SCHEMA_VERSION = 1


def scenario_stack(base_stack, entry: ScenarioModel):
    """A Stack on the scenario model sharing the base stack's policy and actor (same thread)."""
    return dataclasses.replace(base_stack, scene_info=entry.scene_info, ids=entry.ids)


def payload_of(scenario: Scenario) -> Payload | None:
    if scenario.payload_mass_kg <= 0:
        return None
    return Payload(scenario.payload_mass_kg, scenario.payload_offset_xyz_m)


def model_for(
    scenario: Scenario, base_stack, cache: ScenarioModelCache = MODEL_CACHE
) -> ScenarioModel:
    return cache.get(base_stack, scenario.ground_friction_mu, payload_of(scenario))


def foot_slip(model: mujoco.MjModel, record, last_tick: int | None = None) -> dict:
    """Mean horizontal speed of each foot site over Ticks where that foot touches the floor at both
    ends of the Tick (recorded contact flags). Computed post hoc with mj_kinematics on recorded
    qpos; up to `last_tick` (inclusive) so motion after a fall is excluded."""
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f) for f in FOOT_GEOMS]
    n = record.n_ticks if last_tick is None else min(record.n_ticks, last_tick + 1)
    data = mujoco.MjData(model)
    xy = np.empty((n, len(sites), 2))
    for k in range(n):
        data.qpos[:] = record.qpos[k]
        mujoco.mj_kinematics(model, data)
        xy[k] = data.site_xpos[sites, :2]
    per_foot = {}
    total_dist, total_ticks = 0.0, 0
    for j, foot in enumerate(FOOT_GEOMS):
        both = record.contact[: n - 1, j] & record.contact[1:n, j] if n > 1 else np.zeros(0, bool)
        steps = np.linalg.norm(np.diff(xy[:, j], axis=0), axis=1)[both] if n > 1 else np.zeros(0)
        per_foot[foot] = {
            "stance_ticks": int(both.sum()),
            "slip_distance_m": float(steps.sum()),
            "mean_slip_speed_mps": float(steps.mean() / record.ctrl_dt) if steps.size else None,
        }
        total_dist += float(steps.sum())
        total_ticks += int(both.sum())
    return {
        "definition": "foot site horizontal displacement during Ticks with floor contact at both "
        "ends, before any fall",
        "per_foot": per_foot,
        "total_slip_distance_m": total_dist,
        "mean_slip_speed_mps": total_dist / (total_ticks * record.ctrl_dt) if total_ticks else None,
    }


def _last_valid_tick(record, outcome: dict) -> int:
    until = outcome.get("progress_until_s") or 0.0
    return int(np.count_nonzero(record.time <= until + 1e-9)) - 1


def case_summary(model, record, outcome: dict, push_start_s: float | None, physics: dict) -> dict:
    summary = episode_summary(record, outcome, push_start_s)
    summary.update(
        max_tilt_deg=outcome["max_tilt_deg"],
        min_trunk_z=outcome["min_trunk_z"],
        mean_forward_speed_mps=outcome["mean_forward_speed"],
        horizontal_disp_m=outcome["horizontal_disp_m"],
        slip=foot_slip(model, record, _last_valid_tick(record, outcome)),
        physics=physics,
    )
    return summary


def _physics_params(scenario: Scenario | None, *, push: bool) -> dict:
    """Exact physical parameters of a simulated case (None scenario = original nominal)."""
    if scenario is None:
        return {"ground_friction_mu": NOMINAL_FRICTION_MU, "friction_source": "trained nominal",
                "payload_kg": 0.0, "payload_offset_xyz_m": [0.0, 0.0, 0.0], "push": None}  # fmt: skip
    return {
        "ground_friction_mu": scenario.effective_mu(),
        "friction_source": "trained nominal" if scenario.ground_friction_mu is None else "scenario",
        "payload_kg": scenario.payload_mass_kg,
        "payload_offset_xyz_m": list(scenario.payload_offset_xyz_m),
        "push": dataclasses.asdict(scenario.push) if push and scenario.push else None,
    }


def suite_config_for(scenario: Scenario, policy: str, *, trials: int, interval_s, seed: int,
                     method: str = "stratified"):  # fmt: skip
    """The M2 suite config that varies only the push start of `scenario`."""
    p = scenario.push
    return load_suite_config(
        {
            "schema_version": 1,
            "name": "scenario-suite",
            "description": f"push start varied for scenario {scenario.name!r}",
            "policy": policy,
            "scene": scenario.scene,
            "command": list(scenario.command),
            "horizon_s": scenario.horizon_s,
            "trials": trials,
            "seed": seed,
            "attack": {
                "kind": "push",
                "force_n": p.force_n,
                "direction": list(p.direction_vector),
                "duration_s": p.duration_s,
                "start": {"method": method, "low_s": interval_s[0], "high_s": interval_s[1]},
            },
        }
    )


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _provenance(base_stack) -> dict:
    import platform

    return {
        "policy": base_stack.policy_info,
        "weeble_commit": base_stack.lock["source"]["commit"],
        "versions": {"python": platform.python_version(), **base_stack.library_versions},
        "git": git_revision(base_stack.root),
    }


def _save(out_dir: Path, name: str, record, stack, scenario: Scenario, outcome, criteria, case):
    manifest = stack.manifest_for(record, run_id=f"{scenario.name}/{name}", command=scenario.command,
                                  seconds=scenario.horizon_s, seed=0, outcome=outcome,
                                  criteria=criteria)  # fmt: skip
    manifest["scenario"] = scenario.to_dict()
    manifest["case"] = case
    write_run(out_dir / name, record, manifest, overwrite=True)
    return name


def run_single(scenario: Scenario, base_stack, out_dir: Path | None, *,
               cache: ScenarioModelCache = MODEL_CACHE, adjustments=None) -> dict:  # fmt: skip
    """Scenario episode, matched baseline (if a push) and original nominal; writes trajectories
    and `scenario_result.json` when `out_dir` is given."""
    started = time.perf_counter()
    criteria = WalkCriteria()
    entry = model_for(scenario, base_stack, cache)
    sstack = scenario_stack(base_stack, entry)
    runner = EpisodeRunner(entry.model, entry.ids, base_stack.policy, base_stack.actor, criteria)
    base_runner = EpisodeRunner(base_stack.model, base_stack.ids, base_stack.policy,
                                base_stack.actor, criteria)  # fmt: skip
    records = {}
    paired = push_record = None
    if scenario.push is not None:
        config = suite_config_for(scenario, base_stack.policy_path.as_posix(), trials=1,
                                  interval_s=(scenario.push.start_s,
                                              scenario.push.start_s + 0.004),
                                  seed=0, method="grid")  # fmt: skip
        k = scenario.push.start_substep
        trial, (matched, m_out), (attacked, a_out) = run_trial(config, runner, 0, k, [k, k + 1],
                                                               criteria)  # fmt: skip
        records["scenario"] = (attacked, a_out, scenario.push.start_s)
        records["matched"] = (matched, m_out, None)
        paired = trial["paired"]
        push_record = trial["push"]
        stance = trial["stance_at_start"]
    else:
        rec = runner.run(EpisodeConfig(scenario.command, scenario.horizon_s, 0))
        records["scenario"] = (rec, evaluate(rec, criteria), None)
        stance = None
    original = base_runner.run(EpisodeConfig(scenario.command, scenario.horizon_s, 0))
    records["original"] = (original, evaluate(original, criteria), None)

    cases = {}
    for name, (rec, out, push_start) in records.items():
        model = base_stack.model if name == "original" else entry.model
        physics = _physics_params(None if name == "original" else scenario,
                                  push=name == "scenario")  # fmt: skip
        cases[name] = case_summary(model, rec, out, push_start, physics)
    primary = "matched" if scenario.push is not None else "original"
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "robustlab-scenario-result",
        "mode": "single",
        "scenario": scenario.to_dict(),
        "scenario_sha256": scenario.sha256(),
        "adjustments": adjustments or [],
        "changed_from_original": scenario.changed_from_nominal(),
        "model": {
            "signature": list(entry.signature),
            "reused_base_model": entry.base,
            "friction": friction_report(
                entry.model, entry.ids, base_stack.policy, scenario.ground_friction_mu
            ),
            "mass": mass_report(entry.model, entry.ids),
            "scene": entry.scene_info.describe(),
        },
        "cases": cases,
        "primary_baseline": primary,
        "baselines": {
            "matched": None
            if scenario.push is None
            else "same friction, payload, command, horizon and reset; push removed",
            "original": "trained nominal: friction 0.6, no payload, no push; same command and horizon",
        },
        "paired": paired,
        "push": push_record,
        "stance_at_push": stance,
        "trial_semantics": "one deterministic paired run; repeating it reproduces it exactly",
    }
    result["results_sha256"] = _digest(
        {k: result[k] for k in ("scenario", "cases", "paired", "model", "adjustments")}
    )
    result["provenance"] = _provenance(base_stack)
    result["wall_s"] = time.perf_counter() - started
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, (rec, out, _) in records.items():
            stack = base_stack if name == "original" else sstack
            paths[name] = _save(out_dir, name, rec, stack, scenario, out, criteria, name)
        result["trajectories"] = paths
        _atomic_write_json(out_dir / RESULT_NAME, result)
    return result


def run_scenario_suite(scenario: Scenario, base_stack, out_dir: Path, *, trials: int = 16,
                       interval_s=(2.0, 3.024), seed: int = 0, method: str = "stratified",
                       cache: ScenarioModelCache = MODEL_CACHE, adjustments=None,
                       on_trial=None, overwrite: bool = False) -> dict:  # fmt: skip
    """Vary only the push start (M2 sampling); everything else fixed. Requires a push."""
    if scenario.push is None:
        raise ValueError(
            "a suite needs a push: without one every trial is the same deterministic episode, "
            "so repeated runs would not be independent trials"
        )
    started = time.perf_counter()
    entry = model_for(scenario, base_stack, cache)
    sstack = scenario_stack(base_stack, entry)
    config = suite_config_for(scenario, base_stack.policy_path.as_posix(), trials=trials,
                              interval_s=interval_s, seed=seed, method=method)  # fmt: skip
    out_dir = Path(out_dir)
    report = run_suite(config, sstack, out_dir, save_trajectories=trials <= 32,
                       overwrite=overwrite, on_trial=on_trial)  # fmt: skip
    criteria = WalkCriteria()
    base_runner = EpisodeRunner(base_stack.model, base_stack.ids, base_stack.policy,
                                base_stack.actor, criteria)  # fmt: skip
    original = base_runner.run(EpisodeConfig(scenario.command, scenario.horizon_s, 0))
    o_out = evaluate(original, criteria)
    _save(out_dir, "original", original, base_stack, scenario, o_out, criteria, "original")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "robustlab-scenario-result",
        "mode": "suite",
        "scenario": scenario.to_dict(),
        "scenario_sha256": scenario.sha256(),
        "adjustments": adjustments or [],
        "changed_from_original": scenario.changed_from_nominal(),
        "trial_semantics": (
            f"only the push start varies ({report['sampling']['kind']} over "
            f"[{interval_s[0]:.3f}, {interval_s[1]:.3f}) s, seed {seed}); friction, payload, "
            "force, direction, duration, command and reset are fixed. The scenario's own "
            f"start_s ({scenario.push.start_s:g} s) is replaced by the sampled starts."
        ),
        "model": {
            "signature": list(entry.signature),
            "reused_base_model": entry.base,
            "friction": friction_report(
                entry.model, entry.ids, base_stack.policy, scenario.ground_friction_mu
            ),
            "mass": mass_report(entry.model, entry.ids),
        },
        "experiment": "experiment.json",
        "experiment_results_sha256": report["results_sha256"],
        "aggregates": report["aggregates"],
        "interpretation": report["interpretation"],
        "primary_baseline": "matched",
        "baselines": {
            "matched": "the suite's nominal case: same friction and payload, push removed",
            "original": "trained nominal reference run once (original/)",
        },
        "original": case_summary(
            base_stack.model, original, o_out, None, _physics_params(None, push=False)
        ),
        "matched_physics": _physics_params(scenario, push=False),
        "provenance": _provenance(base_stack),
        "wall_s": time.perf_counter() - started,
    }
    _atomic_write_json(out_dir / RESULT_NAME, result)
    return result


def outcome_line(case: dict) -> str:
    if case["outcome"] == "fell":
        return f"FELL {case['reason']} at {case['fall_time_s']:.3f} s"
    if case["outcome"] == "numerical":
        return f"NUMERICAL {case['reason']}"
    return "survived"
