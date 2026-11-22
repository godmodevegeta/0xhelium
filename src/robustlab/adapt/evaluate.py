"""Deterministic episode metrics, the hardening objective, calibration and a process pool.

All metrics are defined over Ticks up to the first physical fall (or the horizon), so stopping an
episode at the fall (`stop_on_fall`) does not change them. Numerical failures are their own class
and are never counted as survival.
"""

from __future__ import annotations

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robustlab.adapt.reflex import (
    RESIDUAL_CAP,
    Calibration,
    ReflexActor,
    envelope,
    features,
    signal_indices,
)
from robustlab.adapt.sets import EpisodeSpec
from robustlab.experiments.suite import classify, trajectory_digest
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner, WalkCriteria, evaluate
from robustlab.physics.scenario_model import MODEL_CACHE

# Objective weights (documented in docs/milestones/M6.md and the candidate record).
OBJECTIVE = {
    "numerical": 4.0,
    "attack_fall_base": 2.0,
    "attack_recovery_window_s": 1.5,
    "attack_survived_tilt_weight": 0.5,
    "attack_progress_weight": 0.5,
    "attack_effort_weight": 0.2,
    "nominal_fall": 3.0,
    "nominal_progress_weight": 2.0,
    "nominal_activation_weight": 1.0,
    "nominal_effort_weight": 0.5,
    "tilt_scale_deg": 45.0,
}

CALIBRATION_CONDITIONS = (
    {"command": (1.0, 0.0, 0.0)}, {"command": (0.5, 0.0, 0.0)}, {"command": (0.0, 0.0, 0.0)},
    {"command": (1.0, 0.0, 0.5)}, {"command": (1.0, 0.0, -0.5)}, {"command": (0.5, 0.3, 0.0)},
    {"command": (0.5, -0.3, 0.0)}, {"command": (1.5, 0.0, 0.0)},
    {"command": (1.0, 0.0, 0.0), "ground_friction_mu": 0.45},
    {"command": (1.0, 0.0, 0.0), "ground_friction_mu": 1.0},
    {"command": (1.0, 0.0, 0.0), "payload_kg": 2.5},
)  # fmt: skip
CALIBRATION_SECONDS = 5.0


class Context:
    """Per-process simulation context: one Stack, cached scenario models and runners."""

    def __init__(self, stack, calibration: Calibration | None) -> None:
        self.stack = stack
        self.calibration = calibration
        self.criteria = WalkCriteria()
        self._runners: dict[tuple, EpisodeRunner] = {}

    def runner(self, spec: EpisodeSpec) -> EpisodeRunner:
        key = spec.physics_key()
        if key not in self._runners:
            entry = MODEL_CACHE.get(self.stack, spec.ground_friction_mu, spec.payload)
            self._runners[key] = EpisodeRunner(entry.model, entry.ids, self.stack.policy,
                                               self.stack.actor, self.criteria)  # fmt: skip
        return self._runners[key]

    def actor(self, params):
        if params is None:
            return self.stack.actor
        if self.calibration is None:
            raise ValueError("a calibration is required for reflex parameters")
        return ReflexActor(self.stack.actor, self.stack.policy, self.calibration, params)


def run_episode(ctx: Context, spec: EpisodeSpec, params=None, *, stop_on_fall: bool = True,
                keep: bool = False):  # fmt: skip
    """Run one episode; returns (metrics, record or None). `params=None` is the frozen base."""
    runner = ctx.runner(spec)
    actor = ctx.actor(params)
    runner.actor = actor
    try:
        push = None if spec.push is None else spec.push.scenario()
        record = runner.run(EpisodeConfig(spec.command, spec.horizon_s, 0, push,
                                          stop_on_fall=stop_on_fall))  # fmt: skip
    finally:
        runner.actor = ctx.stack.actor
    metrics = episode_metrics(record, spec, ctx.criteria, actor if params is not None else None)
    return metrics, (record if keep else None)


def episode_metrics(record, spec: EpisodeSpec, criteria: WalkCriteria, reflex_actor) -> dict:
    out = evaluate(record, criteria)
    fall = record.fall
    outcome = classify(fall.kind)
    cutoff_s = out["progress_until_s"]
    last = int(np.count_nonzero(record.time <= cutoff_s + 1e-9)) - 1
    push_start = None if spec.push is None else spec.push.start_s
    upto = slice(0, last + 1)
    up_z = record.up_z[upto]
    if push_start is not None:
        up_z = up_z[record.time[upto] >= push_start - 1e-9]
    max_tilt = (float(np.degrees(np.arccos(np.clip(np.min(up_z), -1, 1))))
                if up_z.size and np.isfinite(up_z).all() else None)  # fmt: skip
    metrics = {
        "id": spec.id,
        "kind": spec.kind,
        "confirmed_failure": spec.confirmed_failure,
        "outcome": outcome,
        "fell": bool(fall.fell),
        "numerical": bool(fall.numerical_failure),
        "reason": fall.reason,
        "fall_time_s": fall.time_s if outcome != "survived" else None,
        "time_after_push_s": None
        if outcome == "survived" or push_start is None or fall.time_s is None
        else fall.time_s - push_start,
        "max_tilt_after_push_deg": max_tilt,
        "min_trunk_z": float(np.min(record.trunk_z[upto])) if last >= 0 else None,
        "progress_m": out["forward_distance_m"],
        "progress_until_s": cutoff_s,
        "ticks_simulated": record.n_ticks,
        "stopped_at_fall": bool(record.truncated and fall.fell and not fall.numerical_failure),
        "digest_up_to_stop": trajectory_digest(record),
        "reflex": None if reflex_actor is None else reflex_actor.summary(max(last, 0)),
    }
    return metrics


def episode_cost(m: dict, progress_ref: float) -> float:
    """Cost of one episode (lower is better); see OBJECTIVE and the M6 design."""
    w = OBJECTIVE
    # progress deficit in [0, 1] relative to the base policy's no-push run (no reference when the
    # reference itself does not move forward, e.g. a zero command)
    ratio = 0.0 if progress_ref <= 0 else min(1.0, max(0.0, 1.0 - m["progress_m"] / progress_ref))
    reflex = m.get("reflex") or {"mean_abs_residual": 0.0, "activation_fraction": 0.0}
    effort = reflex["mean_abs_residual"] / RESIDUAL_CAP
    if m["numerical"] or m["outcome"] == "numerical":
        return w["numerical"]
    if m["kind"] == "nominal":
        if m["fell"]:
            return w["nominal_fall"]
        return (w["nominal_progress_weight"] * ratio
                + w["nominal_activation_weight"] * reflex["activation_fraction"]
                + w["nominal_effort_weight"] * effort)  # fmt: skip
    if m["fell"]:
        dt = m["time_after_push_s"] or 0.0
        base = w["attack_fall_base"] - min(max(dt, 0.0) / w["attack_recovery_window_s"], 1.0)
    else:
        tilt = m["max_tilt_after_push_deg"]
        tilt = w["tilt_scale_deg"] if tilt is None else tilt
        base = w["attack_survived_tilt_weight"] * min(tilt / w["tilt_scale_deg"], 1.0)
    return base + w["attack_progress_weight"] * ratio + w["attack_effort_weight"] * effort


def objective(metrics: list[dict], refs: dict[str, float]) -> float:
    """J = mean attack cost + mean nominal cost (either mean is 0 if that kind is absent)."""
    costs = {"attack": [], "nominal": []}
    for m in metrics:
        costs[m["kind"]].append(episode_cost(m, refs[m["id"]]))
    value = sum(float(np.mean(v)) for v in costs.values() if v)
    return value if math.isfinite(value) else float("inf")


def aggregate(metrics: list[dict]) -> dict:
    attacks = [m for m in metrics if m["kind"] == "attack"]
    nominal = [m for m in metrics if m["kind"] == "nominal"]

    def count(rows, key):
        return sum(1 for m in rows if m["outcome"] == key)

    return {
        "attacks": len(attacks),
        "attack_survived": count(attacks, "survived"),
        "attack_fell": count(attacks, "fell"),
        "attack_numerical": count(attacks, "numerical"),
        "nominal": len(nominal),
        "nominal_survived": count(nominal, "survived"),
        "nominal_fell": count(nominal, "fell"),
        "nominal_numerical": count(nominal, "numerical"),
        "mean_attack_tilt_deg": _mean([m["max_tilt_after_push_deg"] for m in attacks]),
        "mean_time_after_push_s_of_falls": _mean(
            [m["time_after_push_s"] for m in attacks if m["outcome"] == "fell"]
        ),
    }


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------- calibration


def calibration_samples(stack, conditions=CALIBRATION_CONDITIONS,
                        seconds: float = CALIBRATION_SECONDS) -> tuple[np.ndarray, list[dict]]:  # fmt: skip
    """Reflex signals at every Tick of base-policy episodes without pushes."""
    from robustlab.physics.scene import Payload

    idx = signal_indices(stack.policy)

    class Recorder:
        def __init__(self, base) -> None:
            self.base, self.rows = base, []

        def act(self, obs):
            self.rows.append(features(obs, idx))
            return self.base.act(obs)

    samples, episodes = [], []
    criteria = WalkCriteria()
    for cond in conditions:
        mu = cond.get("ground_friction_mu")
        payload = Payload(cond["payload_kg"], (0.0, 0.0, 0.0)) if cond.get("payload_kg") else None
        entry = MODEL_CACHE.get(stack, mu, payload)
        rec_actor = Recorder(stack.actor)
        runner = EpisodeRunner(entry.model, entry.ids, stack.policy, rec_actor, criteria)
        record = runner.run(EpisodeConfig(tuple(cond["command"]), seconds, 0))
        if record.fall.fell or record.fall.numerical_failure:
            raise ValueError(f"calibration episode {cond} did not stay upright")
        rows = np.asarray(rec_actor.rows)
        samples.append(rows)
        episodes.append({**{k: list(v) if isinstance(v, tuple) else v for k, v in cond.items()},
                         "ticks": int(rows.shape[0]),
                         "digest": trajectory_digest(record)})  # fmt: skip
    return np.concatenate(samples), episodes


def calibrate(stack) -> Calibration:
    samples, episodes = calibration_samples(stack)
    return envelope(samples, {
        "method": "median and largest absolute deviation from it, per signal, over all Ticks "
        "of base-policy episodes without pushes",
        "seconds": CALIBRATION_SECONDS, "episodes": episodes,
        "policy_sha256": stack.policy.sha256, "scene": stack.scene_info.name,
    })  # fmt: skip


# ---------------------------------------------------------------- parallel evaluation

_WORKER: Context | None = None


def _init_worker(root: str, policy: str, scene: str, calibration: dict | None) -> None:
    global _WORKER
    from robustlab.experiments.stack import load_stack

    stack = load_stack(Path(root), Path(policy), scene)
    _WORKER = Context(stack, None if calibration is None else Calibration.from_dict(calibration))


def _work(task: tuple) -> dict:
    params, spec = task
    assert _WORKER is not None
    metrics, _ = run_episode(_WORKER, spec, None if params is None else np.asarray(params))
    return metrics


@dataclass
class Evaluator:
    """Evaluate (params, spec) tasks in order, in-process or on a spawn-started process pool."""

    ctx: Context
    workers: int = 1

    def __post_init__(self) -> None:
        self.pool: ProcessPoolExecutor | None = None
        self.rollouts = 0
        if self.workers > 1:
            stack = self.ctx.stack
            cal = None if self.ctx.calibration is None else self.ctx.calibration.to_dict()
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(
                    str(Path(stack.root).resolve()),
                    str(stack.policy_path),
                    stack.scene_info.name,
                    cal,
                ),
            )

    def map(self, tasks: list[tuple]) -> list[dict]:
        self.rollouts += len(tasks)
        if self.pool is None:
            return [run_episode(self.ctx, spec, None if p is None else np.asarray(p))[0]
                    for p, spec in tasks]  # fmt: skip
        chunk = max(1, len(tasks) // (4 * self.workers))
        return list(self.pool.map(_work, tasks, chunksize=chunk))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
