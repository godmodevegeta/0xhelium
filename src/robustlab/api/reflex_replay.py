"""Baseline-vs-reflex-candidate replay shared by the Hardening (M6) and Gate (M7) APIs.

Both controllers are re-simulated on the server on the identical episode (model, reset, command,
push). The re-simulated outcomes are compared with a recorded row when one exists. With
`digest_stop_on_fall` set, both controllers are also re-run exactly as the recording was made
(stopping at the first fall or not) and their trajectory digests are compared.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from robustlab.adapt.evaluate import Context, episode_metrics, run_episode
from robustlab.adapt.harden import DEFAULT_CALIBRATION, load_calibration
from robustlab.adapt.reflex import Calibration, ReflexActor
from robustlab.adapt.sets import EpisodeSpec
from robustlab.api.jobs import Worker
from robustlab.experiments.suite import episode_summary
from robustlab.experiments.trace import trajectory_dict
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner, WalkCriteria, evaluate
from robustlab.physics.scenario_model import MODEL_CACHE
from robustlab.replay import replay_payload

OUTCOME_KEYS = ("outcome", "reason", "fall_time_s", "numerical_time_s", "time_after_start_s",
                "events", "first_nonfoot_body", "forward_progress_m")  # fmt: skip
GHOSTS = ("baseline", "matched")


def runner_for(stack, entry) -> EpisodeRunner:
    return EpisodeRunner(entry.model, entry.ids, stack.policy, stack.actor, WalkCriteria())


def _check(want: dict, got: dict, digest: str | None) -> dict:
    same = want["outcome"] == got["outcome"] and want["fall_time_s"] == got["fall_time_s"]
    check = {"recorded_outcome": want["outcome"], "resimulated_outcome": got["outcome"],
             "recorded_fall_time_s": want["fall_time_s"],
             "resimulated_fall_time_s": got["fall_time_s"], "outcome_match": same}  # fmt: skip
    if digest is None:
        check["digest_match"] = same  # M6 records carry no digest: outcome and fall time only
    else:
        check["recorded_digest"] = want["digest_up_to_stop"]
        check["resimulated_digest"] = digest
        check["digest_match"] = digest == want["digest_up_to_stop"]
    return check


def paired_reflex_replay(
    root: Path,
    spec: EpisodeSpec,
    params,
    calibration: Calibration,
    ghost: str,
    *,
    mode: str,
    labels: dict[str, str],
    extra: dict,
    recorded: dict | None = None,
    recorded_commit: str | None = None,
    what: str = "candidate record",
    digest_stop_on_fall: bool | None = None,
) -> Callable[[Worker], dict]:
    """Return the worker computation. Raises LookupError for an impossible ghost."""
    if ghost not in GHOSTS:
        raise LookupError(f"unknown ghost {ghost!r}")
    if spec.push is None and ghost == "matched":
        raise LookupError("a nominal episode has no matched no-shove ghost")
    params = np.asarray(params, np.float64)

    def compute(worker: Worker) -> dict:
        stack = worker.stack("fall-contact")
        load_calibration(Path(root) / DEFAULT_CALIBRATION, stack.policy.sha256)  # policy check
        ctx = worker.cache.get("hardening_ctx")
        if ctx is None or ctx.calibration.sha256() != calibration.sha256():
            ctx = worker.cache["hardening_ctx"] = Context(stack, calibration)
        base_m, base_rec = run_episode(ctx, spec, None, stop_on_fall=False, keep=True)
        # The candidate run keeps its ReflexActor so the activation trace can be shown.
        actor = ReflexActor(stack.actor, stack.policy, calibration, params)
        entry = MODEL_CACHE.get(stack, spec.ground_friction_mu, spec.payload)
        runner = EpisodeRunner(entry.model, entry.ids, stack.policy, actor, WalkCriteria())
        push = None if spec.push is None else spec.push.scenario()
        cand_rec = runner.run(EpisodeConfig(spec.command, spec.horizon_s, 0, push))
        cand_m = episode_metrics(cand_rec, spec, WalkCriteria(), actor)
        push_start = None if spec.push is None else spec.push.start_s
        if ghost == "matched":
            ghost_rec = runner_for(stack, entry).run(EpisodeConfig(spec.command, spec.horizon_s, 0))
            ghost_out = episode_summary(ghost_rec, evaluate(ghost_rec), None)
        else:
            ghost_rec = base_rec
            ghost_out = episode_summary(base_rec, evaluate(base_rec), push_start)
        cand_out = episode_summary(cand_rec, evaluate(cand_rec), push_start)
        digests: dict[str, str | None] = {"baseline": None, "candidate": None}
        if recorded and digest_stop_on_fall is not None:
            if digest_stop_on_fall:  # re-run exactly as recorded (stopped at the first fall)
                digests["baseline"] = run_episode(ctx, spec, None)[0]["digest_up_to_stop"]
                digests["candidate"] = run_episode(ctx, spec, params)[0]["digest_up_to_stop"]
            else:
                digests = {"baseline": base_m["digest_up_to_stop"],
                           "candidate": cand_m["digest_up_to_stop"]}  # fmt: skip
        checks = {}
        for case, metrics in (("baseline", base_m), ("candidate", cand_m)):
            if recorded and case in recorded:
                checks[case] = _check(recorded[case], metrics, digests[case])
        activation = [0.0, *actor.activations]
        residual = [0.0, *(float(np.abs(r).max()) for r in actor.residuals)]
        summary = actor.summary()
        return replay_payload(
            stack.model,
            (trajectory_dict(ghost_rec), {k: ghost_out.get(k) for k in OUTCOME_KEYS}),
            (trajectory_dict(cand_rec), {k: cand_out.get(k) for k in OUTCOME_KEYS}),
            None if cand_rec.push is None else {
                "start_s": cand_rec.push["start_s"], "end_s": cand_rec.push["end_s"],
                "world_force_n": cand_rec.push["world_force_n"],
                "force_n": cand_rec.push["force_n"]},
            source="re-simulated",
            verification=None if not checks else {
                "all_match": all(c["outcome_match"] and c["digest_match"]
                                 for c in checks.values()),
                "recorded_commit": recorded_commit, "cases": checks, "what": what},
            case_models={"attacked": entry.model, "nominal": entry.model},
            extra={
                "mode": mode,
                **extra,
                "episode": spec.to_dict(),
                "start_s": push_start,
                "ghost": {"kind": ghost, "label": labels[ghost], "parameters": None},
                "labels": {"attacked": labels["candidate"], "nominal": labels[ghost]},
                "baseline": {k: base_m[k] for k in ("outcome", "reason", "fall_time_s",
                                                    "max_tilt_after_push_deg", "progress_m")},
                "reflex": {"activation": [round(a, 4) for a in activation],
                           "max_abs_residual": [round(r, 4) for r in residual],
                           "affected_joints": summary["affected_joints"],
                           "first_activation_s": summary["first_activation_s"],
                           "activation_fraction": summary["activation_fraction"],
                           "frame_note": "frame f > 0 shows the activation used during "
                           "Tick f - 1"},
                "environment": None if spec.ground_friction_mu is None else {
                    "mu": spec.ground_friction_mu, "ghost_mu": spec.ground_friction_mu,
                    "nominal_mu": 0.6},
            },
        )  # fmt: skip

    return compute
