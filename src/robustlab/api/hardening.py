"""Policy Hardening API service (M6): attack choices, the precomputed sample candidate, and the
baseline-vs-candidate replay.

Replays re-simulate both controllers on the server on the identical episode (model, reset,
command, push) and compare the outcomes with the candidate record. Locked M7 episodes cannot be
replayed: `HardeningSets.split` refuses them.
"""

from __future__ import annotations

import numpy as np

from robustlab.adapt.evaluate import Context, episode_metrics, run_episode
from robustlab.adapt.harden import (
    DEFAULT_CALIBRATION,
    DEFAULT_SETS,
    HardenConfig,
    load_calibration,
)
from robustlab.adapt.reflex import Calibration, ReflexActor
from robustlab.adapt.sets import load_sets
from robustlab.api.jobs import JobManager, Worker
from robustlab.api.presets import PresetError, load_preset
from robustlab.experiments.suite import episode_summary
from robustlab.experiments.trace import trajectory_dict
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner, WalkCriteria, evaluate
from robustlab.physics.scenario_model import MODEL_CACHE
from robustlab.replay import replay_payload

PRESET_DIR = "presets/hardening-candidate-v1"
PRESET_KEY = "preset-v1"
OUTCOME_KEYS = ("outcome", "reason", "fall_time_s", "numerical_time_s", "time_after_start_s",
                "events", "first_nonfoot_body", "forward_progress_m")  # fmt: skip
LIMITS = {"max_generations": (1, 30), "popsize": 8, "max_rollouts": 4500}
LABELS = {
    "candidate": "Candidate: frozen policy + reflex (pending M7 regression evaluation)",
    "baseline": "Frozen baseline controller, same shove",
    "matched": "Matched condition: frozen baseline, no shove",
}


class HardeningService:
    def __init__(self, settings, jobs: JobManager, wall_clock_s: float, workers: int) -> None:
        self.jobs = jobs
        self.root = settings.root
        self.wall_clock_s = wall_clock_s
        self.workers = workers
        self.sets = load_sets(self.root / DEFAULT_SETS, self.root)
        try:
            self.preset = load_preset(self.root, PRESET_DIR)
            self.preset_record = self.preset.json("candidate.json")
            self.preset_grids = self.preset.json("grids.json")
        except PresetError:
            self.preset = self.preset_record = self.preset_grids = None

    def episodes(self) -> list[dict]:
        out = []
        for split in ("train", "validation_preview"):
            for e in self.sets.split(split):
                out.append({**e.to_dict(), "selectable": split == "train" and e.kind == "attack"
                            and e.confirmed_failure})  # fmt: skip
        return out

    def config(self) -> dict:
        return {
            "sets": self.sets.summary(),
            "episodes": self.episodes(),
            "defaults": {
                "max_generations": 12,
                "popsize": LIMITS["popsize"],
                "max_rollouts": 2000,
                "wall_clock_s": self.wall_clock_s,
                "workers": self.workers,
            },
            "limits": {
                "max_generations": list(LIMITS["max_generations"]),
                "max_rollouts": LIMITS["max_rollouts"],
                "wall_clock_s": self.wall_clock_s,
            },
            "preset": None
            if self.preset is None
            else {
                "key": PRESET_KEY,
                "result_origin": "precomputed",
                "meta": {k: self.preset.meta[k] for k in ("id", "version", "title", "source")},
                "record": self.preset_record,
                "grids": self.preset_grids,
            },
            "runs": self.jobs.list_hardenings(),
            "semantics": "A candidate is the frozen go1-walk-s0 actor plus a bounded residual "
            "reflex. It is never promoted here: the M7 regression gate decides.",
        }

    def make_config(self, seed: int, generations: int, focus: str | None) -> HardenConfig:
        return HardenConfig(seed=seed, popsize=LIMITS["popsize"], max_generations=generations,
                            max_rollouts=LIMITS["max_rollouts"], wall_clock_s=self.wall_clock_s,
                            workers=self.workers, focus=focus)  # fmt: skip

    def record_for(self, key: str) -> dict | None:
        if key == PRESET_KEY:
            return self.preset_record
        return self.jobs.hardening_record(key)

    def replay_compute(self, key: str, record: dict, episode_id: str, ghost: str):
        """Raises LookupError for an unknown or locked episode."""
        spec = next((e for split in ("train", "validation_preview")
                     for e in self.sets.split(split) if e.id == episode_id), None)  # fmt: skip
        if spec is None:
            raise LookupError(
                f"unknown episode {episode_id!r} (locked M7 episodes cannot be replayed)"
            )
        if spec.push is None and ghost == "matched":
            raise LookupError("a nominal episode has no matched no-shove ghost")
        recorded = {}
        for split in ("train", "validation_preview"):
            for row in record["measurements"][split].get("comparison") or []:
                recorded[row["id"]] = row
        params = np.asarray(record["reflex"]["vector"], np.float64)
        calibration = Calibration.from_dict(record["calibration"])

        def compute(worker: Worker) -> dict:
            stack = worker.stack("fall-contact")
            load_calibration(self.root / DEFAULT_CALIBRATION, stack.policy.sha256)  # policy check
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
                ghost_rec = runner_for(stack, entry).run(
                    EpisodeConfig(spec.command, spec.horizon_s, 0))  # fmt: skip
                ghost_out = episode_summary(ghost_rec, evaluate(ghost_rec), None)
            else:
                ghost_rec = base_rec
                ghost_out = episode_summary(base_rec, evaluate(base_rec), push_start)
            cand_out = episode_summary(cand_rec, evaluate(cand_rec), push_start)
            checks = {}
            row = recorded.get(spec.id)
            for case, metrics in (("baseline", base_m), ("candidate", cand_m)):
                if row is None:
                    continue
                want = row[case]
                same = (want["outcome"] == metrics["outcome"]
                        and want["fall_time_s"] == metrics["fall_time_s"])  # fmt: skip
                checks[case] = {"recorded_outcome": want["outcome"],
                                "resimulated_outcome": metrics["outcome"],
                                "recorded_fall_time_s": want["fall_time_s"],
                                "resimulated_fall_time_s": metrics["fall_time_s"],
                                "digest_match": same, "outcome_match": same}  # fmt: skip
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
                    "all_match": all(c["outcome_match"] for c in checks.values()),
                    "recorded_commit": record["provenance"]["git"]["commit"],
                    "cases": checks, "what": "candidate record"},
                case_models={"attacked": entry.model, "nominal": entry.model},
                extra={
                    "mode": "hardening",
                    "run_id": key,
                    "candidate_id": record["candidate_id"],
                    "candidate_status": record["status"],
                    "episode": spec.to_dict(),
                    "start_s": push_start,
                    "ghost": {"kind": ghost, "label": LABELS[ghost], "parameters": None},
                    "labels": {"attacked": LABELS["candidate"], "nominal": LABELS[ghost]},
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


def runner_for(stack, entry) -> EpisodeRunner:
    return EpisodeRunner(entry.model, entry.ids, stack.policy, stack.actor, WalkCriteria())
