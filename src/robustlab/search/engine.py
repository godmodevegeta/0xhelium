"""Search engine: Random Search and CMA-ES over the M4 shove space, confirmation, shrinking, record.

Search logic takes `evaluate(candidate) -> EvalResult` and `confirm(candidate, digest) ->
Confirmation` callables. `PhysicsEvaluator` provides them from the M1 runner (one shoved episode
per evaluation) and the M2 `run_trial` (paired nominal + shoved re-simulation per confirmation).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robustlab.experiments.suite import (
    classify,
    load_suite_config,
    run_trial,
    stance_at,
    trajectory_digest,
)
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner, WalkCriteria, evaluate
from robustlab.search.objective import objective, robustness_score
from robustlab.search.shrink import shrink
from robustlab.search.space import (
    CMA_LOWER,
    CMA_UPPER,
    Candidate,
    bounds_record,
    decode_cma,
    sample_uniform,
)

RECORD_SCHEMA_VERSION = 1
ALGORITHMS = ("random", "cmaes")
POPSIZE = 8
SIGMA0 = 0.4
X0 = (0.0, 0.0, 0.5, 0.5)
MIN_BUDGET, MAX_BUDGET = 8, 256
MAX_SHRINK_BUDGET = 128
MAX_CONFIRM_ATTEMPTS = 3
FIXED = {"command": (1.0, 0.0, 0.0), "horizon_s": 5.0, "scene": "fall-contact"}


class SearchConfigError(ValueError):
    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class SearchConfig:
    algorithm: str
    budget: int = 64
    seed: int = 0
    shrink_budget: int = 32
    timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise SearchConfigError("algorithm", f"expected one of {ALGORITHMS}")
        for name, value, lo, hi in (
            ("budget", self.budget, MIN_BUDGET, MAX_BUDGET),
            ("shrink_budget", self.shrink_budget, 0, MAX_SHRINK_BUDGET),
            ("seed", self.seed, 0, 2**63 - 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise SearchConfigError(name, f"must be an integer in [{lo}, {hi}]")
        if self.budget % POPSIZE:
            raise SearchConfigError("budget", f"must be a multiple of {POPSIZE}")
        if not (isinstance(self.timeout_s, (int, float)) and 0 < self.timeout_s <= 3600):
            raise SearchConfigError("timeout_s", "must be in (0, 3600]")


@dataclass(frozen=True)
class EvalResult:
    candidate: Candidate
    outcome: str  # survived | fell | numerical
    reason: str | None
    fall_time_s: float | None
    min_up_z: float | None
    min_trunk_z: float | None
    max_tilt_deg: float | None
    score: float | None  # robustness score for survivors
    objective: float
    digest: str
    first_nonfoot_body: str | None = None
    stance: str | None = None
    events: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "outcome": self.outcome,
            "reason": self.reason,
            "fall_time_s": self.fall_time_s,
            "time_after_start_s": None
            if self.fall_time_s is None
            else self.fall_time_s - self.candidate.start_s,
            "min_up_z": self.min_up_z,
            "min_trunk_z": self.min_trunk_z,
            "max_tilt_deg": self.max_tilt_deg,
            "robustness_score": self.score,
            "objective": self.objective,
            "digest": self.digest,
            "first_nonfoot_body": self.first_nonfoot_body,
            "stance_at_start": self.stance,
            "events": self.events,
        }


@dataclass(frozen=True)
class Confirmation:
    candidate: Candidate
    confirmed: bool
    attacked_outcome: str
    digest_match: bool
    pairing_ok: bool
    rollouts: int
    trial: dict | None = None

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "confirmed": self.confirmed,
            "attacked_outcome": self.attacked_outcome,
            "digest_match": self.digest_match,
            "pairing_ok": self.pairing_ok,
            "rollouts": self.rollouts,
            "trial": self.trial,
        }


def stance_key(stance: dict | None) -> str | None:
    if stance is None:
        return None
    down = [foot for foot in ("FR", "FL", "RR", "RL") if stance.get(foot)]
    return "+".join(down) if down else "none"


def _finite(value) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def scenario_config(candidate: Candidate, *, name: str, policy: str) -> dict:
    """The candidate as a one-trial M2 suite config (`robustlab experiment --config`)."""
    return {
        "schema_version": 1,
        "name": name,
        "description": (
            f"M4 discovered shove: {candidate.force_n:g} N at {candidate.direction_deg:g} deg "
            f"(initial-heading frame), {candidate.start_s:.3f}-"
            f"{candidate.start_s + candidate.duration_s:.3f} s, impulse "
            f"{candidate.impulse_ns:.3f} N*s"
        ),
        "policy": policy,
        "scene": FIXED["scene"],
        "command": list(FIXED["command"]),
        "horizon_s": FIXED["horizon_s"],
        "trials": 1,
        "seed": 0,
        "attack": {
            "kind": "push",
            "force_n": candidate.force_n,
            "direction": list(candidate.direction_vector),
            "duration_s": candidate.duration_s,
            "start": {
                "method": "grid",
                "low_s": candidate.start_s,
                "high_s": (candidate.start_substep + 1) * 0.004,
            },
        },
    }


class PhysicsEvaluator:
    """Evaluations and confirmations on one Stack. Not thread-safe: use from one thread."""

    def __init__(self, stack, criteria: WalkCriteria | None = None) -> None:
        self.stack = stack
        self.criteria = criteria or WalkCriteria()
        self.runner = EpisodeRunner(
            stack.model, stack.ids, stack.policy, stack.actor, self.criteria
        )
        self.reference = self.runner.run(self._episode(None))
        self.reference_outcome = classify(self.reference.fall.kind)
        self.reference_digest = trajectory_digest(self.reference)
        self.rollouts = 1  # the reference nominal episode

    def _episode(self, push) -> EpisodeConfig:
        return EpisodeConfig(FIXED["command"], FIXED["horizon_s"], 0, push)

    def evaluate(self, candidate: Candidate) -> EvalResult:
        record = self.runner.run(self._episode(candidate.push()))
        self.rollouts += 1
        outcome_dict = evaluate(record, self.criteria)
        outcome = classify(record.fall.kind)
        score = None
        if outcome == "survived":
            score = robustness_score(
                outcome_dict["min_up_z"],
                outcome_dict["min_trunk_z"],
                self.criteria.tilt_deg,
                self.criteria.min_trunk_z,
            )
        contact = record.fall.events.get("nonfoot_contact")
        return EvalResult(
            candidate=candidate,
            outcome=outcome,
            reason=record.fall.reason,
            fall_time_s=record.fall.time_s if outcome == "fell" else None,
            min_up_z=_finite(outcome_dict["min_up_z"]),
            min_trunk_z=_finite(outcome_dict["min_trunk_z"]),
            max_tilt_deg=_finite(outcome_dict["max_tilt_deg"]),
            score=score,
            objective=objective(outcome, candidate.impulse_ns, score),
            digest=trajectory_digest(record),
            first_nonfoot_body=contact.detail["body"] if contact else None,
            stance=stance_key(stance_at(self.reference, candidate.start_s)),
            events={k: e.time_s for k, e in sorted(record.fall.events.items())},
        )

    def suite_config(self, candidate: Candidate, name: str = "m4-candidate"):
        return load_suite_config(
            scenario_config(candidate, name=name, policy=self.stack.policy_path.as_posix())
        )

    def paired_trial(self, candidate: Candidate):
        """Authoritative paired re-simulation via the M2 `run_trial`."""
        config = self.suite_config(candidate)
        k = candidate.start_substep
        result = run_trial(config, self.runner, 0, k, [k, k + 1], self.criteria)
        self.rollouts += 2
        return result

    def confirm(self, candidate: Candidate, expected_digest: str) -> Confirmation:
        trial, _, _ = self.paired_trial(candidate)
        attacked = trial["attacked"]
        digest_match = attacked["digest"] == expected_digest
        pairing_ok = bool(trial["paired"]["pairing_ok"])
        return Confirmation(
            candidate=candidate,
            confirmed=attacked["outcome"] == "fell" and digest_match and pairing_ok,
            attacked_outcome=attacked["outcome"],
            digest_match=digest_match,
            pairing_ok=pairing_ok,
            rollouts=2,
            trial={k: v for k, v in trial.items() if k != "trajectories"},
        )


# ----------------------------------------------------------------------------- searchers


class _Search:
    def __init__(self, config, evaluate_fn, should_stop, on_eval) -> None:
        self.config = config
        self.evaluate_fn = evaluate_fn
        self.should_stop = should_stop
        self.on_eval = on_eval
        self.evaluations: list[dict] = []
        self.cache: dict[tuple, dict] = {}
        self.cache_hits: list[dict] = []
        self.stop_reason: str | None = None

    @property
    def rollouts(self) -> int:
        return len(self.evaluations)

    def can_evaluate(self) -> bool:
        if self.rollouts >= self.config.budget:
            self.stop_reason = self.stop_reason or "budget"
            return False
        reason = self.should_stop()
        if reason:
            self.stop_reason = reason
            return False
        return True

    def get(self, candidate: Candidate, generation: int | None) -> dict | None:
        """Cached result, or a new evaluation; None if no budget/stop."""
        key = candidate.key()
        if key in self.cache:
            self.cache_hits.append(
                {
                    "generation": generation,
                    "candidate": candidate.to_dict(),
                    "same_as_evaluation": self.cache[key]["index"],
                }
            )
            return self.cache[key]
        if not self.can_evaluate():
            return None
        result = self.evaluate_fn(candidate)
        entry = {"index": self.rollouts + 1, "generation": generation, **result.to_dict()}
        self.evaluations.append(entry)
        self.cache[key] = entry
        if self.on_eval is not None:
            self.on_eval(entry)
        return entry


def _random_search(search: _Search) -> dict:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(search.config.seed)))
    asks = 0
    max_asks = 4 * search.config.budget
    while asks < max_asks:
        asks += 1
        if search.get(sample_uniform(rng), None) is None:
            break
    else:
        search.stop_reason = "ask_cap"
    return {
        "generator": "numpy Generator(PCG64(SeedSequence(seed)))",
        "asks": asks,
        "prior": "force uniform [10,150] N, direction uniform [0,360) deg, start and duration "
        "uniform over their 4 ms substeps",
    }


def _cma_search(search: _Search) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pycma warns when matplotlib is absent
        import cma
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([search.config.seed, 1])))
    options = {
        "seed": np.nan,  # do not touch numpy's global RNG
        "randn": lambda *shape: rng.standard_normal(shape),
        "bounds": [list(CMA_LOWER), list(CMA_UPPER)],
        "popsize": POPSIZE,
        "verbose": -9,
        "tolfun": 0,
        "tolx": 0,
        "tolfunhist": 0,
        "tolstagnation": 10**9,
    }
    es = cma.CMAEvolutionStrategy(list(X0), SIGMA0, options)
    max_generations = 4 * search.config.budget // POPSIZE
    generation = 0
    told = 0
    while generation < max_generations:
        xs = es.ask()
        generation += 1
        fitness = []
        for x in xs:
            entry = search.get(decode_cma(x), generation)
            if entry is None:
                break
            fitness.append(entry["objective"])
        if len(fitness) < len(xs):
            break  # partial generation: never told
        es.tell(xs, fitness)
        told += 1
    else:
        search.stop_reason = search.stop_reason or "generation_cap"
    if search.stop_reason is None:
        search.stop_reason = "budget"
    return {
        "library": f"pycma {cma.__version__}",
        "internal_parameters": "x = (fx/150 N, fy/150 N, u_start, u_duration) in "
        "[-1,1]^2 x [0,1]^2; (fx, fy) projected radially onto 10 <= |f| <= 150 N",
        "popsize": POPSIZE,
        "sigma0": SIGMA0,
        "x0": list(X0),
        "bounds": [list(CMA_LOWER), list(CMA_UPPER)],
        "bound_handling": "pycma BoundTransform (default)",
        "sampler": "private numpy Generator(PCG64(SeedSequence([seed, 1]))) via the randn option",
        "generations_asked": generation,
        "generations_told": told,
        "max_generations": max_generations,
    }


def search_summary(evaluations: list[dict]) -> dict:
    fails = [e for e in evaluations if e["outcome"] == "fell"]
    best = min(fails, key=lambda e: (e["candidate"]["impulse_ns"], e["index"])) if fails else None
    return {
        "found_failure": bool(fails),
        "candidate_failures": len(fails),
        "survived": sum(e["outcome"] == "survived" for e in evaluations),
        "numerical": sum(e["outcome"] == "numerical" for e in evaluations),
        "evaluations_to_first_failure": fails[0]["index"] if fails else None,
        "best_candidate_index": None if best is None else best["index"],
        "best_candidate_impulse_ns": None if best is None else best["candidate"]["impulse_ns"],
        "best_objective": min((e["objective"] for e in evaluations), default=None),
    }


def results_digest(record: dict) -> str:
    payload = {
        k: record[k]
        for k in (
            "config",
            "evaluations",
            "cache_hits",
            "search",
            "confirmation",
            "shrink",
            "final",
        )
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def discover(
    config: SearchConfig,
    evaluate_fn: Callable[[Candidate], EvalResult],
    confirm_fn: Callable[[Candidate, str], Confirmation],
    *,
    cancelled: Callable[[], bool] = lambda: False,
    on_eval: Callable[[dict], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Run one bounded search, confirm its best candidate failure, shrink it. Pure orchestration
    over the injected callables; returns the record without provenance."""
    started = clock()

    def should_stop() -> str | None:
        if cancelled():
            return "cancelled"
        if clock() - started > config.timeout_s:
            return "timeout"
        return None

    stage = on_stage or (lambda _s: None)
    stage("search")
    search = _Search(config, evaluate_fn, should_stop, on_eval)
    details = _random_search(search) if config.algorithm == "random" else _cma_search(search)
    search_wall = clock() - started
    summary = search_summary(search.evaluations)

    stage("confirm")
    attempts = []
    confirmed: dict | None = None
    fails = sorted(
        (e for e in search.evaluations if e["outcome"] == "fell"),
        key=lambda e: (e["candidate"]["impulse_ns"], e["index"]),
    )
    for entry in fails[:MAX_CONFIRM_ATTEMPTS]:
        if should_stop():
            break
        c = Candidate(
            **{
                k: entry["candidate"][k]
                for k in ("force_n", "direction_deg", "start_substep", "duration_substeps")
            }
        )
        result = confirm_fn(c, entry["digest"])
        attempts.append({"evaluation": entry["index"], **result.to_dict()})
        if result.confirmed:
            confirmed = {
                "evaluation": entry["index"],
                **result.to_dict(),
                "digest": entry["digest"],
            }
            break

    shrink_record = None
    if confirmed is not None and config.shrink_budget > 0 and not should_stop():
        stage("shrink")
        start = Candidate(
            **{
                k: confirmed["candidate"][k]
                for k in ("force_n", "direction_deg", "start_substep", "duration_substeps")
            }
        )
        shrink_record = shrink(
            start, evaluate_fn, confirm_fn, config.shrink_budget, should_stop=should_stop
        )

    if confirmed is None:
        final = None
    elif shrink_record is not None and shrink_record["improved"]:
        final = {"source": "shrunk", **shrink_record["final_confirmed"]}
    else:
        final = {
            "source": "search",
            "candidate": confirmed["candidate"],
            "digest": confirmed["digest"],
            "trial": confirmed["trial"],
        }

    accounting = {
        "budget": config.budget,
        "search_rollouts": search.rollouts,
        "cache_hits": len(search.cache_hits),
        "confirmation_rollouts": sum(a["rollouts"] for a in attempts),
        "shrink_rollouts": 0 if shrink_record is None else shrink_record["rollouts"],
        "shrink_confirmation_rollouts": 0
        if shrink_record is None
        else shrink_record["confirmation_rollouts"],
    }
    accounting["total_rollouts_excluding_reference"] = (
        accounting["search_rollouts"]
        + accounting["confirmation_rollouts"]
        + accounting["shrink_rollouts"]
        + accounting["shrink_confirmation_rollouts"]
    )
    record = {
        "config": {
            "algorithm": config.algorithm,
            "budget": config.budget,
            "seed": config.seed,
            "shrink_budget": config.shrink_budget,
            "timeout_s": config.timeout_s,
            "space": bounds_record(),
            "fixed": {**FIXED, "command": list(FIXED["command"]), "initial_state": "nominal reset"},
            "objective": "fell: -2 + impulse/30 N*s; survived: min(tilt margin, height margin) "
            "in [0, 1]; numerical: +2 (invalid)",
            "warm_start": None,
        },
        "search": {
            **details,
            **summary,
            "stop_reason": search.stop_reason,
            "wall_s": search_wall,
        },
        "evaluations": search.evaluations,
        "cache_hits": search.cache_hits,
        "confirmation": {"attempts": attempts, "confirmed": confirmed},
        "shrink": shrink_record,
        "final": final,
        "accounting": accounting,
        "wall_s": clock() - started,
    }
    record["results_sha256"] = results_digest(_without_wall(record))
    return record


def _without_wall(record: dict) -> dict:
    out = dict(record)
    out["search"] = {k: v for k, v in record["search"].items() if k != "wall_s"}
    return out


def provenance(stack, evaluator: PhysicsEvaluator) -> dict[str, Any]:
    import platform

    from robustlab.experiments.trace import git_revision

    return {
        "policy": stack.policy_info,
        "scene": stack.scene_info.describe(),
        "weeble_commit": stack.lock["source"]["commit"],
        "versions": {"python": platform.python_version(), **stack.library_versions},
        "platform": {"platform": platform.platform(), "machine": platform.machine()},
        "git": git_revision(stack.root),
        "sim_dt": float(stack.model.opt.timestep),
        "ctrl_dt": stack.policy.control_dt,
        "reference_nominal": {
            "outcome": evaluator.reference_outcome,
            "digest": evaluator.reference_digest,
        },
    }


def run_discovery(config: SearchConfig, stack, **kwargs) -> dict:
    """Full physics discovery on `stack`, with provenance and the exported scenario."""
    evaluator = PhysicsEvaluator(stack)
    record = discover(config, evaluator.evaluate, evaluator.confirm, **kwargs)
    record["schema_version"] = RECORD_SCHEMA_VERSION
    record["kind"] = "robustlab-failure-discovery"
    record["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["provenance"] = provenance(stack, evaluator)
    record["accounting"]["reference_rollouts"] = 1
    record["accounting"]["physics_rollouts_total"] = evaluator.rollouts
    if record["final"] is not None:
        c = record["final"]["candidate"]
        cand = Candidate(
            c["force_n"], c["direction_deg"], c["start_substep"], c["duration_substeps"]
        )
        record["final"]["scenario_config"] = scenario_config(
            cand, name="m4-discovered", policy=stack.policy_path.as_posix()
        )
    record["statement"] = (
        "Lowest-impulse confirmed physical failure found within this search budget in this "
        "simulation. It is not shown to be globally minimal and carries no real-world claim."
    )
    return record
