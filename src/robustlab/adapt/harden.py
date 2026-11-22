"""CPU hardening: seeded CMA-ES over the 10 reflex parameters, and the candidate record.

The result is a *candidate* (frozen base + reflex parameters) with baseline and candidate
measurements on the training and validation-preview sets. It is never promoted here; M7 owns the
regression gate. Numerical failures are their own outcome and never count as recovery.

Determinism: CMA-ES samples come from a private `Generator(PCG64(SeedSequence([seed, 6])))`,
episodes are deterministic, and evaluation results are consumed in task order. A run that stops on
the generation cap, the rollout budget or a cancellation point at the same generation reproduces
the same parameters; a run stopped by the wall-clock cap depends on machine speed (recorded).
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from robustlab.adapt.evaluate import (
    OBJECTIVE,
    Context,
    Evaluator,
    aggregate,
    episode_cost,
    objective,
)
from robustlab.adapt.reflex import (
    N_GAINS,
    N_PARAMS,
    PARAMS,
    Calibration,
    check_action_bounds,
    params_dict,
    schema,
    validate_params,
)
from robustlab.adapt.sets import EpisodeSpec, HardeningSets
from robustlab.experiments.trace import _atomic_write_json, git_revision

CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_NAME = "candidate.json"
GAIN_MAX = np.array([max(abs(p.lo), abs(p.hi)) for p in PARAMS[:N_GAINS]])
AMPLITUDE_HALF = PARAMS[9].hi / 2


class HardenConfigError(ValueError):
    pass


@dataclass(frozen=True)
class HardenConfig:
    seed: int = 0
    popsize: int = 8
    sigma0: float = 0.3
    max_generations: int = 12
    max_rollouts: int = 2000  # optimization rollouts (baseline/reference/validation are extra)
    wall_clock_s: float = 60.0
    workers: int = 1
    focus: str | None = None  # one training attack id (plus its variants and the nominal set)

    def __post_init__(self) -> None:
        checks = (
            ("seed", 0 <= self.seed < 2**63),
            ("popsize", 4 <= self.popsize <= 32),
            ("sigma0", 0.01 <= self.sigma0 <= 1.0),
            ("max_generations", 1 <= self.max_generations <= 200),
            ("max_rollouts", 1 <= self.max_rollouts <= 50_000),
            ("wall_clock_s", 1.0 <= self.wall_clock_s <= 1800.0),
            ("workers", 1 <= self.workers <= 32),
        )
        for name, ok in checks:
            if not ok:
                raise HardenConfigError(f"{name}: out of range ({getattr(self, name)!r})")

    def deterministic_dict(self) -> dict:
        """Settings that determine the result (workers and the wall cap do not, unless hit)."""
        d = asdict(self)
        d.pop("workers")
        d.pop("wall_clock_s")
        return d


def decode(x) -> np.ndarray:
    """CMA-ES coordinates x in [-1, 1]^10 -> physical reflex parameters (x = 0: zero gains)."""
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    values = np.concatenate([x[:N_GAINS] * GAIN_MAX, [x[7] + 1.0, x[8] + 1.0],
                             [AMPLITUDE_HALF * (x[9] + 1.0)]])  # fmt: skip
    hi = np.array([p.hi for p in PARAMS])
    lo = np.array([p.lo for p in PARAMS])
    return validate_params(np.clip(values, lo, hi))


def training_episodes(sets: HardeningSets, focus: str | None) -> tuple[EpisodeSpec, ...]:
    train = sets.split("train")
    if focus is None:
        return train
    ids = {e.id for e in train if e.kind == "attack" and e.confirmed_failure}
    if focus not in ids:
        raise HardenConfigError(f"focus: {focus!r} is not a confirmed training attack")
    return tuple(e for e in train if e.kind == "nominal" or e.id == focus
                 or e.id.startswith(f"{focus}-v"))  # fmt: skip


def reference_specs(episodes) -> dict[tuple, EpisodeSpec]:
    """One no-push base-policy episode per (physics, command, horizon): the progress reference."""
    refs = {}
    for e in episodes:
        key = e.reference_key()
        if key not in refs:
            refs[key] = replace(e, id=f"ref-{len(refs)}", kind="nominal", push=None,
                                source=f"no-push reference for {e.id}", confirmed_failure=False,
                                tags=("reference",))  # fmt: skip
    return refs


def compare(baseline: list[dict], candidate: list[dict]) -> list[dict]:
    """Per-episode paired comparison on identical episodes."""
    rows = []
    for b, c in zip(baseline, candidate, strict=True):
        if b["id"] != c["id"]:
            raise ValueError("baseline and candidate episodes are not aligned")
        if "numerical" in (b["outcome"], c["outcome"]):
            change = "numerical"
        elif b["outcome"] == "fell" and c["outcome"] == "survived":
            change = "recovered"
        elif b["outcome"] == "survived" and c["outcome"] == "fell":
            change = "regressed"
        else:
            change = f"both_{b['outcome']}"
        rows.append({"id": b["id"], "kind": b["kind"], "change": change,
                     "baseline": _brief(b), "candidate": _brief(c)})  # fmt: skip
    return rows


def _brief(m: dict) -> dict:
    keys = ("outcome", "reason", "fall_time_s", "time_after_push_s", "max_tilt_after_push_deg",
            "progress_m", "min_trunk_z")  # fmt: skip
    brief = {k: m[k] for k in keys}
    if m.get("reflex"):
        r = m["reflex"]
        brief["reflex"] = {k: r[k] for k in ("activation_fraction", "first_activation_s",
                                              "mean_abs_residual", "max_abs_residual",
                                              "affected_joints")}  # fmt: skip
    return brief


def change_counts(rows: list[dict]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        bucket = out.setdefault(r["kind"], {})
        bucket[r["change"]] = bucket.get(r["change"], 0) + 1
    return out


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=float).encode()).hexdigest()


def run_hardening(
    stack,
    calibration: Calibration,
    sets: HardeningSets,
    config: HardenConfig,
    *,
    cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run the bounded search and return the candidate record (not yet written)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pycma warns when matplotlib is absent
        import cma
    check_action_bounds(stack.model, stack.policy)
    started = time.perf_counter()
    train = training_episodes(sets, config.focus)
    val = sets.split("validation_preview")
    evaluator = Evaluator(Context(stack, calibration), config.workers)
    try:
        refs = reference_specs((*train, *val))
        ref_metrics = evaluator.map([(None, s) for s in refs.values()])
        ref_progress = {k: m["progress_m"] for k, m in zip(refs, ref_metrics, strict=True)}
        progress_ref = {e.id: ref_progress[e.reference_key()] for e in (*train, *val)}
        base_train = evaluator.map([(None, s) for s in train])
        base_val = evaluator.map([(None, s) for s in val])
        baseline_rollouts = evaluator.rollouts
        j_base = objective(base_train, progress_ref)

        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.seed, 6])))
        options = {
            "seed": np.nan, "randn": lambda *shape: rng.standard_normal(shape),
            "bounds": [[-1.0] * N_PARAMS, [1.0] * N_PARAMS], "popsize": config.popsize,
            "verbose": -9, "tolfun": 0, "tolx": 0, "tolfunhist": 0, "tolstagnation": 10**9,
        }  # fmt: skip
        es = cma.CMAEvolutionStrategy([0.0] * N_PARAMS, config.sigma0, options)
        per_generation = config.popsize * len(train)
        opt_rollouts = 0
        history = []
        best = None
        stop = None
        opt_started = time.perf_counter()
        last_gen_s = 0.0
        for generation in range(1, config.max_generations + 1):
            if cancelled is not None and cancelled():
                stop = "cancelled"
                break
            if opt_rollouts + per_generation > config.max_rollouts:
                stop = "rollout_budget"
                break
            elapsed = time.perf_counter() - started
            if elapsed + last_gen_s > config.wall_clock_s:
                stop = "wall_clock"
                break
            gen_started = time.perf_counter()
            xs = es.ask()
            candidates = [decode(x) for x in xs]
            tasks = [(tuple(p), s) for p in candidates for s in train]
            results = evaluator.map(tasks)
            opt_rollouts += len(tasks)
            fitness = []
            for i, p in enumerate(candidates):
                metrics = results[i * len(train) : (i + 1) * len(train)]
                j = objective(metrics, progress_ref)
                fitness.append(j)
                if best is None or j < best["objective"]:
                    best = {"objective": j, "generation": generation, "index": i,
                            "params": p, "metrics": metrics}  # fmt: skip
            es.tell(xs, fitness)
            last_gen_s = time.perf_counter() - gen_started
            agg = aggregate(best["metrics"])
            entry = {
                "generation": generation,
                "rollouts": opt_rollouts,
                "best_objective": best["objective"],
                "generation_best": float(min(fitness)),
                "generation_median": float(np.median(fitness)),
                "best_attack_survived": agg["attack_survived"],
                "best_nominal_fell": agg["nominal_fell"],
                "elapsed_s": time.perf_counter() - started,
            }
            history.append(entry)
            if on_progress is not None:
                on_progress({**entry, "baseline_objective": j_base, "attacks": agg["attacks"]})
        else:
            stop = "generation_cap"
        opt_wall = time.perf_counter() - opt_started

        improved = best is not None and best["objective"] < j_base
        cand_val = (evaluator.map([(tuple(best["params"]), s) for s in val])
                    if best is not None else None)  # fmt: skip
        validation_rollouts = len(val) if best is not None else 0
    finally:
        evaluator.close()

    wall = time.perf_counter() - started
    params = best["params"] if best is not None else validate_params(np.zeros(N_PARAMS))
    record = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "kind": "robustlab-candidate",
        "status": "candidate",
        "regression_gate": "pending: M7 owns evaluation and promotion",
        "base_policy": {
            "sha256": stack.policy.sha256,
            "path": stack.policy_path.as_posix(),
            "run_id": stack.policy.run_id,
            "weights": "frozen, unmodified",
        },
        "reflex": {
            "schema": schema(),
            "parameters": params_dict(params),
            "vector": [float(v) for v in params],
        },
        "calibration": {**calibration.to_dict(), "sha256": calibration.sha256()},
        "sets": {**sets.summary(), "training_ids": [e.id for e in train]},
        "optimizer": {
            "library": f"pycma {cma.__version__}",
            "algorithm": "CMA-ES",
            "config": asdict(config),
            "x0": [0.0] * N_PARAMS,
            "bounds": "x in [-1, 1]^10 (pycma BoundTransform)",
            "decode": "gains = x*max|range|; margin = x+1; width = x+1; amplitude = 0.15*(x+1)",
            "sampler": "private Generator(PCG64(SeedSequence([seed, 6]))) via randn",
        },
        "objective": {
            "weights": OBJECTIVE,
            "definition": "J = mean attack cost + mean nominal cost; see docs/milestones/M6.md",
            "baseline_training": j_base,
            "candidate_training": None if best is None else best["objective"],
        },
        "result": {
            "stop_reason": stop,
            "generations_completed": len(history),
            "improved_training_objective": bool(improved),
            "best_generation": None if best is None else best["generation"],
            "statement": _statement(improved, best, base_train, base_val, cand_val),
        },
        "accounting": {
            "reference_rollouts": len(refs),
            "baseline_rollouts": baseline_rollouts - len(refs),
            "optimization_rollouts": opt_rollouts,
            "validation_preview_rollouts": validation_rollouts,
            "total_rollouts": baseline_rollouts + opt_rollouts + validation_rollouts,
            "rollouts_per_generation": per_generation,
            "max_rollouts": config.max_rollouts,
        },
        "timing": {
            "wall_s": wall,
            "optimization_wall_s": opt_wall,
            "workers": config.workers,
            "wall_clock_cap_s": config.wall_clock_s,
        },
        "history": history,
        "progress_reference_m": progress_ref,
        "measurements": {
            "train": _measure(base_train, best["metrics"] if best else None, progress_ref),
            "validation_preview": _measure(base_val, cand_val, progress_ref),
        },
        "provenance": {
            "git": git_revision(stack.root),
            "scene": stack.scene_info.describe(),
            "versions": {"python": platform.python_version(), **stack.library_versions},
            "weeble_commit": stack.lock["source"]["commit"],
        },
    }
    record["reproducibility_sha256"] = _digest({
        "config": config.deterministic_dict(), "sets": sets.sha256,
        "policy": stack.policy.sha256, "calibration": calibration.sha256(),
        "reflex_schema": schema(), "vector": record["reflex"]["vector"],
        "stop_reason": stop if stop != "wall_clock" else "wall_clock (machine dependent)",
        "measurements": {k: [(m["id"], m["outcome"], m["digest_up_to_stop"])
                             for m in (v["candidate"] or [])]
                         for k, v in record["measurements"].items()},
    })  # fmt: skip
    record["candidate_id"] = f"cand-{record['reproducibility_sha256'][:12]}"
    return record


def _measure(baseline: list[dict], candidate: list[dict] | None, refs: dict) -> dict:
    out = {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_summary": aggregate(baseline),
        "candidate_summary": None if candidate is None else aggregate(candidate),
        "baseline_costs": {m["id"]: episode_cost(m, refs[m["id"]]) for m in baseline},
    }
    if candidate is not None:
        rows = compare(baseline, candidate)
        out.update(comparison=rows, changes=change_counts(rows),
                   candidate_costs={m["id"]: episode_cost(m, refs[m["id"]]) for m in candidate})  # fmt: skip
        out["confirmed_failures"] = {
            "ids": [m["id"] for m in baseline if m["confirmed_failure"]],
            "changes": change_counts(
                [r for r, m in zip(rows, baseline, strict=True) if m["confirmed_failure"]]
            ),
        }
    return out


def _statement(improved, best, base_train, base_val, cand_val) -> str:
    if best is None:
        return "No candidate was evaluated (stopped before the first generation)."
    bt, ct = aggregate(base_train), aggregate(best["metrics"])
    text = (f"Training: baseline survived {bt['attack_survived']}/{bt['attacks']} attacks, "
            f"candidate {ct['attack_survived']}/{ct['attacks']}; nominal falls "
            f"{bt['nominal_fell']} -> {ct['nominal_fell']}.")  # fmt: skip
    if cand_val is not None:
        bv, cv = aggregate(base_val), aggregate(cand_val)
        text += (f" Validation preview: baseline survived {bv['attack_survived']}/"
                 f"{bv['attacks']}, candidate {cv['attack_survived']}/{cv['attacks']}; nominal "
                 f"falls {bv['nominal_fell']} -> {cv['nominal_fell']}.")  # fmt: skip
    if not improved:
        text += " No parameter vector beat the frozen baseline's training objective."
    text += " Status: candidate, pending the M7 regression gate."
    return text


DEFAULT_SETS = "examples/m6/hardening-sets-v1.json"
DEFAULT_CALIBRATION = "examples/m6/reflex-calibration-v1.json"


def default_workers() -> int:
    import os

    return max(1, min(8, (os.cpu_count() or 2) - 2))


def load_calibration(path: Path, policy_sha256: str) -> Calibration:
    calibration = Calibration.from_dict(json.loads(Path(path).read_text()))
    recorded = calibration.source.get("policy_sha256")
    if recorded != policy_sha256:
        raise HardenConfigError(f"calibration was made for policy {recorded}, not {policy_sha256}")
    return calibration


def write_candidate(out_dir: Path, record: dict) -> Path:
    """Write `candidate.json` once; an existing record is never overwritten."""
    out_dir = Path(out_dir)
    path = out_dir / CANDIDATE_NAME
    if path.exists():
        raise FileExistsError(f"{path} exists; candidate records are immutable")
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, record)
    return path


def load_candidate(path: Path) -> dict:
    record = json.loads(Path(path).read_text())
    if record.get("kind") != "robustlab-candidate" or record.get("status") != "candidate":
        raise ValueError(f"{path}: not a robustlab candidate record")
    validate_params(record["reflex"]["vector"])
    return record
