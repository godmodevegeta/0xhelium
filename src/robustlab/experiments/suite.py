"""Paired nominal-vs-shove experiment suite over shove start times.

The simulation is deterministic, so a seed label alone cannot create independent trials. Each trial
here differs physically: the shove starts at a different substep of the documented interval, which
samples a different phase of the gait. Everything else is fixed. See
`.kiro/specs/experiment-runner/design.md`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robustlab.experiments.push import (
    DIRECTIONS,
    PushError,
    PushScenario,
    parse_direction,
    steps_of,
)
from robustlab.experiments.trace import _atomic_write_json, git_revision, write_run
from robustlab.experiments.walk import (
    EpisodeConfig,
    EpisodeRecord,
    EpisodeRunner,
    WalkCriteria,
    evaluate,
)
from robustlab.physics.scene import SCENES

CONFIG_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
REPORT_NAME = "experiment.json"
METHODS = ("stratified", "grid")
OUTCOMES = ("survived", "fell", "numerical")
MAX_TRIALS = 1024
SAVE_TRAJECTORIES_LIMIT = 32
SIM_DT = 0.004
CTRL_DT = 0.02
MAX_HORIZON_S = 600.0
GENERATOR = (
    "numpy.random.SeedSequence(seed).generate_state(trials, uint32); "
    "start_substep_i = stratum_lo_i + (word_i * stratum_size_i >> 32)"
)
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class SuiteConfigError(ValueError):
    """An invalid Suite_Config field."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class StartConfig:
    method: str
    low_s: float
    high_s: float


@dataclass(frozen=True)
class AttackConfig:
    kind: str
    force_n: float
    direction: tuple[float, float, float]
    duration_s: float
    start: StartConfig


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    policy: str
    scene: str
    command: tuple[float, float, float]
    horizon_s: float
    trials: int
    seed: int
    attack: AttackConfig
    description: str = ""
    schema_version: int = CONFIG_SCHEMA_VERSION

    @property
    def low_step(self) -> int:
        return steps_of(self.attack.start.low_s, SIM_DT, "attack.start.low_s")

    @property
    def high_step(self) -> int:
        return steps_of(self.attack.start.high_s, SIM_DT, "attack.start.high_s")

    def push_at(self, start_substep: int) -> PushScenario:
        return PushScenario(
            force_n=self.attack.force_n,
            direction=self.attack.direction,
            start_s=start_substep * SIM_DT,
            duration_s=self.attack.duration_s,
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["command"] = list(self.command)
        out["attack"]["direction"] = list(self.attack.direction)
        return out


# ----------------------------------------------------------------------------- loading


def _check_keys(obj: Any, where: str, required: set[str], optional: set[str] = frozenset()):
    if not isinstance(obj, dict):
        raise SuiteConfigError(where or "config", "expected an object")
    missing = sorted(required - obj.keys())
    if missing:
        raise SuiteConfigError(f"{where}{missing[0]}", "missing")
    unknown = sorted(obj.keys() - required - optional)
    if unknown:
        raise SuiteConfigError(f"{where}{unknown[0]}", "unknown key")


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SuiteConfigError(field, "expected a number")
    if not math.isfinite(value):
        raise SuiteConfigError(field, "must be finite")
    return float(value)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SuiteConfigError(field, "expected an integer")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise SuiteConfigError(field, "expected a string")
    return value


def load_suite_config(
    source: Path | str | dict, *, trials: int | None = None, seed: int | None = None
) -> SuiteConfig:
    """Validate a Suite_Config file or dict; `trials` and `seed` override the file's values."""
    if isinstance(source, dict):
        raw = json.loads(json.dumps(source))  # detach from the caller's object
    else:
        try:
            raw = json.loads(Path(source).read_text())
        except json.JSONDecodeError as exc:
            raise SuiteConfigError("config", f"invalid JSON: {exc}") from exc
    top = {"schema_version", "name", "policy", "scene", "command", "horizon_s", "trials", "seed"}
    _check_keys(raw, "", top | {"attack"}, {"description"})
    if raw["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise SuiteConfigError("schema_version", f"expected {CONFIG_SCHEMA_VERSION}")
    if trials is not None:
        raw["trials"] = trials
    if seed is not None:
        raw["seed"] = seed

    name = _string(raw["name"], "name")
    if not _NAME.match(name):
        raise SuiteConfigError("name", "use 1-64 characters from A-Z a-z 0-9 . _ -")
    description = _string(raw.get("description", ""), "description")
    policy = _string(raw["policy"], "policy")
    scene = _string(raw["scene"], "scene")
    if scene not in SCENES:
        raise SuiteConfigError("scene", f"expected one of {SCENES}")
    command = raw["command"]
    if not isinstance(command, list) or len(command) != 3:
        raise SuiteConfigError("command", "expected [vx, vy, yaw]")
    command = tuple(_number(c, "command") for c in command)
    horizon = _number(raw["horizon_s"], "horizon_s")
    ticks = round(horizon / CTRL_DT)
    if not (0 < horizon <= MAX_HORIZON_S) or abs(ticks * CTRL_DT - horizon) > 1e-9:
        raise SuiteConfigError("horizon_s", f"must be a multiple of {CTRL_DT} s in (0, 600]")
    n = _integer(raw["trials"], "trials")
    if not 1 <= n <= MAX_TRIALS:
        raise SuiteConfigError("trials", f"must be in [1, {MAX_TRIALS}]")
    seed_value = _integer(raw["seed"], "seed")
    if not 0 <= seed_value < 2**64:
        raise SuiteConfigError("seed", "must be in [0, 2**64)")

    attack = raw["attack"]
    _check_keys(attack, "attack.", {"kind", "force_n", "direction", "duration_s", "start"})
    if attack["kind"] != "push":
        raise SuiteConfigError("attack.kind", "only 'push' is supported")
    direction = attack["direction"]
    try:
        if isinstance(direction, str):
            direction = parse_direction(direction)
        elif isinstance(direction, list) and len(direction) == 3:
            direction = tuple(_number(d, "attack.direction") for d in direction)
        else:
            raise SuiteConfigError("attack.direction", f"use {sorted(DIRECTIONS)} or [x, y, z]")
    except PushError as exc:
        raise SuiteConfigError("attack.direction", exc.reason) from exc
    start = attack["start"]
    _check_keys(start, "attack.start.", {"method", "low_s", "high_s"})
    method = _string(start["method"], "attack.start.method")
    if method not in METHODS:
        raise SuiteConfigError("attack.start.method", f"expected one of {METHODS}")
    force = _number(attack["force_n"], "attack.force_n")
    duration = _number(attack["duration_s"], "attack.duration_s")
    low = _number(start["low_s"], "attack.start.low_s")
    high = _number(start["high_s"], "attack.start.high_s")
    try:  # reuse the M1 push rules for force, direction, duration and the 4 ms grid
        PushScenario(force, direction, low, duration)
    except PushError as exc:
        field = {"start_s": "attack.start.low_s"}.get(exc.field, f"attack.{exc.field}")
        raise SuiteConfigError(field, exc.reason) from exc
    try:
        high_step = steps_of(high, SIM_DT, "high_s")
    except PushError as exc:
        raise SuiteConfigError("attack.start.high_s", exc.reason) from exc
    low_step = round(low / SIM_DT)
    width = high_step - low_step
    if width <= 0:
        raise SuiteConfigError("attack.start.high_s", "must be greater than low_s")
    if width < n:
        raise SuiteConfigError(
            "trials", f"{n} trials need at least {n} start substeps, have {width}"
        )
    if method == "stratified" and width % n:
        raise SuiteConfigError(
            "trials",
            f"stratified sampling needs the {width} start substeps to divide evenly into {n} "
            "equal strata; change trials or high_s",
        )
    latest_end = (high_step - 1) * SIM_DT + duration
    if latest_end > horizon + 1e-9:
        raise SuiteConfigError(
            "horizon_s", f"latest shove ends at {latest_end:.3f} s, after the {horizon} s horizon"
        )
    return SuiteConfig(
        name=name,
        policy=policy,
        scene=scene,
        command=command,  # type: ignore[arg-type]
        horizon_s=horizon,
        trials=n,
        seed=seed_value,
        attack=AttackConfig(
            kind="push",
            force_n=force,
            direction=tuple(float(d) for d in direction),  # type: ignore[arg-type]
            duration_s=duration,
            start=StartConfig(method=method, low_s=low, high_s=high),
        ),
        description=description,
    )


# ----------------------------------------------------------------------------- sampling


def strata(low_step: int, high_step: int, n: int) -> list[tuple[int, int]]:
    """n contiguous half-open substep ranges covering [low_step, high_step)."""
    width = high_step - low_step
    return [(low_step + i * width // n, low_step + (i + 1) * width // n) for i in range(n)]


def sample_start_steps(
    method: str, seed: int, low_step: int, high_step: int, n: int
) -> list[tuple[int, tuple[int, int]]]:
    """(start substep, stratum) per trial. Integer arithmetic only."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if not 1 <= n <= high_step - low_step:
        raise ValueError("need 1 <= n <= number of start substeps")
    bounds = strata(low_step, high_step, n)
    if method == "grid":
        return [(lo, (lo, hi)) for lo, hi in bounds]
    words = np.random.SeedSequence(seed).generate_state(n, np.uint32)
    return [(lo + ((int(w) * (hi - lo)) >> 32), (lo, hi)) for w, (lo, hi) in zip(words, bounds)]


# ----------------------------------------------------------------------------- measurement


def classify(kind: str) -> str:
    """FallResult.kind -> Outcome."""
    return {"none": "survived", "physical": "fell", "numerical": "numerical"}[kind]


def trajectory_digest(record: EpisodeRecord) -> str:
    h = hashlib.sha256()
    for array in (record.qpos, record.qvel, record.action):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def episode_summary(record: EpisodeRecord, outcome: dict, push_start_s: float | None) -> dict:
    fall = record.fall
    result = classify(fall.kind)
    contact = fall.events.get("nonfoot_contact")
    summary = {
        "outcome": result,
        "reason": fall.reason,
        "fall_time_s": fall.time_s if result == "fell" else None,
        "numerical_time_s": fall.time_s if result == "numerical" else None,
        "time_after_start_s": None,
        "numerical_failure_any": fall.numerical_failure,
        "events": {name: event.time_s for name, event in sorted(fall.events.items())},
        "first_nonfoot_body": contact.detail["body"] if contact else None,
        "forward_progress_m": outcome["forward_distance_m"],
        "progress_until_s": outcome["progress_until_s"],
        "truncated": record.truncated,
        "n_ticks": record.n_ticks,
        "digest": trajectory_digest(record),
    }
    if push_start_s is not None and result == "fell":
        summary["time_after_start_s"] = fall.time_s - push_start_s
    return summary


def pre_attack_parity(nominal: EpisodeRecord, attacked: EpisodeRecord, start_s: float) -> dict:
    """Max |difference| over Ticks whose state precedes any push effect (time <= start_s)."""
    n = min(nominal.n_ticks, attacked.n_ticks)
    ticks = int(np.count_nonzero(nominal.time[:n] <= start_s + 1e-9))
    diff = 0.0
    for name in ("qpos", "qvel", "action"):
        a = getattr(nominal, name)[:ticks].astype(np.float64)
        b = getattr(attacked, name)[:ticks].astype(np.float64)
        if ticks:
            diff = max(diff, float(np.max(np.abs(a - b))))
    return {"pre_attack_ticks": ticks, "pre_attack_max_abs_diff": diff, "pairing_ok": diff == 0.0}


def stance_at(record: EpisodeRecord, start_s: float) -> dict | None:
    """Foot-floor contact flags at the last Tick at or before `start_s` (None before Tick 1)."""
    ticks = int(np.count_nonzero(record.time <= start_s + 1e-9))
    if ticks == 0:
        return None
    flags = record.contact[ticks - 1]
    return {foot: bool(flag) for foot, flag in zip(("FR", "FL", "RR", "RL"), flags)}


# ----------------------------------------------------------------------------- aggregation


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    """Wilson score interval for k successes in n Bernoulli trials (None if n == 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return [min(low, p), max(high, p)]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def aggregate_case(episodes: list[dict]) -> dict:
    counts = {o: sum(1 for e in episodes if e["outcome"] == o) for o in OUTCOMES}
    total = len(episodes)
    valid = total - counts["numerical"]
    progress = [e["forward_progress_m"] for e in episodes if e["outcome"] != "numerical"]
    return {
        "total_trials": total,
        "survived": counts["survived"],
        "physical_falls": counts["fell"],
        "numerical_failures": counts["numerical"],
        "valid_trials": valid,
        "fall_rate": counts["fell"] / valid if valid else None,
        "fall_rate_denominator": "valid_trials (numerical failures excluded)",
        "fall_time_s": _stats(sorted(e["fall_time_s"] for e in episodes if e["outcome"] == "fell")),
        "time_after_start_s": _stats(
            sorted(e["time_after_start_s"] for e in episodes if e["time_after_start_s"] is not None)
        ),
        "mean_forward_progress_m": statistics.fmean(progress) if progress else None,
        "distinct_trajectories": len({e["digest"] for e in episodes}),
    }


def paired_table(trials: list[dict]) -> dict[str, int]:
    table = {f"{a}->{b}": 0 for a in OUTCOMES for b in OUTCOMES}
    for t in trials:
        table[f"{t['nominal']['outcome']}->{t['attacked']['outcome']}"] += 1
    return table


def aggregate(trials: list[dict], method: str) -> dict:
    nominal = aggregate_case([t["nominal"] for t in trials])
    attacked = aggregate_case([t["attacked"] for t in trials])
    if method == "stratified":
        attacked["wilson_95"] = wilson_interval(
            attacked["physical_falls"], attacked["valid_trials"]
        )
        attacked["wilson_note"] = (
            "treats the valid trials as independent Bernoulli draws; stratified sampling with "
            "equal strata has variance no larger than that"
        )
    else:
        attacked["wilson_95"] = None
        attacked["wilson_note"] = "deterministic grid: no sampling interval"
    both = [
        t
        for t in trials
        if t["nominal"]["outcome"] != "numerical" and t["attacked"]["outcome"] != "numerical"
    ]
    diffs = [t["paired"]["progress_difference_m"] for t in both]
    return {
        "nominal": nominal,
        "attacked": attacked,
        "paired": {
            "table_nominal_to_attacked": paired_table(trials),
            "valid_pairs": len(both),
            "mean_progress_difference_m": statistics.fmean(diffs) if diffs else None,
            "pairing_ok_all": all(t["paired"]["pairing_ok"] for t in trials),
            "max_pre_attack_abs_diff": max(
                (t["paired"]["pre_attack_max_abs_diff"] for t in trials), default=0.0
            ),
        },
    }


def interpretation(config: SuiteConfig, aggregates: dict) -> list[str]:
    a = config.attack
    width = config.high_step - config.low_step
    dist = (
        f"shove start uniform over the {width} substeps (4 ms grid) of "
        f"[{a.start.low_s:.3f}, {a.start.high_s:.3f}) s; force {a.force_n:g} N, direction "
        f"{list(a.direction)} (initial-heading frame), duration {a.duration_s:g} s, command "
        f"{list(config.command)}, scene {config.scene}, policy {config.policy}, nominal reset "
        "state, all fixed"
    )
    att = aggregates["attacked"]
    rate = att["fall_rate"]
    counts = f"{att['physical_falls']}/{att['valid_trials']}"
    lines = []
    if rate is None:
        lines.append("Attacked: no valid trials; no fall rate.")
    elif a.start.method == "stratified":
        lo, hi = att["wilson_95"]
        lines.append(
            f"Attacked fall-rate estimate {counts} = {rate:.3f} over the test distribution "
            f"({dist}). Wilson 95% interval [{lo:.3f}, {hi:.3f}] "
            f"({att['wilson_note']})."
        )
    elif config.trials == width:
        lines.append(
            f"Attacked failure frequency {counts} = {rate:.3f}, exact over the test distribution "
            f"({dist}): every start substep was simulated once; the model is deterministic."
        )
    else:
        lines.append(
            f"Attacked failure frequency {counts} = {rate:.3f} over a deterministic grid of "
            f"{config.trials} evenly spaced start times (every {width // config.trials}th substep "
            f"of the interval; {dist}). Not a random sample; no confidence interval."
        )
    nom = aggregates["nominal"]
    if nom["distinct_trajectories"] == 1:
        verdict = "did not fall" if nom["physical_falls"] == 0 else "fell"
        lines.append(
            f"Nominal: all {nom['total_trials']} nominal runs produced one identical trajectory "
            "(repeated identical simulation of a deterministic model). They are one observation, "
            f"not {nom['total_trials']}: the unperturbed walk {verdict} within the horizon."
        )
    else:
        lines.append(
            f"Nominal: {nom['distinct_trajectories']} distinct trajectories in "
            f"{nom['total_trials']} runs; expected 1 for identical nominal conditions."
        )
    lines.append(
        f"Falls are counted only within the {config.horizon_s:g} s horizon. These numbers say "
        "nothing about other forces, directions, durations, commands, terrains, start times outside "
        "the interval, or the real robot."
    )
    return lines


def results_digest(sampling: dict, trials: list[dict], aggregates: dict) -> str:
    stripped = [{k: v for k, v in t.items() if k != "trajectories"} for t in trials]
    payload = {"sampling": sampling, "trials": stripped, "aggregates": aggregates}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------- running


def sampling_record(config: SuiteConfig) -> dict:
    s = config.attack.start
    starts = sample_start_steps(
        s.method, config.seed, config.low_step, config.high_step, config.trials
    )
    width = config.high_step - config.low_step
    return {
        "method": s.method,
        "kind": "randomized stratified sampling"
        if s.method == "stratified"
        else (
            "exhaustive deterministic sweep"
            if config.trials == width
            else "deterministic grid sweep"
        ),
        "seed": config.seed,
        "seed_used": s.method == "stratified",
        "generator": GENERATOR if s.method == "stratified" else "k_i = stratum_lo_i (seed unused)",
        "interval_s": [s.low_s, s.high_s],
        "interval_substeps": [config.low_step, config.high_step],
        "start_substeps_in_interval": width,
        "sim_dt": SIM_DT,
        "trials": config.trials,
        "stratum_size_substeps": width // config.trials if width % config.trials == 0 else None,
        "start_substeps": [k for k, _ in starts],
        "strata": [list(b) for _, b in starts],
    }


def check_out_dir(out_dir: Path, overwrite: bool) -> None:
    if (Path(out_dir) / REPORT_NAME).exists() and not overwrite:
        raise FileExistsError(f"{Path(out_dir) / REPORT_NAME} exists; use overwrite")


def run_suite(
    config: SuiteConfig,
    stack,
    out_dir: Path,
    *,
    save_trajectories: bool | None = None,
    overwrite: bool = False,
    criteria: WalkCriteria | None = None,
) -> dict:
    """Run every trial, write trajectories (optional) and `experiment.json`; return the report."""
    out_dir = Path(out_dir)
    check_out_dir(out_dir, overwrite)
    criteria = criteria or WalkCriteria()
    error = stack.command_error(config.command)
    if error:
        raise SuiteConfigError("command", error)
    if save_trajectories is None:
        save_trajectories = config.trials <= SAVE_TRAJECTORIES_LIMIT
    sampling = sampling_record(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and (out_dir / "trials").exists():
        shutil.rmtree(out_dir / "trials")  # stale trials from an earlier run of this directory

    runner = EpisodeRunner(stack.model, stack.ids, stack.policy, stack.actor, criteria)
    width = max(2, len(str(config.trials - 1)))
    trials: list[dict] = []
    wall = time.perf_counter()
    for i, (k, stratum) in enumerate(zip(sampling["start_substeps"], sampling["strata"])):
        push = config.push_at(k)
        nominal = runner.run(EpisodeConfig(config.command, config.horizon_s, config.seed))
        nominal_out = evaluate(nominal, criteria)
        attacked = runner.run(EpisodeConfig(config.command, config.horizon_s, config.seed, push))
        attacked_out = evaluate(attacked, criteria)
        nom = episode_summary(nominal, nominal_out, None)
        att = episode_summary(attacked, attacked_out, push.start_s)
        paired = pre_attack_parity(nominal, attacked, push.start_s)
        paired["progress_difference_m"] = att["forward_progress_m"] - nom["forward_progress_m"]
        trial = {
            "trial": i,
            "start_substep": k,
            "start_s": push.start_s,
            "stratum_substeps": stratum,
            "push": attacked.push,
            "stance_at_start": stance_at(nominal, push.start_s),
            "nominal": nom,
            "attacked": att,
            "paired": paired,
            "trajectories": None,
        }
        if save_trajectories:
            paths = {}
            for case, record, outcome in (
                ("nominal", nominal, nominal_out),
                ("attacked", attacked, attacked_out),
            ):
                rel = f"trials/{i:0{width}d}/{case}"
                manifest = stack.manifest_for(
                    record,
                    run_id=f"{config.name}/trial-{i:0{width}d}/{case}",
                    command=config.command,
                    seconds=config.horizon_s,
                    seed=config.seed,
                    outcome=outcome,
                    criteria=criteria,
                )
                manifest["experiment"] = {"name": config.name, "trial": i, "case": case}
                write_run(out_dir / rel, record, manifest, overwrite=overwrite)
                paths[case] = rel
            trial["trajectories"] = paths
        trials.append(trial)
    wall = time.perf_counter() - wall

    aggregates = aggregate(trials, config.attack.start.method)
    config_dict = config.to_dict()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "robustlab-experiment-report",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config_dict,
        "config_sha256": hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest(),
        "provenance": {
            "policy": stack.policy_info,
            "scene": stack.scene_info.describe(),
            "weeble_commit": stack.lock["source"]["commit"],
            "playground_commit": stack.provenance.get("playground_commit"),
            "menagerie_commit": stack.provenance.get("menagerie_commit"),
            "versions": {"python": _python_version(), **stack.library_versions},
            "platform": _platform(),
            "git": git_revision(stack.root),
            "sim_dt": float(stack.model.opt.timestep),
            "ctrl_dt": stack.policy.control_dt,
            "n_substeps": stack.policy.n_substeps,
            "initial_state": "policy default_qpos / default_qvel for every episode",
            "criteria": asdict(criteria),
        },
        "fixed_parameters": {
            "force_n": config.attack.force_n,
            "direction_heading_frame": list(config.attack.direction),
            "duration_s": config.attack.duration_s,
            "command": list(config.command),
            "horizon_s": config.horizon_s,
            "scene": config.scene,
            "policy_sha256": stack.policy.sha256,
            "initial_state": "nominal reset",
            "varied": "shove start time only",
        },
        "sampling": sampling,
        "trials": trials,
        "aggregates": aggregates,
        "interpretation": interpretation(config, aggregates),
        "results_sha256": results_digest(sampling, trials, aggregates),
        "trajectories_saved": save_trajectories,
        "timing": {
            "wall_s": wall,
            "rollouts": 2 * config.trials,
            "mean_rollout_s": wall / (2 * config.trials),
            "load_s": stack.load_s,
        },
    }
    _atomic_write_json(out_dir / REPORT_NAME, report)
    return report


def read_report(out_dir: Path) -> dict:
    report = json.loads((Path(out_dir) / REPORT_NAME).read_text())
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(f"unsupported experiment report schema {report.get('schema_version')!r}")
    return report


def _python_version() -> str:
    import platform

    return platform.python_version()


def _platform() -> dict:
    import platform

    return {"platform": platform.platform(), "machine": platform.machine()}


# ----------------------------------------------------------------------------- summary


def _fmt(value, spec=".3f") -> str:
    return "-" if value is None else format(value, spec)


def _episode_cell(e: dict) -> str:
    if e["outcome"] == "survived":
        return "survived"
    if e["outcome"] == "numerical":
        return f"NUMERICAL {e['reason']} @{_fmt(e['numerical_time_s'])}"
    return f"FELL {e['reason']} @{_fmt(e['fall_time_s'])}"


def format_summary(report: dict) -> str:
    cfg, s, agg = report["config"], report["sampling"], report["aggregates"]
    seed_note = "" if s["seed_used"] else " unused"
    low, high = s["interval_s"]
    attack = cfg["attack"]
    lines = [
        f"experiment       {cfg['name']}  ({s['kind']}, seed {s['seed']}{seed_note})",
        f"varied           shove start in [{low:.3f}, {high:.3f}) s, 4 ms grid, {s['trials']} trials",
        (
            f"fixed            {attack['force_n']:g} N toward {attack['direction']} for "
            f"{attack['duration_s']:g} s; command {cfg['command']}; {cfg['horizon_s']:g} s; "
            f"scene {cfg['scene']}"
        ),
        "",
        "trial  start_s  stance FR FL RR RL  nominal   attacked                     +t_fall  "
        + "progress nom/att m",
    ]
    for t in report["trials"]:
        stance = t["stance_at_start"]
        feet = (
            " ".join("x " if stance[f] else ". " for f in ("FR", "FL", "RR", "RL"))
            if stance
            else "?"
        )
        att = t["attacked"]
        lines.append(
            f"{t['trial']:>5}  {t['start_s']:7.3f}  {feet:>18}  {t['nominal']['outcome']:<8}  "
            f"{_episode_cell(att):<27}  {_fmt(att['time_after_start_s']):>7}  "
            f"{t['nominal']['forward_progress_m']:6.3f} / {att['forward_progress_m']:6.3f}"
            f"{'' if t['paired']['pairing_ok'] else '  PAIRING MISMATCH'}"
        )
    lines.append("")
    for case in ("nominal", "attacked"):
        a = agg[case]
        rate = "n/a" if a["fall_rate"] is None else f"{a['fall_rate']:.3f}"
        extra = ""
        if case == "attacked" and a.get("wilson_95"):
            extra = f", Wilson 95% [{a['wilson_95'][0]:.3f}, {a['wilson_95'][1]:.3f}]"
        lines.append(
            f"{case:<9} falls {a['physical_falls']}/{a['valid_trials']} valid = {rate}{extra}; "
            f"survived {a['survived']}; numerical {a['numerical_failures']}; "
            f"distinct trajectories {a['distinct_trajectories']}"
        )
    p = agg["paired"]
    lines.append(
        f"paired    {p['table_nominal_to_attacked']['survived->fell']} pairs survived->fell, "
        f"{p['table_nominal_to_attacked']['survived->survived']} survived->survived; mean progress "
        f"difference {_fmt(p['mean_progress_difference_m'])} m; pre-attack max |diff| "
        f"{p['max_pre_attack_abs_diff']:g}"
    )
    ft = agg["attacked"]["time_after_start_s"]
    if ft["n"]:
        lines.append(
            f"attacked  time from shove start to fall: min {ft['min']:.3f} / median "
            f"{ft['median']:.3f} / max {ft['max']:.3f} s"
        )
    lines.append("")
    lines += [f"- {line}" for line in report["interpretation"]]
    lines.append(f"results_sha256   {report['results_sha256']}")
    return "\n".join(lines)
