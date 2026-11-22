"""Run the pre-registered gate once and decide.

Every episode goes through the M6 metric code (`adapt.evaluate.run_episode`), so the gate adds no
physics. `assess` is pure: it turns per-episode outcomes into criteria, statistics and a decision,
and is unit-tested with synthetic outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Callable

import numpy as np

from robustlab.adapt.evaluate import Context, Evaluator, run_episode
from robustlab.experiments.suite import wilson_interval
from robustlab.experiments.trace import git_revision
from robustlab.experiments.walk import yaw_from_quat
from robustlab.gate.config import Gate

RESULT_SCHEMA_VERSION = 1
BASELINE = "baseline"
KEEP = ("id", "outcome", "reason", "fall_time_s", "time_after_push_s", "max_tilt_after_push_deg",
        "progress_m", "progress_until_s", "digest_up_to_stop")  # fmt: skip


def compact(m: dict) -> dict:
    out = {k: m[k] for k in KEEP}
    r = m.get("reflex")
    if r:
        out["reflex"] = {"activation_fraction": r["activation_fraction"],
                         "max_abs_residual": r["max_abs_residual"]}  # fmt: skip
    return out


def displacement(record, until_s: float) -> list[float]:
    """Trunk displacement at the last valid Tick in the initial-heading frame (x fwd, y left)."""
    last = int(np.count_nonzero(record.time <= until_s + 1e-9)) - 1
    if last < 0:
        return [0.0, 0.0]
    d = record.qpos[last, 0:2] - record.initial_qpos[0:2]
    yaw = float(yaw_from_quat(record.initial_qpos[3:7]))
    c, s = math.cos(-yaw), math.sin(-yaw)
    return [float(c * d[0] - s * d[1]), float(s * d[0] + c * d[1])]


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(X >= b) for X ~ Binomial(b + c, 1/2): exact one-sided McNemar for improvement."""
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(b, n + 1)) / 2**n)


def run_gate(stack, gate: Gate, *, workers: int = 1,
             on_progress: Callable[[str], None] | None = None) -> dict:  # fmt: skip
    """Evaluate the baseline and every registered candidate on every suite."""
    say = on_progress or (lambda _m: None)
    started = time.perf_counter()
    controllers = [(BASELINE, None)] + [(c.candidate_id, c.vector) for c in gate.candidates]
    ctx = Context(stack, gate.calibration)
    out: dict = {cid: {} for cid, _ in controllers}
    rollouts = 0
    # nominal: in-process, records kept for displacement
    say("nominal walking")
    for cid, vector in controllers:
        rows = []
        for spec in gate.nominal:
            metrics, record = run_episode(ctx, spec, None if vector is None else np.asarray(vector),
                                          stop_on_fall=False, keep=True)  # fmt: skip
            row = compact(metrics)
            row["displacement_m"] = displacement(record, metrics["progress_until_s"])
            rows.append(row)
            rollouts += 1
        out[cid]["nominal"] = rows
    # attacks: process pool, metrics only, results in task order
    evaluator = Evaluator(ctx, workers)
    try:
        for suite, specs in (("holdout", gate.holdout), ("gauntlet", gate.gauntlet),
                             ("known_failures", gate.known_failures)):  # fmt: skip
            say(suite)
            tasks = [(vector, s) for _, vector in controllers for s in specs]
            results = evaluator.map(tasks)
            n = len(specs)
            for i, (cid, _) in enumerate(controllers):
                out[cid][suite] = [compact(m) for m in results[i * n : (i + 1) * n]]
        rollouts += evaluator.rollouts
    finally:
        evaluator.close()
    return {"outcomes": out, "rollouts": rollouts, "wall_s": time.perf_counter() - started,
            "workers": workers}  # fmt: skip


def _rate(rows: list[dict]) -> dict:
    n = len(rows)
    fell = sum(r["outcome"] == "fell" for r in rows)
    numerical = sum(r["outcome"] == "numerical" for r in rows)
    valid = n - numerical
    return {"episodes": n, "fell": fell, "numerical": numerical, "valid": valid,
            "fall_rate": fell / valid if valid else None,
            "wilson_95": wilson_interval(fell, valid) if valid else None}  # fmt: skip


def _paired(base: list[dict], cand: list[dict]) -> dict:
    assert [b["id"] for b in base] == [c["id"] for c in cand]
    recovered = [
        b["id"]
        for b, c in zip(base, cand, strict=True)
        if b["outcome"] == "fell" and c["outcome"] == "survived"
    ]
    regressed = [
        b["id"]
        for b, c in zip(base, cand, strict=True)
        if b["outcome"] == "survived" and c["outcome"] != "survived"
    ]
    numerical = [c["id"] for c in cand if c["outcome"] == "numerical"]
    return {"recovered": recovered, "regressed": regressed, "candidate_numerical": numerical,
            "mcnemar_one_sided_p": mcnemar_one_sided(len(recovered), len(regressed))}  # fmt: skip


def assess(gate: Gate, outcomes: dict) -> dict:
    """Criteria G1-G4 for every candidate; the decision concerns the primary candidate only."""
    crit = gate.criteria
    base = outcomes[BASELINE]
    per_candidate = {}
    for cand in gate.candidates:
        mine = outcomes[cand.candidate_id]
        # G1 nominal
        g1 = crit["G1_nominal"]
        nominal_rows = []
        for b, c in zip(base["nominal"], mine["nominal"], strict=True):
            db, dc = np.asarray(b["displacement_m"]), np.asarray(c["displacement_m"])
            diff = float(np.linalg.norm(dc - db))
            tol = max(
                g1["displacement_tolerance_m"],
                g1["displacement_tolerance_rel"] * float(np.linalg.norm(db)),
            )
            nominal_rows.append({"id": b["id"], "baseline": b["outcome"], "candidate": c["outcome"],
                                 "baseline_displacement_m": b["displacement_m"],
                                 "candidate_displacement_m": c["displacement_m"],
                                 "displacement_difference_m": diff, "tolerance_m": tol,
                                 "within_tolerance": diff <= tol,
                                 "reflex_activation": (c.get("reflex") or {}).get("activation_fraction", 0.0)})  # fmt: skip
        nom = _rate(mine["nominal"])
        g1_failures = [
            r["id"]
            for r in nominal_rows
            if r["candidate"] != "survived" or not r["within_tolerance"]
        ]
        g1_pass = (
            nom["fell"] <= g1["max_falls"]
            and nom["numerical"] <= g1["max_numerical"]
            and not g1_failures
        )
        # G2 holdout
        g2 = crit["G2_holdout"]
        hb, hc = _rate(base["holdout"]), _rate(mine["holdout"])
        hp = _paired(base["holdout"], mine["holdout"])
        cap = max(g2["max_fall_rate_floor"], (hb["fall_rate"] or 0.0) - g2["required_rate_drop"])
        g2_checks = {
            "fewer_falls": hc["fell"] < hb["fell"],
            "rate_cap": hc["fall_rate"] is not None and hc["fall_rate"] <= cap + 1e-12,
            "mcnemar": hp["mcnemar_one_sided_p"] <= g2["mcnemar_one_sided_alpha"],
        }
        # G3 regression gauntlet
        g3 = crit["G3_regression"]
        conditions = {}
        for cond in gate.doc["suites"]["gauntlet"]["conditions"]:
            ids = [i for i, s in enumerate(gate.gauntlet) if s.tags == (cond["id"],)]
            rb, rc = [base["gauntlet"][i] for i in ids], [mine["gauntlet"][i] for i in ids]
            b_rate, c_rate = _rate(rb), _rate(rc)
            paired = _paired(rb, rc)
            conditions[cond["id"]] = {"baseline": b_rate, "candidate": c_rate,
                                      "increase": (c_rate["fall_rate"] or 0) - (b_rate["fall_rate"] or 0),
                                      "recovered": len(paired["recovered"]), "regressed": paired["regressed"],
                                      "mcnemar_one_sided_p": paired["mcnemar_one_sided_p"]}  # fmt: skip
        gb, gc = _rate(base["gauntlet"]), _rate(mine["gauntlet"])
        g_paired = _paired(base["gauntlet"], mine["gauntlet"])
        limit = g3["max_fall_rate_increase_abs"]
        g3_checks = {"overall": (gc["fall_rate"] or 0) <= (gb["fall_rate"] or 0) + limit + 1e-12}
        if g3["per_condition"]:
            g3_checks.update(
                {f"condition {k}": v["increase"] <= limit + 1e-12 for k, v in conditions.items()}
            )
        # G4 numerical, any suite
        numerical = sum(
            _rate(mine[s])["numerical"]
            for s in ("nominal", "holdout", "gauntlet", "known_failures")
        )
        g4_pass = numerical <= crit["G4_numerical"]["max_numerical"]
        # known failures (reported)
        kb, kc = _rate(base["known_failures"]), _rate(mine["known_failures"])
        kp = _paired(base["known_failures"], mine["known_failures"])
        criteria = {
            "G1_nominal": {"pass": g1_pass, "falls": nom["fell"], "numerical": nom["numerical"],
                           "episodes": nom["episodes"], "failures": g1_failures, "rows": nominal_rows},
            "G2_holdout": {"pass": all(g2_checks.values()), "checks": g2_checks, "baseline": hb, "candidate": hc,
                           "rate_cap": cap, **hp},
            "G3_regression": {"pass": all(g3_checks.values()), "checks": g3_checks, "baseline": gb, "candidate": gc,
                              "limit": limit, "regressed": g_paired["regressed"],
                              "recovered": len(g_paired["recovered"]), "conditions": conditions},
            "G4_numerical": {"pass": g4_pass, "candidate_numerical": numerical},
        }  # fmt: skip
        failed = [k for k, v in criteria.items() if not v["pass"]]
        per_candidate[cand.candidate_id] = {
            "seed": cand.seed, "primary": cand.primary, "eligible_for_promotion": cand.primary,
            "criteria": criteria, "failed_criteria": failed, "passes_all": not failed,
            "known_failures": {"baseline": kb, "candidate": kc, "recovered": kp["recovered"],
                               "still_failing": [c["id"] for c in mine["known_failures"] if c["outcome"] == "fell"],
                               "regressed": kp["regressed"]},
        }  # fmt: skip
    primary = per_candidate[gate.primary_id]
    decision = {
        "candidate_id": gate.primary_id,
        "verdict": "promote" if primary["passes_all"] else "reject",
        "failed_criteria": primary["failed_criteria"],
        "statement": (
            f"{gate.primary_id} passed G1-G4 and is promoted."
            if primary["passes_all"]
            else f"{gate.primary_id} failed {', '.join(primary['failed_criteria'])}; the frozen baseline "
            "stays active."
        ),
        "note": "Only the pre-registered primary candidate is eligible; other candidates are reported "
        "and cannot be promoted by this gate version, so the holdout is not used to pick a winner.",
    }
    return {"candidates": per_candidate, "decision": decision}


def build_result(stack, gate: Gate, run: dict, verdict: dict) -> dict:
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "robustlab-gate-result",
        "gate": {
            "name": gate.name,
            "path": gate.path,
            "sha256": gate.sha256,
            "locked_sha256": gate.doc["locked_set"]["locked_sha256"],
        },
        "suites": {k: len(v) for k, v in gate.suite_ids().items()},
        "criteria_definition": gate.criteria,
        **verdict,
        "outcomes": run["outcomes"],
        "accounting": {"rollouts": run["rollouts"], "controllers": 1 + len(gate.candidates)},
        "timing": {"wall_s": run["wall_s"], "workers": run["workers"]},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git": git_revision(stack.root),
            "policy_sha256": stack.policy.sha256,
            "versions": {"python": platform.python_version(), **stack.library_versions},
            "scene": stack.scene_info.describe(),
        },
    }
    result["results_sha256"] = hashlib.sha256(json.dumps(
        {"outcomes": {c: {s: [(r["id"], r["outcome"], r["digest_up_to_stop"]) for r in rows]
                          for s, rows in suites.items()} for c, suites in run["outcomes"].items()},
         "decision": verdict["decision"]}, sort_keys=True).encode()).hexdigest()  # fmt: skip
    return result
