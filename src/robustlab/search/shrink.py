"""Bounded coarse-to-fine local shrinking of a confirmed failure toward lower impulse.

The fall/no-fall boundary is not monotonic in force or duration, so no bisection is used and no
threshold is inferred. The shrinker is a local pattern search:

1. Descent, direction and start fixed. For force, then duration, try reductions to 50%, 70%, 85%
   and 95% of the current value (coarse to fine). The first reduction that falls *and confirms*
   becomes the new best, and that dimension starts again from 50%. A pass with no confirmed
   improvement in either dimension ends the descent.
2. Release. At one fine step lower force (95%), try neighbouring starts (+/-1, +/-2 substeps) and
   directions (+/-5 deg). A confirmed improvement restarts the descent from there.

The best failure is only ever replaced by a newly confirmed failure with lower impulse. Evaluated
falls that do not confirm are recorded as `unconfirmed` and never adopted.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from robustlab.search.space import DURATION_MIN_SUBSTEPS, FORCE_MIN_N, Candidate

EPS = 1e-9
REDUCTIONS = (0.5, 0.7, 0.85, 0.95)
RELEASE_FORCE_FACTOR = 0.95
RELEASE_START_OFFSETS = (-2, -1, 1, 2)
RELEASE_DIRECTION_OFFSETS_DEG = (-5.0, 5.0)
MAX_RELEASE_ROUNDS = 3


def reduce(candidate: Candidate, dimension: str, factor: float) -> Candidate:
    """`candidate` with force or duration scaled by `factor` (quantized, never below bounds)."""
    if dimension == "force":
        return Candidate.make(
            max(FORCE_MIN_N, candidate.force_n * factor),
            candidate.direction_deg,
            candidate.start_substep,
            candidate.duration_substeps,
        )
    return Candidate.make(
        candidate.force_n,
        candidate.direction_deg,
        candidate.start_substep,
        max(DURATION_MIN_SUBSTEPS, math.floor(candidate.duration_substeps * factor)),
    )


def shrink(
    start: Candidate,
    evaluate: Callable,
    confirm: Callable,
    budget: int,
    *,
    should_stop: Callable[[], str | None] = lambda: None,
) -> dict:
    """Shrink a confirmed failing `start`. `evaluate(c) -> EvalResult`,
    `confirm(c, digest) -> Confirmation`. Returns the shrink record."""
    best = start
    best_confirmation: dict | None = None
    tested: list[dict] = []
    seen = {start.key()}
    history = [{"step": 0, "candidate": start.to_dict(), "via": "start (confirmed search failure)"}]
    state = {"rollouts": 0, "confirmation_rollouts": 0, "stopped": None}

    def attempt(point: Candidate, phase: str) -> bool | None:
        """Evaluate `point`; True if it is a confirmed improvement, False if not, None if the
        budget or a stop condition prevents evaluating it."""
        nonlocal best, best_confirmation
        if point.impulse_ns >= best.impulse_ns - EPS or point.key() in seen:
            return False
        if state["rollouts"] >= budget:
            state["stopped"] = "budget"
            return None
        reason = should_stop()
        if reason:
            state["stopped"] = reason
            return None
        seen.add(point.key())
        result = evaluate(point)
        state["rollouts"] += 1
        entry = {
            "phase": phase,
            "candidate": point.to_dict(),
            "outcome": result.outcome,
            "reason": result.reason,
            "objective": result.objective,
            "digest": result.digest,
            "status": result.outcome,
        }
        improved = False
        if result.outcome == "fell":
            conf = confirm(point, result.digest)
            state["confirmation_rollouts"] += conf.rollouts
            entry["confirmed"] = conf.confirmed
            if conf.confirmed:
                entry["status"] = "confirmed_improvement"
                best = point
                best_confirmation = {
                    "candidate": point.to_dict(),
                    "digest": result.digest,
                    "trial": conf.trial,
                }
                history.append({"step": len(history), "candidate": point.to_dict(), "via": phase})
                improved = True
            else:
                entry["status"] = "unconfirmed"
        tested.append(entry)
        return improved

    def descent() -> None:
        while state["stopped"] is None:
            any_improved = False
            for dimension in ("force", "duration"):
                restart = True
                while restart and state["stopped"] is None:
                    restart = False
                    for factor in REDUCTIONS:
                        outcome = attempt(reduce(best, dimension, factor), f"descent-{dimension}")
                        if outcome is None:
                            return
                        if outcome:
                            any_improved = restart = True
                            break
            if not any_improved:
                return

    descent()
    for _ in range(MAX_RELEASE_ROUNDS):
        if state["stopped"] is not None:
            break
        lower = reduce(best, "force", RELEASE_FORCE_FACTOR)
        points = [
            Candidate.make(
                lower.force_n, best.direction_deg, best.start_substep + dk, best.duration_substeps
            )
            for dk in RELEASE_START_OFFSETS
        ] + [
            Candidate.make(
                lower.force_n, best.direction_deg + dd, best.start_substep, best.duration_substeps
            )
            for dd in RELEASE_DIRECTION_OFFSETS_DEG
        ]
        improved = False
        for point in points:
            outcome = attempt(point, "release")
            if outcome is None:
                break
            if outcome:
                improved = True
                break
        if not improved:
            break
        descent()

    surviving_below = [
        t
        for t in tested
        if t["outcome"] == "survived" and t["candidate"]["impulse_ns"] < best.impulse_ns - EPS
    ]
    untested = [
        (
            "Only reductions to 50/70/85/95% of the current force or duration were tried at each "
            "step; values between them were not tested."
        ),
        (
            "Joint force-and-duration reductions were not tried except through successive "
            "single-dimension steps."
        ),
        (
            f"Directions other than {best.direction_deg:g} deg and start times other than substep "
            f"{best.start_substep} were tried only in the release phase (+/-5 deg, +/-1-2 substeps, "
            "at 95% force)."
        ),
        "Any disturbance other than one constant horizontal trunk force.",
    ]
    return {
        "method": "local coarse-to-fine pattern search (reductions 50/70/85/95%), then release",
        "start": start.to_dict(),
        "final": best.to_dict(),
        "final_confirmed": best_confirmation,
        "improved": best != start,
        "impulse_reduction_ns": start.impulse_ns - best.impulse_ns,
        "history": history,
        "tested": tested,
        "surviving_below_final": surviving_below,
        "untested": untested,
        "budget": budget,
        "rollouts": state["rollouts"],
        "confirmation_rollouts": state["confirmation_rollouts"],
        "budget_exhausted": state["stopped"] == "budget",
        "stopped": state["stopped"],
        "statement": (
            "Lowest confirmed impulse among the points tested here; not shown to be minimal. "
            "Lower-impulse failures may exist at untested points."
        ),
    }
