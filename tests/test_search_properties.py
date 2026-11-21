"""Hypothesis property tests for the M4 search (Requirement 8.1). Physics is replaced by fast fake
evaluators so the properties exercise the search, budget and shrinking logic itself."""

from __future__ import annotations

import itertools
import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from robustlab.experiments.suite import load_suite_config
from robustlab.search.engine import (
    Confirmation,
    EvalResult,
    SearchConfig,
    discover,
    scenario_config,
)
from robustlab.search.objective import NUMERICAL_OBJECTIVE, objective
from robustlab.search.shrink import shrink
from robustlab.search.space import (
    DURATION_MAX_SUBSTEPS,
    DURATION_MIN_SUBSTEPS,
    FORCE_MAX_N,
    FORCE_MIN_N,
    MAX_IMPULSE_NS,
    START_MIN_SUBSTEP,
    START_SUBSTEPS,
    Candidate,
    decode_cma,
    quantize_direction,
    sample_uniform,
)

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def in_space(c: Candidate) -> bool:
    return (
        FORCE_MIN_N <= c.force_n <= FORCE_MAX_N
        and 0.0 <= c.direction_deg < 360.0
        and START_MIN_SUBSTEP <= c.start_substep < START_MIN_SUBSTEP + START_SUBSTEPS
        and DURATION_MIN_SUBSTEPS <= c.duration_substeps <= DURATION_MAX_SUBSTEPS
        and round(c.force_n * 10) == c.force_n * 10
        and round(c.direction_deg * 10) == round(c.direction_deg * 10, 6)
    )


def circular_gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# Feature: m4-failure-discovery, Property 20: parameter bounds and angular wrapping
@given(st.tuples(finite, finite, finite, finite), st.integers(-5, 5))
def test_property_bounds_and_wrapping(x, turns) -> None:
    c = decode_cma(x)
    assert in_space(c)
    r = math.hypot(150 * x[0], 150 * x[1])
    if 10.0 <= r <= 150.0:
        assert abs(c.force_n - r) <= 0.05 + 1e-9  # inside the annulus only rounding applies
    if r > 0:
        expected = math.degrees(math.atan2(x[1], x[0])) % 360.0
        assert circular_gap(c.direction_deg, expected) <= 0.05 + 1e-9
    same = Candidate.make(c.force_n, c.direction_deg + 360.0 * turns, c.start_substep,
                          c.duration_substeps)  # fmt: skip
    assert same == c


@given(st.floats(min_value=-0.004, max_value=0.004), st.floats(min_value=0.1, max_value=1.0))
def test_property_no_seam_at_zero_degrees(dy, radius) -> None:
    """Crossing the +x axis in the Cartesian force plane moves the direction across 0/360 deg
    continuously: a tiny change in x gives a tiny circular change in direction."""
    a = decode_cma([radius, dy, 0.5, 0.5])
    b = decode_cma([radius, -dy, 0.5, 0.5])
    bound = 2 * math.degrees(math.atan2(abs(dy), radius)) + 0.1 + 1e-9
    assert circular_gap(a.direction_deg, b.direction_deg) <= bound


# Feature: m4-failure-discovery, Property 21: time quantization
@given(st.tuples(finite, finite, finite, finite))
def test_property_time_quantization(x) -> None:
    c = decode_cma(x)
    assert c.start_s == c.start_substep * 0.004
    assert c.duration_s == c.duration_substeps * 0.004
    assert c.push().window() == (c.start_substep, c.start_substep + c.duration_substeps)
    assert math.isclose(c.push().impulse_ns, c.impulse_ns, rel_tol=1e-12)
    assert c.impulse_ns <= MAX_IMPULSE_NS + 1e-9


@given(st.integers(0, 2**32))
def test_property_random_sampler_in_space(seed) -> None:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    for _ in range(20):
        assert in_space(sample_uniform(rng))


# ----------------------------------------------------------------------------- fake physics


class FakePhysics:
    """Deterministic fake: falls when a direction-weighted impulse exceeds `threshold` inside a
    gait-phase window; numerical on a sparse set of start substeps."""

    def __init__(self, threshold: float, numerical_mod: int = 0, confirm_ok: float = 1.0,
                 seed: int = 0) -> None:  # fmt: skip
        self.threshold = threshold
        self.numerical_mod = numerical_mod
        self.confirm_ok = confirm_ok
        self.confirm_rng = np.random.default_rng(seed)
        self.calls = 0
        self.confirm_calls = 0
        self.confirmed: set[tuple] = set()

    def outcome(self, c: Candidate) -> str:
        if self.numerical_mod and c.start_substep % self.numerical_mod == 0:
            return "numerical"
        weight = 1.0 + 0.5 * math.sin(math.radians(c.direction_deg))
        phase_ok = (c.start_substep % 110) < 60
        return "fell" if phase_ok and c.impulse_ns * weight > self.threshold else "survived"

    def evaluate(self, c: Candidate) -> EvalResult:
        self.calls += 1
        outcome = self.outcome(c)
        score = None if outcome != "survived" else 0.5
        return EvalResult(
            candidate=c, outcome=outcome, reason=None, fall_time_s=None, min_up_z=None,
            min_trunk_z=None, max_tilt_deg=None, score=score,
            objective=objective(outcome, c.impulse_ns, score), digest=f"d{c.key()}",
        )  # fmt: skip

    def confirm(self, c: Candidate, digest: str) -> Confirmation:
        self.confirm_calls += 1
        ok = self.outcome(c) == "fell" and self.confirm_rng.random() < self.confirm_ok
        if ok:
            self.confirmed.add(c.key())
        return Confirmation(c, ok, self.outcome(c), ok, True, 2, {"fake": True})


configs = st.builds(
    SearchConfig,
    algorithm=st.sampled_from(["random", "cmaes"]),
    budget=st.sampled_from([8, 16, 32, 64]),
    seed=st.integers(0, 2**31),
    shrink_budget=st.integers(0, 24),
)


# Feature: m4-failure-discovery, Property 22: determinism
@settings(max_examples=25)
@given(configs, st.floats(min_value=1.0, max_value=25.0))
def test_property_determinism(config, threshold) -> None:
    a = discover(config, FakePhysics(threshold).evaluate, FakePhysics(threshold).confirm)
    b = discover(config, FakePhysics(threshold).evaluate, FakePhysics(threshold).confirm)
    assert a["results_sha256"] == b["results_sha256"]
    assert [e["candidate"] for e in a["evaluations"]] == [e["candidate"] for e in b["evaluations"]]


# Feature: m4-failure-discovery, Property 23: budget accounting
@settings(max_examples=40)
@given(configs, st.floats(min_value=1.0, max_value=40.0), st.sampled_from([0, 7, 13]))
def test_property_budget_accounting(config, threshold, numerical_mod) -> None:
    fake = FakePhysics(threshold, numerical_mod)
    record = discover(config, fake.evaluate, fake.confirm)
    acct = record["accounting"]
    assert acct["search_rollouts"] == len(record["evaluations"]) <= config.budget
    assert [e["index"] for e in record["evaluations"]] == list(
        range(1, acct["search_rollouts"] + 1)
    )
    assert acct["cache_hits"] == len(record["cache_hits"])
    shrink_rollouts = acct["shrink_rollouts"]
    assert fake.calls == acct["search_rollouts"] + shrink_rollouts  # every simulation counted
    assert (
        acct["confirmation_rollouts"] + acct["shrink_confirmation_rollouts"]
        == 2 * fake.confirm_calls
    )
    if record["shrink"] is not None:
        assert shrink_rollouts <= config.shrink_budget
    keys = [tuple(e["candidate"][k] for k in ("force_n", "direction_deg", "start_substep",
                                                "duration_substeps")) for e in record["evaluations"]]  # fmt: skip
    assert len(keys) == len(set(keys))  # no quantized candidate simulated twice
    if record["search"]["stop_reason"] == "budget":
        assert acct["search_rollouts"] == config.budget


outcomes = st.sampled_from(["fell", "survived", "numerical"])
impulses = st.floats(min_value=0.4, max_value=MAX_IMPULSE_NS)
scores = st.floats(min_value=0.0, max_value=1.0)


# Feature: m4-failure-discovery, Property 24: fitness ordering
@given(outcomes, impulses, scores, outcomes, impulses, scores)
def test_property_fitness_ordering(o1, i1, s1, o2, i2, s2) -> None:
    f1, f2 = objective(o1, i1, s1), objective(o2, i2, s2)
    rank = {"fell": 0, "survived": 1, "numerical": 2}
    if rank[o1] < rank[o2]:
        assert f1 < f2
    if o1 == o2 == "fell" and i1 < i2:
        assert f1 < f2
    if o1 == o2 == "survived" and s1 < s2:
        assert f1 < f2
    assert objective("numerical", i1, None) == NUMERICAL_OBJECTIVE


# Feature: m4-failure-discovery, Property 25: numerical failures are never attacks
@settings(max_examples=30)
@given(configs, st.floats(min_value=1.0, max_value=25.0), st.sampled_from([1, 2, 3, 5]))
def test_property_numerical_excluded(config, threshold, numerical_mod) -> None:
    fake = FakePhysics(threshold, numerical_mod)
    record = discover(config, fake.evaluate, fake.confirm)
    summary = record["search"]
    numerical = [e for e in record["evaluations"] if e["outcome"] == "numerical"]
    assert summary["numerical"] == len(numerical)
    assert summary["candidate_failures"] == sum(
        e["outcome"] == "fell" for e in record["evaluations"]
    )
    best = summary["best_candidate_index"]
    if best is not None:
        assert record["evaluations"][best - 1]["outcome"] == "fell"
    if numerical_mod == 1:  # every candidate numerical
        assert not summary["found_failure"] and record["final"] is None
    for attempt in record["confirmation"]["attempts"]:
        assert record["evaluations"][attempt["evaluation"] - 1]["outcome"] == "fell"


# Feature: m4-failure-discovery, Property 26: shrinking never adopts an unconfirmed candidate
@settings(max_examples=60)
@given(
    st.floats(min_value=0.5, max_value=20.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.integers(0, 2**31),
    st.integers(0, 40),
    st.tuples(st.floats(min_value=60, max_value=150), st.floats(min_value=0, max_value=359.9),
              st.integers(500, 559), st.integers(30, 50)),
)  # fmt: skip
def test_property_shrink_only_confirmed(threshold, confirm_ok, seed, budget, start_tuple) -> None:
    fake = FakePhysics(threshold, confirm_ok=confirm_ok, seed=seed)
    start = Candidate.make(*start_tuple)
    record = shrink(start, fake.evaluate, fake.confirm, budget)
    final = Candidate(**{k: record["final"][k] for k in ("force_n", "direction_deg",
                                                          "start_substep", "duration_substeps")})  # fmt: skip
    assert final.impulse_ns <= start.impulse_ns + 1e-12
    assert record["rollouts"] <= budget
    if record["improved"]:
        assert final.key() in fake.confirmed
        assert record["final_confirmed"]["candidate"] == record["final"]
    else:
        assert final == start and record["final_confirmed"] is None
    impulses_seen = [h["candidate"]["impulse_ns"] for h in record["history"]]
    assert all(b < a for a, b in itertools.pairwise(impulses_seen))
    for h in record["history"][1:]:
        key = tuple(h["candidate"][k] for k in ("force_n", "direction_deg", "start_substep",
                                                 "duration_substeps"))  # fmt: skip
        assert key in fake.confirmed
    for t in record["tested"]:
        if t["status"] == "unconfirmed":
            assert t["candidate"] != record["final"] or not record["improved"]


# Feature: m4-failure-discovery, Property 27: the exported scenario is the discovered one
@given(st.tuples(finite, finite, finite, finite))
def test_property_scenario_export_matches(x) -> None:
    c = decode_cma(x)
    config = load_suite_config(scenario_config(c, name="p27", policy="policies/go1-walk-s0.json"))
    push = config.push_at(c.start_substep)
    original = c.push()
    assert push.window() == original.window()
    assert push.force_n == original.force_n and push.direction == original.direction
    assert push.start_s == original.start_s and push.duration_s == original.duration_s
    assert config.trials == 1 and config.low_step == c.start_substep


def test_quantize_direction_edges() -> None:
    assert quantize_direction(359.96) == 0.0
    assert quantize_direction(-0.04) == 0.0
    assert quantize_direction(-90) == 270.0
    assert quantize_direction(720.06) == 0.1
