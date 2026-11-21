"""Hypothesis property-based tests for M2 sampling and aggregation (Requirements 2, 4, 8.1)."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from robustlab.experiments.push import PushScenario, steps_of
from robustlab.experiments.suite import (
    OUTCOMES,
    SIM_DT,
    aggregate,
    sample_start_steps,
    wilson_interval,
)

seeds = st.integers(min_value=0, max_value=2**64 - 1)
methods = st.sampled_from(["stratified", "grid"])


@st.composite
def intervals(draw):
    low = draw(st.integers(min_value=0, max_value=5000))
    width = draw(st.integers(min_value=1, max_value=600))
    n = draw(st.integers(min_value=1, max_value=width))
    return low, low + width, n


# Feature: experiment-runner, Property 18: start times are quantized to the 4 ms substep grid
@given(intervals(), methods, seeds)
def test_property_quantized_distinct_in_stratum(interval, method, seed) -> None:
    low, high, n = interval
    starts = sample_start_steps(method, seed, low, high, n)
    assert len(starts) == n
    ks = [k for k, _ in starts]
    assert all(isinstance(k, int) for k in ks)
    assert ks == sorted(set(ks))  # distinct and increasing, so no two trials repeat a timing
    covered = []
    for k, (lo, hi) in starts:
        assert low <= lo <= k < hi <= high
        covered.extend(range(lo, hi))
    assert covered == list(range(low, high))  # strata tile the interval exactly
    for k in ks[:5]:
        start_s = k * SIM_DT
        assert steps_of(start_s, SIM_DT, "start_s") == k  # the float time maps back to k
        assert PushScenario(10.0, (0, -1, 0), start_s, 0.1).window()[0] == k


# Feature: experiment-runner, Property 19: seed determinism
@given(intervals(), seeds, seeds)
def test_property_seed_determinism(interval, seed, other) -> None:
    low, high, n = interval
    first = sample_start_steps("stratified", seed, low, high, n)
    assert first == sample_start_steps("stratified", seed, low, high, n)
    grid = sample_start_steps("grid", seed, low, high, n)
    assert grid == sample_start_steps("grid", other, low, high, n)  # grid ignores the seed
    if n == high - low:  # one substep per stratum: both methods enumerate every substep
        assert [k for k, _ in first] == [k for k, _ in grid] == list(range(low, high))


def _episode(outcome: str, t: float) -> dict:
    return {
        "outcome": outcome,
        "fall_time_s": t if outcome == "fell" else None,
        "time_after_start_s": t - 1.0 if outcome == "fell" else None,
        "forward_progress_m": t,
        "digest": f"{outcome}-{t}",
    }


trial_lists = st.lists(
    st.tuples(
        st.sampled_from(OUTCOMES),
        st.sampled_from(OUTCOMES),
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False),
    ),
    max_size=80,
)


# Feature: experiment-runner, Property 17: aggregation invariants
@given(trial_lists, methods)
def test_property_aggregation_invariants(rows, method) -> None:
    trials = [
        {
            "nominal": _episode(a, t),
            "attacked": _episode(b, t),
            "paired": {"pairing_ok": True, "pre_attack_max_abs_diff": 0.0,
                       "progress_difference_m": 0.0},
        }
        for a, b, t in rows
    ]  # fmt: skip
    agg = aggregate(trials, method)
    for case in ("nominal", "attacked"):
        a = agg[case]
        assert a["physical_falls"] + a["survived"] + a["numerical_failures"] == a["total_trials"]
        assert a["total_trials"] == len(rows)
        assert a["valid_trials"] == a["total_trials"] - a["numerical_failures"]
        if a["valid_trials"] == 0:
            assert a["fall_rate"] is None
        else:
            assert 0.0 <= a["fall_rate"] <= 1.0
            assert math.isclose(a["fall_rate"] * a["valid_trials"], a["physical_falls"])
        assert a["fall_time_s"]["n"] == a["physical_falls"]
    table = agg["paired"]["table_nominal_to_attacked"]
    assert sum(table.values()) == len(rows) and len(table) == 9
    attacked = agg["attacked"]
    if method == "stratified" and attacked["valid_trials"]:
        lo, hi = attacked["wilson_95"]
        assert 0.0 <= lo <= attacked["fall_rate"] <= hi <= 1.0
    else:
        assert attacked["wilson_95"] is None


@given(st.integers(min_value=0, max_value=500), st.integers(min_value=0, max_value=500))
def test_property_wilson_bounds(k, extra) -> None:
    n = k + extra
    interval = wilson_interval(k, n)
    if n == 0:
        assert interval is None
        return
    lo, hi = interval
    assert 0.0 <= lo <= k / n <= hi <= 1.0
