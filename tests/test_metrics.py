"""Walking metric tests (Requirements 10.3, 10.4, 11.2)."""

from __future__ import annotations

import math

import numpy as np

from robustlab.experiments.walk import latency_summary, mean_forward_speed, touchdowns

SEED = 10
CTRL_DT = 0.02


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# Feature: m0-walking-go1, Property 10: Forward speed is heading-invariant
def test_property_forward_speed_heading_invariant() -> None:
    rng = np.random.default_rng(SEED)
    yaws = [0.0, math.pi / 2, math.pi, -math.pi, -math.pi / 2, math.pi - 1e-9]
    yaws += list(rng.uniform(-math.pi, math.pi, 120))
    for case, yaw in enumerate(yaws):
        v = float(rng.uniform(-1.5, 1.5)) if case else 1.0
        n = int(rng.integers(500, 700))
        t = np.arange(1, n + 1) * CTRL_DT
        qpos = np.zeros((n, 19))
        qpos[:, 0:2] = rng.normal(size=2) * 5 + np.outer(v * t, [math.cos(yaw), math.sin(yaw)])
        qpos[:, 2] = 0.28
        qpos[:, 3:7] = _yaw_quat(yaw)
        got = mean_forward_speed(t, qpos, 2.0, 10.0)
        assert got is not None and abs(got - v) <= 1e-6, f"seed={SEED} case={case} yaw={yaw}"


def test_forward_speed_needs_two_ticks_in_window() -> None:
    t = np.arange(1, 50) * CTRL_DT  # ends before 2 s
    qpos = np.zeros((t.size, 19))
    qpos[:, 3] = 1.0  # identity orientation
    assert mean_forward_speed(t, qpos, 2.0, 10.0) is None


def _reference_touchdowns(time_s, contact, start, end):
    counts = [0] * contact.shape[1]
    for k in range(1, len(time_s)):
        if start - 1e-9 <= time_s[k] <= end + 1e-9:
            for foot in range(contact.shape[1]):
                counts[foot] += int(not contact[k - 1, foot] and contact[k, foot])
    return counts


# Feature: m0-walking-go1, Property 11: Touchdown counting
def test_property_touchdown_counting() -> None:
    rng = np.random.default_rng(SEED)
    for case in range(200):
        n = int(rng.integers(2, 700))
        t = np.arange(1, n + 1) * CTRL_DT
        contact = rng.random((n, 4)) < rng.uniform(0.05, 0.95)
        if case % 4 == 0:
            # Boundary: transitions landing exactly on the 2 s and 10 s Ticks (indices 99, 499).
            for k in (98, 99, 498, 499):
                if k < n:
                    contact[k] = k in (99, 499)
        got = touchdowns(t, contact, 2.0, 10.0)
        assert got == _reference_touchdowns(t, contact, 2.0, 10.0), f"seed={SEED} case={case}"


def test_touchdown_at_window_edges_counted() -> None:
    t = np.arange(1, 501) * CTRL_DT
    contact = np.zeros((500, 1), bool)
    contact[99] = True  # t = 2.00 s, rise from t = 1.98 s
    contact[499] = True  # t = 10.00 s
    contact[50] = True  # t = 1.02 s, outside the window
    assert touchdowns(t, contact, 2.0, 10.0) == [2]


def test_latency_summary_known_array() -> None:
    warmup = [10**9] * 50  # 1 s per warm-up Tick, must be discarded
    measured = [i * 10**6 for i in range(1, 101)]  # 1..100 ms
    summary = latency_summary(np.array(warmup + measured), CTRL_DT)
    assert summary["warmup_ticks"] == 50 and summary["measured_ticks"] == 100
    assert summary["median_ms"] == 50.5
    assert math.isclose(summary["p95_ms"], 95.05)
    assert summary["max_ms"] == 100.0
    assert summary["target_ms"] == 25.0
    assert summary["median_within_target"] is False
    # 100 Ticks of simulated time (2 s) took 5.05 s of wall time.
    assert math.isclose(summary["real_time_factor"], 2.0 / 5.05)
