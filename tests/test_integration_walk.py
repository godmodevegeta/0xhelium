"""The real policy walks forward and stands still under the fixed M0 criteria (Requirement 10),
and Tick latency is measured on >= 2,500 Ticks (Requirement 11)."""

from __future__ import annotations

import math

import pytest

from conftest import MODEL_XML
from robustlab.experiments.walk import (
    EpisodeConfig,
    EpisodeRunner,
    WalkCriteria,
    evaluate,
    latency_summary,
)
from robustlab.physics.model import apply_model_parameters, load_model, resolve_ids
from robustlab.policy.actor import Actor

pytestmark = pytest.mark.integration
CRITERIA = WalkCriteria()


def _run(policy, command, seconds):
    model = load_model(MODEL_XML)
    apply_model_parameters(model, policy.model_parameters)
    runner = EpisodeRunner(model, resolve_ids(model), policy, Actor(policy))
    record = runner.run(EpisodeConfig(command=command, seconds=seconds))
    return record, evaluate(record, CRITERIA)


def test_forward_episode_walks(policy) -> None:
    record, out = _run(policy, (1.0, 0.0, 0.0), 10.0)
    print(f"\nforward episode outcome: {out}")
    assert record.n_ticks == 500
    assert out["criteria_set"] == "forward"
    assert out["upright_45"] and out["tilt_ok"] and out["height_ok"]
    assert out["min_trunk_z"] >= 0.18
    assert out["finite"] and out["warnings_ok"] and out["warnings"] == {}
    assert out["mean_forward_speed"] >= 0.7
    assert min(out["touchdowns"]) >= 5
    assert out["passed"], out["failed_checks"]


def test_zero_episode_stands(policy) -> None:
    _, out = _run(policy, (0.0, 0.0, 0.0), 10.0)
    print(f"\nzero-command episode outcome: {out}")
    assert out["criteria_set"] == "zero"
    assert out["upright_45"]
    assert out["horizontal_disp_m"] < 0.3
    assert out["passed"], out["failed_checks"]


def test_latency_benchmark(policy) -> None:
    record, out = _run(policy, (1.0, 0.0, 0.0), 51.0)
    summary = latency_summary(record.tick_ns, record.ctrl_dt, warmup=50)
    print(f"\nlatency (51 s episode): {summary}")
    assert summary["warmup_ticks"] == 50 and summary["measured_ticks"] >= 2500
    for key in ("median_ms", "p95_ms", "max_ms", "real_time_factor"):
        assert math.isfinite(summary[key]) and summary[key] > 0
    assert out["upright_45"]
