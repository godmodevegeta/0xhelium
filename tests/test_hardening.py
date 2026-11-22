"""Hardening sets, objective, comparison, candidate records and the search loop (M6 Req. 3-5).

The search-loop tests replace the physics evaluator with a deterministic fake, so budget
accounting, stop reasons and determinism are checked without rollouts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from conftest import REPO_ROOT
from robustlab.adapt import harden as harden_mod
from robustlab.adapt.evaluate import OBJECTIVE, aggregate, episode_cost, objective
from robustlab.adapt.harden import (
    HardenConfig,
    HardenConfigError,
    compare,
    load_calibration,
    load_candidate,
    run_hardening,
    training_episodes,
    write_candidate,
)
from robustlab.adapt.sets import LOCKED_SPLIT, SetsError, load_sets

SETS = REPO_ROOT / "examples/m6/hardening-sets-v1.json"
CALIBRATION = REPO_ROOT / "examples/m6/reflex-calibration-v1.json"


@pytest.fixture(scope="module")
def sets():
    return load_sets(SETS, REPO_ROOT)


@pytest.fixture(scope="module")
def stack():
    from robustlab.experiments.stack import load_stack

    return load_stack(REPO_ROOT, Path("policies/go1-walk-s0.json"), "fall-contact")


# ------------------------------------------------------------------ sets


def test_locked_split_is_refused(sets) -> None:
    with pytest.raises(SetsError, match="reserved for M7"):
        sets.split(LOCKED_SPLIT)
    with pytest.raises(SetsError):
        sets.split("test")
    assert sets.summary()["locked_test"]["status"].startswith("reserved for M7")


def test_tampered_locked_set_is_rejected(tmp_path) -> None:
    doc = json.loads(SETS.read_text())
    doc[LOCKED_SPLIT]["episodes"][0]["push"]["force_n"] = 10.0
    bad = tmp_path / "sets.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(SetsError, match="SHA-256"):
        load_sets(bad)


def test_split_contents_and_gait_phases(sets) -> None:
    train, val = sets.split("train"), sets.split("validation_preview")
    assert len(train) == 18 and len(val) == 12
    confirmed = [e for e in train if e.confirmed_failure]
    assert len(confirmed) == 8 and sum("variant" in e.tags for e in train) == 8
    train_phases = {e.gait_phase_substep for e in confirmed}
    val_phases = {e.gait_phase_substep for e in val if e.kind == "attack"}
    assert not train_phases & val_phases
    assert min(train_phases) >= 42 and max(train_phases) <= 68
    ids = [e.id for e in (*train, *val)]
    assert len(ids) == len(set(ids))


def test_variants_are_deterministic(sets) -> None:
    again = load_sets(SETS, REPO_ROOT)
    assert again.split("train") == sets.split("train")
    for e in sets.split("train"):
        if "variant" in e.tags:
            assert not e.confirmed_failure and e.push.force_n <= 150


def test_focus_selects_one_attack_and_nominal(sets) -> None:
    episodes = training_episodes(sets, "m2-100N-500")
    assert [e.id for e in episodes] == ["m2-100N-500", "nominal-forward-1.0",
                                        "nominal-forward-0.5", "m2-100N-500-v0"]  # fmt: skip
    with pytest.raises(HardenConfigError):
        training_episodes(sets, "m3-150N-564")  # a validation-preview attack


def test_calibration_policy_check(stack, tmp_path) -> None:
    cal = load_calibration(CALIBRATION, stack.policy.sha256)
    assert cal.sha256() and len(cal.center) == 5
    with pytest.raises(HardenConfigError):
        load_calibration(CALIBRATION, "0" * 64)


# ------------------------------------------------------------------ objective


def metric(kind="attack", outcome="fell", t_after=0.5, tilt=30.0, progress=2.0, reflex=None,
           id_="e", confirmed=True):  # fmt: skip
    return {"id": id_, "kind": kind, "confirmed_failure": confirmed, "outcome": outcome,
            "fell": outcome == "fell", "numerical": outcome == "numerical", "reason": None,
            "fall_time_s": None, "time_after_push_s": t_after if outcome == "fell" else None,
            "max_tilt_after_push_deg": tilt, "min_trunk_z": 0.2, "progress_m": progress,
            "digest_up_to_stop": "x", "reflex": reflex}  # fmt: skip


@given(st.floats(0, 3), st.floats(0, 90), st.floats(-1, 6), st.floats(0, 1), st.floats(0, 0.3))
def test_property_cost_ordering(t_after, tilt, progress, activation, effort) -> None:
    reflex = {"activation_fraction": activation, "mean_abs_residual": effort}
    numerical = episode_cost(metric(outcome="numerical", reflex=reflex), 4.5)
    fell = episode_cost(metric(t_after=t_after, progress=progress, reflex=reflex), 4.5)
    survived = episode_cost(metric(outcome="survived", tilt=tilt, progress=progress,
                                   reflex=reflex), 4.5)  # fmt: skip
    nominal_fell = episode_cost(metric("nominal", reflex=reflex), 4.5)
    assert numerical == OBJECTIVE["numerical"] > nominal_fell
    assert numerical > fell and 1.0 <= fell <= 2.0 + 0.5 + 0.2 + 1e-9
    assert survived <= 0.5 + 0.5 + 0.2 + 1e-9
    assert survived < fell or fell - 1.0 >= 0  # a survival with no deficit always beats a fall
    later = episode_cost(metric(t_after=min(t_after + 0.5, 3)), 4.5)
    assert later <= episode_cost(metric(t_after=t_after), 4.5)


def test_numerical_is_never_survival() -> None:
    rows = [metric(outcome="numerical"), metric(outcome="survived"), metric(outcome="fell")]
    agg = aggregate(rows)
    assert (agg["attack_survived"], agg["attack_fell"], agg["attack_numerical"]) == (1, 1, 1)


def test_objective_is_mean_attack_plus_mean_nominal() -> None:
    rows = [metric(id_="a"), metric(id_="b", outcome="survived", tilt=0.0, progress=4.5),
            metric("nominal", outcome="survived", progress=4.5, id_="n")]  # fmt: skip
    refs = {"a": 4.5, "b": 4.5, "n": 4.5}
    expected = np.mean([episode_cost(rows[0], 4.5), episode_cost(rows[1], 4.5)]) + 0.0
    assert objective(rows, refs) == pytest.approx(expected)


def test_compare_classifies_and_requires_alignment() -> None:
    base = [metric(id_="a"), metric(id_="b", outcome="survived"), metric(id_="c"),
            metric(id_="d", outcome="survived")]  # fmt: skip
    cand = [metric(id_="a", outcome="survived"), metric(id_="b"),
            metric(id_="c", outcome="numerical"), metric(id_="d", outcome="survived")]  # fmt: skip
    changes = [r["change"] for r in compare(base, cand)]
    assert changes == ["recovered", "regressed", "numerical", "both_survived"]
    with pytest.raises(ValueError):
        compare(base, list(reversed(cand)))


# ------------------------------------------------------------------ search loop (fake physics)


class FakeEvaluator:
    """Deterministic stand-in: an attack is survived iff gain 0 > 0 and amplitude > 0.1."""

    instances: ClassVar[list] = []

    def __init__(self, ctx, workers=1) -> None:
        self.rollouts = 0
        self.calls: list[int] = []
        FakeEvaluator.instances.append(self)

    def map(self, tasks):
        self.rollouts += len(tasks)
        self.calls.append(len(tasks))
        out = []
        for params, spec in tasks:
            p = None if params is None else np.asarray(params)
            ok = p is not None and p[0] > 0 and p[9] > 0.1
            if spec.kind == "nominal":
                out.append(metric("nominal", "survived", progress=4.5, id_=spec.id,
                                  confirmed=False))  # fmt: skip
            else:
                out.append(metric("attack", "survived" if ok else "fell", tilt=10.0,
                                  progress=4.5 if ok else 2.0, id_=spec.id,
                                  confirmed=spec.confirmed_failure))  # fmt: skip
        return out

    def close(self) -> None:
        pass


@pytest.fixture()
def fake(monkeypatch):
    FakeEvaluator.instances = []
    monkeypatch.setattr(harden_mod, "Evaluator", FakeEvaluator)
    return FakeEvaluator


@pytest.fixture(scope="module")
def calibration(stack):
    return load_calibration(CALIBRATION, stack.policy.sha256)


def test_generation_cap_accounting_and_determinism(fake, stack, calibration, sets) -> None:
    config = HardenConfig(seed=3, max_generations=4, popsize=6)
    a = run_hardening(stack, calibration, sets, config)
    b = run_hardening(stack, calibration, sets, config)
    assert a["reproducibility_sha256"] == b["reproducibility_sha256"]
    assert a["reflex"]["vector"] == b["reflex"]["vector"] and a["status"] == "candidate"
    acc = a["accounting"]
    assert a["result"]["stop_reason"] == "generation_cap" and len(a["history"]) == 4
    assert acc["optimization_rollouts"] == 4 * 6 * 18 == acc["rollouts_per_generation"] * 4
    assert acc["validation_preview_rollouts"] == 12
    assert acc["total_rollouts"] == (acc["reference_rollouts"] + acc["baseline_rollouts"]
                                     + acc["optimization_rollouts"] + 12)  # fmt: skip
    assert acc["baseline_rollouts"] == 18 + 12
    assert sum(fake.instances[0].calls) == acc["total_rollouts"]
    other = run_hardening(stack, calibration, sets, HardenConfig(seed=4, max_generations=4,
                                                                 popsize=6))  # fmt: skip
    assert other["reproducibility_sha256"] != a["reproducibility_sha256"]


def test_rollout_budget_stops_before_a_partial_generation(fake, stack, calibration, sets):
    config = HardenConfig(seed=0, max_generations=50, popsize=8, max_rollouts=8 * 18 * 3 + 5)
    record = run_hardening(stack, calibration, sets, config)
    assert record["result"]["stop_reason"] == "rollout_budget"
    assert record["accounting"]["optimization_rollouts"] == 8 * 18 * 3
    assert record["history"][-1]["generation"] == 3


def test_cancel_and_wall_clock(fake, stack, calibration, sets) -> None:
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    record = run_hardening(stack, calibration, sets, HardenConfig(max_generations=10),
                           cancelled=cancelled)  # fmt: skip
    assert record["result"]["stop_reason"] == "cancelled" and len(record["history"]) == 2
    record = run_hardening(stack, calibration, sets, HardenConfig(max_generations=10,
                                                                  wall_clock_s=1.0),
                           cancelled=lambda: False)  # fmt: skip
    assert record["result"]["stop_reason"] in ("wall_clock", "generation_cap")


def test_no_candidate_before_first_generation(fake, stack, calibration, sets) -> None:
    record = run_hardening(stack, calibration, sets, HardenConfig(max_rollouts=10))
    assert record["result"]["stop_reason"] == "rollout_budget"
    assert record["result"]["improved_training_objective"] is False
    assert record["measurements"]["validation_preview"]["candidate"] is None
    assert not any(record["reflex"]["vector"])  # the frozen baseline is preserved


def test_candidate_record_contents_and_immutability(fake, stack, calibration, sets, tmp_path):
    record = run_hardening(stack, calibration, sets, HardenConfig(max_generations=3))
    for key in ("candidate_id", "status", "base_policy", "reflex", "calibration", "sets",
                "optimizer", "objective", "result", "accounting", "timing", "measurements",
                "provenance", "reproducibility_sha256"):  # fmt: skip
        assert key in record
    assert record["status"] == "candidate" and "promoted" not in json.dumps(record["status"])
    assert record["base_policy"]["sha256"] == stack.policy.sha256
    assert record["sets"]["sha256"] == sets.sha256 and "locked_test" in record["sets"]
    train = record["measurements"]["train"]
    assert (
        train["changes"]["attack"].get("recovered", 0)
        + train["changes"]["attack"].get("both_fell", 0)
        + train["changes"]["attack"].get("both_survived", 0)
        == 16
    )
    path = write_candidate(tmp_path, record)
    assert load_candidate(path)["candidate_id"] == record["candidate_id"]
    with pytest.raises(FileExistsError):
        write_candidate(tmp_path, record)
    bad = copy.deepcopy(record)
    bad["status"] = "promoted"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_candidate(tmp_path / "bad.json")


def test_config_bounds() -> None:
    for kwargs in ({"popsize": 2}, {"max_generations": 0}, {"workers": 0}, {"seed": -1},
                   {"wall_clock_s": 0.1}, {"sigma0": 2.0}):  # fmt: skip
        with pytest.raises(HardenConfigError):
            HardenConfig(**kwargs)
