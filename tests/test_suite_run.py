"""Paired suite execution, persistence and determinism on a small fast config (Requirements 3,
4, 6, 7)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import POLICY_PATH, REPO_ROOT
from robustlab.experiments.stack import load_stack
from robustlab.experiments.suite import (
    SuiteConfigError,
    aggregate,
    format_summary,
    load_suite_config,
    read_report,
    run_suite,
)
from robustlab.experiments.trace import read_run
from test_suite_config import BASE


@pytest.fixture(scope="module")
def stack():
    return load_stack(REPO_ROOT, POLICY_PATH, "fall-contact")


@pytest.fixture(scope="module")
def first(stack, tmp_path_factory):
    out = tmp_path_factory.mktemp("suite") / "a"
    return out, run_suite(load_suite_config(BASE), stack, out)


def test_paired_construction_and_pre_attack_parity(first) -> None:
    out, report = first
    trials = report["trials"]
    assert [t["start_substep"] for t in trials] == report["sampling"]["start_substeps"]
    assert len({t["start_substep"] for t in trials}) == 4
    assert len({t["nominal"]["digest"] for t in trials}) == 1  # identical nominal episodes
    assert report["aggregates"]["nominal"]["distinct_trajectories"] == 1
    for t in trials:
        lo, hi = t["stratum_substeps"]
        assert lo <= t["start_substep"] < hi
        assert t["start_s"] == t["start_substep"] * 0.004
        assert t["push"]["active_substeps"] == [t["start_substep"], t["start_substep"] + 25]
        # Ticks at 0.02 s spacing up to and including the shove start are untouched by it.
        assert t["paired"]["pre_attack_ticks"] == int((t["start_s"] + 1e-9) // 0.02)
        assert t["paired"]["pre_attack_max_abs_diff"] == 0.0 and t["paired"]["pairing_ok"]
        assert t["nominal"]["outcome"] in ("survived", "fell", "numerical")
    assert report["aggregates"]["paired"]["pairing_ok_all"]
    assert "Nominal: all 4 nominal runs produced one identical trajectory" in " ".join(
        report["interpretation"]
    )
    assert read_report(out)["results_sha256"] == report["results_sha256"]
    text = format_summary(report)
    assert "results_sha256" in text and "falls" in text


def test_saved_trajectories_are_m1_runs_matching_the_report(first) -> None:
    out, report = first
    for t in report["trials"]:
        for case in ("nominal", "attacked"):
            trajectory, manifest = read_run(out / t["trajectories"][case])
            assert manifest["schema_version"] == 2 and manifest["experiment"]["trial"] == t["trial"]
            assert manifest["outcome"]["fall"]["kind"] == {
                "survived": "none", "fell": "physical", "numerical": "numerical"
            }[t[case]["outcome"]]  # fmt: skip
            if case == "attacked":
                assert manifest["push"]["start_s"] == t["start_s"]
            else:
                assert manifest["push"] is None
            qpos = np.asarray(trajectory["qpos"])
            assert qpos.shape == (t[case]["n_ticks"], 19)


def test_rerun_reproduces_parameters_and_outcomes(stack, first, tmp_path) -> None:
    _, report = first
    again = run_suite(load_suite_config(BASE), stack, tmp_path / "b", save_trajectories=False)
    assert again["results_sha256"] == report["results_sha256"]
    strip = [{k: v for k, v in t.items() if k != "trajectories"} for t in report["trials"]]
    assert [{k: v for k, v in t.items() if k != "trajectories"} for t in again["trials"]] == strip
    other = run_suite(
        load_suite_config(BASE, seed=8), stack, tmp_path / "c", save_trajectories=False
    )
    assert other["sampling"]["start_substeps"] != report["sampling"]["start_substeps"]
    assert other["results_sha256"] != report["results_sha256"]


def test_existing_report_is_not_overwritten(stack, first) -> None:
    out, _ = first
    with pytest.raises(FileExistsError):
        run_suite(load_suite_config(BASE), stack, out)


def test_command_outside_trained_range_rejected(stack, tmp_path) -> None:
    cfg = json.loads(json.dumps(BASE))
    cfg["command"] = [2.0, 0.0, 0.0]
    with pytest.raises(SuiteConfigError, match="trained range"):
        run_suite(load_suite_config(cfg), stack, tmp_path / "x")
    assert not (tmp_path / "x" / "experiment.json").exists()


def _ep(outcome, t=None, progress=1.0, digest="d"):
    return {
        "outcome": outcome,
        "fall_time_s": t,
        "time_after_start_s": None if t is None else t - 2.0,
        "forward_progress_m": progress,
        "digest": digest,
    }


def test_aggregation_known_example_keeps_numerical_separate() -> None:
    ok = {"pairing_ok": True, "pre_attack_max_abs_diff": 0.0}
    trials = [
        {"nominal": _ep("survived"), "attacked": _ep("fell", 2.5, 0.4, "a"),
         "paired": {**ok, "progress_difference_m": -0.6}},
        {"nominal": _ep("survived"), "attacked": _ep("fell", 2.7, 0.6, "b"),
         "paired": {**ok, "progress_difference_m": -0.4}},
        {"nominal": _ep("survived"), "attacked": _ep("survived", None, 1.0, "c"),
         "paired": {**ok, "progress_difference_m": 0.0}},
        {"nominal": _ep("survived"), "attacked": _ep("numerical", None, 9.0, "d"),
         "paired": {**ok, "progress_difference_m": 8.0}},
    ]  # fmt: skip
    agg = aggregate(trials, "stratified")
    att = agg["attacked"]
    assert (att["physical_falls"], att["survived"], att["numerical_failures"]) == (2, 1, 1)
    assert att["valid_trials"] == 3 and att["fall_rate"] == pytest.approx(2 / 3)
    assert att["fall_time_s"] == {"n": 2, "min": 2.5, "median": 2.6, "max": 2.7}
    assert att["time_after_start_s"]["median"] == pytest.approx(0.6)
    assert att["mean_forward_progress_m"] == pytest.approx((0.4 + 0.6 + 1.0) / 3)
    assert agg["paired"]["valid_pairs"] == 3
    assert agg["paired"]["mean_progress_difference_m"] == pytest.approx(-1 / 3)
    assert agg["paired"]["table_nominal_to_attacked"]["survived->numerical"] == 1
    assert agg["nominal"]["distinct_trajectories"] == 1


def test_m0_schema_1_runs_still_read(tmp_path) -> None:
    run = tmp_path / "m0"
    run.mkdir()
    (run / "trajectory.json").write_text(json.dumps({"schema_version": 1, "qpos": [[0.0] * 19]}))
    (run / "manifest.json").write_text(json.dumps({"schema_version": 1, "run_id": "m0"}))
    trajectory, manifest = read_run(run)
    assert trajectory["schema_version"] == 1 and manifest["run_id"] == "m0"
    (run / "manifest.json").write_text(json.dumps({"schema_version": 3}))
    with pytest.raises(ValueError):
        read_run(run)
