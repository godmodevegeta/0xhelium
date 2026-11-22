"""M7 gate: pre-registration checks, criteria, statistics and the registry. No locked-set rollouts
here: outcomes are synthetic, so these tests never look at real holdout results."""

from __future__ import annotations

import copy
import json
import shutil

import pytest

from conftest import REPO_ROOT
from robustlab.gate.config import GateConfigError, load_gate
from robustlab.gate.evaluate import BASELINE, assess, mcnemar_one_sided
from robustlab.gate.registry import RegistryError, apply_decision, ensure_baseline, read_active

GATE = REPO_ROOT / "examples/m7/gate-v1.json"


@pytest.fixture(scope="module")
def gate():
    return load_gate(GATE, REPO_ROOT)


def test_config_is_verified_and_complete(gate) -> None:
    assert {k: len(v) for k, v in gate.suite_ids().items()} == {
        "holdout": 10, "nominal": 13, "gauntlet": 1280, "known_failures": 34}  # fmt: skip
    assert gate.primary_id == "cand-ea7b9dbc0bbd" and sum(c.primary for c in gate.candidates) == 1
    ranking = gate.doc["primary"]["ranking"]
    assert (
        ranking[0]["candidate_id"] == gate.primary_id
        and ranking[-1]["validation_nominal_falls"] == 1
    )
    assert all(e.id.startswith("lt-") for e in gate.holdout)


@pytest.mark.parametrize("edit", ["candidate", "locked", "criteria"])
def test_tampering_is_refused(tmp_path, edit) -> None:
    root = tmp_path / "repo"
    for rel in ("policies/go1-walk-s0.json", "examples/m6/reflex-calibration-v1.json",
                "examples/m6/hardening-sets-v1.json", "examples/m7/gate-v1.json"):  # fmt: skip
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, root / rel)
    shutil.copytree(REPO_ROOT / "presets/m6-candidates-v1", root / "presets/m6-candidates-v1")
    load_gate(root / "examples/m7/gate-v1.json", root)  # the untouched copy verifies
    if edit == "candidate":
        path = root / "presets/m6-candidates-v1/seed0.json"
        record = json.loads(path.read_text())
        record["reflex"]["vector"][0] = 0.0
        path.write_text(json.dumps(record))
    elif edit == "locked":
        path = root / "examples/m6/hardening-sets-v1.json"
        doc = json.loads(path.read_text())
        doc["locked_test"]["episodes"][0]["push"]["force_n"] = 10.0
        path.write_text(json.dumps(doc))
    else:  # editing the criteria changes the config; a recorded hash elsewhere would catch it,
        path = root / "examples/m7/gate-v1.json"  # and here the primary no longer exists
        doc = json.loads(path.read_text())
        doc["primary"]["candidate_id"] = "cand-000000000000"
        path.write_text(json.dumps(doc))
    with pytest.raises(GateConfigError):
        load_gate(root / "examples/m7/gate-v1.json", root)


def test_mcnemar_exact_values() -> None:
    assert mcnemar_one_sided(0, 0) == 1.0
    assert mcnemar_one_sided(10, 0) == pytest.approx(1 / 1024)
    assert mcnemar_one_sided(5, 0) == pytest.approx(1 / 32)
    assert mcnemar_one_sided(4, 0) == pytest.approx(1 / 16)  # 4 recoveries alone are not enough
    assert mcnemar_one_sided(6, 2) == pytest.approx(37 / 256)


def row(id_, outcome="survived", disp=(4.5, 0.0)):
    return {"id": id_, "outcome": outcome, "reason": None, "fall_time_s": None, "time_after_push_s": None,
            "max_tilt_after_push_deg": 5.0, "progress_m": disp[0], "progress_until_s": 5.0,
            "digest_up_to_stop": "x", "displacement_m": list(disp)}  # fmt: skip


def outcomes(gate, *, cand_holdout_fell=0, nominal_fall=None, nominal_drift=None, gauntlet_regress=0,
             numerical=None):  # fmt: skip
    base = {"nominal": [row(e.id) for e in gate.nominal],
            "holdout": [row(e.id, "fell") for e in gate.holdout],
            "gauntlet": [row(e.id, "fell" if i % 10 == 0 else "survived") for i, e in enumerate(gate.gauntlet)],
            "known_failures": [row(e.id, "fell") for e in gate.known_failures]}  # fmt: skip
    cand = copy.deepcopy(base)
    for i, r in enumerate(cand["holdout"]):
        r["outcome"] = "fell" if i < cand_holdout_fell else "survived"
    if nominal_fall is not None:
        cand["nominal"][nominal_fall]["outcome"] = "fell"
    if nominal_drift is not None:
        cand["nominal"][nominal_drift]["displacement_m"] = [3.0, 0.0]
    survivors = [r for r in cand["gauntlet"] if r["outcome"] == "survived"]
    for r in survivors[:gauntlet_regress]:
        r["outcome"] = "fell"
    if numerical is not None:
        cand["known_failures"][numerical]["outcome"] = "numerical"
    out = {BASELINE: base}
    for c in gate.candidates:
        out[c.candidate_id] = copy.deepcopy(base)
    out[gate.primary_id] = cand
    return out


def test_passing_candidate_is_promoted(gate) -> None:
    v = assess(gate, outcomes(gate, cand_holdout_fell=1, gauntlet_regress=3))
    p = v["candidates"][gate.primary_id]
    assert p["passes_all"] and v["decision"]["verdict"] == "promote"
    assert (
        len(p["criteria"]["G3_regression"]["regressed"]) == 3
    )  # listed even when within tolerance
    # unchanged candidates fail G2 (no improvement) and are never eligible
    other = next(c for c in gate.candidates if not c.primary).candidate_id
    assert (
        not v["candidates"][other]["passes_all"]
        and not v["candidates"][other]["eligible_for_promotion"]
    )


@pytest.mark.parametrize(
    ("kwargs", "failed"),
    [
        ({"cand_holdout_fell": 0, "nominal_fall": 2}, "G1_nominal"),
        ({"cand_holdout_fell": 0, "nominal_drift": 0}, "G1_nominal"),
        (
            {"cand_holdout_fell": 6},
            "G2_holdout",
        ),  # 60% after 100%: fewer and under the 0.75 cap, but p = 1/16 > 0.05
        ({"cand_holdout_fell": 10}, "G2_holdout"),
        ({"cand_holdout_fell": 0, "gauntlet_regress": 60}, "G3_regression"),
        ({"cand_holdout_fell": 0, "numerical": 0}, "G4_numerical"),
    ],
)
def test_each_criterion_can_reject(gate, kwargs, failed) -> None:
    v = assess(gate, outcomes(gate, **kwargs))
    assert failed in v["decision"]["failed_criteria"] and v["decision"]["verdict"] == "reject"


def test_gauntlet_per_condition_rule(gate) -> None:
    # 9 regressions all in the first condition (256 episodes): +3.5% there, +0.7% overall
    out = outcomes(gate, cand_holdout_fell=0)
    first = [r for r in out[gate.primary_id]["gauntlet"][:256] if r["outcome"] == "survived"]
    for r in first[:9]:
        r["outcome"] = "fell"
    g3 = assess(gate, out)["candidates"][gate.primary_id]["criteria"]["G3_regression"]
    assert g3["checks"]["overall"] and not g3["checks"]["condition m2-100N"] and not g3["pass"]


def test_registry_promote_and_reject(tmp_path) -> None:
    cand = {"candidate_id": "cand-x", "record": "r.json", "record_sha256": "a" * 64}
    ensure_baseline(tmp_path, "p.json", "b" * 64)
    assert read_active(tmp_path)["active"] == "v0-baseline"
    reject = {"verdict": "reject", "failed_criteria": ["G2_holdout"], "statement": "no"}
    apply_decision(tmp_path, "gate-v1", "c" * 64, reject, cand, "p.json", "b" * 64)
    assert read_active(tmp_path)["active"] == "v0-baseline"
    assert (tmp_path / "policies/registry/rejected/cand-x-gate-v1.json").exists()
    with pytest.raises(RegistryError):  # one decision per candidate and gate
        apply_decision(tmp_path, "gate-v1", "c" * 64, reject, cand, "p.json", "b" * 64)
    promote = {"verdict": "promote", "failed_criteria": [], "statement": "ok"}
    entry = apply_decision(tmp_path, "gate-v2", "d" * 64, promote, cand, "p.json", "b" * 64)
    active = read_active(tmp_path)
    assert (
        active["active"] == entry["version"] == "v1-cand-x"
        and active["history"][-1]["version"] == "v1-cand-x"
    )
    version = json.loads((tmp_path / "policies/registry/versions/v1-cand-x.json").read_text())
    assert (
        version["parent"] == "v0-baseline" and version["policy"]["weights"] == "frozen, unmodified"
    )


def test_cli_refuses_a_second_evaluation(tmp_path, capsys) -> None:
    """gate-v1 already has a committed result: the CLI stops before any rollout."""
    from robustlab.cli import EXIT_WRITE, main

    assert (REPO_ROOT / "presets/m7-gate-v1/gate_result.json").exists()
    code = main(["gate", "--root", str(REPO_ROOT), "--out", str(tmp_path / "gate")])
    assert code == EXIT_WRITE
    assert "already evaluated" in capsys.readouterr().err
    assert not (tmp_path / "gate").exists()
