"""Write the pre-registered M7 gate configuration `examples/m7/gate-v1.json`.

Run once, before any gate evaluation; the file is committed and its SHA-256 is checked by
`robustlab gate`. The primary candidate is chosen here from M6 evidence only (training and
validation-preview measurements inside each candidate record); no locked-set result exists yet.

Usage: uv run python scripts/make_gate_config.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from robustlab.adapt.reflex import Calibration
from robustlab.adapt.sets import LOCKED_SPLIT, locked_digest

ROOT = Path(__file__).resolve().parents[1]
SETS = "examples/m6/hardening-sets-v1.json"
CALIBRATION = "examples/m6/reflex-calibration-v1.json"
POLICY = "policies/go1-walk-s0.json"
CANDIDATES = "presets/m6-candidates-v1"
RIGHT = [0.0, -1.0, 0.0]


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def m4_finals() -> list[dict]:
    """The confirmed M4 discovery finals (34), with the exact direction vectors M4 used."""
    out = []
    for path in sorted((ROOT / "runs/m4/comparison").glob("*/discovery.json")):
        record = json.loads(path.read_text())
        final = record.get("final")
        if not final:
            continue
        c = final["candidate"]
        a = math.radians(c["direction_deg"])
        out.append({
            "id": f"kf-m4-{path.parent.name}", "kind": "attack", "confirmed_failure": True,
            "source": f"M4 discovery {path.parent.name} final ({final['source']}); results "
            f"{record['results_sha256'][:12]}",
            "push": {"force_n": c["force_n"], "direction": [math.cos(a), math.sin(a), 0.0],
                     "start_substep": c["start_substep"], "duration_substeps": c["duration_substeps"]},
        })  # fmt: skip
    return out


def nominal(
    id_: str, source: str, command=(1.0, 0.0, 0.0), horizon_s: float = 5.0, **physics
) -> dict:
    return {"id": id_, "kind": "nominal", "source": source, "command": list(command),
            "horizon_s": horizon_s, **physics}  # fmt: skip


def primary(records: dict[str, dict]) -> tuple[str, list[dict]]:
    rows = []
    for cid, r in records.items():
        val = r["measurements"]["validation_preview"]["candidate_summary"]
        rows.append({"candidate_id": cid, "validation_nominal_falls": val["nominal_fell"],
                     "validation_attacks_survived": val["attack_survived"],
                     "validation_attacks": val["attacks"],
                     "training_objective": r["objective"]["candidate_training"]})  # fmt: skip
    rows.sort(key=lambda x: (x["validation_nominal_falls"] > 0, -x["validation_attacks_survived"],
                             x["training_objective"]))  # fmt: skip
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows[0]["candidate_id"], rows


def main() -> None:
    sets = json.loads((ROOT / SETS).read_text())
    locked = sets[LOCKED_SPLIT]
    assert locked_digest(locked["episodes"]) == locked["sha256"]
    calibration = Calibration.from_dict(json.loads((ROOT / CALIBRATION).read_text()))
    meta = json.loads((ROOT / CANDIDATES / "preset.json").read_text())
    records, registry = {}, []
    for name, digest in meta["files"].items():
        path = f"{CANDIDATES}/{name}"
        assert sha(path) == digest
        record = json.loads((ROOT / path).read_text())
        assert record["calibration"]["sha256"] == calibration.sha256()
        assert record["sets"]["sha256"] == sha(SETS)
        records[record["candidate_id"]] = record
        registry.append({"candidate_id": record["candidate_id"], "record": path,
                         "record_sha256": digest, "seed": record["optimizer"]["config"]["seed"],
                         "reproducibility_sha256": record["reproducibility_sha256"]})  # fmt: skip
    chosen, ranking = primary(records)
    locked_nominal = [e["id"] for e in locked["episodes"] if e["kind"] == "nominal"]
    locked_attacks = [e["id"] for e in locked["episodes"] if e["kind"] == "attack"]
    doc = {
        "schema_version": 1,
        "kind": "robustlab-gate",
        "name": "gate-v1",
        "created_before_evaluation": True,
        "base_policy": {"path": POLICY, "sha256": sha(POLICY)},
        "calibration": {
            "path": CALIBRATION,
            "file_sha256": sha(CALIBRATION),
            "calibration_sha256": calibration.sha256(),
        },
        "locked_set": {
            "path": SETS,
            "file_sha256": sha(SETS),
            "locked_sha256": locked["sha256"],
            "attack_ids": locked_attacks,
            "nominal_ids": locked_nominal,
            "status": "first evaluated by this gate",
        },
        "candidates": registry,
        "primary": {
            "candidate_id": chosen,
            "rule": "rank M6 candidates by M6 evidence only: no validation-preview nominal fall "
            "first, then more validation-preview attacks survived, then lower training objective; "
            "only the top-ranked candidate is eligible for promotion, the others are evaluated "
            "and reported",
            "ranking": ranking,
        },
        "suites": {
            "holdout_attacks": {"source": "locked test set attacks", "ids": locked_attacks},
            "nominal": {
                "locked_ids": locked_nominal,
                "episodes": [
                    nominal("g-forward-10s", "M0 forward walk, 10 s", horizon_s=10.0),
                    nominal(
                        "g-zero-10s",
                        "M0 zero command, 10 s",
                        command=(0.0, 0.0, 0.0),
                        horizon_s=10.0,
                    ),
                    nominal("g-sidestep-left", "sidestep, not used in M6", command=(0.0, 0.5, 0.0)),
                    nominal(
                        "g-backward", "walk backward, not used in M6", command=(-0.5, 0.0, 0.0)
                    ),
                    nominal("g-mu0.1", "M5 friction 0.1, no push", ground_friction_mu=0.1),
                    nominal("g-mu0.2", "M5 friction 0.2, no push", ground_friction_mu=0.2),
                    nominal("g-mu1.5", "M5 friction 1.5, no push", ground_friction_mu=1.5),
                    nominal(
                        "g-5kg-forward",
                        "M5 5 kg at x +0.10",
                        payload_kg=5.0,
                        payload_offset_xyz_m=[0.10, 0.0, 0.0],
                    ),
                    nominal(
                        "g-5kg-left",
                        "M5 5 kg at y +0.08",
                        payload_kg=5.0,
                        payload_offset_xyz_m=[0.0, 0.08, 0.0],
                    ),
                    nominal(
                        "g-mu0.3-3kg",
                        "M5 combined physics without the push (also M6 validation preview)",
                        ground_friction_mu=0.3,
                        payload_kg=3.0,
                        payload_offset_xyz_m=[0.0, 0.0, 0.04],
                    ),
                ],
            },
            "gauntlet": {
                "starts_substeps": [500, 756],
                "duration_substeps": 25,
                "direction": RIGHT,
                "note": "every 4 ms start in [2.000, 3.024) s; includes M6 training and "
                "validation timings; regression detection only",
                "conditions": [
                    {"id": "m2-100N", "force_n": 100.0},
                    {"id": "m3-150N", "force_n": 150.0},
                    {"id": "m5-mu0.3-100N", "force_n": 100.0, "ground_friction_mu": 0.3},
                    {
                        "id": "m5-3kg-100N",
                        "force_n": 100.0,
                        "payload_kg": 3.0,
                        "payload_offset_xyz_m": [0.0, 0.0, 0.04],
                    },
                    {
                        "id": "m5-mu0.3-3kg-100N",
                        "force_n": 100.0,
                        "ground_friction_mu": 0.3,
                        "payload_kg": 3.0,
                        "payload_offset_xyz_m": [0.0, 0.0, 0.04],
                    },
                ],
            },
            "known_failures": {
                "note": "confirmed M4 discovery finals; include M6 training "
                "attacks; reported, not a criterion",
                "episodes": m4_finals(),
            },
        },
        "criteria": {
            "G1_nominal": {
                "max_falls": 0,
                "max_numerical": 0,
                "displacement_tolerance_m": 0.10,
                "displacement_tolerance_rel": 0.10,
            },
            "G2_holdout": {
                "require_fewer_falls": True,
                "max_fall_rate_floor": 0.50,
                "required_rate_drop": 0.25,
                "mcnemar_one_sided_alpha": 0.05,
            },
            "G3_regression": {"max_fall_rate_increase_abs": 0.03, "per_condition": True},
            "G4_numerical": {"max_numerical": 0},
        },
        "statistics": {
            "paired_test": "exact one-sided McNemar on discordant pairs",
            "interval": "Wilson 95% for every fall rate",
        },
    }
    out = ROOT / "examples/m7/gate-v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(
        out, "primary", chosen, "known failures", len(doc["suites"]["known_failures"]["episodes"])
    )
    for row in ranking:
        print(row)


if __name__ == "__main__":
    main()
