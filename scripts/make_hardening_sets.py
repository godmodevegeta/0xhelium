"""Write `examples/m6/hardening-sets-v1.json` from the recorded failures of M2-M5.

Sources (all on the `fall-contact` scene, command (1, 0, 0), 5 s, nominal reset):
- M2 hidden-phase grid (100 N toward the right, 0.1 s): failing start substeps from
  `presets/hidden-phase-v1/experiment.json`
- M3 150 N grid: failing substeps from `presets/live-stress-v1/reference-grid-256.json`
- M4 discovery finals (confirmed, possibly shrunk) from `runs/m4/comparison/<run>/discovery.json`
- M5 friction 0.3 grid failures (`docs/milestones/M5.md`)

Usage: uv run python scripts/make_hardening_sets.py   (needs the local M4 comparison runs)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from robustlab.adapt.sets import LOCKED_SPLIT, SCHEMA_VERSION, SETS_KIND, locked_digest

ROOT = Path(__file__).resolve().parents[1]
RIGHT = [0.0, -1.0, 0.0]  # M1/M2 `right`


def grid_failures(path: str) -> set[int]:
    report = json.loads((ROOT / path).read_text())
    return {t["start_substep"] for t in report["trials"] if t["attacked"]["outcome"] == "fell"}


M2_FAILS = grid_failures("presets/hidden-phase-v1/experiment.json")
M3_FAILS = grid_failures("presets/live-stress-v1/reference-grid-256.json")
M5_MU03_FAILS = {567, 574, 593, 594, 595, 596, 597, 601, 602, 701, 702, 754}


def m4(run: str) -> dict:
    record = json.loads((ROOT / "runs/m4/comparison" / run / "discovery.json").read_text())
    c = record["final"]["candidate"]
    a = math.radians(c["direction_deg"])
    return {
        "id": f"m4-{run}",
        "source": f"M4 discovery {run} final ({record['final']['source']}), confirmed fall; "
        f"results {record['results_sha256'][:12]}",
        "push": {"force_n": c["force_n"], "direction": [math.cos(a), math.sin(a), 0.0],
                 "start_substep": c["start_substep"],
                 "duration_substeps": c["duration_substeps"]},
    }  # fmt: skip


def grid(milestone: str, force: float, substep: int, fails: set[int], **extra) -> dict:
    assert substep in fails, (milestone, substep)
    return {
        "id": f"{milestone}-{force:g}N-{substep}" + ("-mu0.3" if extra else ""),
        "source": f"{milestone} grid: {force:g} N toward the right, 0.1 s, start substep "
        f"{substep}; recorded fall",
        "push": {"force_n": force, "direction": RIGHT, "start_substep": substep,
                 "duration_substeps": 25},
        **extra,
    }  # fmt: skip


def attack(entry: dict) -> dict:
    return {"kind": "attack", "confirmed_failure": True, **entry}


def nominal(id_: str, command=(1.0, 0.0, 0.0), source="nominal walking", **extra) -> dict:
    return {"id": id_, "kind": "nominal", "command": list(command), "source": source, **extra}


TRAIN = [
    attack(grid("m2", 100, 500, M2_FAILS)),
    attack(grid("m2", 100, 611, M2_FAILS)),
    attack(grid("m2", 100, 702, M2_FAILS)),
    attack(grid("m3", 150, 610, M3_FAILS)),
    attack(m4("cmaes-s1")),
    attack(m4("cmaes-s12")),
    attack(m4("cmaes-s3")),
    attack(m4("cmaes-s14")),
    nominal("nominal-forward-1.0"),
    nominal("nominal-forward-0.5", (0.5, 0.0, 0.0)),
]
VALIDATION = [
    attack(grid("m3", 150, 564, M3_FAILS)),
    attack(grid("m3", 150, 590, M3_FAILS)),
    attack(grid("m3", 150, 694, M3_FAILS)),
    attack(m4("cmaes-s0")),
    attack(m4("random-s11")),
    attack(m4("random-s13")),
    attack(m4("random-s0")),
    attack(m4("random-s15")),
    attack(grid("m5", 100, 567, M5_MU03_FAILS, ground_friction_mu=0.3)),
    nominal("val-nominal-turn-left", (1.0, 0.0, 0.5)),
    nominal("val-nominal-stand", (0.0, 0.0, 0.0)),
    nominal(
        "val-nominal-mu0.3-3kg",
        source="M5 combined physics without the push",
        ground_friction_mu=0.3,
        payload_kg=3.0,
        payload_offset_xyz_m=[0.0, 0.0, 0.04],
    ),
]
LOCKED = [
    attack(m4("cmaes-s7")),
    attack(m4("cmaes-s17")),
    attack(m4("random-s8")),
    attack(m4("random-s19")),
    attack(m4("random-s5")),
    attack(grid("m3", 150, 588, M3_FAILS)),
    attack(grid("m3", 150, 514, M3_FAILS)),
    attack(grid("m5", 100, 754, M5_MU03_FAILS, ground_friction_mu=0.3)),
    attack(grid("m5", 100, 593, M5_MU03_FAILS, ground_friction_mu=0.3)),
    attack(grid("m2", 100, 725, M2_FAILS)),
    nominal("lt-nominal-turn-right", (1.0, 0.0, -0.5)),
    nominal("lt-nominal-fast", (1.5, 0.0, 0.0)),
    nominal(
        "lt-nominal-5kg-top",
        source="M5 payload without a push",
        payload_kg=5.0,
        payload_offset_xyz_m=[0.0, 0.0, 0.08],
    ),
]
for e in LOCKED:
    e["id"] = "lt-" + e["id"].removeprefix("lt-")

DOC = {
    "schema_version": SCHEMA_VERSION,
    "kind": SETS_KIND,
    "name": "hardening-sets-v1",
    "gait_period_s": 0.44,
    "notes": [
        (
            "Confirmed failures were recorded as baseline falls by M2-M5 on the fall-contact "
            "scene; every M6 run re-measures the baseline."
        ),
        (
            "Training push starts lie at gait phases 42-68 substeps (of 110); validation-preview "
            "phases are 14-40 and 84. Timings at the same phase of the 0.44 s gait are strongly "
            "correlated, so different timestamps are not independent tests."
        ),
        (
            "Training variants are seeded perturbations (force +-10%, direction +-10 deg, start "
            "+-4 and duration +-3 substeps) and are not confirmed failures."
        ),
        "The locked test set is for M7 and is refused by all M6 evaluation code.",
    ],
    "train": {
        "episodes": TRAIN,
        "variants": {
            "seed": 6,
            "per_attack": 1,
            "jitter": {
                "force_rel": 0.10,
                "direction_deg": 10.0,
                "start_substeps": 4,
                "duration_substeps": 3,
            },
        },
    },
    "validation_preview": {"episodes": VALIDATION},
    LOCKED_SPLIT: {"episodes": LOCKED, "sha256": locked_digest(LOCKED)},
}

if __name__ == "__main__":
    out = ROOT / "examples/m6/hardening-sets-v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(DOC, indent=1) + "\n")
    print(out, "locked", DOC[LOCKED_SPLIT]["sha256"][:12])
