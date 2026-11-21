"""M5a measurements: does each physical axis change the observed behaviour?

1. Single deterministic runs (command 1,0,0, 5 s): nominal; friction sweep; payload sweep;
   a combined low-friction + payload + shove scenario.
2. Contact mechanics with the robot standing and a 150 N, 0.3 s sideways shove: observed contact
   friction, tangential/normal force ratio and foot slip at mu 0.1 / 0.3 / 0.6 / 1.0.
3. Exhaustive 256-start sweeps of the M2 shove (100 N toward the robot's right, 0.1 s, starts on
   the 4 ms grid in [2.000, 3.024) s) under nominal physics, low friction, payload, and both.

Usage: uv run python scripts/scenario_measurements.py [--out runs/m5/measurements] [--skip-grids]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from robustlab.experiments.push import PushScenario
from robustlab.experiments.scenario_run import foot_slip, run_scenario_suite, run_single
from robustlab.experiments.stack import load_stack
from robustlab.experiments.walk import EpisodeConfig, EpisodeRunner
from robustlab.physics.scenario_model import MODEL_CACHE
from robustlab.scenarios.schema import nominal_dict, validate

ROOT = Path(__file__).resolve().parent.parent
SHOVE = {"force_n": 100.0, "direction_deg": 270.0, "start_s": 2.0, "duration_s": 0.1}


def scenario(name, mu=None, mass=0.0, offset=(0.0, 0.0, 0.0), push=None):
    d = nominal_dict(name)
    d["environment"]["ground_friction_mu"] = mu
    d["payload"] = {"mass_kg": mass, "offset_xyz_m": list(offset)}
    d["push"] = push
    return validate(d).scenario


def single_row(result: dict) -> dict:
    case = result["cases"]["scenario"]
    original = result["cases"]["original"]
    return {
        "name": result["scenario"]["name"],
        "mu_configured": result["model"]["friction"]["configured_mu"],
        "mu_observed_at_reset": result["model"]["friction"]["observed_at_reset"][
            "sliding_mu_values"
        ],
        "total_mass_kg": result["model"]["mass"]["total_mass_kg"],
        "outcome": case["outcome"],
        "reason": case["reason"],
        "fall_time_s": case["fall_time_s"],
        "progress_m": case["forward_progress_m"],
        "mean_forward_speed_mps": case["mean_forward_speed_mps"],
        "max_tilt_deg": case["max_tilt_deg"],
        "min_trunk_z_m": case["min_trunk_z"],
        "slip_mps": case["slip"]["mean_slip_speed_mps"],
        "same_trajectory_as_original": case["digest"] == original["digest"],
        "matched": None
        if "matched" not in result["cases"]
        else {
            k: result["cases"]["matched"][k]
            for k in ("outcome", "forward_progress_m", "max_tilt_deg")
        },
        "results_sha256": result["results_sha256"],
    }


def contact_mechanics(stack, mu: float) -> dict:
    entry = MODEL_CACHE.get(stack, mu, None)
    runner = EpisodeRunner(entry.model, entry.ids, stack.policy, stack.actor)
    push = PushScenario(150.0, (0.0, -1.0, 0.0), 1.0, 0.3)
    runner.reset((0.0, 0.0, 0.0), push)
    floor = entry.ids.floor_geom
    frictions, ratios = set(), []
    force = np.zeros(6)
    for _ in range(100):
        runner.tick()
        d = runner.data
        for i in range(d.ncon):
            c = d.contact[i]
            if floor in (c.geom1, c.geom2):
                frictions.add(round(float(c.friction[0]), 12))
                mujoco.mj_contactForce(entry.model, d, i, force)
                if force[0] > 1e-6:
                    ratios.append(float(np.hypot(force[1], force[2]) / force[0]))
    record = runner.run(EpisodeConfig((0.0, 0.0, 0.0), 2.0, push=push))
    return {
        "mu": mu,
        "observed_contact_mu": sorted(frictions),
        "max_tangential_over_normal": max(ratios),
        "fall": record.fall.kind,
        "trunk_lateral_disp_m": float(record.qpos[-1, 1] - record.initial_qpos[1]),
        "foot_slip_m": foot_slip(entry.model, record)["total_slip_distance_m"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/m5/measurements")
    parser.add_argument("--skip-grids", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    stack = load_stack(ROOT, Path("policies/go1-walk-s0.json"), "fall-contact")
    summary: dict = {"git": None}

    singles = [scenario("nominal")]
    singles += [scenario(f"mu-{mu:g}", mu=mu) for mu in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 1.0, 1.5)]
    singles += [scenario(f"payload-{m:g}kg-centre", mass=m) for m in (1.0, 2.5, 5.0)]
    singles += [
        scenario("payload-5kg-forward", mass=5.0, offset=(0.10, 0.0, 0.0)),
        scenario("payload-5kg-back", mass=5.0, offset=(-0.10, 0.0, 0.0)),
        scenario("payload-5kg-left", mass=5.0, offset=(0.0, 0.08, 0.0)),
        scenario("payload-5kg-top", mass=5.0, offset=(0.0, 0.0, 0.08)),
        scenario("shove-only", push=SHOVE),
        scenario("combined-mu0.3-3kg-shove", mu=0.3, mass=3.0, offset=(0.0, 0.0, 0.04), push=SHOVE),
    ]
    rows = []
    for s in singles:
        result = run_single(s, stack, out / "single" / s.name)
        summary["git"] = result["provenance"]["git"]
        row = single_row(result)
        rows.append(row)
        print(f"{row['name']:<26} {row['outcome']:<9} progress {row['progress_m']:.3f} m  "
              f"speed {row['mean_forward_speed_mps']:.3f}  tilt {row['max_tilt_deg']:5.1f}  "
              f"slip {row['slip_mps']:.4f}  mu_obs {row['mu_observed_at_reset']}  "
              f"mass {row['total_mass_kg']:.2f}", file=sys.stderr, flush=True)  # fmt: skip
    summary["single"] = rows

    summary["contact_mechanics"] = [contact_mechanics(stack, mu) for mu in (0.1, 0.3, 0.6, 1.0)]
    for r in summary["contact_mechanics"]:
        print(r, file=sys.stderr, flush=True)

    grids = []
    if not args.skip_grids:
        for name, mu, mass in (("grid-nominal", None, 0.0), ("grid-mu0.3", 0.3, 0.0),
                               ("grid-3kg", None, 3.0), ("grid-mu0.3-3kg", 0.3, 3.0)):  # fmt: skip
            s = scenario(name, mu=mu, mass=mass, offset=(0.0, 0.0, 0.04) if mass else (0, 0, 0),
                         push=SHOVE)  # fmt: skip
            t = time.perf_counter()
            result = run_scenario_suite(s, stack, out / name, trials=256, interval_s=(2.0, 3.024),
                                        method="grid", overwrite=True)  # fmt: skip
            report = json.loads((out / name / "experiment.json").read_text())
            agg = report["aggregates"]
            fails = [t_["start_substep"] for t_ in report["trials"]
                     if t_["attacked"]["outcome"] == "fell"]  # fmt: skip
            grids.append({
                "name": name, "mu": mu, "payload_kg": mass,
                "matched_falls": agg["nominal"]["physical_falls"],
                "pushed_falls": agg["attacked"]["physical_falls"],
                "valid": agg["attacked"]["valid_trials"],
                "numerical": agg["attacked"]["numerical_failures"],
                "failing_substeps": fails,
                "results_sha256": report["results_sha256"],
                "wall_s": time.perf_counter() - t,
            })  # fmt: skip
            print(grids[-1], file=sys.stderr, flush=True)
    summary["grids"] = grids
    summary["model_cache"] = MODEL_CACHE.stats()
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
