"""Bounded rightward-shove force sweep with the M2 experiment runner (M3 preset selection).

For each force: the exhaustive 256-start grid (exact failure frequency over the M2 test
distribution) and the 16-trial stratified suite with the M2 example seed. Everything except the force
is the M2 configuration. Writes one experiment directory per run plus `summary.json`.

Usage: uv run python scripts/force_sweep.py [--forces 110,120,130,140,150] [--out runs/m3/force-sweep]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robustlab.experiments.stack import load_stack
from robustlab.experiments.suite import load_suite_config, run_suite

ROOT = Path(__file__).resolve().parent.parent

BASES = {
    "grid-all-256": ROOT / "examples/m2/sideways-shove-grid-all-256.json",
    "stratified-16": ROOT / "examples/m2/sideways-shove-stratified-16.json",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forces", default="110,120,130,140,150")
    parser.add_argument("--out", default="runs/m3/force-sweep")
    args = parser.parse_args()
    forces = [float(f) for f in args.forces.split(",")]
    out = ROOT / args.out
    stack = load_stack(ROOT, Path("policies/go1-walk-s0.json"), "fall-contact")
    summary = []
    for force in forces:
        row = {"force_n": force}
        for kind, base in BASES.items():
            raw = json.loads(base.read_text())
            raw["attack"]["force_n"] = force
            raw["name"] = f"{raw['name']}-{force:g}N"
            config = load_suite_config(raw)
            report = run_suite(
                config, stack, out / f"{force:g}N" / kind, save_trajectories=False, overwrite=True
            )
            att = report["aggregates"]["attacked"]
            row[kind] = {
                "physical_falls": att["physical_falls"],
                "valid_trials": att["valid_trials"],
                "numerical_failures": att["numerical_failures"],
                "fall_rate": att["fall_rate"],
                "wilson_95": att.get("wilson_95"),
                "results_sha256": report["results_sha256"],
                "git": report["provenance"]["git"],
            }
            if kind == "grid-all-256":
                fails = {t["start_substep"] for t in report["trials"]
                         if t["attacked"]["outcome"] == "fell"}  # fmt: skip
                row["per_stratum_fail_fraction"] = [
                    sum(k in fails for k in range(500 + 16 * i, 516 + 16 * i)) / 16
                    for i in range(16)
                ]
            print(
                f"{force:5g} N {kind:14s} falls {att['physical_falls']:3d}/{att['valid_trials']:<3d}"
                f" numerical {att['numerical_failures']}",
                flush=True,
            )
        summary.append(row)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
