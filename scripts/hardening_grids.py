"""Exploratory dense check of a candidate: the M2 (100 N) and M3 (150 N) start-time grids.

Each grid is every 4 ms push start in [2.000, 3.024) s, toward the robot's right for 0.1 s,
command (1, 0, 0), 5 s. Baseline and candidate run the identical episodes. The grids overlap the
training and validation-preview timings (marked) and skip the locked M7 timings, which are never
evaluated in M6. This is not a holdout test.

Usage: uv run python scripts/hardening_grids.py runs/m6/seed0/candidate.json [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from robustlab.adapt.evaluate import Context, Evaluator
from robustlab.adapt.harden import default_workers, load_calibration, load_candidate
from robustlab.adapt.sets import LOCKED_SPLIT, EpisodeSpec, PushSpec, load_sets
from robustlab.experiments.stack import load_stack
from robustlab.experiments.trace import _atomic_write_json, git_revision

ROOT = Path(__file__).resolve().parents[1]
GRIDS = {"m2-100N": 100.0, "m3-150N": 150.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--out", default="runs/m6/grids")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    record = load_candidate(Path(args.candidate))
    stack = load_stack(ROOT, Path(record["base_policy"]["path"]), "fall-contact")
    calibration = load_calibration(ROOT / "examples/m6/reflex-calibration-v1.json",
                                   stack.policy.sha256)  # fmt: skip
    sets_doc = json.loads((ROOT / record["sets"]["path"]).read_text())
    locked = {(e["push"]["force_n"], e["push"]["start_substep"]) for e in
              sets_doc[LOCKED_SPLIT]["episodes"] if e.get("push") and e["push"]["direction"] ==
              [0.0, -1.0, 0.0] and e.get("ground_friction_mu") is None}  # fmt: skip
    sets = load_sets(ROOT / record["sets"]["path"], ROOT)
    used = {}
    for split in ("train", "validation_preview"):
        for e in sets.split(split):
            if e.push and e.push.direction == (0.0, -1.0, 0.0) and e.ground_friction_mu is None:
                used[(e.push.force_n, e.push.start_substep)] = split
    params = tuple(record["reflex"]["vector"])
    evaluator = Evaluator(Context(stack, calibration), args.workers or default_workers())
    started = time.perf_counter()
    out = {"candidate_id": record["candidate_id"], "git": git_revision(ROOT), "grids": {}}
    try:
        for name, force in GRIDS.items():
            specs = []
            for k in range(500, 756):
                if (force, k) in locked:
                    continue
                specs.append(EpisodeSpec(id=f"{name}-{k}", kind="attack", split="exploratory",
                                         source=f"{name} grid start {k}",
                                         push=PushSpec(force, (0.0, -1.0, 0.0), k, 25)))  # fmt: skip
            base = evaluator.map([(None, s) for s in specs])
            cand = evaluator.map([(params, s) for s in specs])
            rows = []
            for s, b, c in zip(specs, base, cand, strict=True):
                rows.append({"start_substep": s.push.start_substep, "baseline": b["outcome"],
                             "candidate": c["outcome"],
                             "in_split": used.get((force, s.push.start_substep)),
                             "candidate_activation": c["reflex"]["activation_fraction"]})  # fmt: skip
            count = {}
            for r in rows:
                key = f"{r['baseline']}->{r['candidate']}"
                count[key] = count.get(key, 0) + 1
            out["grids"][name] = {
                "force_n": force,
                "tested": len(rows),
                "skipped_locked": sorted(k for f, k in locked if f == force),
                "baseline_fell": sum(r["baseline"] == "fell" for r in rows),
                "candidate_fell": sum(r["candidate"] == "fell" for r in rows),
                "numerical": sum("numerical" in (r["baseline"], r["candidate"]) for r in rows),
                "transitions": count,
                "regressions": [r["start_substep"] for r in rows
                                if r["baseline"] == "survived" and r["candidate"] != "survived"],
                "candidate_failures": [r["start_substep"] for r in rows if r["candidate"] == "fell"],
                "rows": rows,
            }  # fmt: skip
            g = out["grids"][name]
            print(f"{name}: baseline fell {g['baseline_fell']}/{g['tested']}, candidate fell "
                  f"{g['candidate_fell']}/{g['tested']}, transitions {count}, regressions "
                  f"{g['regressions']}")  # fmt: skip
    finally:
        evaluator.close()
    out["wall_s"] = time.perf_counter() - started
    out["rollouts"] = evaluator.rollouts
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / f"{record['candidate_id']}.json", out)
    print(f"{evaluator.rollouts} rollouts in {out['wall_s']:.1f} s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
