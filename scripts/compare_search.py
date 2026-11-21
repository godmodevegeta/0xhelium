"""Matched comparison of Random Search and CMA-ES (M4).

Both algorithms get the same search seeds, the same rollout budget and shrink budget, the same
fixed experiment (command, horizon, policy, scene, nominal reset) and no warm start. Writes one
discovery record per (algorithm, seed) and `summary.json` with per-seed rows, aggregates and
paired tests.

Usage: uv run python scripts/compare_search.py [--seeds 0-19] [--budget 64] [--shrink-budget 32]
       [--out runs/m4/comparison] [--from-records]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from robustlab.experiments.stack import load_stack
from robustlab.search.engine import SearchConfig, run_discovery
from robustlab.search.report import read_discovery, write_discovery

ROOT = Path(__file__).resolve().parent.parent


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = (int(x) for x in text.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in text.split(",")]


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial sign test, ties excluded."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher test for the 2x2 table [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)

    observed = prob(a)
    support = range(max(0, k - n2), min(k, n1) + 1)
    return min(1.0, sum(prob(x) for x in support if prob(x) <= observed * (1 + 1e-9)))


def row(record: dict) -> dict:
    s, final, shrunk = record["search"], record["final"], record["shrink"]
    conf = record["confirmation"]["confirmed"]
    return {
        "found_failure": s["found_failure"],
        "candidate_failures": s["candidate_failures"],
        "numerical": s["numerical"],
        "evaluations_to_first_failure": s["evaluations_to_first_failure"],
        "search_confirmed_impulse_ns": conf["candidate"]["impulse_ns"] if conf else None,
        "final_impulse_ns": final["candidate"]["impulse_ns"] if final else None,
        "final_source": final["source"] if final else None,
        "shrink_reduction_ns": shrunk["impulse_reduction_ns"] if shrunk else None,
        "search_rollouts": record["accounting"]["search_rollouts"],
        "physics_rollouts_total": record["accounting"]["physics_rollouts_total"],
        "search_wall_s": s["wall_s"],
        "wall_s": record["wall_s"],
        "results_sha256": record["results_sha256"],
        "stop_reason": s["stop_reason"],
    }


def aggregate(rows: list[dict]) -> dict:
    found = [r for r in rows if r["found_failure"]]

    def med(key):
        values = [r[key] for r in found if r[key] is not None]
        return statistics.median(values) if values else None

    return {
        "runs": len(rows),
        "found_failure": len(found),
        "median_evaluations_to_first_failure": med("evaluations_to_first_failure"),
        "median_search_confirmed_impulse_ns": med("search_confirmed_impulse_ns"),
        "median_final_impulse_ns": med("final_impulse_ns"),
        "min_final_impulse_ns": min((r["final_impulse_ns"] for r in found), default=None),
        "shrink_improved": sum(1 for r in found if r["final_source"] == "shrunk"),
        "total_search_rollouts": sum(r["search_rollouts"] for r in rows),
        "total_physics_rollouts": sum(r["physics_rollouts_total"] for r in rows),
        "median_wall_s": statistics.median(r["wall_s"] for r in rows),
        "total_wall_s": sum(r["wall_s"] for r in rows),
    }


def paired(rows_a: list[dict], rows_b: list[dict], key: str, both_only: bool = False) -> dict:
    """a = cmaes, b = random; lower `key` wins. A missing value (no confirmed failure) loses to a
    present one, unless `both_only`, which keeps only seeds where both have a value."""
    wins = losses = ties = pairs = 0
    for a, b in zip(rows_a, rows_b, strict=True):
        va, vb = a[key], b[key]
        if both_only and (va is None or vb is None):
            continue
        pairs += 1
        if va is None and vb is None:
            ties += 1
        elif vb is None or (va is not None and va < vb - 1e-12):
            wins += 1
        elif va is None or vb < va - 1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "pairs": pairs,
        "cmaes_better": wins,
        "random_better": losses,
        "ties": ties,
        "sign_test_p_two_sided": sign_test_p(wins, losses),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--shrink-budget", type=int, default=32)
    parser.add_argument("--out", default="runs/m4/comparison")
    parser.add_argument(
        "--from-records", action="store_true", help="rebuild summary.json from saved records"
    )
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    out = ROOT / args.out
    stack = None
    if not args.from_records:
        stack = load_stack(ROOT, Path("policies/go1-walk-s0.json"), "fall-contact")
    rows: dict[str, list[dict]] = {"random": [], "cmaes": []}
    for seed in seeds:
        for algorithm in ("random", "cmaes"):
            if args.from_records:
                record = read_discovery(out / f"{algorithm}-s{seed}")
                cfg = record["config"]
                expected = (algorithm, args.budget, args.shrink_budget, seed)
                if (cfg["algorithm"], cfg["budget"], cfg["shrink_budget"], cfg["seed"]) != expected:
                    raise SystemExit(f"{algorithm}-s{seed}: record does not match the protocol")
            else:
                config = SearchConfig(algorithm, args.budget, seed, args.shrink_budget)
                record = run_discovery(config, stack)
                write_discovery(out / f"{algorithm}-s{seed}", record)
            r = {"seed": seed, **row(record)}
            rows[algorithm].append(r)
            print(
                f"{algorithm:6s} seed {seed:3d}: found {r['found_failure']!s:5s} first@"
                f"{r['evaluations_to_first_failure']!s:>4} search {r['search_confirmed_impulse_ns']}"
                f" final {r['final_impulse_ns']} ({r['final_source']}) {r['wall_s']:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    found = {alg: sum(r["found_failure"] for r in rows[alg]) for alg in rows}
    summary = {
        "schema_version": 1,
        "kind": "robustlab-search-comparison",
        "protocol": {
            "seeds": seeds,
            "budget": args.budget,
            "shrink_budget": args.shrink_budget,
            "matched": (
                "same seeds, budgets, fixed experiment and code for both algorithms; no warm start"
            ),
            "metric_lower_is_better": (
                "confirmed failing impulse (N*s); in the unrestricted paired test a run without "
                "a confirmed failure loses to one with"
            ),
        },
        "git": record["provenance"]["git"],
        "versions": record["provenance"]["versions"],
        "rows": rows,
        "aggregate": {alg: aggregate(r) for alg, r in rows.items()},
        "found_failure_fisher_p_two_sided": fisher_exact_p(
            found["cmaes"],
            len(seeds) - found["cmaes"],
            found["random"],
            len(seeds) - found["random"],
        ),
        "paired": {
            "search_confirmed_impulse": paired(
                rows["cmaes"], rows["random"], "search_confirmed_impulse_ns"
            ),
            "final_impulse": paired(rows["cmaes"], rows["random"], "final_impulse_ns"),
            "search_confirmed_impulse_both_found": paired(
                rows["cmaes"], rows["random"], "search_confirmed_impulse_ns", both_only=True
            ),
            "final_impulse_both_found": paired(
                rows["cmaes"], rows["random"], "final_impulse_ns", both_only=True
            ),
            "evaluations_to_first_failure_both_found": paired(
                rows["cmaes"], rows["random"], "evaluations_to_first_failure", both_only=True
            ),
        },
        "wall_s": sum(r["wall_s"] for a in rows.values() for r in a),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("aggregate", "found_failure_fisher_p_two_sided",
                                              "paired")}, indent=2))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
