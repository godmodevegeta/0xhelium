"""Writing and summarizing discovery records."""

from __future__ import annotations

import json
from pathlib import Path

from robustlab.experiments.trace import _atomic_write_json

RECORD_NAME = "discovery.json"


def write_discovery(out_dir: Path, record: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if record.get("final") and record["final"].get("scenario_config"):
        _atomic_write_json(out_dir / "scenario.json", record["final"]["scenario_config"])
    path = out_dir / RECORD_NAME
    _atomic_write_json(path, record)
    return path


def read_discovery(out_dir: Path) -> dict:
    record = json.loads((Path(out_dir) / RECORD_NAME).read_text())
    if record.get("schema_version") != 1:
        raise ValueError(f"unsupported discovery schema {record.get('schema_version')!r}")
    return record


def candidate_text(c: dict) -> str:
    return (
        f"{c['force_n']:g} N at {c['direction_deg']:g} deg, start {c['start_s']:.3f} s, "
        f"{c['duration_s']:.3f} s -> {c['impulse_ns']:.3f} N*s"
    )


def format_discovery(record: dict) -> str:
    cfg, s, acct = record["config"], record["search"], record["accounting"]
    lines = [
        (
            f"algorithm        {cfg['algorithm']}  seed {cfg['seed']}  budget {cfg['budget']}  "
            f"shrink budget {cfg['shrink_budget']}"
        ),
        (
            f"search           {s['stop_reason']}: {acct['search_rollouts']} rollouts, "
            f"{acct['cache_hits']} cache hits, {s['candidate_failures']} candidate failures, "
            f"{s['numerical']} numerical, {s['wall_s']:.2f} s"
        ),
    ]
    if not s["found_failure"]:
        lines.append("result           NO PHYSICAL FAILURE FOUND within the budget")
    else:
        lines.append(f"first failure    evaluation {s['evaluations_to_first_failure']}")
        best = record["evaluations"][s["best_candidate_index"] - 1]
        lines.append(f"best candidate   #{best['index']}: {candidate_text(best['candidate'])}")
        conf = record["confirmation"]["confirmed"]
        lines.append(
            "confirmed        "
            + (f"#{conf['evaluation']}: {candidate_text(conf['candidate'])}" if conf else "NONE")
        )
        sh = record["shrink"]
        if sh is not None:
            lines.append(
                f"shrunk           {candidate_text(sh['final'])} "
                f"({'improved' if sh['improved'] else 'no confirmed improvement'}; "
                f"{sh['rollouts']} rollouts + {sh['confirmation_rollouts']} confirmation)"
            )
        final = record["final"]
        if final:
            lines.append(f"final ({final['source']:>6})  {candidate_text(final['candidate'])}")
            trial = final.get("trial") or {}
            stance = trial.get("stance_at_start")
            if stance:
                down = "+".join(k for k in ("FR", "FL", "RR", "RL") if stance[k]) or "none"
                lines.append(f"stance at shove  {down}")
    lines.append(
        f"rollouts         total {acct.get('physics_rollouts_total', '?')} "
        f"(search {acct['search_rollouts']}, confirmation {acct['confirmation_rollouts']}, "
        f"shrink {acct['shrink_rollouts']} + {acct['shrink_confirmation_rollouts']}, reference "
        f"{acct.get('reference_rollouts', 0)})"
    )
    lines.append(f"results_sha256   {record['results_sha256']}")
    lines.append(f"note             {record.get('statement', '')}")
    return "\n".join(lines)
