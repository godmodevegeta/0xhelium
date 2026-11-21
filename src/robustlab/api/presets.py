"""Versioned scenario presets: integrity checks and the summaries the viewer shows.

A preset directory holds `preset.json` (id, version, file hashes, source) and the files it lists.
Results in a preset are precomputed: they were produced by the M2 runner at the recorded commit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FOOT_ORDER = ("FR", "FL", "RR", "RL")
HIDDEN_PHASE_DIR = "presets/hidden-phase-v1"
LIVE_STRESS_DIR = "presets/live-stress-v1"
DISCOVERY_COMPARISON_DIR = "presets/discovery-comparison-v1"


class PresetError(Exception):
    """A preset is missing, malformed or does not match its recorded hashes."""


@dataclass(frozen=True)
class Preset:
    directory: Path
    meta: dict

    def path(self, name: str) -> Path:
        return self.directory / name

    def json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text())


def load_preset(root: Path, relative_dir: str) -> Preset:
    directory = Path(root) / relative_dir
    try:
        meta = json.loads((directory / "preset.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PresetError(f"{relative_dir}/preset.json: {exc}") from exc
    files = meta.get("files")
    if not isinstance(files, dict) or not files:
        raise PresetError(f"{relative_dir}/preset.json: no files listed")
    for name, expected in files.items():
        if "/" in name or name.startswith("."):
            raise PresetError(f"{relative_dir}: invalid file name {name!r}")
        try:
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        except OSError as exc:
            raise PresetError(f"{relative_dir}/{name}: {exc}") from exc
        if actual != expected:
            raise PresetError(f"{relative_dir}/{name}: SHA-256 {actual} != recorded {expected}")
    return Preset(directory, meta)


def stance_key(stance: dict | None) -> str:
    if not stance:
        return "unknown"
    down = [foot for foot in FOOT_ORDER if stance.get(foot)]
    return "+".join(down) if down else "none"


def stance_table(trials: list[dict]) -> list[dict]:
    tested: Counter[str] = Counter()
    fell: Counter[str] = Counter()
    for t in trials:
        key = stance_key(t["stance_at_start"])
        tested[key] += 1
        fell[key] += t["attacked"]["outcome"] == "fell"
    rows = [{"stance": k, "tested": tested[k], "fell": fell[k]} for k in tested]
    return sorted(rows, key=lambda r: (-r["fell"], -r["tested"], r["stance"]))


def compact_trial(t: dict) -> dict:
    att = t["attacked"]
    return {
        "trial": t["trial"],
        "start_substep": t["start_substep"],
        "start_s": t["start_s"],
        "stance": stance_key(t["stance_at_start"]),
        "nominal_outcome": t["nominal"]["outcome"],
        "outcome": att["outcome"],
        "reason": att["reason"],
        "fall_time_s": att["fall_time_s"],
        "numerical_time_s": att["numerical_time_s"],
        "time_after_start_s": att["time_after_start_s"],
        "first_nonfoot_body": att["first_nonfoot_body"],
        "events": att["events"],
        "forward_progress_m": att["forward_progress_m"],
        "nominal_progress_m": t["nominal"]["forward_progress_m"],
        "pairing_ok": t["paired"]["pairing_ok"],
        "has_trajectories": t.get("trajectories") is not None,
    }


def report_summary(report: dict) -> dict:
    """What the viewer needs from an M2 experiment report, without per-trial digests."""
    agg = report["aggregates"]
    return {
        "name": report["config"]["name"],
        "config": report["config"],
        "config_sha256": report["config_sha256"],
        "results_sha256": report["results_sha256"],
        "sampling": {
            k: report["sampling"][k]
            for k in ("method", "kind", "seed", "seed_used", "interval_s", "trials", "generator")
        },
        "fixed_parameters": report["fixed_parameters"],
        "aggregates": {
            "nominal": agg["nominal"],
            "attacked": agg["attacked"],
            "paired": agg["paired"],
        },
        "stance_table": stance_table(report["trials"]),
        "interpretation": report["interpretation"],
        "provenance": {
            "git": report["provenance"]["git"],
            "versions": report["provenance"]["versions"],
            "created_utc": report["created_utc"],
            "policy_sha256": report["provenance"]["policy"]["sha256"],
            "scene": report["provenance"]["scene"]["name"],
            "scene_transform_sha256": report["provenance"]["scene"]["transform_sha256"],
        },
        "trials": [compact_trial(t) for t in report["trials"]],
    }


def hidden_phase_summary(preset: Preset, report: dict) -> dict:
    summary = report_summary(report)
    failing = [t for t in summary["trials"] if t["outcome"] == "fell"]
    verified = preset.meta["verified_failure_substep"]
    if verified not in {t["start_substep"] for t in failing}:
        raise PresetError(f"verified failure substep {verified} is not a recorded failure")
    diag = next((r for r in summary["stance_table"] if r["stance"] == "FR+RL"), None)
    fell_total = summary["aggregates"]["attacked"]["physical_falls"]
    summary.update(
        preset={
            "id": preset.meta["id"],
            "version": preset.meta["version"],
            "title": preset.meta["title"],
            "source": preset.meta["source"],
            "files": preset.meta["files"],
            "result_origin": "precomputed",
        },
        verified_failure_substep=verified,
        verified_failure_note=preset.meta["verified_failure_note"],
        failing_substeps=[t["start_substep"] for t in failing],
        stance_finding=None
        if diag is None
        else {
            "stance": "FR+RL",
            "fell": diag["fell"],
            "tested": diag["tested"],
            "all_failures_in_stance": diag["fell"] == fell_total,
        },
    )
    return summary
