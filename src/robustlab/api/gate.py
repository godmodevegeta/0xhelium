"""Regression Gate API service (M7): the committed gate-v1 result, the policy registry and
re-simulated gate replays.

The gate is evaluated once by `robustlab gate`; this service only reads the committed result
preset and the registry, and never evaluates or promotes anything. Because the gate result already
exists, locked episodes may be replayed here (the Hardening API still refuses them).
"""

from __future__ import annotations

import json
from pathlib import Path

from robustlab.api.presets import PresetError, load_preset
from robustlab.api.reflex_replay import paired_reflex_replay
from robustlab.gate.config import GateConfigError, load_gate
from robustlab.gate.evaluate import BASELINE
from robustlab.gate.registry import read_active, registry_dir

GATE_PRESET_DIR = "presets/m7-gate-v1"
GATE_CONFIG = "examples/m7/gate-v1.json"
SUITES = ("holdout", "nominal", "gauntlet", "known_failures")
SUBSTEP_S = 0.004
ROW_KEYS = ("outcome", "reason", "fall_time_s", "time_after_push_s", "max_tilt_after_push_deg",
            "progress_m")  # fmt: skip
SEMANTICS = (
    "Stress test, Discover failure and Design scenario evaluate the frozen baseline (v0) so that "
    "their evidence stays comparable with M2-M5. The gate decides which reflex candidate the "
    "registry points at; it does not change any earlier result."
)


def _row(r: dict) -> dict:
    out = {k: r.get(k) for k in ROW_KEYS}
    out["reflex_activation"] = (r.get("reflex") or {}).get("activation_fraction")
    return out


class GateService:
    """Loaded once at startup. Missing preset = gate not available; inconsistent files raise."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        try:
            self.preset = load_preset(self.root, GATE_PRESET_DIR)
        except PresetError:
            if (self.root / GATE_PRESET_DIR).exists():
                raise
            self.preset = None
            self.view = {"available": False, "semantics": SEMANTICS,
                         "registry": self._registry()}  # fmt: skip
            return
        result = self.preset.json("gate_result.json")
        try:
            self.gate = load_gate(self.root / GATE_CONFIG, self.root)
        except GateConfigError as exc:
            raise PresetError(f"gate config: {exc}") from exc
        if result["gate"]["sha256"] != self.gate.sha256:
            raise PresetError("the gate result was produced from another gate configuration")
        if result["results_sha256"] != self.preset.meta["source"]["results_sha256"]:
            raise PresetError("gate result hash differs from the preset record")
        self.result = result
        self.outcomes = result["outcomes"]
        self.specs = {}
        for suite in SUITES:
            for i, spec in enumerate(getattr(self.gate, suite)):
                self.specs[spec.id] = (suite, i, spec)
        self.candidates = {c.candidate_id: c for c in self.gate.candidates}
        registry = self._registry()
        self.status = {cid: self._status(cid, registry) for cid in self.candidates}
        self.view = self._view(registry)

    # ---------------------------------------------------------------- registry and status

    def _registry(self) -> dict:
        reg = registry_dir(self.root)
        versions = [json.loads(p.read_text()) for p in sorted((reg / "versions").glob("v*.json"))]
        rejected = [json.loads(p.read_text()) for p in sorted((reg / "rejected").glob("*.json"))]
        active = read_active(self.root)
        return {
            "active": None if active is None else active["active"],
            "history": [] if active is None else active["history"],
            "versions": [
                {k: v.get(k) for k in ("version", "parent", "candidate_id", "gate",
                                       "gate_result_sha256", "statement", "created_utc",
                                       "source")} | {"policy": v["policy"], "reflex": v["reflex"]}
                for v in versions
            ],
            "rejected": [{k: r.get(k) for k in ("candidate_id", "gate", "failed_criteria",
                                                 "statement", "active_unchanged", "created_utc")}
                         for r in rejected],
            "path": "policies/registry",
        }  # fmt: skip

    def _status(self, cid: str, registry: dict) -> dict:
        c = self.result["candidates"][cid]
        name = self.gate.name
        entry = next((v for v in registry["versions"]
                      if v["candidate_id"] == cid and v["gate"] == name), None)  # fmt: skip
        version = None if entry is None else entry["version"]
        if entry is not None and entry["gate_result_sha256"] != self.result["results_sha256"]:
            raise PresetError(f"registry version {version} cites another gate result")
        failed = c["failed_criteria"]
        if c["eligible_for_promotion"]:
            if self.result["decision"]["verdict"] == "promote":
                if version is None:
                    raise PresetError(f"{cid} was promoted but has no registry version")
                status, label = "promoted", f"promoted by {name} as {version}"
            else:
                status, label = "rejected", f"rejected by {name} (failed {', '.join(failed)})"
        else:
            status = "not_eligible"
            verdict = "passed G1-G4" if not failed else f"failed {', '.join(failed)}"
            label = f"evaluated by {name}, not eligible for promotion ({verdict})"
        return {"gate": name, "candidate_id": cid, "status": status, "label": label,
                "version": version, "eligible": c["eligible_for_promotion"],
                "passes_all": c["passes_all"], "failed_criteria": failed}  # fmt: skip

    def status_for(self, candidate_id: str | None) -> dict | None:
        """Gate status of a candidate id (ids are content hashes, so equal id = same inputs)."""
        if self.preset is None or candidate_id is None:
            return None
        return self.status.get(candidate_id)

    # ---------------------------------------------------------------- view

    def _regression_rows(self, cid: str, ids: list[str]) -> list[dict]:
        rows = []
        for eid in ids:
            suite, i, spec = self.specs[eid]
            rows.append({"id": eid, "start_substep": spec.push.start_substep,
                         "start_s": round(spec.push.start_substep * SUBSTEP_S, 3),
                         "baseline": _row(self.outcomes[BASELINE][suite][i]),
                         "candidate": _row(self.outcomes[cid][suite][i])})  # fmt: skip
        return rows

    def _candidate_view(self, cid: str) -> dict:
        c = self.result["candidates"][cid]
        cr = c["criteria"]
        g3 = cr["G3_regression"]
        conditions = []
        for cond in self.gate.doc["suites"]["gauntlet"]["conditions"]:
            v = g3["conditions"][cond["id"]]
            conditions.append({**cond, "baseline": v["baseline"], "candidate": v["candidate"],
                               "increase": v["increase"], "recovered": v["recovered"],
                               "mcnemar_one_sided_p": v["mcnemar_one_sided_p"],
                               "pass": g3["checks"].get(f"condition {cond['id']}", True),
                               "regressions": self._regression_rows(cid, v["regressed"])})  # fmt: skip
        kf = c["known_failures"]
        return {
            "candidate_id": cid,
            "seed": c["seed"],
            "primary": c["primary"],
            "record": self.candidates[cid].record_path,
            "gate_status": self.status[cid],
            "passes_all": c["passes_all"],
            "failed_criteria": c["failed_criteria"],
            "G1_nominal": cr["G1_nominal"],
            "G2_holdout": cr["G2_holdout"],
            "G3_regression": {k: g3[k] for k in ("pass", "checks", "baseline", "candidate",
                                                 "limit", "recovered")}
            | {"regressed": len(g3["regressed"]), "conditions": conditions},
            "G4_numerical": cr["G4_numerical"],
            "known_failures": {"baseline": kf["baseline"], "candidate": kf["candidate"],
                               "recovered": len(kf["recovered"]),
                               "still_failing": kf["still_failing"],
                               "regressed": kf["regressed"]},
        }  # fmt: skip

    def _view(self, registry: dict) -> dict:
        r = self.result
        ids = list(self.candidates)
        holdout = [
            {"episode": spec.to_dict(), "baseline": _row(self.outcomes[BASELINE]["holdout"][i]),
             "candidates": {cid: _row(self.outcomes[cid]["holdout"][i]) for cid in ids}}
            for i, spec in enumerate(self.gate.holdout)
        ]  # fmt: skip
        g = self.gate.doc["suites"]["gauntlet"]
        return {
            "available": True,
            "result_origin": "precomputed",
            "preset": {k: self.preset.meta[k] for k in ("id", "version", "title", "source")},
            "gate": r["gate"],
            "config_path": GATE_CONFIG,
            "created_before_evaluation": self.gate.doc.get("created_before_evaluation"),
            "primary_rule": self.gate.doc["primary"]["rule"],
            "primary_ranking": self.gate.doc["primary"]["ranking"],
            "suites": r["suites"],
            "suite_notes": {
                "holdout": "the 10 locked M6 attacks; never used for tuning or selection",
                "nominal": "3 locked nominal walks + 10 gate-defined walks (commands, friction, "
                "payload)",
                "gauntlet": g["note"],
                "known_failures": self.gate.doc["suites"]["known_failures"]["note"],
            },
            "gauntlet": {k: g[k] for k in ("starts_substeps", "duration_substeps", "direction")},
            "criteria_definition": r["criteria_definition"],
            "statistics": self.gate.doc.get("statistics"),
            "decision": r["decision"],
            "candidates": [self._candidate_view(cid) for cid in ids],
            "holdout": holdout,
            "nominal_episodes": [s.to_dict() for s in self.gate.nominal],
            "results_sha256": r["results_sha256"],
            "accounting": r["accounting"],
            "timing": r["timing"],
            "created_utc": r["created_utc"],
            "provenance": {
                "git": r["provenance"]["git"],
                "policy_sha256": r["provenance"]["policy_sha256"],
                "versions": r["provenance"]["versions"],
            },
            "registry": registry,
            "reproduce": "uv run robustlab gate --verify",
            "semantics": SEMANTICS,
        }

    # ---------------------------------------------------------------- replay

    def replay_compute(self, candidate_id: str, episode_id: str, ghost: str):
        """Raises LookupError for an unknown candidate, episode or impossible ghost."""
        if self.preset is None:
            raise LookupError("no gate result is available")
        cand = self.candidates.get(candidate_id)
        if cand is None:
            raise LookupError(f"{candidate_id!r} is not a gate candidate")
        found = self.specs.get(episode_id)
        if found is None:
            raise LookupError(f"unknown gate episode {episode_id!r}")
        suite, i, spec = found
        status = self.status[candidate_id]
        short = candidate_id[5:13]
        what = "same episode" if spec.push is None else "same shove"
        labels = {
            "candidate": f"Candidate {short} ({status['label']}): frozen policy + reflex",
            "baseline": f"Frozen baseline v0, {what}",
            "matched": "Matched condition: frozen baseline v0, no shove",
        }
        recorded = {"baseline": self.outcomes[BASELINE][suite][i],
                    "candidate": self.outcomes[candidate_id][suite][i]}  # fmt: skip
        return paired_reflex_replay(
            self.root, spec, cand.vector, self.gate.calibration, ghost,
            mode="gate", labels=labels, recorded=recorded,
            recorded_commit=self.result["provenance"]["git"]["commit"],
            what=f"{self.gate.name} result", digest_stop_on_fall=suite != "nominal",
            extra={"gate": self.gate.name, "suite": suite, "candidate_id": candidate_id,
                   "candidate_status": status["status"], "gate_status": status,
                   "evidence_compute": {
                       "label": f"{self.gate.name} evaluation",
                       "rollouts": self.result["accounting"]["rollouts"],
                       "wall_s": self.result["timing"]["wall_s"],
                       "workers": self.result["timing"]["workers"]}},
        )  # fmt: skip
