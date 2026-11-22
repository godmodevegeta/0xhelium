"""Scenario Designer service (M5): presets, previews, OpenRouter compile audit and replays.

Physics previews and replays run on the JobManager's replay worker (`preview(worker, ...)` and
the replay `compute` callables are only called there). The OpenRouter call runs in a thread
(`asyncio.to_thread`), never on the event loop or a simulation worker.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from pathlib import Path

from robustlab.api.jobs import JobManager, ScenarioRunError, Worker
from robustlab.experiments.trace import read_run
from robustlab.replay import replay_payload
from robustlab.scenarios.schema import (
    BOUNDS,
    NOMINAL_FRICTION_MU,
    ScenarioError,
    json_schema,
    validate,
)

PRESETS_DIR = "examples/m5"
PRESET_NOTES = {
    "nominal": "Trained conditions: friction 0.6, no payload, no push.",
    "low-friction-0.2": "Floor friction 0.2 (trained 0.6). No payload, no push.",
    "payload-5kg-top": "5 kg box welded on top of the trunk. Trained friction, no push.",
    "icy-heavy-shove": "Friction 0.3, 3 kg payload, 100 N shove from the left at 2.0 s.",
}
OUTCOME_KEYS = ("outcome", "reason", "fall_time_s", "numerical_time_s", "time_after_start_s",
                "events", "first_nonfoot_body", "forward_progress_m")  # fmt: skip


def _outcome(summary: dict) -> dict:
    return {k: summary.get(k) for k in OUTCOME_KEYS}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_presets(root: Path) -> list[dict]:
    """Every example scenario must validate; a broken preset is a startup error."""
    out = []
    for path in sorted((Path(root) / PRESETS_DIR).glob("*.json")):
        document = json.loads(path.read_text())
        validated = validate(document)  # raises ScenarioError
        out.append({
            "id": path.stem,
            "file": f"{PRESETS_DIR}/{path.name}",
            "note": PRESET_NOTES.get(path.stem, ""),
            "scenario": validated.scenario.to_dict(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })  # fmt: skip
    return out


def ghost_parameters(result: dict, kind: str) -> dict:
    if kind == "original":
        return {"ground_friction_mu": NOMINAL_FRICTION_MU, "payload_kg": 0.0,
                "payload_offset_xyz_m": [0.0, 0.0, 0.0], "push": None}  # fmt: skip
    s = result["scenario"]
    mu = s["environment"]["ground_friction_mu"]
    return {"ground_friction_mu": NOMINAL_FRICTION_MU if mu is None else mu,
            "payload_kg": s["payload"]["mass_kg"],
            "payload_offset_xyz_m": s["payload"]["offset_xyz_m"], "push": None}  # fmt: skip


GHOST_LABELS = {
    "matched": "Matched baseline: same friction and payload, push removed",
    "original": "Original nominal: friction 0.6, no payload, no push",
}


class ScenarioService:
    def __init__(self, settings, jobs: JobManager, compiler=None) -> None:
        from robustlab.llm.openrouter import OpenRouterCompiler

        self.settings = settings
        self.jobs = jobs
        self.compiler = compiler if compiler is not None else OpenRouterCompiler.from_env()
        self.presets = load_presets(settings.root)
        self.audit_path = Path(settings.runs_dir) / "llm" / "compiles.jsonl"
        self._compiled: dict[str, dict] = {}  # compile_id -> {scenario_sha256, model}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ config and preview

    def config(self) -> dict:
        from robustlab.llm.prompt import EXAMPLES

        return {
            "bounds": {k: list(v) for k, v in BOUNDS.items()},
            "nominal_mu": NOMINAL_FRICTION_MU,
            "presets": self.presets,
            "examples": EXAMPLES,
            "compiler": self.compiler.status(),
            "schema": json_schema(),
            "suite_defaults": {
                "trials": 16,
                "interval_s": [2.0, 3.024],
                "method": "stratified",
                "max_trials": 32,
            },
            "semantics": {
                "single": "one deterministic paired run: scenario, matched baseline (push removed, "
                "only when there is a push) and the original nominal reference",
                "suite": "only the push start varies over the trials; everything else is fixed. "
                "Needs a push: without one every trial would be the same deterministic episode",
                "ghost": "the translucent robot is the matched baseline when the scenario has a "
                "push, otherwise the original nominal",
            },
            "runs": self.jobs.list_scenarios(),
        }

    def preview(self, worker: Worker, validated) -> dict:
        """Validated parameters plus the compiled model's friction and mass facts (replay worker)."""
        from robustlab.experiments.scenario_run import model_for
        from robustlab.physics.scenario_model import MODEL_CACHE, friction_report, mass_report

        scenario = validated.scenario
        stack = worker.stack(scenario.scene)
        entry = model_for(scenario, stack, MODEL_CACHE)
        push = scenario.push
        return {
            "valid": True,
            "errors": [],
            **validated.to_dict(),
            "changed_from_original": scenario.changed_from_nominal(),
            "effective_mu": scenario.effective_mu(),
            "friction": friction_report(
                entry.model, entry.ids, stack.policy, scenario.ground_friction_mu
            ),
            "mass": mass_report(entry.model, entry.ids),
            "model": {
                "signature": list(entry.signature),
                "reused_base_model": entry.base,
                "cache": MODEL_CACHE.stats(),
            },
            "push": None
            if push is None
            else {
                "force_n": push.force_n,
                "direction_deg": push.direction_deg,
                "direction_vector": list(push.direction_vector),
                "start_s": push.start_s,
                "duration_s": push.duration_s,
                "impulse_ns": push.impulse_ns,
                "frame": "initial heading: +x forward, +y left; 270 deg pushes toward the "
                "robot's right",
            },
            "runnable": {
                "single": True,
                "suite": push is not None,
                "suite_reason": None
                if push is not None
                else "a suite varies the push start; without a push every trial is the same "
                "deterministic episode",
            },
            "primary_baseline": "matched" if push is not None else "original",
        }

    # ------------------------------------------------------------------ OpenRouter

    def compile(self, prompt: str, client: str) -> dict:
        """Call the compiler (blocking; run in a thread) and append an audit record."""
        from robustlab.llm.openrouter import CompileError

        compile_id = f"cp-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{secrets.token_hex(4)}"
        record = {"compile_id": compile_id, "utc": _utc(),
                  "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "client_sha256": hashlib.sha256(client.encode()).hexdigest()[:16],
                  "requested_model": self.compiler.model}  # fmt: skip
        try:
            result = self.compiler.compile(prompt, client=client)
        except CompileError as exc:
            self._audit({**record, "status": "error", "error": exc.to_dict()})
            raise
        result["compile_id"] = compile_id
        self._audit({**record, "status": "valid" if result["valid"] else "invalid",
                     "llm": result["llm"], "proposal": result["proposal"],
                     "errors": result["errors"], "adjustments": result.get("adjustments"),
                     "scenario_sha256": result.get("scenario_sha256")})  # fmt: skip
        if result["valid"]:
            with self._lock:
                self._compiled[compile_id] = {
                    "scenario_sha256": result["scenario_sha256"],
                    "model": result["llm"].get("model") or self.compiler.model,
                }
        return result

    def _audit(self, record: dict) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.audit_path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def origin_for(self, compile_id: str | None, scenario) -> dict:
        if compile_id is None:
            return {"kind": "manual"}
        with self._lock:
            compiled = self._compiled.get(compile_id)
        if compiled is None:
            raise ScenarioRunError(
                f"unknown compile_id {compile_id} (compiled in another server session?)"
            )
        return {
            "kind": "openrouter",
            "compile_id": compile_id,
            "model": compiled["model"],
            "edited_after_compile": compiled["scenario_sha256"] != scenario.sha256(),
        }

    # ------------------------------------------------------------------ replays

    def replay_plan(self, run_dir: Path, result: dict, trial: int, ghost: str) -> dict:
        """Resolve which recorded trajectories to show; raises LookupError (404) if unavailable.
        Returns {"ghost": {...}, "compute": callable(worker) -> replay payload}."""
        has_push = result["scenario"]["push"] is not None
        kind = ("matched" if has_push else "original") if ghost == "primary" else ghost
        if kind == "matched" and not has_push:
            raise LookupError("no matched baseline: the scenario has no push (compare original)")
        paths = {}
        if result["mode"] == "single":
            if trial != 0:
                raise LookupError("a single run has only trial 0")
            traj = result.get("trajectories") or {}
            paths = {"attacked": traj.get("scenario"), "nominal": traj.get(kind)}
            summaries = {"attacked": result["cases"]["scenario"], "nominal": result["cases"][kind]}
            push = result.get("push")
            start_s = None if push is None else push["start_s"]
        else:
            from robustlab.experiments.suite import read_report

            report = read_report(run_dir)
            if not 0 <= trial < len(report["trials"]):
                raise LookupError("unknown trial")
            t = report["trials"][trial]
            if not t.get("trajectories"):
                raise LookupError("trajectories were not saved for this run")
            paths = {"attacked": t["trajectories"]["attacked"],
                     "nominal": t["trajectories"]["nominal"] if kind == "matched" else "original"}  # fmt: skip
            summaries = {"attacked": t["attacked"],
                         "nominal": t["nominal"] if kind == "matched" else result["original"]}  # fmt: skip
            push = t["push"]
            start_s = t["start_s"]
        if None in paths.values():
            raise LookupError("trajectories missing for this run")
        try:
            validated = validate(result["scenario"])
        except ScenarioError as exc:  # a stored record that no longer validates
            raise LookupError(f"stored scenario no longer validates: {exc}") from exc
        scenario = validated.scenario
        ghost_info = {"kind": kind, "label": GHOST_LABELS[kind],
                      "parameters": ghost_parameters(result, kind)}  # fmt: skip
        scenario_label = "Scenario (pushed)" if has_push else "Scenario"

        def compute(worker: Worker) -> dict:
            from robustlab.experiments.scenario_run import model_for
            from robustlab.physics.scenario_model import MODEL_CACHE

            stack = worker.stack(scenario.scene)
            entry = model_for(scenario, stack, MODEL_CACHE)
            loaded = {case: read_run(run_dir / paths[case])[0] for case in paths}
            return replay_payload(
                stack.model,
                (loaded["nominal"], _outcome(summaries["nominal"])),
                (loaded["attacked"], _outcome(summaries["attacked"])),
                push,
                source="recorded",
                case_models={
                    "attacked": entry.model,
                    "nominal": entry.model if kind == "matched" else stack.model,
                },
                extra={
                    "run_id": run_dir.name,
                    "trial": trial,
                    "start_s": start_s,
                    "mode": result["mode"],
                    "scenario": result["scenario"],
                    "ghost": ghost_info,
                    "labels": {"attacked": scenario_label, "nominal": ghost_info["label"]},
                    "environment": {
                        "mu": scenario.effective_mu(),
                        "ghost_mu": ghost_info["parameters"]["ground_friction_mu"],
                        "nominal_mu": NOMINAL_FRICTION_MU,
                    },
                },
            )

        return {"ghost": ghost_info, "compute": compute}
