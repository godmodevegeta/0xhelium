"""FastAPI application: presets, live experiment runs and replays.

Start with `uv run robustlab serve` (or `uvicorn robustlab.api.app:create_app --factory`).
Simulations run on the JobManager's worker threads, never on the event loop.
"""

from __future__ import annotations

import asyncio
import os
import platform
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from robustlab import __version__
from robustlab.api.jobs import BusyError, JobManager, Worker
from robustlab.api.presets import (
    DISCOVERY_COMPARISON_DIR,
    HIDDEN_PHASE_DIR,
    LIVE_STRESS_DIR,
    PresetError,
    hidden_phase_summary,
    load_preset,
    report_summary,
)
from robustlab.experiments.suite import load_suite_config, run_trial
from robustlab.experiments.trace import read_run, trajectory_dict
from robustlab.experiments.walk import EpisodeRunner, WalkCriteria
from robustlab.replay import model_payload, replay_payload


@dataclass(frozen=True)
class ServerSettings:
    root: Path
    runs_dir: Path
    live_suite: Path
    web_dist: Path
    discovery_timeout_s: float = 120.0
    hardening_wall_clock_s: float = 60.0
    hardening_workers: int = 0  # 0 = automatic (cpu_count - 2, at most 8)

    @classmethod
    def from_env(cls) -> ServerSettings:
        root = Path(os.environ.get("ROBUSTLAB_ROOT", ".")).resolve()
        return cls(
            root=root,
            runs_dir=Path(os.environ.get("ROBUSTLAB_RUNS_DIR", root / "runs" / "web")),
            live_suite=Path(
                os.environ.get("ROBUSTLAB_LIVE_SUITE", root / LIVE_STRESS_DIR / "suite.json")
            ),
            web_dist=Path(os.environ.get("ROBUSTLAB_WEB_DIST", root / "web" / "dist")),
            discovery_timeout_s=float(os.environ.get("ROBUSTLAB_DISCOVERY_TIMEOUT_S", "120")),
            hardening_wall_clock_s=float(os.environ.get("ROBUSTLAB_HARDENING_WALL_S", "60")),
            hardening_workers=int(os.environ.get("ROBUSTLAB_HARDENING_WORKERS", "0")),
        )


class LiveRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int | None = Field(default=None, ge=0, lt=2**64)


class DiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    algorithm: Literal["random", "cmaes"]
    budget: int = Field(default=64, ge=8, le=256, multiple_of=8)
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    shrink_budget: int = Field(default=32, ge=0, le=128)


def _episode_outcome(summary: dict) -> dict:
    keys = ("outcome", "reason", "fall_time_s", "numerical_time_s", "time_after_start_s",
            "events", "first_nonfoot_body", "forward_progress_m")  # fmt: skip
    return {k: summary.get(k) for k in keys}


class ScenarioBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: dict


class CompileBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(max_length=4000)  # the compiler enforces its own, smaller limit


class ScenarioRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: dict
    mode: Literal["single", "suite"] = "single"
    trials: int = Field(default=16, ge=2, le=32)
    interval_s: tuple[float, float] = (2.0, 3.024)
    method: Literal["stratified", "grid"] = "stratified"
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    compile_id: str | None = Field(default=None, pattern=r"^cp-\d{8}T\d{6}Z-[0-9a-f]{8}$")


class HardeningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    generations: int = Field(default=12, ge=1, le=30)
    focus: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,64}$")


def create_app(settings: ServerSettings | None = None, compiler=None) -> FastAPI:
    """`compiler` defaults to `OpenRouterCompiler.from_env()` (tests inject a mocked one)."""
    settings = settings or ServerSettings.from_env()
    hidden = load_preset(settings.root, HIDDEN_PHASE_DIR)
    live = load_preset(settings.root, LIVE_STRESS_DIR)
    hidden_report = hidden.json("experiment.json")
    hidden_view = hidden_phase_summary(hidden, hidden_report)
    hidden_trials = {t["start_substep"]: t for t in hidden_report["trials"]}
    hidden_config = load_suite_config(hidden_report["config"])
    live_reference = report_summary(live.json("reference-grid-256.json"))
    comparison = load_preset(settings.root, DISCOVERY_COMPARISON_DIR)
    comparison_summary = comparison.json("summary.json")
    jobs = JobManager(
        settings.root,
        settings.runs_dir,
        settings.live_suite,
        live.meta,
        discovery_timeout_s=settings.discovery_timeout_s,
    )
    model_cache: dict[str, dict] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="Robustness Studio", version=__version__, lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.state.jobs = jobs
    app.state.settings = settings

    @app.exception_handler(PresetError)
    async def preset_error(_request: Request, exc: PresetError):
        return JSONResponse(status_code=500, content={"detail": f"preset error: {exc}"})

    async def wait(future):
        return await asyncio.wrap_future(future)

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "robustlab": __version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "presets": {
                "hidden-phase": hidden.meta["version"],
                "live-stress": live.meta["version"],
            },
            "physics": "native MuJoCo on the server; the browser only replays recorded poses",
        }

    @app.get("/api/model")
    async def model():
        if "model" not in model_cache:

            def build(worker: Worker) -> dict:
                return model_payload(worker.stack(jobs.live_config.scene).model)

            model_cache["model"] = await wait(jobs.replays.submit(build))
        return model_cache["model"]

    @app.get("/api/presets/hidden-phase")
    async def hidden_phase():
        return hidden_view

    @app.get("/api/presets/hidden-phase/replay/{start_substep}")
    async def hidden_phase_replay(start_substep: int):
        recorded = hidden_trials.get(start_substep)
        if recorded is None:
            raise HTTPException(404, f"start substep {start_substep} was not tested")

        def compute(worker: Worker) -> dict:
            stack = worker.stack(hidden_config.scene)
            criteria = WalkCriteria()
            runner = EpisodeRunner(stack.model, stack.ids, stack.policy, stack.actor, criteria)
            trial, (nominal, _), (attacked, _) = run_trial(
                hidden_config, runner, recorded["trial"], start_substep,
                recorded["stratum_substeps"], criteria,
            )  # fmt: skip
            checks = {
                case: {
                    "recorded_digest": recorded[case]["digest"],
                    "resimulated_digest": trial[case]["digest"],
                    "digest_match": recorded[case]["digest"] == trial[case]["digest"],
                    "recorded_outcome": recorded[case]["outcome"],
                    "resimulated_outcome": trial[case]["outcome"],
                    "outcome_match": recorded[case]["outcome"] == trial[case]["outcome"],
                }
                for case in ("nominal", "attacked")
            }
            verification = {
                "all_match": all(c["digest_match"] and c["outcome_match"] for c in checks.values()),
                "cases": checks,
                "recorded_commit": hidden_report["provenance"]["git"]["commit"],
            }
            return replay_payload(
                stack.model,
                (trajectory_dict(nominal), _episode_outcome(trial["nominal"])),
                (trajectory_dict(attacked), _episode_outcome(trial["attacked"])),
                trial["push"],
                source="re-simulated",
                verification=verification,
                extra={"start_substep": start_substep, "start_s": trial["start_s"]},
            )

        return await wait(jobs.replay(("hidden-phase", start_substep), compute))

    @app.get("/api/presets/live-stress")
    async def live_stress():
        return {
            "preset": {
                "id": live.meta["id"],
                "version": live.meta["version"],
                "title": live.meta["title"],
                "source": live.meta["source"],
                "files": live.meta["files"],
            },
            "config": jobs.live_config.to_dict(),
            "trials": jobs.live_config.trials,
            "reference": {
                "result_origin": "precomputed",
                "name": live_reference["name"],
                "aggregates": live_reference["aggregates"],
                "stance_table": live_reference["stance_table"],
                "provenance": live_reference["provenance"],
                "interpretation": live_reference["interpretation"],
            },
            "runs": jobs.list_runs(),
        }

    @app.post("/api/presets/live-stress/runs", status_code=202)
    async def start_live_run(body: LiveRunRequest | None = None):
        try:
            job = jobs.start_live_run(None if body is None else body.seed)
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"run_id": job.run_id, "seed": job.seed, "total": job.total, "status": job.status}

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str):
        status = jobs.run_status(run_id)
        if status is None:
            raise HTTPException(404, "unknown run")
        if status.get("report"):
            report = status.pop("report")
            status["summary"] = report_summary(report)
            status["timing"] = report["timing"]
        return status

    @app.get("/api/runs/{run_id}/trials/{trial}/replay")
    async def run_replay(run_id: str, trial: int):
        status = jobs.run_status(run_id)
        if status is None or status.get("status") != "done":
            raise HTTPException(404, "unknown or unfinished run")
        report = status["report"]
        if not 0 <= trial < len(report["trials"]):
            raise HTTPException(404, "unknown trial")
        record = report["trials"][trial]
        paths = record.get("trajectories")
        if not paths:
            raise HTTPException(404, "trajectories were not saved for this run")
        run_dir = jobs.run_dir(run_id)

        def compute(worker: Worker) -> dict:
            stack = worker.stack(report["config"]["scene"])
            loaded = {case: read_run(run_dir / paths[case])[0] for case in ("nominal", "attacked")}
            return replay_payload(
                stack.model,
                (loaded["nominal"], _episode_outcome(record["nominal"])),
                (loaded["attacked"], _episode_outcome(record["attacked"])),
                record["push"],
                source="recorded",
                extra={"run_id": run_id, "trial": trial, "start_s": record["start_s"]},
            )

        return await wait(jobs.replay((run_id, trial), compute))

    # ------------------------------------------------------------------ failure discovery

    @app.get("/api/discovery")
    async def discovery_info():
        from robustlab.search.space import bounds_record

        return {
            "bounds": bounds_record(),
            "defaults": {"budget": 64, "shrink_budget": 32},
            "limits": {
                "budget": {"min": 8, "max": 256, "multiple_of": 8},
                "shrink_budget": {"min": 0, "max": 128},
                "timeout_s": settings.discovery_timeout_s,
            },
            "fixed": {
                "command": [1.0, 0.0, 0.0],
                "horizon_s": 5.0,
                "scene": "fall-contact",
                "initial_state": "nominal reset",
            },
            "comparison": {
                "result_origin": "precomputed",
                "preset": {k: comparison.meta[k] for k in ("id", "version", "title", "source")},
                "protocol": comparison_summary["protocol"],
                "aggregate": comparison_summary["aggregate"],
                "paired": comparison_summary["paired"],
                "found_failure_fisher_p_two_sided": comparison_summary[
                    "found_failure_fisher_p_two_sided"
                ],
                "rows": comparison_summary["rows"],
                "git": comparison_summary["git"],
            },
            "runs": jobs.list_discoveries(),
        }

    @app.post("/api/discovery/runs", status_code=202)
    async def start_discovery(body: DiscoveryRequest):
        from robustlab.search.engine import SearchConfigError

        try:
            job = jobs.start_discovery(body.algorithm, body.budget, body.seed, body.shrink_budget)
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except SearchConfigError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"run_id": job.run_id, "request": job.request, "status": job.status}

    @app.get("/api/discovery/runs/{run_id}")
    async def discovery_status(run_id: str):
        status = jobs.discovery_status(run_id)
        if status is None:
            raise HTTPException(404, "unknown discovery run")
        return status

    @app.post("/api/discovery/runs/{run_id}/cancel", status_code=202)
    async def cancel_discovery(run_id: str):
        if jobs.discovery_status(run_id) is None:
            raise HTTPException(404, "unknown discovery run")
        if not jobs.cancel_discovery(run_id):
            raise HTTPException(409, "run is not active")
        return {"run_id": run_id, "cancel_requested": True}

    @app.get("/api/discovery/runs/{run_id}/replay")
    async def discovery_replay(run_id: str, which: Literal["final", "search"] = "final"):
        from robustlab.search.engine import PhysicsEvaluator
        from robustlab.search.space import Candidate

        record = jobs.discovery_record(run_id)
        if record is None:
            raise HTTPException(404, "unknown or unfinished discovery run")
        if which == "final":
            source = record.get("final")
        else:
            source = record["confirmation"]["confirmed"]
        if source is None:
            raise HTTPException(404, "this run has no confirmed failure to replay")
        c = source["candidate"]
        candidate = Candidate(c["force_n"], c["direction_deg"], c["start_substep"],
                              c["duration_substeps"])  # fmt: skip
        recorded_digest = source["digest"]

        def compute(worker: Worker) -> dict:
            evaluator = worker.cache.get("evaluator")
            if evaluator is None:
                evaluator = worker.cache["evaluator"] = PhysicsEvaluator(
                    worker.stack("fall-contact")
                )
            trial, (nominal, _), (attacked, _) = evaluator.paired_trial(candidate)
            match = trial["attacked"]["digest"] == recorded_digest
            verification = {
                "all_match": match and trial["attacked"]["outcome"] == "fell",
                "recorded_commit": record["provenance"]["git"]["commit"],
                "cases": {
                    "attacked": {
                        "recorded_digest": recorded_digest,
                        "resimulated_digest": trial["attacked"]["digest"],
                        "digest_match": match,
                        "recorded_outcome": "fell",
                        "resimulated_outcome": trial["attacked"]["outcome"],
                        "outcome_match": trial["attacked"]["outcome"] == "fell",
                    }
                },
            }
            return replay_payload(
                evaluator.stack.model,
                (trajectory_dict(nominal), _episode_outcome(trial["nominal"])),
                (trajectory_dict(attacked), _episode_outcome(trial["attacked"])),
                trial["push"],
                source="re-simulated",
                verification=verification,
                extra={"run_id": run_id, "which": which, "candidate": source["candidate"],
                       "stance_at_start": trial["stance_at_start"], "start_s": trial["start_s"]},
            )  # fmt: skip

        return await wait(jobs.replay(("discovery", run_id, which), compute))

    # ------------------------------------------------------------------ scenario designer (M5)

    from robustlab.api.scenarios import ScenarioService

    scenarios = ScenarioService(settings, jobs, compiler)
    app.state.scenarios = scenarios

    def _validation_failure(exc) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": {
            "code": "invalid_scenario", "message": "the scenario did not validate",
            "errors": exc.errors}})  # fmt: skip

    @app.get("/api/scenarios/config")
    async def scenarios_config():
        return scenarios.config()

    @app.post("/api/scenarios/validate")
    async def scenarios_validate(body: ScenarioBody):
        from robustlab.scenarios.schema import ScenarioError, validate

        try:
            validated = validate(body.scenario)
        except ScenarioError as exc:
            return {"valid": False, "errors": exc.errors}
        return await wait(jobs.replays.submit(lambda w: scenarios.preview(w, validated)))

    @app.post("/api/scenarios/compile")
    async def scenarios_compile(body: CompileBody, request: Request):
        from robustlab.llm.openrouter import CompileError

        client = request.client.host if request.client else "unknown"
        try:
            result = await asyncio.to_thread(scenarios.compile, body.prompt, client)
        except CompileError as exc:
            return JSONResponse(status_code=exc.status, content={"detail": exc.to_dict()})
        if result.get("valid"):
            from robustlab.scenarios.schema import validate

            validated = validate(result["scenario"])
            preview = await wait(jobs.replays.submit(lambda w: scenarios.preview(w, validated)))
            # The preview validates the normalized document; keep the compile-time adjustments.
            preview["adjustments"] = result["adjustments"]
            result["preview"] = preview
        return result

    @app.post("/api/scenarios/runs", status_code=202)
    async def scenarios_start(body: ScenarioRunBody):
        from robustlab.api.jobs import ScenarioRunError
        from robustlab.experiments.suite import SuiteConfigError
        from robustlab.scenarios.schema import ScenarioError, validate

        try:
            validated = validate(body.scenario)
        except ScenarioError as exc:
            return _validation_failure(exc)
        try:
            origin = scenarios.origin_for(body.compile_id, validated.scenario)
            job = jobs.start_scenario_run(
                validated,
                mode=body.mode,
                trials=body.trials,
                interval_s=body.interval_s,
                seed=body.seed,
                method=body.method,
                origin=origin,
            )
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ScenarioRunError, SuiteConfigError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"run_id": job.run_id, "request": job.request, "status": job.status}

    @app.get("/api/scenarios/runs/{run_id}")
    async def scenarios_status(run_id: str):
        status = jobs.scenario_status(run_id)
        if status is None:
            raise HTTPException(404, "unknown scenario run")
        if status.get("status") == "done" and status["result"]["mode"] == "suite":
            from robustlab.experiments.suite import read_report

            status["summary"] = report_summary(read_report(jobs.scenario_path(run_id)))
        return status

    @app.get("/api/scenarios/runs/{run_id}/replay")
    async def scenarios_replay(
        run_id: str, trial: int = 0, ghost: Literal["primary", "matched", "original"] = "primary"
    ):
        status = jobs.scenario_status(run_id)
        if status is None or status.get("status") != "done":
            raise HTTPException(404, "unknown or unfinished scenario run")
        try:
            plan = scenarios.replay_plan(jobs.scenario_path(run_id), status["result"], trial, ghost)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        return await wait(
            jobs.replay(("scenario", run_id, trial, plan["ghost"]["kind"]), plan["compute"])
        )

    # ------------------------------------------------------------------ policy hardening (M6)

    from robustlab.adapt.harden import HardenConfigError, default_workers
    from robustlab.api.hardening import PRESET_KEY, HardeningService

    hardening = HardeningService(settings, jobs, settings.hardening_wall_clock_s,
                                 settings.hardening_workers or default_workers())  # fmt: skip
    app.state.hardening = hardening

    @app.get("/api/hardening")
    async def hardening_info():
        return hardening.config()

    @app.post("/api/hardening/runs", status_code=202)
    async def start_hardening(body: HardeningRequest | None = None):
        import secrets as _secrets

        body = body or HardeningRequest()
        seed_source = "supplied" if body.seed is not None else "drawn"
        seed = body.seed if body.seed is not None else _secrets.randbelow(2**32)
        try:
            config = hardening.make_config(seed, body.generations, body.focus)
            if body.focus is not None:
                from robustlab.adapt.harden import training_episodes

                training_episodes(hardening.sets, body.focus)
            job = jobs.start_hardening(config, seed_source=seed_source,
                                       wall_clock_s=settings.hardening_wall_clock_s)  # fmt: skip
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except HardenConfigError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"run_id": job.run_id, "request": job.request, "status": job.status}

    @app.get("/api/hardening/runs/{run_id}")
    async def hardening_status(run_id: str):
        status = jobs.hardening_status(run_id)
        if status is None:
            raise HTTPException(404, "unknown hardening run")
        return status

    @app.post("/api/hardening/runs/{run_id}/cancel", status_code=202)
    async def cancel_hardening(run_id: str):
        if jobs.hardening_status(run_id) is None:
            raise HTTPException(404, "unknown hardening run")
        if not jobs.cancel_hardening(run_id):
            raise HTTPException(409, "run is not active")
        return {"run_id": run_id, "cancel_requested": True}

    @app.get("/api/hardening/runs/{run_id}/replay")
    async def hardening_replay(
        run_id: str, episode: str, ghost: Literal["baseline", "matched"] = "baseline"
    ):
        record = hardening.record_for(run_id) if (run_id == PRESET_KEY or
                                                  jobs.hardening_path(run_id)) else None  # fmt: skip
        if record is None:
            raise HTTPException(404, "unknown or unfinished hardening run")
        try:
            compute = hardening.replay_compute(run_id, record, episode, ghost)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        return await wait(jobs.replay(("hardening", run_id, episode, ghost), compute))

    if settings.web_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="web")
    return app
