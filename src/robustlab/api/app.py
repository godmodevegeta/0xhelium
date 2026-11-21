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
        )


class LiveRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int | None = Field(default=None, ge=0, lt=2**64)


def _episode_outcome(summary: dict) -> dict:
    keys = ("outcome", "reason", "fall_time_s", "numerical_time_s", "time_after_start_s",
            "events", "first_nonfoot_body", "forward_progress_m")  # fmt: skip
    return {k: summary.get(k) for k in keys}


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings.from_env()
    hidden = load_preset(settings.root, HIDDEN_PHASE_DIR)
    live = load_preset(settings.root, LIVE_STRESS_DIR)
    hidden_report = hidden.json("experiment.json")
    hidden_view = hidden_phase_summary(hidden, hidden_report)
    hidden_trials = {t["start_substep"]: t for t in hidden_report["trials"]}
    hidden_config = load_suite_config(hidden_report["config"])
    live_reference = report_summary(live.json("reference-grid-256.json"))
    jobs = JobManager(settings.root, settings.runs_dir, settings.live_suite, live.meta)
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

    if settings.web_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="web")
    return app
