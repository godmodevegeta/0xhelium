"""Bounded background execution: one experiment worker and one replay worker.

Each worker is a single thread that owns its own `Stack` (the NumPy actor has preallocated buffers,
so a stack is never shared between threads). Job state is in memory; the run directory is the
durable record: `request.json` is written before simulating and `experiment.json` last.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robustlab.experiments.stack import Stack, load_stack
from robustlab.experiments.suite import SuiteConfig, load_suite_config, read_report, run_suite
from robustlab.experiments.trace import _atomic_write_json

RUN_ID = re.compile(r"^ls-\d{8}T\d{6}Z-\d{1,20}$")
DISCOVERY_ID = re.compile(r"^ds-\d{8}T\d{6}Z-(random|cmaes)-\d{1,20}$")
SCENARIO_ID = re.compile(r"^sc-\d{8}T\d{6}Z-[0-9a-f]{8}$")
LIVE_SUBDIR = "live-stress"
DISCOVERY_SUBDIR = "discovery"
SCENARIO_SUBDIR = "scenarios"
SCENARIO_MAX_TRIALS = 32  # trajectories are saved (and replayable) up to this many trials


class BusyError(Exception):
    """An experiment is already queued or running."""


class ScenarioRunError(ValueError):
    """Scenario run parameters that cannot be executed (reported as 422)."""


class Worker:
    """Single-thread executor with a lazily loaded Stack per scene, used only on that thread."""

    def __init__(self, name: str, root: Path, policy: str) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self.root = root
        self.policy = policy
        self._stacks: dict[str, Stack] = {}
        self.cache: dict[str, Any] = {}  # per-thread helpers (e.g. a PhysicsEvaluator)

    def stack(self, scene: str) -> Stack:  # call only from the worker thread
        if scene not in self._stacks:
            self._stacks[scene] = load_stack(self.root, Path(self.policy), scene)
        return self._stacks[scene]

    def submit(self, fn: Callable[[Worker], Any]) -> Future:
        return self.executor.submit(fn, self)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class Job:
    run_id: str
    seed: int
    total: int
    status: str = "queued"  # queued | running | done | failed
    completed: int = 0
    error: str | None = None
    started_utc: str | None = None
    finished_utc: str | None = None
    trials: list = field(default_factory=list)


@dataclass
class DiscoveryJob:
    run_id: str
    request: dict
    status: str = "queued"  # queued | running | done | failed
    stage: str = "queued"  # queued | search | confirm | shrink | writing | finished
    evaluations: list = field(default_factory=list)
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    started_utc: str | None = None
    finished_utc: str | None = None


@dataclass
class ScenarioJob:
    run_id: str
    request: dict
    total: int
    status: str = "queued"  # queued | running | done | failed
    completed: int = 0
    trials: list = field(default_factory=list)
    error: str | None = None
    started_utc: str | None = None
    finished_utc: str | None = None


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class JobManager:
    def __init__(
        self,
        root: Path,
        runs_dir: Path,
        live_suite: Path,
        live_preset_meta: dict,
        discovery_timeout_s: float = 120.0,
    ) -> None:
        self.root = Path(root)
        self.runs_dir = Path(runs_dir) / LIVE_SUBDIR
        self.live_suite = Path(live_suite)
        self.live_config: SuiteConfig = load_suite_config(self.live_suite)
        self.live_preset_meta = live_preset_meta
        self.experiments = Worker("experiment", self.root, self.live_config.policy)
        self.replays = Worker("replay", self.root, self.live_config.policy)
        self.jobs: dict[str, Job] = {}
        self.discoveries: dict[str, DiscoveryJob] = {}
        self.discovery_dir = Path(runs_dir) / DISCOVERY_SUBDIR
        self.discovery_timeout_s = discovery_timeout_s
        self.scenarios: dict[str, ScenarioJob] = {}
        self.scenario_dir = Path(runs_dir) / SCENARIO_SUBDIR
        self._lock = threading.Lock()
        self._replay_cache: OrderedDict[tuple, dict] = OrderedDict()

    def _busy(self) -> bool:
        """One expensive job (live run, discovery or scenario run) at a time per server."""
        active = ("queued", "running")
        return (
            any(j.status in active for j in self.jobs.values())
            or any(d.status in active for d in self.discoveries.values())
            or any(s.status in active for s in self.scenarios.values())
        )

    # ------------------------------------------------------------------ live runs

    def start_live_run(self, seed: int | None) -> Job:
        with self._lock:
            if self._busy():
                raise BusyError("an experiment or discovery is already running")
            seed_source = "supplied" if seed is not None else "drawn"
            if seed is None:
                seed = secrets.randbelow(2**32)
            config = load_suite_config(self.live_suite, seed=seed)
            run_id = f"ls-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{seed}"
            while run_id in self.jobs or (self.runs_dir / run_id).exists():
                time.sleep(1.0)
                run_id = f"ls-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{seed}"
            job = Job(run_id=run_id, seed=seed, total=config.trials)
            self.jobs[run_id] = job
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True)
        _atomic_write_json(
            run_dir / "request.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "preset": {k: self.live_preset_meta[k] for k in ("id", "version", "files")},
                "suite_config": str(self.live_suite.relative_to(self.root))
                if self.live_suite.is_relative_to(self.root)
                else str(self.live_suite),
                "seed": seed,
                "seed_source": seed_source,
                "requested_utc": _utc(),
                "reproduce": self.reproduce_command(seed),
            },
        )

        def work(worker: Worker) -> None:
            job.status, job.started_utc = "running", _utc()
            try:
                stack = worker.stack(config.scene)

                def progress(i: int, trial: dict) -> None:
                    job.completed = i + 1
                    job.trials.append(trial)

                run_suite(config, stack, run_dir, save_trajectories=True, on_trial=progress)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - reported to the client, never swallowed
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                (run_dir / "error.txt").write_text(traceback.format_exc())
            finally:
                job.finished_utc = _utc()

        self.experiments.submit(work)
        return job

    def reproduce_command(self, seed: int) -> str:
        suite = (
            self.live_suite.relative_to(self.root)
            if self.live_suite.is_relative_to(self.root)
            else self.live_suite
        )
        return (
            f"uv run robustlab experiment --config {suite} --seed {seed} "
            f"--out runs/m2/reproduce-{seed}"
        )

    def run_dir(self, run_id: str) -> Path | None:
        if not RUN_ID.match(run_id):
            return None
        path = self.runs_dir / run_id
        return path if path.is_dir() else None

    def run_status(self, run_id: str) -> dict | None:
        path = self.run_dir(run_id)
        if path is None:
            return None
        job = self.jobs.get(run_id)
        request = _read_json(path / "request.json") or {}
        status: dict[str, Any] = {"run_id": run_id, "seed": request.get("seed"),
                                  "seed_source": request.get("seed_source"),
                                  "requested_utc": request.get("requested_utc"),
                                  "reproduce": request.get("reproduce")}  # fmt: skip
        if (path / "experiment.json").exists() and (job is None or job.status == "done"):
            status.update(status="done", report=read_report(path))
        elif job is not None:
            status.update(status=job.status, completed=job.completed, total=job.total,
                          error=job.error, trials=list(job.trials))  # fmt: skip
        else:
            status.update(status="incomplete", error="no experiment.json (server restarted?)")
        return status

    def list_runs(self, limit: int = 20) -> list[dict]:
        if not self.runs_dir.is_dir():
            return []
        ids = sorted(
            (p.name for p in self.runs_dir.iterdir() if RUN_ID.match(p.name)), reverse=True
        )
        out = []
        for run_id in ids[:limit]:
            path = self.runs_dir / run_id
            request = _read_json(path / "request.json") or {}
            entry = {"run_id": run_id, "seed": request.get("seed"),
                     "requested_utc": request.get("requested_utc")}  # fmt: skip
            report = _read_json(path / "experiment.json")
            if report:
                att = report["aggregates"]["attacked"]
                entry.update(status="done", physical_falls=att["physical_falls"],
                             valid_trials=att["valid_trials"],
                             numerical_failures=att["numerical_failures"])  # fmt: skip
            else:
                job = self.jobs.get(run_id)
                entry["status"] = job.status if job else "incomplete"
            out.append(entry)
        return out

    # ------------------------------------------------------------------ discovery

    def start_discovery(
        self, algorithm: str, budget: int, seed: int | None, shrink_budget: int
    ) -> DiscoveryJob:
        from robustlab.search.engine import SearchConfig  # raises SearchConfigError on bad input

        seed_source = "supplied" if seed is not None else "drawn"
        if seed is None:
            seed = secrets.randbelow(2**32)
        config = SearchConfig(algorithm, budget, seed, shrink_budget, self.discovery_timeout_s)
        with self._lock:
            if self._busy():
                raise BusyError("an experiment or discovery is already running")
            run_id = f"ds-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{algorithm}-{seed}"
            while run_id in self.discoveries or (self.discovery_dir / run_id).exists():
                time.sleep(1.0)
                run_id = f"ds-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{algorithm}-{seed}"
            request = {
                "schema_version": 1,
                "run_id": run_id,
                "algorithm": algorithm,
                "budget": budget,
                "seed": seed,
                "seed_source": seed_source,
                "shrink_budget": shrink_budget,
                "timeout_s": self.discovery_timeout_s,
                "requested_utc": _utc(),
                "reproduce": (
                    f"uv run robustlab discover --algorithm {algorithm} --budget {budget} "
                    f"--seed {seed} --shrink-budget {shrink_budget} "
                    f"--timeout {self.discovery_timeout_s:g} --out runs/m4/reproduce-{run_id}"
                ),
            }
            job = DiscoveryJob(run_id=run_id, request=request)
            self.discoveries[run_id] = job
        run_dir = self.discovery_dir / run_id
        run_dir.mkdir(parents=True)
        _atomic_write_json(run_dir / "request.json", request)

        def work(worker: Worker) -> None:
            from robustlab.search.engine import run_discovery
            from robustlab.search.report import write_discovery

            job.status, job.started_utc = "running", _utc()
            try:
                stack = worker.stack("fall-contact")

                def stage(name: str) -> None:
                    job.stage = name

                record = run_discovery(
                    config,
                    stack,
                    cancelled=job.cancel.is_set,
                    on_eval=job.evaluations.append,
                    on_stage=stage,
                )
                record["request"] = request
                job.stage = "writing"
                write_discovery(run_dir, record)
                job.status, job.stage = "done", "finished"
            except Exception as exc:  # noqa: BLE001 - reported to the client
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                (run_dir / "error.txt").write_text(traceback.format_exc())
            finally:
                job.finished_utc = _utc()

        self.experiments.submit(work)
        return job

    def discovery_path(self, run_id: str) -> Path | None:
        if not DISCOVERY_ID.match(run_id):
            return None
        path = self.discovery_dir / run_id
        return path if path.is_dir() else None

    def discovery_record(self, run_id: str) -> dict | None:
        path = self.discovery_path(run_id)
        return None if path is None else _read_json(path / "discovery.json")

    def discovery_status(self, run_id: str) -> dict | None:
        path = self.discovery_path(run_id)
        if path is None:
            return None
        job = self.discoveries.get(run_id)
        request = (job.request if job else None) or _read_json(path / "request.json") or {}
        status: dict[str, Any] = {"run_id": run_id, "request": request}
        record = _read_json(path / "discovery.json")
        if record is not None and (job is None or job.status == "done"):
            status.update(status="done", stage="finished", record=record)
        elif job is not None:
            status.update(
                status=job.status,
                stage=job.stage,
                evaluations=list(job.evaluations),
                completed=len(job.evaluations),
                budget=request.get("budget"),
                cancel_requested=job.cancel.is_set(),
                error=job.error,
            )
        else:
            status.update(status="incomplete", error="no discovery.json (server restarted?)")
        return status

    def cancel_discovery(self, run_id: str) -> bool:
        job = self.discoveries.get(run_id)
        if job is None or job.status not in ("queued", "running"):
            return False
        job.cancel.set()
        return True

    def list_discoveries(self, limit: int = 20) -> list[dict]:
        if not self.discovery_dir.is_dir():
            return []
        ids = sorted(
            (p.name for p in self.discovery_dir.iterdir() if DISCOVERY_ID.match(p.name)),
            reverse=True,
        )
        out = []
        for run_id in ids[:limit]:
            path = self.discovery_dir / run_id
            request = _read_json(path / "request.json") or {}
            entry = {k: request.get(k) for k in ("algorithm", "budget", "seed", "requested_utc")}
            entry["run_id"] = run_id
            record = _read_json(path / "discovery.json")
            if record:
                final = record.get("final")
                entry.update(
                    status="done",
                    stop_reason=record["search"]["stop_reason"],
                    found_failure=record["search"]["found_failure"],
                    final_impulse_ns=final["candidate"]["impulse_ns"] if final else None,
                    final_source=final["source"] if final else None,
                )
            else:
                job = self.discoveries.get(run_id)
                entry["status"] = job.status if job else "incomplete"
            out.append(entry)
        return out

    # ------------------------------------------------------------------ scenario runs (M5)

    def start_scenario_run(
        self,
        validated,
        *,
        mode: str,
        trials: int = 16,
        interval_s: tuple[float, float] = (2.0, 3.024),
        seed: int | None = None,
        method: str = "stratified",
        origin: dict | None = None,
    ) -> ScenarioJob:
        """Queue a validated fixed scenario: `single` (one deterministic paired run) or `suite`
        (push start varied over `trials`). Raises ScenarioRunError / SuiteConfigError on bad
        parameters before anything is written, BusyError if another job is active."""
        from robustlab.experiments.scenario_run import (
            run_scenario_suite,
            run_single,
            suite_config_for,
        )

        scenario = validated.scenario
        seed_source = None
        if mode == "suite":
            if scenario.push is None:
                raise ScenarioRunError(
                    "a suite needs a push: without one every trial is the same deterministic "
                    "episode, so repeating it would not give independent trials"
                )
            if not 2 <= trials <= SCENARIO_MAX_TRIALS:
                raise ScenarioRunError(f"trials must be in [2, {SCENARIO_MAX_TRIALS}]")
            seed_source = "supplied" if seed is not None else "drawn"
            if seed is None:
                seed = secrets.randbelow(2**32)
            # Validates interval, method and timing against the horizon (SuiteConfigError).
            suite_config_for(scenario, self.live_config.policy, trials=trials,
                             interval_s=interval_s, seed=seed, method=method)  # fmt: skip
        elif mode != "single":
            raise ScenarioRunError("mode must be 'single' or 'suite'")
        with self._lock:
            if self._busy():
                raise BusyError("an experiment, discovery or scenario run is already running")
            run_id = f"sc-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{secrets.token_hex(4)}"
            while run_id in self.scenarios or (self.scenario_dir / run_id).exists():
                run_id = (
                    f"sc-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{secrets.token_hex(4)}"
                )
            run_dir = self.scenario_dir / run_id
            scenario_file = run_dir / "scenario.json"
            shown = (
                scenario_file.relative_to(self.root)
                if scenario_file.is_relative_to(self.root)
                else scenario_file
            )
            reproduce = f"uv run robustlab scenario --file {shown} --out runs/m5/reproduce-{run_id}"
            if mode == "suite":
                reproduce += (
                    f" --suite --trials {trials} --interval {interval_s[0]:g},{interval_s[1]:g}"
                    f" --method {method} --seed {seed}"
                )
            request = {
                "schema_version": 1,
                "run_id": run_id,
                "mode": mode,
                "scenario": scenario.to_dict(),
                "scenario_sha256": scenario.sha256(),
                "adjustments": validated.adjustments,
                "origin": origin or {"kind": "manual"},
                "suite": None
                if mode == "single"
                else {
                    "trials": trials,
                    "interval_s": list(interval_s),
                    "method": method,
                    "seed": seed,
                    "seed_source": seed_source,
                },
                "requested_utc": _utc(),
                "reproduce": reproduce,
            }
            job = ScenarioJob(run_id=run_id, request=request,
                              total=trials if mode == "suite" else 1)  # fmt: skip
            self.scenarios[run_id] = job
        run_dir.mkdir(parents=True)
        _atomic_write_json(scenario_file, scenario.to_dict())
        _atomic_write_json(run_dir / "request.json", request)

        def work(worker: Worker) -> None:
            job.status, job.started_utc = "running", _utc()
            try:
                stack = worker.stack(scenario.scene)
                if mode == "single":
                    run_single(scenario, stack, run_dir, adjustments=validated.adjustments)
                    job.completed = 1
                else:

                    def progress(i: int, trial: dict) -> None:
                        job.completed = i + 1
                        job.trials.append(trial)

                    run_scenario_suite(scenario, stack, run_dir, trials=trials,
                                       interval_s=interval_s, seed=seed, method=method,
                                       adjustments=validated.adjustments,
                                       on_trial=progress)  # fmt: skip
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - reported to the client
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                (run_dir / "error.txt").write_text(traceback.format_exc())
            finally:
                job.finished_utc = _utc()

        self.experiments.submit(work)
        return job

    def scenario_path(self, run_id: str) -> Path | None:
        if not SCENARIO_ID.match(run_id):
            return None
        path = self.scenario_dir / run_id
        return path if path.is_dir() else None

    def scenario_status(self, run_id: str) -> dict | None:
        path = self.scenario_path(run_id)
        if path is None:
            return None
        job = self.scenarios.get(run_id)
        request = (job.request if job else None) or _read_json(path / "request.json") or {}
        status: dict[str, Any] = {"run_id": run_id, "request": request}
        result = _read_json(path / "scenario_result.json")
        if result is not None and (job is None or job.status == "done"):
            status.update(status="done", result=result)
        elif job is not None:
            status.update(status=job.status, completed=job.completed, total=job.total,
                          trials=list(job.trials), error=job.error)  # fmt: skip
        else:
            status.update(status="incomplete", error="no scenario_result.json (server restarted?)")
        return status

    def list_scenarios(self, limit: int = 20) -> list[dict]:
        if not self.scenario_dir.is_dir():
            return []
        ids = sorted(
            (p.name for p in self.scenario_dir.iterdir() if SCENARIO_ID.match(p.name)),
            reverse=True,
        )
        out = []
        for run_id in ids[:limit]:
            path = self.scenario_dir / run_id
            request = _read_json(path / "request.json") or {}
            entry = {"run_id": run_id, "mode": request.get("mode"),
                     "name": (request.get("scenario") or {}).get("name"),
                     "origin": (request.get("origin") or {}).get("kind"),
                     "requested_utc": request.get("requested_utc")}  # fmt: skip
            result = _read_json(path / "scenario_result.json")
            if result is None:
                job = self.scenarios.get(run_id)
                entry["status"] = job.status if job else "incomplete"
            elif result["mode"] == "single":
                case = result["cases"]["scenario"]
                entry.update(status="done", outcome=case["outcome"], reason=case["reason"])
            else:
                att = result["aggregates"]["attacked"]
                entry.update(status="done", physical_falls=att["physical_falls"],
                             valid_trials=att["valid_trials"],
                             numerical_failures=att["numerical_failures"])  # fmt: skip
            out.append(entry)
        return out

    # ------------------------------------------------------------------ replays

    def replay(self, key: tuple, compute: Callable[[Worker], dict]) -> Future:
        """Run `compute` on the replay worker, caching up to 32 results by `key`."""
        with self._lock:
            if key in self._replay_cache:
                self._replay_cache.move_to_end(key)
                done: Future = Future()
                done.set_result(self._replay_cache[key])
                return done

        def run(worker: Worker) -> dict:
            result = compute(worker)
            with self._lock:
                self._replay_cache[key] = result
                while len(self._replay_cache) > 32:
                    self._replay_cache.popitem(last=False)
            return result

        return self.replays.submit(run)

    def shutdown(self) -> None:
        self.experiments.shutdown()
        self.replays.shutdown()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
