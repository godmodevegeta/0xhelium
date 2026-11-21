# Design Document: M3 — Web application with 3D replays

## Overview

```text
web/ (React + Three.js)  --HTTP JSON-->  robustlab.api.app (FastAPI)
                                             |  presets.py  (fixture verify, summaries, stance stats)
                                             |  jobs.py     (1 experiment worker, 1 replay worker)
                                             |  replay.py   (mj_kinematics poses, mesh export)
                                             v
                         robustlab.experiments: suite.run_suite / run_trial, stack.load_stack
                                             v
                                M1 EpisodeRunner + FallMonitor + PushScenario (native MuJoCo)
```

The browser receives poses, never physics state to integrate. The only simulation paths are
`run_suite` (live runs) and `run_trial` (single-timing re-simulation); `run_trial` is factored out
of `run_suite` so both use the same code.

## Presets

```text
presets/hidden-phase-v1/preset.json        {id, version, title, files: {name: sha256}, source}
presets/hidden-phase-v1/experiment.json    copy of runs/m2/grid-all-256/experiment.json (0b57e70)
presets/live-stress-v1/preset.json
presets/live-stress-v1/suite.json          M2 stratified config with force_n 150
presets/live-stress-v1/reference-grid-256.json  150 N exhaustive report (force sweep, c682baa)
```

Verified failing timing for "Replay a failure": substep 500 (2.000 s), the M1 demonstration
timing; the integration test re-simulates it and requires matching digests.

## API

| Method | Path | Result |
| --- | --- | --- |
| GET | `/api/health` | versions, preset status |
| GET | `/api/model` | visual geoms (body, mesh) and meshes (base64 float32 vertices, int32 faces) |
| GET | `/api/presets/hidden-phase` | provenance, aggregates, stance table, 256 compact trials, verified failure |
| GET | `/api/presets/hidden-phase/replay/{start_substep}` | re-simulated replay + verification |
| GET | `/api/presets/live-stress` | config, reference (150 N grid) summary, latest runs |
| POST | `/api/presets/live-stress/runs` `{seed?}` | `{run_id}` (202), 409 if busy |
| GET | `/api/runs/{run_id}` | status, progress, summary when done |
| GET | `/api/runs/{run_id}/trials/{i}/replay` | replay from the run's saved trajectories |

Run id: `ls-YYYYMMDDTHHMMSSZ-<seed>`; pattern `^ls-\d{8}T\d{6}Z-\d{1,20}$`.

Replay payload: `{fps, frames, time[F], cases: {nominal|attacked: {pos: b64 f32 [F,G,3],
quat: b64 f32 [F,G,4] (wxyz), up_z[F], trunk_z[F], nonfoot[F], feet[F][4], outcome, events}},
push: {start_s, end_s, world_force_n}, source: "re-simulated"|"recorded", verification?}`. Frame 0
is the reset pose (`initial_qpos`), frame k ≥ 1 is Tick k−1.

## Jobs

`JobManager` holds two single-thread executors (experiment, replay), each with its own `Stack`
(the Actor has preallocated buffers, so stacks are not shared across threads). A job record
`{status: queued|running|done|failed, completed, total, error}` lives in memory; the run directory
on disk is the durable record (`request.json` first, `experiment.json` last). After a restart a
directory without `experiment.json` is reported `incomplete`. `run_suite` gains an optional
`on_trial(i, trial)` callback for progress.

## Viewer

- `App`: two tabs (Hidden Phase, Live Stress) and a shared `ReplayPanel`.
- `SweepChart`: 256 bars coloured by outcome over start time, stance band underneath; click selects.
- `TrialDetail`: recorded outcome, labelled "precomputed (M2, commit …)".
- `StressPanel`: Run 16 Trials button, progress bar, 16 coloured trial chips, counts, Wilson
  interval, seed and reproduce command; reference 65/256 card labelled precomputed.
- `ReplayViewer`: Three.js scene (z-up), floor grid, attacked robot opaque, nominal robot
  translucent; per-frame geom transforms; push arrow during `[start, end)`; camera target follows the
  two trunks; OrbitControls. Timeline with play/pause, speed and markers.

Colours: survived `#2e9d5b`, fell `#d64545`, numerical `#7b5cd6`, not run `#c9ced6`.

## Testing

Python: FastAPI `TestClient` tests with a small live config injected through settings (1 s horizon,
4 trials) so runs finish in under a second; integration test for the real failing-timing replay.
Frontend: `tsc --noEmit` and `vite build`; the built app served by the Backend.
