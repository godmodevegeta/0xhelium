# Requirements Document

## Introduction

Milestone M3 of `delivery.md`: a first web application with 3D replays of recorded MuJoCo
trajectories and two scenario presets backed by M2 experiments. The browser never simulates
physics; it plays back poses computed by the native MuJoCo stack. All experiment logic stays in
`robustlab.experiments` (M2 runner, M1 fall monitor and push). The backend is a thin FastAPI layer;
the frontend is React + TypeScript + Vite + Three.js.

Preset selection evidence (bounded force sweep through the M2 runner, `scripts/force_sweep.py`,
same duration, command, scene and timing distribution as M2):

| Force | Exhaustive 256-start frequency | 16-trial stratified, seed 20261121 |
| --- | --- | --- |
| 100 N (M2) | 11/256 = 4.3% | 0/16 |
| 110 N | 32/256 = 12.5% | 3/16 |
| 120 N | 45/256 = 17.6% | 3/16 |
| 130 N | 53/256 = 20.7% | 3/16 |
| 140 N | 56/256 = 21.9% | 3/16 |
| 150 N | 65/256 = 25.4% | 3/16 |

150 N is the only tested force inside the 25–75% band and is the upper bound of the M1 push
envelope, so the Live Stress Test uses 150 N. No force above 150 N was tested.

Out of scope: browser physics, WASM, LLM features, search, training, adapters, friction, payload,
deployment hosting.

## Glossary

- **Backend**: `robustlab.api` (FastAPI), started with `robustlab serve`.
- **Viewer**: the React/Three.js single-page app in `web/`.
- **Preset**: a versioned directory under `presets/` with a manifest listing its files and SHA-256.
- **Hidden_Phase_Preset**: the M2 exhaustive experiment (100 N, 0.1 s, 1 m/s, 256 start times in
  [2.000, 3.024) s), served from its committed report.
- **Live_Stress_Preset**: the same experiment at 150 N with 16 stratified trials, executed on request,
  plus the committed 150 N exhaustive report as a labelled reference.
- **Replay**: per-frame world poses of the 13 visual geoms for a nominal and an attacked episode,
  computed by MuJoCo forward kinematics from recorded `qpos`, with per-Tick tilt, height, contacts
  and the push window.
- **Precomputed**: a result read from a committed fixture or an earlier run directory.
- **Executed_Now**: a simulation run by the Backend in response to the current request.

## Requirements

### Requirement 1: Presets and fixtures

1. THE Repository SHALL contain `presets/hidden-phase-v1/` and `presets/live-stress-v1/`, each with
   a `preset.json` naming every file, its SHA-256, its source run and git commit.
2. WHEN the Backend starts or serves a preset, THE Backend SHALL verify the preset files against
   `preset.json` and refuse to serve a preset whose hashes do not match.
3. THE Live_Stress_Preset suite config SHALL equal the M2 stratified example except
   `force_n = 150`, `name` and `description`.

### Requirement 2: Hidden_Phase_Preset

1. THE Backend SHALL serve the 256 trials of the committed report (start time, stance, outcome, fall
   reason and time, time after shove, first non-foot body, events) without simulating.
2. THE Viewer SHALL show the measured 11/256 failure frequency and the full 256-point sweep, and let
   the user select any tested timing and inspect its recorded outcome.
3. THE Viewer SHALL offer a prominent "Replay a failure" action using a failing timing whose replay
   a test re-simulates and checks.
4. THE Backend SHALL compute, from the report, outcome counts per stance pattern and the Viewer SHALL
   explain that all failures occurred in FR+RL stance but only 11 of the 110 FR+RL timings fell.
5. WHEN a timing is replayed, THE Backend SHALL re-simulate that one paired trial with the M2 runner,
   compare both trajectory digests and outcomes with the report, and return the comparison.
6. THE Viewer SHALL label recorded outcomes as precomputed (with source commit) and replays as
   re-simulated now, showing whether the digests matched.

### Requirement 3: Live_Stress_Preset

1. WHEN the user presses "Run 16 Trials", THE Backend SHALL start a real 16-trial stratified run of
   the preset config through `run_suite` in a background worker and return a run id immediately.
2. THE Backend SHALL draw a fresh seed for each run unless one is supplied, and record it.
3. WHILE a run is in progress, THE Backend SHALL report the number of completed trials, and THE
   Viewer SHALL show progress.
4. IF a run is already in progress, THEN THE Backend SHALL refuse a new run with HTTP 409.
5. WHEN the run completes, THE Viewer SHALL show each trial coloured green (survived), red (physical
   fall) or a separate colour (numerical failure), the falls over valid trials, the Wilson interval,
   and the reproduce command; each trial SHALL open its recorded replay.
6. THE Backend SHALL persist each run under `runs/web/live-stress/<run_id>/` with `request.json`,
   `experiment.json` and per-trial trajectories in the M1 format.
7. THE Viewer SHALL show the committed 150 N exhaustive frequency (65/256) as a precomputed
   reference, separate from live results.

### Requirement 4: Replay

1. THE Backend SHALL compute replay poses from recorded `qpos` with `mj_kinematics` on the scene
   model, for frame 0 (the reset state) and every Tick.
2. THE Backend SHALL serve the compiled visual meshes (vertices and faces) so the poses and meshes
   share MuJoCo's frames.
3. THE Viewer SHALL show the nominal and attacked robots in one scene with one synchronized
   timeline (play, pause, scrub, speed), the shove as an arrow during its active interval, and
   timeline markers for shove start and end, first fall, and every recorded fall event.
4. THE Viewer SHALL show the current time, attacked tilt and trunk height, and non-foot contact.

### Requirement 5: API robustness

1. THE Backend SHALL validate run ids against a fixed pattern and trial indices against the run, and
   return 404 otherwise, without building paths from unvalidated input.
2. THE Backend SHALL run simulations off the event loop with at most one experiment and one replay
   simulation at a time.
3. THE Backend SHALL report failures as JSON errors and the Viewer SHALL display them.
4. THE Backend SHALL serve the built Viewer from `web/dist` when present.

### Requirement 6: Tests and evidence

1. THE test suite SHALL cover preset integrity, hidden-phase summary and stance counts, re-simulated
   replay digest match for the failing timing, replay geometry against direct MuJoCo kinematics, a
   live run end to end (small config), the 409 guard, id validation and fixture tampering.
2. THE Viewer SHALL pass `tsc --noEmit` and `vite build`.
3. THE `docs/milestones/M3.md` SHALL record the force sweep, measured live runs, latency of the
   main endpoints, test output and limitations.
