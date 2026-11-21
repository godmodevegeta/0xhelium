# Implementation Plan: M3 — Web application with 3D replays

## Tasks

- [x] 1. Preconditions and preset selection
  - [x] 1.1 M2 merged into `main` (`451fef8`, tag `m2-baseline` = `64165ea`); branch `m3-web-replay`
  - [x] 1.2 Force sweep 110–150 N (exhaustive + 16-trial) with `scripts/force_sweep.py`; choose 150 N
  - [x] 1.3 Add `fastapi` + `uvicorn` as the `server` extra and `httpx` to the dev group
- [x] 2. Core support
  - [x] 2.1 Factor `run_trial` out of `run_suite`; add `on_trial` progress callback
  - [x] 2.2 `robustlab/replay.py`: kinematic frames, mesh export
  - _Requirements: 2.5, 4.1, 4.2_
- [x] 3. Presets
  - [x] 3.1 `presets/hidden-phase-v1`, `presets/live-stress-v1` with `preset.json` hashes
  - [x] 3.2 `robustlab/api/presets.py`: verification, summaries, stance table, verified failure
  - _Requirements: 1.1–1.3, 2.1, 2.4_
- [x] 4. Backend
  - [x] 4.1 `robustlab/api/jobs.py` and `robustlab/api/app.py`; `robustlab serve`
  - [x] 4.2 API tests
  - _Requirements: 2.5, 3.1–3.6, 5.1–5.4, 6.1_
- [x] 5. Viewer
  - [x] 5.1 `web/` scaffold (React 18, TS 5.6, Vite 6, Three)
  - [x] 5.2 Hidden Phase tab, Live Stress tab, ReplayViewer
  - [x] 5.3 `tsc --noEmit`, `vite build`, served by the Backend
  - _Requirements: 2.2, 2.3, 2.6, 3.5, 3.7, 4.3, 4.4, 6.2_
- [x] 6. Evidence and stop
  - [x] 6.1 Live runs through the API, endpoint latency, screenshot if a headless browser is available
  - [x] 6.2 `docs/milestones/M3.md`, README, build log; commit; stop before M4
  - _Requirements: 6.3_
