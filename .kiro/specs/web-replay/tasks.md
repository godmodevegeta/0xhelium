# Implementation Plan: M3 — Web application with 3D replays

## Tasks

- [x] 1. Preconditions and preset selection
  - [x] 1.1 M2 merged into `main` (`451fef8`, tag `m2-baseline` = `64165ea`); branch `m3-web-replay`
  - [x] 1.2 Force sweep 110–150 N (exhaustive + 16-trial) with `scripts/force_sweep.py`; choose 150 N
  - [x] 1.3 Add `fastapi` + `uvicorn` as the `server` extra and `httpx` to the dev group
- [ ] 2. Core support
  - [ ] 2.1 Factor `run_trial` out of `run_suite`; add `on_trial` progress callback
  - [ ] 2.2 `robustlab/replay.py`: kinematic frames, mesh export
  - _Requirements: 2.5, 4.1, 4.2_
- [ ] 3. Presets
  - [ ] 3.1 `presets/hidden-phase-v1`, `presets/live-stress-v1` with `preset.json` hashes
  - [ ] 3.2 `robustlab/api/presets.py`: verification, summaries, stance table, verified failure
  - _Requirements: 1.1–1.3, 2.1, 2.4_
- [ ] 4. Backend
  - [ ] 4.1 `robustlab/api/jobs.py` and `robustlab/api/app.py`; `robustlab serve`
  - [ ] 4.2 API tests
  - _Requirements: 2.5, 3.1–3.6, 5.1–5.4, 6.1_
- [ ] 5. Viewer
  - [ ] 5.1 `web/` scaffold (React 18, TS 5.6, Vite 6, Three)
  - [ ] 5.2 Hidden Phase tab, Live Stress tab, ReplayViewer
  - [ ] 5.3 `tsc --noEmit`, `vite build`, served by the Backend
  - _Requirements: 2.2, 2.3, 2.6, 3.5, 3.7, 4.3, 4.4, 6.2_
- [ ] 6. Evidence and stop
  - [ ] 6.1 Live runs through the API, endpoint latency, screenshot if a headless browser is available
  - [ ] 6.2 `docs/milestones/M3.md`, README, build log; commit; stop before M4
  - _Requirements: 6.3_
