# Implementation Plan: M6 — CPU controller hardening and simulation-first workspace

## Tasks

- [x] 1. Preconditions
  - [x] 1.1 Tag `m5-baseline` = `dd68ed6`; merge into `main` (`9f2839e`); branch `m6-cpu-hardening`
  - [x] 1.2 Probe action/ctrl bounds, joint axes, kinematic signs, nominal signal ranges, shove timing
- [x] 2. M6a controller and feasibility
  - [x] 2.1 `adapt/reflex.py`: parameters, bases, calibration, ReflexActor; runner reset/stop hooks
  - [x] 2.2 Calibration from nominal episodes; stored calibration file with hash
  - [x] 2.3 `adapt/sets.py` + `examples/m6/hardening-sets-v1.json` (train, validation preview, locked)
  - [x] 2.4 `adapt/evaluate.py` metrics/objective, process pool; `adapt/harden.py` CMA-ES + record
  - [x] 2.5 Property and unit tests; integration tests
  - [x] 2.6 Feasibility measurements: budget, wall time, training vs validation results
  - _Requirements: 1–5_
- [x] 3. M6b integration
  - [x] 3.1 `robustlab harden` CLI; reproducibility check
  - [x] 3.2 Hardening API on the JobManager; replay baseline vs candidate; API tests
  - [x] 3.3 Workspace layout refactor; shared viewer; sample replay; Policy Hardening mode
  - [x] 3.4 Browser screenshots 1440×900, 1280×800, 390×844; interaction checks
  - _Requirements: 6–7_
- [x] 4. Evidence and stop
  - [x] 4.1 Full validation; Stop hook timing
  - [x] 4.2 `docs/milestones/M6.md`, README, build log; commit; stop before M7
