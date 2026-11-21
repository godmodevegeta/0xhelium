# Implementation Plan: M5 — Multi-axis scenarios and OpenRouter designer

## Tasks

- [x] 1. Preconditions
  - [x] 1.1 Tag `m4-baseline` = `7127526`; merge into `main` (`26ef65a`); branch `m5-openrouter-scenarios`
  - [x] 1.2 Inspect contact friction resolution (priority, condim, pairs, override) and runtime contacts
  - [x] 1.3 Probe MjSpec payload body (ids unchanged, mass +m)
- [x] 2. M5a physics and schema
  - [x] 2.1 `scenarios/schema.py` (Param, validation, adjustments)
  - [x] 2.2 `build_scene` friction/payload modifications; `physics/scenario_model.py` (reports, LRU cache)
  - [x] 2.3 `experiments/scenario_run.py` (single, suite, foot slip, records); `robustlab scenario`
  - [x] 2.4 Hypothesis + integration tests (golden parity, friction contacts, slip, mass, inertia)
  - [x] 2.5 Measurements: nominal, friction sweep, payload sweep, combined; 256-start grids
  - [x] 2.6 M5a checkpoint: full suites pass before M5b
  - _Requirements: 1–5, 8.1, 8.3_
- [ ] 3. M5b OpenRouter and web
  - [ ] 3.1 `llm/openrouter.py`, `llm/prompt.py`; mocked tests; real smoke test if a key exists
  - [ ] 3.2 `/api/scenarios/*` endpoints and tests
  - [ ] 3.3 Scenario Designer tab; replay payload box, floor tint, ghost label
  - [ ] 3.4 End-to-end check in the browser
  - _Requirements: 6, 7, 8.2_
- [ ] 4. Evidence and stop
  - [ ] 4.1 Full validation, Stop hook timing
  - [ ] 4.2 `docs/milestones/M5.md`, README, build log; commit; stop before M6
  - _Requirements: 8.4_
