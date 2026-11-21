# Implementation Plan: M4 — Failure discovery and shrinking

## Tasks

- [x] 1. Preconditions
  - [x] 1.1 Tag `m3-baseline` = `deeb350`; merge `m3-web-replay` (incl. playback fix `e455db2`) into
    `main` (`8bb69f6`); branch `m4-failure-discovery`
  - [x] 1.2 Add pycma 4.6.0 (BSD-3-Clause) to the `search`/`server` extras and dev group; check
    private-generator reproducibility
- [ ] 2. Core search
  - [ ] 2.1 `search/space.py`, `search/objective.py`
  - [ ] 2.2 `search/engine.py` (evaluator, random, CMA-ES, confirmation, record)
  - [ ] 2.3 `search/shrink.py`
  - [ ] 2.4 Hypothesis tests (Properties 20–27) with fake physics
  - _Requirements: 1–5, 8.1_
- [ ] 3. CLI, comparison and evidence
  - [ ] 3.1 `robustlab discover`; scenario export; integration test (replay digest)
  - [ ] 3.2 `scripts/compare_search.py`; matched runs; commit comparison fixture
  - _Requirements: 5.3, 6, 8.2_
- [ ] 4. Web
  - [ ] 4.1 API endpoints on the JobManager; busy guard; cancel; timeout; tests
  - [ ] 4.2 Discovery tab; build; browser check
  - _Requirements: 7_
- [ ] 5. Evidence and stop
  - [ ] 5.1 Full validation incl. Stop-hook timing
  - [ ] 5.2 `docs/milestones/M4.md`, README, build log; commit; stop before M5
