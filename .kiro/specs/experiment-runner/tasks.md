# Implementation Plan: M2 — Reproducible experiment runner

## Overview

Branch `m2-experiment-runner` from `main` after merging M1 (`m1-baseline` = `fee2cc9`). No new
runtime dependencies; Hypothesis is added to the dev group for property-based tests.

## Tasks

- [x] 1. Preconditions
  - [x] 1.1 Single writer checked; M1 merged into `main` with `--no-ff`; tag `m1-baseline`
  - [x] 1.2 Measure the nominal gait period (0.44 s) to choose the Start_Interval
  - [x] 1.3 Add `hypothesis` to the dev dependency group and lock
- [ ] 2. Shared stack
  - [ ] 2.1 `experiments/stack.py` (`load_stack`, `Stack.manifest_for`); `walk` uses it
  - _Requirements: 6.1_
- [ ] 3. Config and sampling
  - [ ] 3.1 `SuiteConfig` loader with strict validation and overrides
  - [ ] 3.2 `sample_start_steps` (stratified, grid)
  - [ ] 3.3 Tests: invalid configs; Properties 18–19
  - _Requirements: 1.1–1.6, 2.1–2.6, 8.1_
- [ ] 4. Running and aggregation
  - [ ] 4.1 `run_suite`: paired episodes, parity, digests, stance, trajectories
  - [ ] 4.2 `aggregate`, Wilson interval, paired table, interpretation, `results_sha256`
  - [ ] 4.3 Tests: Property 17; small-suite pairing, parity, rerun determinism, saved trajectories
    readable, M0 schema-1 compatibility
  - _Requirements: 3.1–3.4, 4.1–4.6, 5.1–5.3, 6.1–6.4, 7.2, 7.3, 8.2_
- [ ] 5. CLI and examples
  - [ ] 5.1 `robustlab experiment` with summary, `--dry-run`, exit codes
  - [ ] 5.2 `examples/m2/` configs: 16-trial stratified, exhaustive 256-point grid
  - [ ] 5.3 CLI tests
  - _Requirements: 1.6, 7.1, 7.4_
- [ ] 6. Evidence
  - [ ] 6.1 Run the 16-trial suite twice and compare digests; run the exhaustive grid
  - [ ] 6.2 Integration test on the example config
- [ ] 7. Documentation and stop
  - [ ] 7.1 Full validation suite
  - [ ] 7.2 `docs/milestones/M2.md`, build log, README
  - [ ] 7.3 Commit; stop for review before M3
