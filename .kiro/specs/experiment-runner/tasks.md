# Implementation Plan: M2 — Reproducible experiment runner

## Overview

Branch `m2-experiment-runner` from `main` after merging M1 (`m1-baseline` = `fee2cc9`). No new
runtime dependencies; Hypothesis is added to the dev group for property-based tests.

## Tasks

- [x] 1. Preconditions
  - [x] 1.1 Single writer checked; M1 merged into `main` with `--no-ff`; tag `m1-baseline`
  - [x] 1.2 Measure the nominal gait period (0.44 s) to choose the Start_Interval
  - [x] 1.3 Add `hypothesis` to the dev dependency group and lock
- [x] 2. Shared stack
  - [x] 2.1 `experiments/stack.py` (`load_stack`, `Stack.manifest_for`); `walk` uses it
  - _Requirements: 6.1_
- [x] 3. Config and sampling
  - [x] 3.1 `SuiteConfig` loader with strict validation and overrides
  - [x] 3.2 `sample_start_steps` (stratified, grid)
  - [x] 3.3 Tests: invalid configs; Properties 18–19
  - _Requirements: 1.1–1.6, 2.1–2.6, 8.1_
- [x] 4. Running and aggregation
  - [x] 4.1 `run_suite`: paired episodes, parity, digests, stance, trajectories
  - [x] 4.2 `aggregate`, Wilson interval, paired table, interpretation, `results_sha256`
  - [x] 4.3 Tests: Property 17; small-suite pairing, parity, rerun determinism, saved trajectories
    readable, M0 schema-1 compatibility
  - _Requirements: 3.1–3.4, 4.1–4.6, 5.1–5.3, 6.1–6.4, 7.2, 7.3, 8.2_
- [x] 5. CLI and examples
  - [x] 5.1 `robustlab experiment` with summary, `--dry-run`, exit codes
  - [x] 5.2 `examples/m2/` configs: 16-trial stratified, exhaustive 256-point grid
  - [x] 5.3 CLI tests
  - _Requirements: 1.6, 7.1, 7.4_
- [x] 6. Evidence
  - [x] 6.1 Run the 16-trial suite twice and compare digests; run the exhaustive grid
  - [x] 6.2 Integration test on the example config
- [x] 7. Documentation and stop
  - [x] 7.1 Full validation suite
  - [x] 7.2 `docs/milestones/M2.md`, build log, README
  - [x] 7.3 Commit; stop for review before M3
