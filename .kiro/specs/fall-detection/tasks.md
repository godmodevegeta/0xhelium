# Implementation Plan: M1 — Deterministic shove and authoritative fall detection

## Overview

Sequential tasks on branch `m1-fall-detection` from tag `m0-baseline` (`cfe4b13`). Python 3.12,
NumPy, `mujoco==3.15.0`, pytest, ruff. No new dependencies.

## Tasks

- [x] 1. Preconditions and baseline
  - [x] 1.1 Confirm a single writer: clean tree, no new commits; work on branch `m1-fall-detection`;
    tag `m0-baseline`
  - [x] 1.2 Commit Golden_Trajectories `tests/fixtures/m0_golden.{npz,json}` from `runs/m0/forward`
    and `runs/m0/zero`, after checking their SHA-256 against `M0.md`
    - _Requirements: 1.1, 1.2_
  - [x] 1.3 Probe collision variants against M0 (all non-foot geoms vs. lower shin excluded); record
    divergence in the design
    - _Requirements: 1.4, 2.1_
- [x] 2. Scene_Builder
  - [x] 2.1 `src/robustlab/physics/scene.py`: `build_scene`, `SceneInfo`, transform hash
  - [x] 2.2 Tests: only allowed model arrays differ; geom counts; `check_model` passes; robot-robot
    collisions stay off; feet-only equals `load_model`
  - _Requirements: 2.1–2.5_
- [x] 3. Fall_Monitor and FallResult
  - [x] 3.1 `src/robustlab/physics/fall.py`; share tilt/height predicates with `upright_checks`
  - [x] 3.2 Tests: Properties 15–16; trunk lowered onto the floor gives `nonfoot_contact` naming
    `trunk`; feet-only scene reports detection unavailable; NaN gives `numerical`
  - _Requirements: 3.1–3.4, 4.1–4.5_
- [x] 4. PushScenario and runner integration
  - [x] 4.1 `src/robustlab/experiments/push.py`
  - [x] 4.2 `walk.py`: substep loop, push window, monitor calls, impulse and non-foot flags,
    truncation, `evaluate` fall integration
  - [x] 4.3 Tests: Properties 12–14; zero force / zero duration equal nominal; reset clears xfrc;
    horizon check
  - _Requirements: 3.5, 4.6–4.8, 5.1–5.3, 6.1–6.5_
- [x] 5. Trace and CLI
  - [x] 5.1 Schema 2 trajectory and manifest; `read_run` accepts 1 and 2
  - [x] 5.2 `walk --scene` and push flags; summary prints the FallResult
  - [x] 5.3 Tests: CLI flag validation, manifest keys, round trip
  - _Requirements: 5.4, 7.1–7.3_
- [x] 6. Integration evidence
  - [x] 6.1 Golden reproduction in both scenes; nominal walking criteria in the Fall_Contact_Scene;
    sideways shove determinism and real FallResult
  - [x] 6.2 Run the two experiments (nominal, sideways shove) and a small shove-magnitude probe;
    latency
  - _Requirements: 1.3, 1.4, 7.4_
- [x] 7. Evidence and stop
  - [x] 7.1 Full validation suite
  - [x] 7.2 `docs/milestones/M1.md`, build log, README, ADR 0002
  - [x] 7.3 Commit and stop for review before M2
  - _Requirements: 1.5, 7.5, 7.6_

## Notes

- Property tests use seeded NumPy generation (≥ 100 cases) and carry
  `# Feature: fall-detection, Property N: <title>`.
- Integration tests use the `integration` marker, as in M0.
