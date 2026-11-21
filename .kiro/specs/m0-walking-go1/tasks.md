# Implementation Plan: M0 — Pretrained Go1 walking in native MuJoCo

## Overview

Seven sequential tasks; each depends on the previous one. Language: Python 3.12, NumPy +
`mujoco==3.15.0`, pytest + ruff. Source: one pinned commit of `mottopanikeiku/weeble`. Nothing from
M1+ (no UI, schema, LLM, optimizer, adapter, push, or module stubs).

## Tasks

- [x] 1. Go/no-go spike on the upstream runner (30-minute time-box)
  - [x] 1.1 Resolve and clone the pinned weeble commit
    - `git ls-remote https://github.com/mottopanikeiku/weeble main` → record the 40-hex SHA
    - Clone at that SHA into a temp dir outside the workspace (e.g. `$TMPDIR/weeble-<sha>`)
    - _Requirements: 1.1, 1.2_
  - [x] 1.2 Run the upstream 10 s forward rollout
    - In the clone: `uv sync --frozen --only-group eval`, then run `eval/sim.py` with
      `policies/go1-walk-s0.json`, command (1.0, 0, 0), 10 s; print displacement and max tilt
    - Time the install and the rollout
    - _Requirements: 1.1_
  - [x] 1.3 Record the go/no-go result and apply the stop-loss
    - Record SHA, timings, displacement, tilt and decision (carried into the build log in Task 2 and
      `docs/milestones/M0.md` in Task 7)
    - If it does not install or walk within 30 min: report the blocker and ask the user before
      switching to the fallback (Playground rss2025 `go1_policy.onnx` + native rough scene)
    - _Requirements: 1.3_
  - **Exit:** upstream `eval/sim.py` prints forward displacement and tilt for go1-walk-s0; go decision
    recorded with the SHA.

- [x] 2. Repository skeleton
  - [x] 2.1 Initialize git and repository hygiene files
    - `git init` if the workspace is not a repo; add MIT `LICENSE`, `README.md` (setup, test, run
      commands), `.gitignore` (`runs/`, `.venv/`, `__pycache__/`, caches); `.gitignore` must not match
      `.kiro/` or `LICENSE`; stage and commit `.kiro/` and `LICENSE`
    - _Requirements: 12.1, 12.2_
  - [x] 2.2 Create `pyproject.toml` and lockfile
    - `src/` layout, `.python-version` 3.12, runtime deps exactly `numpy` and `mujoco==3.15.0`, dev
      group `pytest` + `ruff`, ruff line length 100, pytest marker `integration` (deselected by
      default), console script `robustlab = robustlab.cli:main`; `uv lock` and commit `uv.lock`
    - _Requirements: 2.1, 2.2_
  - [x] 2.3 Implement `src/robustlab/__init__.py` and `src/robustlab/cli.py` with `doctor`
    - `doctor` prints Python, NumPy, MuJoCo versions, platform and NumPy BLAS config
    - _Requirements: 2.5_
  - [x] 2.4 Add documentation scaffolding
    - `docs/kiro_build_log.md` (with the Task 1 spike record), `docs/decision-log/0001-m0-policy-source.md`
      ADR: weeble as policy source, MuJoCo 3.15.0 instead of 3.17, `src/` layout per `delivery.md`
    - _Requirements: 12.3_
  - [x] 2.5 Kiro hooks for ruff and pytest
    - Confirm the hook file format locally; create ruff/pytest hooks only if confirmed, otherwise skip
      and record why in `docs/kiro_build_log.md`; no GPU or network calls in hooks
    - _Requirements: 12.3_
  - [x] 2.6 Write skeleton tests
    - CLI smoke test for `doctor`; subprocess test that `import robustlab` leaves `jax`, `brax`, `flax`,
      `torch` out of `sys.modules`; repo test: `git ls-files` lists `.kiro/` files, `LICENSE` and
      `uv.lock`, `git check-ignore` matches neither `.kiro/` nor `LICENSE`, `.gitignore` lists `runs/`;
      `pyproject.toml` runtime deps are exactly the two pinned packages
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 12.2_
  - **Exit/demo:** `uv sync && uv run pytest -q && uv run robustlab doctor`

- [x] 3. Pinned assets and model loader
  - [x] 3.1 Implement `scripts/fetch_assets.py`
    - Download from the pinned commit: `assets/go1/walk.xml`, the 5 STLs under `assets/go1/assets/`,
      `LICENSE-Menagerie`, `LICENSE-Playground`, `walk.provenance.json`, `policies/go1-walk-s0.json`;
      write `assets/assets.lock.json` (commit, URL, SHA-256 per file) atomically; abort without
      touching committed files on network error
    - _Requirements: 3.1, 3.2_
  - [x] 3.2 Fetch, commit pinned files and write `THIRD_PARTY_NOTICES.md`
    - Notices: weeble MIT, Playground-derived XML Apache-2.0, Menagerie meshes BSD-3-Clause, and the
      assumption that the policy JSON is covered by the weeble repo MIT license
    - _Requirements: 3.1, 12.1, 12.4_
  - [x] 3.3 Implement asset verification and `robustlab verify-assets`
    - `src/robustlab/assets.py`: `sha256_file`, `verify_assets(root)` against the lock and
      `walk.provenance.json`; CLI prints per-file result, exits non-zero on any issue
    - _Requirements: 3.3, 3.4_
  - [x] 3.4 Implement `src/robustlab/physics/model.py`
    - `load_model`, `apply_model_parameters` (arrays + `opt`), `resolve_ids` (trunk, imu site, four
      feet, floor), `check_model` (nq 19, nv 18, nu 12, timestep 0.004, n_substeps 5, actuator order ==
      policy joint order, read from the raw policy JSON until Task 4)
    - _Requirements: 4.1, 4.2, 3.3, 4.4_
  - [x] 3.5 Write model and asset tests
    - Example tests for model facts and ID resolution (missing name raises)
    - Property 1: Tampered or missing assets are detected — **Validates: Requirements 3.3, 3.4**
    - Property 2: Applying model parameters is idempotent — **Validates: Requirements 4.4, 4.5**
    - _Requirements: 3.4, 4.1, 4.2, 4.3, 4.5_
  - **Exit/demo:** `uv run pytest -q && uv run robustlab verify-assets`

- [x] 4. Policy loader and NumPy actor
  - [x] 4.1 Add the test oracle
    - Copy weeble `eval/sim.py` to `tests/reference/weeble_sim.py` with its MIT copyright and
      permission notice and the source commit in a header; tests only
    - Test that no module under `src/` imports `tests`
    - _Requirements: 8.1, 8.2_
  - [x] 4.2 Implement `load_policy` in `src/robustlab/policy/actor.py`
    - Accept only schema_version 1, activation `swish|silu`, encoding `base64-f32-le`; chained layer
      shapes; obs_dim 48, act_dim 12; finite weights and normalizer; file SHA-256 equals lock;
      normalizer selected by `obs_key`; raise `PolicyError(field, reason)`
    - Switch `check_model` to the validated `Policy`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 2.4_
  - [x] 4.3 Implement `Actor` with stable swish
    - `(x-mean)/std` float32, no epsilon or clip; hidden `swish(xW+b)` via stable sigmoid; linear last
      layer; `tanh(logits[:12])`; preallocated float32 buffers
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [x] 4.4 Write policy and actor unit/property tests
    - Normalizer selection and buffer reuse examples
    - Property 3: Malformed policies are rejected — **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    - Property 4: Swish is finite and accurate — **Validates: Requirements 6.3**
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.3, 6.4_
  - [x] 4.5 Write actor oracle-parity test
    - Property 5: Actor matches the Oracle (1,000 seeded observations, atol 1e-7, record max error) —
      **Validates: Requirements 6.1, 6.2, 6.5**
    - Reset-observation demo test printing our action, the Oracle action and max abs error
    - _Requirements: 6.5_
  - **Exit/demo:** `uv run pytest -q && uv run pytest -q -s -k reset_parity`

- [x] 5. Observation, control loop, trace and manifest
  - [x] 5.1 Implement `src/robustlab/policy/observation.py`
    - 48 fields in `observation.fields` order: velocimeter local_linvel, gyro, gravity
      `R_imu.T @ [0,0,-1]`, `qpos[7:]-default_pose`, `qvel[6:]`, delayed action, command; read after
      `mj_step(..., nstep=5)` with no `mj_forward`
    - _Requirements: 7.1, 7.2_
  - [x] 5.2 Implement the episode runner in `src/robustlab/experiments/walk.py`
    - Reset (`mj_resetData`, default qpos/qvel, `ctrl=qpos[7:]`, command, zero `last_action`,
      `mj_forward`); Tick: clear `xfrc_applied`, `ctrl = default_pose + 0.5*action` held for 5
      substeps, then `obs_last_action ← last_action`, `last_action ← a`; preallocated per-Tick
      recording (time, qpos, qvel, action, command, 4 foot-floor contacts, up_z, trunk z); MuJoCo
      warning counts; per-Tick `perf_counter_ns`; provisional upright outcome (45° gate, 60°
      recorded, trunk z ≥ 0.18 m, finite, no warnings); `latency_summary`
    - _Requirements: 7.3, 7.4, 7.5, 7.6, 9.1, 10.1, 10.2, 11.2_
  - [x] 5.3 Implement `src/robustlab/experiments/trace.py`
    - `write_run` → `runs/m0/<run_id>/trajectory.json` (50 Hz) and `manifest.json` (all fields in
      Requirement 9.3) via temp file + `os.replace`; JSON-write timing; `read_run`
    - _Requirements: 9.2, 9.3, 9.4, 11.3_
  - [x] 5.4 Add the `robustlab walk` subcommand
    - `--policy --command vx,vy,yaw --seconds --out [--seed]`; verify assets first; run, evaluate,
      write run, print summary table
    - _Requirements: 9.2, 3.4_
  - [x] 5.5 Write loop and trace unit/property tests
    - Reset observation example (gravity ≈ [0,0,-1], zero joint offsets, zero action field);
      manifest required-keys check; subprocess short walk leaves `jax`/`torch` out of `sys.modules`
    - Property 6: Observation carries the delayed action — **Validates: Requirements 7.3**
    - Property 7: Tick control invariant — **Validates: Requirements 7.5, 7.6**
    - Property 8: Trace serialization round trip — **Validates: Requirements 9.4, 9.6**
    - Property 9: Upright predicate — **Validates: Requirements 10.1, 10.2**
    - _Requirements: 2.3, 7.1, 7.3, 7.4, 7.5, 7.6, 9.3, 9.4, 9.6, 10.1, 10.2_
  - [x] 5.6 Write closed-loop integration tests (`-m integration`)
    - 500-Tick parity with the Oracle: actions and qpos atol 1e-6
    - Two runs of the same configuration give identical trajectories
    - _Requirements: 7.7, 9.5_
  - **Exit/demo:** `uv run pytest -q && uv run pytest -q -m integration && uv run robustlab walk
    --policy policies/go1-walk-s0.json --command 1,0,0 --seconds 10 --out runs/m0/forward`

- [x] 6. Walking verification, latency and optional render
  - [x] 6.1 Extend `evaluate` with the walking metrics
    - Mean body-frame forward speed over 2–10 s, touchdowns and duty factor per foot, distance, yaw
      drift, horizontal displacement; `passed` and `failed_checks`; criteria constants written to the
      manifest
    - _Requirements: 10.3, 10.4, 10.5, 10.6_
  - [x] 6.2 Write metric property tests
    - Property 10: Forward speed is heading-invariant — **Validates: Requirements 10.3**
    - Property 11: Touchdown counting — **Validates: Requirements 10.4**
    - _Requirements: 10.3, 10.4_
  - [x] 6.3 Write the walking integration test (`-m integration`)
    - Forward_Episode (1.0, 0, 0), 10 s: upright (45°), trunk z ≥ 0.18 m, finite, no warnings, mean
      forward speed ≥ 0.7 m/s, ≥ 5 touchdowns per foot in 2–10 s
    - Zero_Episode (0, 0, 0), 10 s: upright and horizontal displacement < 0.3 m
    - _Requirements: 10.1, 10.3, 10.4, 10.5_
  - [x] 6.4 Measure latency
    - `latency_summary` example test on a known array; run a ≥ 51 s episode so 50 warm-up Ticks are
      dropped and ≥ 2,500 are measured; report median, p95, max vs 25 ms target, real-time factor,
      load time and JSON-write time
    - _Requirements: 11.1, 11.2, 11.3_
  - [ ]* 6.5 Cross-check tracking against weeble's committed table
    - Run (1.5, 0, 0) and compare forward-speed error to weeble's tracking table for go1-walk-s0;
      comparison only, not a pass/fail criterion
    - _Requirements: 10.6_
  - [x]* 6.6 Render a PNG with `mujoco.Renderer`
    - Outside any timed loop; if macOS offscreen OpenGL raises any error or friction, skip with no
      retries or workarounds and record the skip in `docs/milestones/M0.md`
    - _Requirements: 13.1_
  - **Exit/demo:** `uv run pytest -q -m integration`, then `uv run robustlab walk --policy
    policies/go1-walk-s0.json --command 1,0,0 --seconds 10 --out runs/m0/forward`, `... --command
    0,0,0 --seconds 10 --out runs/m0/zero` and `... --command 1,0,0 --seconds 51 --out
    runs/m0/latency`, each printing the summary table

- [ ] 7. M0 evidence and stop
  - [ ] 7.1 Run the full validation suite
    - `uv run ruff check . && uv run ruff format --check . && uv run pytest -q && uv run pytest -q -m
      integration`; capture output
    - _Requirements: 13.1_
  - [ ] 7.2 Write `docs/milestones/M0.md`
    - Exact commands, go/no-go result, all hashes, versions, criteria with measured values, oracle
      parity max errors, latency stats, `uv run pytest -q` output, trace paths, render skip if any
    - Limitations: `walk.xml` collides feet only so non-foot contacts are not modelled (M1 needs a
      collision approach; examine weeble `getup.xml` and its fall rule); trained in MJX Warp, run in
      C MuJoCo; identical results claimed only within one build and machine; policy JSON license
      inherited from repo MIT. If any criterion failed, do not call the policy walking
    - _Requirements: 13.1, 13.2, 10.7_
  - [ ] 7.3 Update `docs/kiro_build_log.md` and `README.md`
    - Map Kiro spec outputs to observed tests; final setup/test/run commands; confirm no M1+ code or
      stubs exist; commit
    - _Requirements: 12.1, 12.3, 13.3_
  - [ ] 7.4 Report and stop
    - Report what works, what failed and what was learned; stop for review before M1
    - _Requirements: 13.1_
  - **Exit:** `docs/milestones/M0.md` committed with measured evidence; stop for review before M1.

## Notes

- Only 6.5 and 6.6 are optional (`*`); all other sub-tasks, including tests, are required.
- Property tests use seeded NumPy generation (≥ 100 cases, 1,000 for Property 5) under pytest, since
  the approved dev group is pytest + ruff; each test carries
  `# Feature: m0-walking-go1, Property N: <title>`.
- Integration tests are marked `integration` and excluded from the default `uv run pytest -q`.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3"] },
    { "id": 3, "tasks": ["2.1"] },
    { "id": 4, "tasks": ["2.2"] },
    { "id": 5, "tasks": ["2.3"] },
    { "id": 6, "tasks": ["2.4"] },
    { "id": 7, "tasks": ["2.5"] },
    { "id": 8, "tasks": ["2.6"] },
    { "id": 9, "tasks": ["3.1"] },
    { "id": 10, "tasks": ["3.2"] },
    { "id": 11, "tasks": ["3.3"] },
    { "id": 12, "tasks": ["3.4"] },
    { "id": 13, "tasks": ["3.5"] },
    { "id": 14, "tasks": ["4.1"] },
    { "id": 15, "tasks": ["4.2"] },
    { "id": 16, "tasks": ["4.3"] },
    { "id": 17, "tasks": ["4.4"] },
    { "id": 18, "tasks": ["4.5"] },
    { "id": 19, "tasks": ["5.1"] },
    { "id": 20, "tasks": ["5.2"] },
    { "id": 21, "tasks": ["5.3"] },
    { "id": 22, "tasks": ["5.4"] },
    { "id": 23, "tasks": ["5.5"] },
    { "id": 24, "tasks": ["5.6"] },
    { "id": 25, "tasks": ["6.1"] },
    { "id": 26, "tasks": ["6.2"] },
    { "id": 27, "tasks": ["6.3"] },
    { "id": 28, "tasks": ["6.4"] },
    { "id": 29, "tasks": ["6.5"] },
    { "id": 30, "tasks": ["6.6"] },
    { "id": 31, "tasks": ["7.1"] },
    { "id": 32, "tasks": ["7.2"] },
    { "id": 33, "tasks": ["7.3"] },
    { "id": 34, "tasks": ["7.4"] }
  ]
}
```
