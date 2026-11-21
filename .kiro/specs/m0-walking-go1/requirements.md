# Requirements Document

## Introduction

Milestone M0 of `delivery.md`: run a verified, existing pretrained Go1 walking policy through native
MuJoCo on a macOS CPU and record a reproducible trajectory with measured evidence. The policy, model
XML and meshes come from one pinned commit of `mottopanikeiku/weeble` (MIT code): policy
`policies/go1-walk-s0.json` (trained on Playground `Go1JoystickFlatTerrain`, no shoves) and the
exact-export model `assets/go1/walk.xml` with its STL meshes and `walk.provenance.json`. Our runner is
written independently in NumPy + MuJoCo 3.15.0; weeble's `eval/sim.py` is used only as a test oracle.

M0 delivers no UI, scenario schema, LLM adapter, optimizer, adapter, push perturbation or module
stubs for later milestones.

## Glossary

- **Robustlab**: the Python package `robustlab` under `src/robustlab/` created by this milestone.
- **Robustlab_CLI**: the `robustlab` console entry point with subcommands `doctor`, `verify-assets`
  and `walk`.
- **Asset_Fetcher**: `scripts/fetch_assets.py`, which downloads pinned upstream files and writes the
  asset lock.
- **Asset_Lock**: `assets/assets.lock.json`, recording the weeble commit, source URLs and SHA-256 of
  every pinned file.
- **Asset_Verifier**: the Robustlab component that checks pinned files against the Asset_Lock and
  against weeble's `walk.provenance.json`.
- **Model_Loader**: `src/robustlab/physics/model.py`, which loads `walk.xml`, applies policy
  `model_parameters` and resolves named MuJoCo IDs (trunk, imu site, feet, floor).
- **Policy_Loader**: the validated JSON loader in `src/robustlab/policy/actor.py`.
- **Actor**: the NumPy MLP in `src/robustlab/policy/actor.py` mapping a 48-dim observation to a
  12-dim action.
- **Observation_Builder**: `src/robustlab/policy/observation.py`, which assembles the 48-dim
  post-step observation including the delayed-action history.
- **Episode_Runner**: `src/robustlab/experiments/walk.py`, which resets the model, runs the control
  loop and computes outcome metrics.
- **Trace_Writer**: `src/robustlab/experiments/trace.py`, which writes `trajectory.json` and
  `manifest.json`.
- **Oracle**: `tests/reference/weeble_sim.py`, an MIT-attributed copy of weeble `eval/sim.py` used only
  by tests.
- **Tick**: one control period: observation build, Actor inference and 5 calls of `mj_step` at
  `sim_dt` 0.004 s, giving `ctrl_dt` 0.02 s.
- **Delayed action**: the observation field holding the action from two ticks earlier, so that
  observation `o[t+1]` contains action `a[t-1]`.
- **up_z**: z component of the trunk's up axis in the world frame; tilt below 45° means
  `up_z >= cos(45°)`.
- **Touchdown**: a transition of a foot-floor contact flag from false to true between consecutive
  recorded ticks.
- **Forward_Episode**: 10 s episode from the nominal pose with command `(1.0, 0, 0)`.
- **Zero_Episode**: 10 s episode from the nominal pose with command `(0, 0, 0)`.
- **Walking_Evaluator**: the Robustlab component that applies the fixed walking criteria to an
  episode's recorded outcomes.
- **Latency_Benchmark**: the Robustlab component that times Ticks, model/policy load and JSON writes.
- **Repository**: the git repository at the workspace root.

## Requirements

### Requirement 1: Go/no-go feasibility check

**User Story:** As the developer, I want to confirm the upstream policy walks in its own runner before
integrating it, so that I do not build on an incompatible checkpoint.

#### Acceptance Criteria

1. WHEN the feasibility check runs, THE developer workflow SHALL record the weeble `main` commit SHA, the install and rollout timings, the 10 s forward displacement and the maximum tilt produced by weeble `eval/sim.py`.
2. THE feasibility check SHALL execute in a temporary directory outside the workspace.
3. IF the weeble runner fails to install or fails to walk within a 30-minute time-box, THEN THE developer workflow SHALL report the blocker and request a user decision before using the fallback policy source.

### Requirement 2: Runtime dependency envelope

**User Story:** As the developer, I want a minimal CPU-only runtime, so that M0 runs on a Mac without
accelerator stacks.

#### Acceptance Criteria

1. THE Robustlab SHALL declare Python 3.12 and exactly two runtime dependencies: `numpy` and `mujoco==3.15.0`.
2. THE Repository SHALL contain a committed `uv.lock` that resolves the runtime and dev dependency groups.
3. WHEN the Robustlab package and its subcommands are imported, THE Robustlab SHALL leave the modules `jax`, `brax`, `flax` and `torch` absent from `sys.modules`.
4. THE Robustlab SHALL load policy data only from JSON files and SHALL exclude all `pickle` and `.brax` sidecar loading.
5. WHEN `robustlab doctor` runs, THE Robustlab_CLI SHALL print the Python, NumPy and MuJoCo versions, the platform and the NumPy BLAS configuration.

### Requirement 3: Pinned and verified assets

**User Story:** As the developer, I want every upstream file pinned and hashed, so that every result
traces to an exact policy and model.

#### Acceptance Criteria

1. WHEN the Asset_Fetcher runs, THE Asset_Fetcher SHALL download `walk.xml`, the five STL meshes, `LICENSE-Menagerie`, `LICENSE-Playground` and `policies/go1-walk-s0.json` from the single weeble commit recorded in the Asset_Lock.
2. WHEN the Asset_Fetcher completes, THE Asset_Fetcher SHALL write the weeble commit, source URL and SHA-256 of each file to the Asset_Lock.
3. WHEN `robustlab verify-assets` runs, THE Asset_Verifier SHALL compare the SHA-256 of each pinned file against both the Asset_Lock and the hashes in `walk.provenance.json`.
4. IF any pinned file is missing or has a SHA-256 that differs from the Asset_Lock or `walk.provenance.json`, THEN THE Asset_Verifier SHALL report the file name and exit with a non-zero status.

### Requirement 4: Model loading

**User Story:** As the developer, I want the model loaded and configured exactly as at training
export, so that the policy sees the dynamics it was trained on.

#### Acceptance Criteria

1. WHEN `walk.xml` is loaded, THE Model_Loader SHALL produce a model with `nq` 19, `nv` 18, `nu` 12 and timestep 0.004 s.
2. THE Model_Loader SHALL resolve the trunk body, imu site, four foot geoms and floor geom IDs by name.
3. THE Model_Loader SHALL verify that actuator order equals the policy joint order.
4. WHEN policy `model_parameters` are applied, THE Model_Loader SHALL set the listed model arrays and `opt` options to the policy values.
5. WHEN policy `model_parameters` are applied twice, THE Model_Loader SHALL produce a model identical to the model after one application.

### Requirement 5: Policy loading and validation

**User Story:** As the developer, I want malformed or unexpected policy files rejected, so that a bad
artifact cannot silently drive the robot.

#### Acceptance Criteria

1. WHEN a policy JSON file is loaded, THE Policy_Loader SHALL accept only `schema_version` 1, activation `swish` or `silu`, and weight encoding `base64-f32-le`.
2. IF consecutive layer shapes do not chain, or `obs_dim` is not 48, or `act_dim` is not 12, THEN THE Policy_Loader SHALL raise a validation error naming the field.
3. IF any decoded weight, bias, normalizer mean or normalizer std is non-finite, THEN THE Policy_Loader SHALL raise a validation error.
4. IF the policy file SHA-256 differs from the Asset_Lock entry, THEN THE Policy_Loader SHALL raise a validation error.
5. THE Policy_Loader SHALL select the normalizer whose key equals the policy `obs_key`.

### Requirement 6: Actor inference

**User Story:** As the developer, I want the NumPy actor to reproduce the trained network exactly,
so that behaviour matches the upstream policy.

#### Acceptance Criteria

1. THE Actor SHALL compute `(x - mean) / std` in float32 without epsilon and without clipping.
2. THE Actor SHALL apply `swish(xW + b)` on hidden layers, a linear last layer, and `tanh` on the first 12 logits.
3. WHEN a hidden pre-activation lies in [-80, 80], THE Actor SHALL return a finite swish value.
4. THE Actor SHALL perform inference using preallocated float32 buffers.
5. WHEN the Actor and the Oracle evaluate the same 1,000 observations, THE Actor SHALL produce actions equal to the Oracle actions within absolute tolerance 1e-7.

### Requirement 7: Observation and control loop

**User Story:** As the developer, I want the observation and control conventions preserved exactly,
so that the closed loop matches the policy's training conditions.

#### Acceptance Criteria

1. THE Observation_Builder SHALL assemble the 48 values in the order given by the policy `observation.fields`: local linear velocity (3, velocimeter site sensor), gyro (3), gravity (3) computed as `R_imu.T @ [0, 0, -1]`, joint offsets `qpos[7:] - default_pose` (12), `qvel[6:]` (12), delayed action (12), command (3).
2. THE Observation_Builder SHALL read the observation after `mj_step(model, data, nstep=5)` without an additional `mj_forward`.
3. WHEN a Tick completes with action `a`, THE Episode_Runner SHALL set the observation action field to the previous `last_action` and then set `last_action` to `a`.
4. WHEN an episode resets, THE Episode_Runner SHALL call `mj_resetData`, set `qpos` and `qvel` to the policy defaults, set `ctrl` to `qpos[7:]`, set the command, zero `last_action`, and call `mj_forward`.
5. WHILE a Tick executes, THE Episode_Runner SHALL hold `ctrl = default_pose + 0.5 * action` constant across 5 physics substeps without extra clipping.
6. WHILE a Tick executes, THE Episode_Runner SHALL clear `xfrc_applied` before stepping.
7. WHEN the Episode_Runner and the Oracle run the same 500 Ticks, THE Episode_Runner SHALL produce actions and `qpos` equal to the Oracle values within absolute tolerance 1e-6.

### Requirement 8: Independent implementation

**User Story:** As the project owner, I want our runner written independently with the upstream
runner used only as a test oracle, so that licensing and provenance stay clear.

#### Acceptance Criteria

1. THE Repository SHALL store weeble `eval/sim.py` only at `tests/reference/weeble_sim.py` with its MIT copyright and permission notice.
2. THE Robustlab SHALL import nothing from `tests/`.

### Requirement 9: Trace and manifest recording

**User Story:** As the developer, I want every rollout saved with full provenance, so that results can
be reconstructed and compared.

#### Acceptance Criteria

1. WHILE an episode runs, THE Episode_Runner SHALL record per Tick, into preallocated arrays, the time, `qpos`, `qvel`, action, command, four foot-floor contact flags, `up_z` and trunk height.
2. WHEN an episode ends, THE Trace_Writer SHALL write `trajectory.json` at 50 Hz and `manifest.json` under `runs/m0/<run_id>/`.
3. THE Trace_Writer SHALL include in `manifest.json` the schema version, asset and policy hashes, the weeble, Playground and Menagerie commits, NumPy, MuJoCo and Python versions, platform, git revision, `ctrl_dt`, `sim_dt`, `n_substeps`, command, horizon, initial state, seed, criteria, outcomes and latency summary.
4. THE Trace_Writer SHALL write each file to a temporary path and rename the file atomically into place.
5. WHEN the same configuration runs twice on the same build and machine, THE Episode_Runner SHALL produce identical recorded trajectories.
6. WHEN `trajectory.json` and `manifest.json` are read back, THE Trace_Writer SHALL yield records equal to the written records.

### Requirement 10: Walking criteria

**User Story:** As the developer, I want walking judged against criteria fixed before any run, so that
"walking" is a measured claim.

#### Acceptance Criteria

1. THE Walking_Evaluator SHALL mark an episode upright only if, at every recorded Tick, `up_z >= cos(45°)`, trunk height is at least 0.18 m, the state is finite and MuJoCo reports no warnings.
2. THE Walking_Evaluator SHALL also record whether `up_z >= cos(60°)` held at every Tick, for comparison with weeble.
3. WHEN the Forward_Episode completes, THE Walking_Evaluator SHALL require mean body-frame forward speed over 2–10 s of at least 0.7 m/s.
4. WHEN the Forward_Episode completes, THE Walking_Evaluator SHALL require at least 5 Touchdowns per foot over 2–10 s.
5. WHEN the Zero_Episode completes, THE Walking_Evaluator SHALL require the episode to be upright and the trunk horizontal displacement to be below 0.3 m.
6. THE Walking_Evaluator SHALL also record forward distance, mean speed, yaw drift, Touchdowns per foot and duty factor per foot.
7. IF any walking criterion fails, THEN THE M0 documentation SHALL describe the policy as not verified walking.

### Requirement 11: Latency measurement

**User Story:** As the developer, I want measured tick latency, so that later live demos are planned on
real numbers.

#### Acceptance Criteria

1. WHEN the Latency_Benchmark runs, THE Latency_Benchmark SHALL discard 50 warm-up Ticks and time at least 2,500 Ticks.
2. THE Latency_Benchmark SHALL report median, p95 and maximum Tick latency against a 25 ms target, and the real-time factor.
3. THE Latency_Benchmark SHALL report model-and-policy load time and JSON write time separately from Tick latency.

### Requirement 12: Repository, licensing and documentation

**User Story:** As a hackathon judge, I want a licensed, documented repository with committed Kiro
artifacts, so that I can set it up and audit its provenance.

#### Acceptance Criteria

1. THE Repository SHALL contain an MIT `LICENSE`, a `THIRD_PARTY_NOTICES.md` covering weeble (MIT), Playground-derived XML (Apache-2.0) and Menagerie meshes (BSD-3-Clause), and a `README.md` with setup, test and run commands.
2. THE Repository SHALL track `.kiro/` and `LICENSE` in git, and `.gitignore` SHALL list `runs/` and exclude `.kiro/` and `LICENSE` from its patterns.
3. THE Repository SHALL contain `docs/kiro_build_log.md` and one ADR in `docs/decision-log/` recording weeble as policy source, MuJoCo 3.15.0 instead of 3.17, and the `src/` layout.
4. THE `THIRD_PARTY_NOTICES.md` SHALL record the assumption that the policy JSON is covered by the weeble repository MIT license.

### Requirement 13: M0 evidence and scope boundary

**User Story:** As the reviewer, I want a single evidence document and nothing from later milestones,
so that M0 is reviewable on its own.

#### Acceptance Criteria

1. WHEN M0 completes, THE Repository SHALL contain `docs/milestones/M0.md` with the exact commands, go/no-go result, hashes, versions, criteria with measured values, Oracle parity error, latency statistics, `uv run pytest -q` output and trace paths.
2. THE `docs/milestones/M0.md` SHALL list the limitations: feet-only collision in `walk.xml`, training in MJX Warp versus evaluation in C MuJoCo, identical results claimed only within one build and machine, and the inherited policy license assumption.
3. THE Repository SHALL contain no UI, scenario schema, LLM adapter, optimizer, adapter, push perturbation or module stubs for M1 and later milestones.
