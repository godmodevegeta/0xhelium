# Requirements Document

## Introduction

Milestone M1 of `delivery.md`: one deterministic, precisely specified shove and one authoritative fall
detector for the M0 Go1 walking policy. The M0 baseline (`m0-baseline` tag, commit `cfe4b13`) stays
reproducible: same policy, same pinned `walk.xml`, same control loop and the same golden
trajectories. M1 adds a collision-enabled scene derived from `walk.xml` in code, a `FallResult`
computed at every physics substep, a typed `PushScenario` applied to the trunk in the robot's
initial-heading frame, and CLI flags to run it. CPU-only, CLI-first.

Out of scope: M2 aggregation over seeds, a JSON scenario schema, other attack dimensions, search or
CMA-ES, web UI, LLM features.

## Glossary

- **Feet_Only_Scene**: the pinned M0 `assets/go1/walk.xml`; only the four foot spheres collide with
  the floor.
- **Fall_Contact_Scene**: the model compiled from `walk.xml` by the Scene_Builder with fall geoms made
  floor-collidable. Nothing else changes.
- **Fall_Geom**: a robot collision geom (MuJoCo group 3, not a foot) that collides with the floor in
  the Fall_Contact_Scene: trunk box and cylinders, hip cylinders, thigh capsules and the upper shin
  capsule (class `calf1`).
- **Lower_Shin_Geom**: the capsule just above each foot (class `calf2`). It stays non-colliding, as in
  the policy's training scene.
- **Scene_Builder**: `src/robustlab/physics/scene.py`.
- **Fall_Monitor**: `src/robustlab/physics/fall.py`, which checks the state after every physics
  substep and produces the FallResult.
- **FallResult**: the authoritative outcome: whether a physical fall or a numerical failure happened,
  the first reason and its time, and the first occurrence of every reason.
- **Physical_Reason**: `nonfoot_contact`, `tilt` or `low_height`.
- **Numerical_Reason**: `nonfinite_state` or `mujoco_warning`.
- **Initial_Heading_Frame**: a frame fixed at reset: +x along the trunk's initial yaw, +y to the left,
  +z up (world z).
- **PushScenario**: force magnitude (N), direction in the Initial_Heading_Frame, start time (s) and
  duration (s), applied as a constant force at the trunk centre of mass.
- **Substep**: one `mj_step` of `sim_dt` = 0.004 s; a Tick is 5 Substeps (0.02 s).
- **Golden_Trajectories**: the M0 forward and zero-command 10 s trajectories, committed as
  `tests/fixtures/m0_golden.npz` with their source hashes.

## Requirements

### Requirement 1: M0 baseline preservation

**User Story:** As the developer, I want M0 results to stay reproducible, so that M1 changes can be
compared against a fixed reference.

#### Acceptance Criteria

1. THE Repository SHALL keep `assets/go1/walk.xml`, its meshes, `policies/go1-walk-s0.json` and
   `assets/assets.lock.json` byte-identical to the `m0-baseline` tag.
2. THE Repository SHALL contain the Golden_Trajectories with the SHA-256 of the M0 `trajectory.json`
   files they came from.
3. WHEN the Episode_Runner runs the Feet_Only_Scene with no PushScenario at the golden commands, THE
   Episode_Runner SHALL reproduce the Golden_Trajectories exactly (qpos, qvel, action, foot contacts).
4. WHEN the Episode_Runner runs the Fall_Contact_Scene with no PushScenario at the golden commands,
   THE Episode_Runner SHALL reproduce the Golden_Trajectories exactly, or the M1 documentation SHALL
   report the measured divergence.
5. THE existing M0 unit, property and integration tests SHALL keep passing.

### Requirement 2: Fall_Contact_Scene

**User Story:** As the developer, I want non-foot body parts to hit the floor in simulation, so that a
fall is observed as contact instead of the body passing through the ground.

#### Acceptance Criteria

1. THE Scene_Builder SHALL compile the Fall_Contact_Scene from the pinned `walk.xml` with MuJoCo
   `MjSpec`, changing only `conaffinity` of the Fall_Geoms from 0 to 1.
2. THE Fall_Contact_Scene SHALL keep robot-robot collisions disabled (robot `contype` stays 0).
3. THE Fall_Contact_Scene SHALL keep every inertial, joint, actuator, sensor, option and keyframe
   array equal to the Feet_Only_Scene; only `geom_conaffinity`, `body_conaffinity` and the derived
   collision bounding-volume arrays may differ.
4. THE Fall_Contact_Scene SHALL pass the M0 `check_model` contract (sizes, timestep, actuator and
   joint order).
5. THE Scene_Builder SHALL report the scene name, transform version, base XML hash, the list of
   Fall_Geoms with body names, the Lower_Shin_Geoms left unchanged, and a SHA-256 of that transform.

### Requirement 3: Contact classification

**User Story:** As the developer, I want foot contacts and non-foot contacts kept separate, so that
normal walking is never counted as a fall.

#### Acceptance Criteria

1. THE Fall_Monitor SHALL derive the non-foot floor-contact geoms from the compiled model: robot
   geoms that can collide with the floor and are not one of the four foot geoms.
2. WHEN a contact between the floor and a non-foot geom has `dist <= 0`, THE Fall_Monitor SHALL
   record a `nonfoot_contact` event with the geom, its body name and the penetration depth.
3. THE Fall_Monitor SHALL NOT treat foot-floor contacts as falls.
4. WHILE the scene has no non-foot floor-collidable geom, THE FallResult SHALL report
   `nonfoot_contact_detection` false.
5. THE Episode_Runner SHALL record per Tick the four foot-floor contact flags (as in M0) and whether
   any non-foot floor contact occurred in that Tick's Substeps.

### Requirement 4: Authoritative FallResult

**User Story:** As the developer, I want one fall calculation used everywhere, so that "fell" has one
meaning with a time and a reason.

#### Acceptance Criteria

1. THE Fall_Monitor SHALL check after every Substep: finite `qpos`, `qvel` and `qacc`; MuJoCo warning
   counters; non-foot floor contacts; trunk `up_z >= cos(45°)`; trunk height `>= 0.18 m`.
2. THE Fall_Monitor SHALL take the 45° and 0.18 m thresholds from the existing `WalkCriteria`.
3. THE FallResult SHALL record, for each Physical_Reason and Numerical_Reason, its first Substep
   index, its time in seconds and a detail value.
4. THE FallResult SHALL name the earliest event as the primary reason, breaking ties in the order
   `nonfinite_state`, `mujoco_warning`, `nonfoot_contact`, `tilt`, `low_height`.
5. THE FallResult SHALL classify the outcome as `none`, `physical` (primary reason is physical) or
   `numerical` (primary reason is numerical), and report `numerical_failure` true whenever any
   numerical event occurred.
6. WHEN a numerical event occurs, THE Episode_Runner SHALL stop the episode after that Tick and mark
   the record truncated.
7. WHEN a physical fall occurs, THE Episode_Runner SHALL keep simulating to the horizon so that the
   fall can be replayed, and progress metrics SHALL stop at the last Tick before the fall.
8. THE Walking_Evaluator SHALL fail an episode that has a physical fall or a numerical failure.

### Requirement 5: PushScenario

**User Story:** As the developer, I want a shove specified in physical units, so that the same
disturbance can be repeated exactly.

#### Acceptance Criteria

1. THE PushScenario SHALL hold `force_n` in [0, 150] N, a finite non-zero `direction` 3-vector in the
   Initial_Heading_Frame, `start_s` >= 0 and `duration_s` in [0, 0.5] s.
2. IF a PushScenario value is non-finite, out of range, or `start_s` or `duration_s` is not an
   integer multiple of 0.004 s, THEN THE PushScenario SHALL be rejected with an error naming the
   field.
3. IF `start_s + duration_s` exceeds the episode horizon, THEN THE Episode_Runner SHALL reject the
   episode before simulating.
4. THE Robustlab_CLI SHALL accept the direction as `forward`, `backward`, `left`, `right` or `x,y,z`,
   where the name is the direction the force points (`right` = −y = a shove from the robot's left).

### Requirement 6: Applying the push

**User Story:** As the developer, I want the push applied exactly when and where it was specified, so
that measured responses are attributable to it.

#### Acceptance Criteria

1. THE Episode_Runner SHALL convert the direction to world coordinates once per episode by rotating it
   about world z by the trunk's yaw at reset, then scale the unit vector by `force_n`.
2. THE Episode_Runner SHALL set `xfrc_applied` of the trunk to that force (zero torque) before every
   Substep `i` with `start_s/sim_dt <= i < (start_s + duration_s)/sim_dt`, and to zero before every
   other Substep.
3. THE Episode_Runner SHALL clear all `xfrc_applied` before every Substep and on reset.
4. THE Episode_Runner SHALL record per Tick the applied impulse (N·s, world frame) and the manifest
   SHALL record the PushScenario, the world force, the active Substep interval and the total impulse.
5. THE Episode_Runner SHALL apply no other external force, noise, command change or reset.

### Requirement 7: Experiments, evidence and tests

**User Story:** As the reviewer, I want two runnable experiments with recorded evidence, so that I can
see whether, why and when the robot fell.

#### Acceptance Criteria

1. THE Robustlab_CLI `walk` subcommand SHALL accept `--scene {fall-contact,feet-only}` (default
   `fall-contact`) and the push flags `--push-force`, `--push-direction`, `--push-start` and
   `--push-duration`, given all together or not at all.
2. WHEN `walk` completes, THE Trace_Writer SHALL write `trajectory.json` and `manifest.json` with the
   scene description, PushScenario, per-Tick impulse and non-foot contact flags, and FallResult.
3. WHEN `walk` completes, THE Robustlab_CLI SHALL print whether the robot fell, the primary reason and
   its time.
4. THE test suite SHALL cover: zero-force and zero-duration equivalence with the nominal rollout;
   world-frame direction; exact start and stop Substeps; no residual force after the push or after
   reset; foot contacts in nominal walking; non-foot contact detection; tilt and height boundaries;
   numerical-failure classification; reproducibility; nominal walking in the Fall_Contact_Scene.
5. THE `docs/milestones/M1.md` SHALL record the commands, measured FallResults, the Fall_Contact_Scene
   versus M0 comparison, latency, test output and limitations.
6. THE Repository SHALL contain no M2 aggregation, scenario schema, other attack dimensions,
   optimizer, web UI or LLM code.
