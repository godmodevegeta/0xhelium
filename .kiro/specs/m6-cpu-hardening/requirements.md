# Requirements: M6 — CPU controller hardening and simulation-first workspace

M6 produces **candidate** controllers only: the frozen `go1-walk-s0` actor plus a small, bounded
residual reflex whose parameters are searched with seeded CMA-ES on CPU MuJoCo. Promotion and
the regression gate belong to M7. The web app is reorganised so the existing 3D replay viewer is
the primary element of every mode.

Measured facts this spec relies on (`docs/milestones/M6.md`, probes at `9f2839e`):

- Action `a ∈ [-1, 1]^12` (tanh), `ctrl = default_pose + 0.5·a`. `default_pose ± 0.5` lies inside
  every actuator `ctrlrange`, so `a ∈ [-1, 1]` is the physically meaningful bound.
- Joint order FR, FL, RR, RL × (hip, thigh, calf). All hip axes are +x; +0.1 rad at any hip moves
  that foot +0.027 m in body y (left). +0.1 rad calf moves the foot −0.016 m in z and −0.014 m in x;
  +0.1 rad thigh moves it −0.026 m in x.
- Actor observation fields (offsets): linvel 0, gyro 3, gravity 6, joint position 9, joint
  velocity 21, last action 33, command 45.
- After a failing training shove the lateral-velocity deviation leaves the nominal envelope
  40–80 ms after onset; falls occur 0.4–0.9 s after onset.

## Requirement 1 — Frozen base and residual wrapper

1.1 THE base actor weights, normalizer and observation assembly SHALL be used unchanged.
1.2 THE controller SHALL compute `a = clip(a_base + r, -1, 1)` where `r` is the residual in the
policy's action units; the result SHALL be applied through the existing `ctrl` mapping.
1.3 WHEN every gain is zero OR the amplitude is zero, THE wrapper SHALL return the base action
object's values unchanged, so episodes are bit-identical to the base policy.
1.4 THE residual SHALL satisfy `|r_j| ≤ 0.30` per joint and `|r_t − r_{t−1}| ≤ 0.05` per joint per
control tick, and SHALL be finite for any finite observation; a non-finite observation SHALL
produce a zero residual.
1.5 THE wrapper SHALL read only fields of the 48-value actor observation (gravity, gyro, local
linear velocity, command), located by the policy's declared field offsets.
1.6 THE parameter vector SHALL have 10 named, bounded, unit-bearing entries; the zero vector SHALL
be valid and SHALL mean "no residual".
1.7 THE corrective joint bases SHALL be derived from joint names and verified against MuJoCo
forward kinematics (lateral shift, left/right roll, front/rear pitch).
1.8 THE wrapper SHALL record, per tick, the activation and the residual, and summarise activation
fraction, first activation time, per-joint peak residual and the affected joints.

## Requirement 2 — Calibrated dead-zone

2.1 THE system SHALL compute a per-signal centre and half-width from nominal episodes of the base
policy over several commands and non-adversarial M5 conditions, and store it with a hash.
2.2 THE activation SHALL be `smoothstep(clip((m − (1 + margin)) / (0.25 + width), 0, 1))` with
`m = max_i |f_i − c_i| / e_i`; it SHALL be 0 on every calibration tick for any margin ≥ 0.
2.3 THE system SHALL measure nominal walking with the candidate rather than assume the dead-zone
keeps it unchanged.

## Requirement 3 — Data sets

3.1 THE sets file SHALL list training attacks (confirmed failures from M2–M4 plus seeded nearby
variants), nominal training episodes, a validation-preview set with gait phases and attack
configurations absent from training, and a locked M7 test set with its SHA-256.
3.2 THE hardening code SHALL refuse to evaluate the locked test set.
3.3 THE documentation SHALL state that timings in the same gait phase (period ≈ 0.44 s) are
correlated and SHALL NOT claim independent generalization from distinct timestamps.

## Requirement 4 — Optimization

4.1 THE optimizer SHALL be seeded pycma CMA-ES with a private random generator, over the
normalized parameter box, starting at the zero residual.
4.2 THE documented objective SHALL rank, in order: no numerical failure, physical survival, later
fall, smaller post-shove tilt, preserved nominal walking and forward progress, small residual and
little nominal activation.
4.3 A numerical failure SHALL cost more than any physical fall and SHALL never count as recovery.
4.4 THE run SHALL stop at the first of: generation cap, rollout budget, wall-clock cap, or
cancellation, and SHALL record which. Partial generations SHALL not be told to CMA-ES.
4.5 Rollout accounting SHALL separate optimization, baseline and validation-preview rollouts.
4.6 Parallel evaluation SHALL give the same results as sequential evaluation.

## Requirement 5 — Candidate artifact

5.1 THE run SHALL write an immutable `candidate.json` with status `candidate` (never `promoted`),
the base policy hash, reflex schema and parameters, calibration, set identifiers and hashes, seed
and optimizer configuration, rollout counts, runtime, termination reason, baseline and candidate
measurements on training and validation-preview sets, provenance and a reproducibility hash.
5.2 THE same configuration and seed SHALL reproduce the same parameters and reproducibility hash
when the run stops on a deterministic criterion.
5.3 `robustlab harden` SHALL run the job from the CLI.

## Requirement 6 — API

6.1 THE hardening job SHALL use the existing JobManager experiment worker, busy guard, run
directory conventions, polling status and cancellation.
6.2 THE API SHALL expose the attack choices, start/status/cancel/list, and a replay that
re-simulates baseline and candidate on the identical attack, model and reset.

## Requirement 7 — Simulation-first workspace

7.1 WHEN the viewport is ≥ 1024 px wide, THE page SHALL show a two-column workspace: controls
(about 35%) and the shared replay viewer (about 65%), with the 3D canvas fully visible without
scrolling at 1440×900 and 1280×800.
7.2 Below 1024 px THE workspace SHALL stack with the viewer near the top and no horizontal
overflow at 390×844.
7.3 ALL modes (Hidden Phase, Live Stress, Failure Discovery, Scenario Designer, Policy Hardening)
SHALL share one ReplayViewer instance; detailed reports SHALL render below the workspace.
7.4 ON first load THE viewer SHALL show a labelled sample replay from an existing preset.
7.5 THE viewer SHALL label each robot, show the shove indicator, timeline, playback controls,
outcome lines and the replay source, offer fullscreen, keep the camera offset across replays, and
not refetch meshes when the mode changes.
7.6 Policy Hardening mode SHALL offer an attack selector, "Harden Controller", progress, best
metrics, baseline-vs-candidate comparison, "Replay Baseline vs. Candidate" and the status
"Candidate — pending regression evaluation".
7.7 Existing deep links (`?tab=`, `?replay=`, `?run=`, `?scenario_replay=`, `?frame=`) SHALL keep
working.
