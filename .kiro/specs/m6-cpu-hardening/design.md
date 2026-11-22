# Design: M6 — CPU controller hardening and simulation-first workspace

## Modules

| Module | Responsibility |
| --- | --- |
| `robustlab/adapt/reflex.py` | Parameter schema, joint bases, calibration record, `ReflexActor` wrapper |
| `robustlab/adapt/sets.py` | Attack specs, the versioned sets file, locked-set guard |
| `robustlab/adapt/evaluate.py` | One episode → metrics (deterministic), objective, process pool |
| `robustlab/adapt/harden.py` | CMA-ES loop, budget accounting, candidate record |
| `robustlab/api/hardening.py` | Hardening jobs on the existing JobManager worker; replay compute |
| `web/src/components/Workspace*` | Shared layout; modes render controls in the left column and reports through a portal below |

`EpisodeRunner` gains two backward-compatible hooks: it calls `actor.reset()` when the actor has
one, and `EpisodeConfig.stop_on_fall` (default False) ends an episode at the first physical fall.
All metrics are defined up to the first fall, so stopping early does not change them.

## Residual reflex

Signals (raw actor observation values, offsets from `policy.fields`):
`f = (gravity_x, gravity_y, gyro_x, gyro_y, linvel_y − command_vy)`. Deviations `d = f − c`.

Bases (12-vectors, policy joint order; entries from names, verified by kinematics):

- `B_shift`: +1 on every hip → all feet move +y (left).
- `B_roll`: legs on the left (FL, RL) extend, right legs (FR, RR) shorten: per leg
  `s·(0, −0.534, 1)`, `s = +1` left, −1 right. Extension = calf +δ, thigh −0.534δ keeps the foot
  under the hip (measured Jacobian).
- `B_pitch`: front legs extend, rear shorten, same leg vector.

Parameters φ (10):

| # | name | range | unit |
| --- | --- | --- | --- |
| 0 | `shift_from_lateral_velocity` | ±2 | action/(m/s) |
| 1 | `shift_from_roll` | ±10 | action per unit gravity_y |
| 2 | `shift_from_roll_rate` | ±1 | action/(rad/s) |
| 3 | `roll_from_roll` | ±10 | action per unit gravity_y |
| 4 | `roll_from_roll_rate` | ±1 | action/(rad/s) |
| 5 | `pitch_from_pitch` | ±10 | action per unit gravity_x |
| 6 | `pitch_from_pitch_rate` | ±1 | action/(rad/s) |
| 7 | `deadzone_margin` | 0–2 | envelope units |
| 8 | `ramp_width` | 0–2 | envelope units (+0.25) |
| 9 | `amplitude` | 0–0.30 | action units |

```text
u_shift = φ0·d4 + φ1·d1 + φ2·d2      u_roll = φ3·d1 + φ4·d2      u_pitch = φ5·d0 + φ6·d3
A = smoothstep(clip((max_i |d_i|/e_i − 1 − φ7) / (0.25 + φ8), 0, 1))
target = clip(φ9·A·(tanh u_shift·B_shift + tanh u_roll·B_roll + tanh u_pitch·B_pitch), ±0.30)
r_t = r_{t−1} + clip(target − r_{t−1}, ±0.05)            a = clip(a_base + r_t, −1, 1)
```

## Objective (minimised)

Per attack episode (fall time and tilt measured after the push starts, up to the first fall):

- numerical failure: 4
- fall: `2 − min(Δt_fall / 1.5 s, 1)` ∈ [1, 2]
- survived: `0.5·min(max_tilt / 45°, 1)` ∈ [0, 0.5]
- plus `0.5·max(0, 1 − progress/progress_ref)` and `0.2·mean|r|/0.30`

Per nominal episode: numerical 4, fall 3, else `2·max(0, 1 − progress/progress_ref) +
1·activation_fraction + 0.5·mean|r|/0.30`. `progress_ref` is the base policy's no-push progress
for the same command and model. `J = mean(attack) + mean(nominal)`.

## Search

CMA-ES over `x ∈ [−1, 1]^10`; gains `= x·max`, margin and width `= x + 1`, amplitude
`= 0.15·(x + 1)`. x0 = 0 (gains 0, margin 1, width 1, amplitude 0.15); σ0 = 0.3; popsize 8;
private PCG64 generator (M4 pattern). Gains at 0 give the zero residual, so the
baseline is the start point. Evaluation of one generation = (popsize × training episodes) tasks on
a process pool (workers initialised once with the stack and scenario models); results are in task
order, so parallel = sequential. Stop at generation cap, rollout budget, wall-clock cap or
cancellation; record the reason. The candidate is the best-ever training objective.

## Records

`runs/<…>/<id>/candidate.json` (schema 1): id `cand-<sha12>`, status `candidate`, base policy
sha, reflex schema version, parameters (named), calibration (values + sha), sets file sha and ids,
optimizer config and seed, accounting, wall time, stop reason, generation history, baseline and
candidate per-episode metrics for training and validation preview, provenance (git, versions,
scene transform), `reproducibility_sha256` over the deterministic content. Written once.

## API

`GET /api/hardening` (attacks, defaults, runs) · `POST /api/hardening/runs` · `GET
/api/hardening/runs/{id}` · `POST /api/hardening/runs/{id}/cancel` · `GET
/api/hardening/runs/{id}/replay?attack=<id>&ghost=baseline|matched`. Ids `hd-<UTC>-<seed>`.

## Web workspace

`App` renders header + mode tabs, then `.workspace` = left `.side` (active mode's controls) and
right `.stage` (one `ReplayViewer`). Each mode renders its long reports into a `#details` node via
a portal, so state stays in one component. `ReplayViewer` creates renderer/scene/camera once per
model and swaps robot meshes per replay, keeping the camera offset. On load the stage shows the
hidden-phase verified failure (re-simulated, labelled "Sample replay").
