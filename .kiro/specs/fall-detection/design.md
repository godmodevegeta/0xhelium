# Design Document: M1 — Deterministic shove and authoritative fall detection

## Overview

M1 adds three things to the M0 runner, and changes nothing about how the policy is driven:

1. A **Fall_Contact_Scene** compiled from the pinned `walk.xml` in code, where trunk, hip, thigh and
   upper-shin collision geoms can touch the floor.
2. A **Fall_Monitor** that inspects the state after every physics substep and produces one
   `FallResult`.
3. A **PushScenario** that applies a constant force to the trunk during an exact substep interval.

## MuJoCo background used by this design

**External forces.** `MjData.xfrc_applied` is an `nbody × 6` array: a Cartesian force (N) and torque
(N·m) in world coordinates, applied at each body's centre of mass. MuJoCo maps it into generalized
forces (`qfrc_applied` adds separately) during every `mj_step`. It persists in `MjData` until changed,
so a force left in place keeps acting. `mj_resetData` zeros it. A force F held for duration T delivers
an impulse F·T (N·s); for the 12.74 kg Go1 a 10 N·s impulse changes the centre-of-mass velocity by
about 0.78 m/s. The trunk centre of mass is not the whole-robot centre of mass, so a pure trunk force
also produces some rotation through the legs.

**Contact geometry.** Every geom has `contype` and `conaffinity` bitmasks. Two geoms are tested for
collision only if `(contype1 & conaffinity2) || (contype2 & conaffinity1)` and they are not in the same
or parent/child body. In `walk.xml` the floor has `contype=1, conaffinity=0`, the four foot spheres
have `conaffinity=1`, and every other robot geom has `0/0`, so only feet can touch the floor and the
trunk would sink through it. Each detected contact appears in `MjData.contact` with the two geom ids
and `dist` (negative = penetration). Collision detection runs inside `mj_step` on the state *before*
integration, so after a step `data.contact` describes the state at the start of that step.

**Falls.** MuJoCo has no notion of a fall. Common simulation proxies, all used here:
non-foot body contact with the ground; trunk tilt, measured as the world-z component of the trunk z
axis, `up_z = 1 − 2(qx² + qy²)` for the free-joint quaternion (a 45° tilt is `up_z < cos 45°`); trunk
height `qpos[2]` below a threshold; and numerical failure (non-finite state, or MuJoCo warnings such
as `BADQACC`, after which MuJoCo resets the data itself and the trajectory is no longer physical).
A numerical failure is not a physical fall and is reported separately.

## Measured constraint that shaped the scene

Enabling floor collision for *every* non-foot geom (as weeble's `getup.xml` does) changed nominal
walking. Probe at the M0 baseline code, 10 s episodes:

| Command | Non-foot floor contacts (substeps) | max \|qpos − M0\| |
| --- | --- | --- |
| (0, 0, 0) | none | 0 |
| (1, 0, 0) | `RR_calf` lower capsule, 5 | 2.0e-6 m |
| (1.5, 0, 0) | lower capsule on all four calves, 43, up to 7.1 mm deep | 0.41 m |

Only the lower shin capsule (class `calf2`, just above the foot) touched the floor. The policy was
trained in the feet-only scene where that capsule passes through the floor, so making it solid
changes the gait the policy was trained for, and counting its contact as a fall would label normal
fast walking a fall. With the lower shin left non-colliding and every other non-foot geom made
floor-collidable, all 29 probed commands (the 27-point ±1.5/±0.8/±1.2 grid plus (1,0,0), (0.5,0,0))
matched M0 with max |qpos − M0| = 0.

**Decision:** Fall_Geoms = trunk (box + 2 cylinders), hip cylinders, thigh capsules (3 per leg) and
the upper shin capsule `calf1` (knee end). Lower_Shin_Geoms (`calf2`) stay as trained. Limitation: a
fall that lands only on lower shins and feet is caught by tilt or height, not by contact.

`getup.xml` was used as a reference only; it also removes the trunk box and keeps one hip cylinder
per leg. We keep all `walk.xml` geoms and change only `conaffinity`, so geometry, mass and inertia are
untouched.

## Components

### `physics/scene.py` — Scene_Builder

- `SCENES = ("fall-contact", "feet-only")`, `SCENE_TRANSFORM_VERSION = 1`.
- `build_scene(xml_path, scene) -> SceneInfo` (model + description).
  - `feet-only`: `load_model` (identical to M0).
  - `fall-contact`: `MjSpec.from_file`; for each body, the group-3 geoms that are not feet; set
    `conaffinity = 1` unless the geom's default class is `calf2`; compile.
- `SceneInfo`: `model`, `name`, `transform_version`, `base_xml_sha256`, `fall_geoms` (list of
  `{body, class, geom_id}`), `unchanged_geoms`, `transform_sha256` (SHA-256 of the canonical JSON of
  the previous fields).
- Selection is by class name, which is stable in the pinned XML; a test asserts the expected counts
  (33 non-foot collision geoms, 29 Fall_Geoms, 4 Lower_Shin_Geoms).

### `physics/fall.py` — Fall_Monitor and FallResult

```text
REASONS = (nonfinite_state, mujoco_warning, nonfoot_contact, tilt, low_height)  # tie order
PHYSICAL = {nonfoot_contact, tilt, low_height}; NUMERICAL = {nonfinite_state, mujoco_warning}

FallEvent  {reason, substep, time_s, detail}
FallResult {fell, numerical_failure, kind: none|physical|numerical, reason, time_s, substep,
            events: {reason: FallEvent}, nonfoot_contact_detection, nonfoot_contact_substeps,
            criteria: {tilt_deg, min_trunk_z}}
```

- `FallMonitor(model, ids, criteria)` precomputes `nonfoot_mask[ngeom]`: robot geoms (body ≠ world)
  that are not feet and collide with the floor by the bitmask rule.
- `check(data, substep) -> bool` after each `mj_step`; `substep` is the 1-based count since reset,
  `time_s = substep · sim_dt`. State checks (finite, tilt, height) use the post-step state at
  `time_s`. Contact checks use `data.contact`, which describes the state at the start of the step,
  so a contact event gets time `(substep − 1) · sim_dt`. Warnings are checked as any non-zero
  `data.warning[i].number`. Only the first occurrence per reason is stored; it returns whether a
  numerical event has occurred.
- Pure predicates `tilt_violated(up_z, cos_tilt)` and `height_violated(z, min_z)` are shared with
  M0's `upright_checks`, so the threshold comparison exists in one place. Non-finite values violate.

### `experiments/push.py` — PushScenario

- Frozen dataclass with `force_n, direction (tuple[3]), start_s, duration_s`; `__post_init__`
  validates (Requirement 5) and raises `PushError(field, reason)`.
- `heading_to_world(direction, yaw) -> np.ndarray` (rotation about z).
- `substep_window(sim_dt) -> (first, end)`, integer substeps `[first, end)` where
  `first = round(start_s / sim_dt)`; multiples are checked with a 1e-9 s tolerance.
- `DIRECTIONS = {forward: (1,0,0), backward: (−1,0,0), left: (0,1,0), right: (0,−1,0)}`.

### `experiments/walk.py` changes

- `EpisodeConfig` gains `push: PushScenario | None = None`.
- `EpisodeRunner.tick()` replaces `mj_step(nstep=5)` by a loop of 5 `mj_step` calls. Before each, it
  zeros `xfrc_applied` and, inside the window, writes the world force to the trunk row. After each,
  it calls the Fall_Monitor. Same order of floating-point operations as `nstep=5` (verified against
  the golden trajectories and the oracle tests).
- `EpisodeRecord` gains `nonfoot_contact [T] bool`, `push_impulse [T,3]`, `fall: FallResult`,
  `truncated: bool`, `push_world_force`, `push_window`.
- `run()` validates the push against the horizon, stops after a Tick with a numerical event and
  trims the arrays.
- `evaluate()` adds `outcome["fall"] = FallResult.to_dict()`; progress cut-off is the earlier of the
  per-Tick first violation and the FallResult time; `failed_checks` gains `fall` or
  `numerical_failure`.

### Trace and CLI

- Trajectory schema 2 adds `nonfoot_contact`, `push_impulse`; manifest schema 2 adds `scene`, `push`
  and `outcome.fall`. `read_run` accepts schema 1 (M0 runs) and 2.
- `walk --scene fall-contact|feet-only` (default `fall-contact`); push flags all-or-none. The summary
  prints `fell`, `reason`, `time`, `numerical failure`, the push, and its world vector and impulse.

## Correctness properties (tests)

| # | Property | Requirements |
| --- | --- | --- |
| 12 | Heading-to-world rotation preserves magnitude and z, and equals R_z(yaw)·d for random yaws | 6.1 |
| 13 | Push window: for random valid (start, duration), xfrc on the trunk is the world force on exactly the substeps in `[first, end)` and zero elsewhere and on other bodies | 6.2, 6.3 |
| 14 | Invalid PushScenarios are rejected naming the field | 5.1, 5.2 |
| 15 | FallResult primary reason is the earliest event with the fixed tie order, for random event sets | 4.3–4.5 |
| 16 | Tilt and height predicates at exact boundaries and for non-finite values | 4.1 |

Example and integration tests: golden reproduction in both scenes; zero force and zero duration equal
nominal; reset clears xfrc; a lowered trunk on the floor triggers `nonfoot_contact` with the trunk
named; a feet-only scene reports detection unavailable; a forced NaN gives `numerical`; nominal
walking has foot contacts and no fall; the sideways-shove run is deterministic.

## Error handling

`PushError` and horizon violations exit 2 from the CLI before any simulation. A numerical failure is
an outcome, not an exception: the run is written with `truncated: true`.
