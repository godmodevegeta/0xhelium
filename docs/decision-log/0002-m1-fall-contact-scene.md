# ADR 0002 — M1 fall-contact scene, substep fall detection and trunk push

Status: accepted for M1 (branch `m1-fall-detection`, from `m0-baseline` = `cfe4b13`).

## Context

The pinned `walk.xml` lets only the four foot spheres touch the floor, so a fallen trunk sinks
through the ground and non-foot contact cannot be observed. `delivery.md` M1 needs an authoritative
fall verdict that includes non-foot ground contact, and a precisely timed shove, without changing
the policy contract or breaking the verified M0 walking.

## Decision

1. **Derive the scene in code, keep the pinned XML.** `physics/scene.py` loads `walk.xml` with
   `MjSpec` and sets `conaffinity = 1` on non-foot collision geoms. Robot `contype` stays 0, so no
   robot-robot pairs are created. Geometry, masses, inertias, joints, actuators, sensors and options
   are unchanged; a test checks that only `geom_conaffinity`, `body_conaffinity` and derived BVH
   arrays differ.
2. **Leave the lower shin capsule (`calf2`) non-colliding.** Measured at the M0 code: making all
   non-foot geoms solid (as weeble's `getup.xml` does) put the lower shins in floor contact during
   normal walking (5 substeps at 1.0 m/s, 43 at 1.5 m/s, up to 7.1 mm deep) and changed the
   trajectory by up to 0.41 m at 1.5 m/s. The policy was trained in the feet-only scene. With
   `calf2` excluded, 29 commands (the 27-point grid at ±1.5/±0.8/±1.2 plus two more) matched M0
   exactly (max |Δqpos| = 0).
3. **Check falls after every physics substep** (`physics/fall.py`), not only at control Ticks, with
   one `FallResult`: first event per reason, the earliest as primary, physical and numerical kinds
   kept apart. Thresholds come from the existing `WalkCriteria` (45°, 0.18 m).
4. **Push through `xfrc_applied` on the trunk, per substep**, inside an exact integer substep
   window; start and duration must be multiples of 0.004 s. Direction is given in the initial-heading
   frame and rotated to world once at reset.
5. **Default `walk --scene fall-contact`.** `--scene feet-only` reproduces the pinned M0 model.

## Consequences

- Nominal runs in both scenes reproduce the M0 golden trajectories exactly, so M0 evidence stays
  valid.
- A fall that touches the floor only with the lower shins and feet is detected by tilt or height,
  not by contact.
- After a fall, the trunk, hips, thighs and knees rest on the floor instead of passing through it,
  which makes replays physically plausible from that point. Those post-fall dynamics were never in
  the policy's training distribution.
- Tick latency roughly doubles (Python substep loop plus monitor), still about 100× under the 25 ms
  budget.

## Revisit when

A different policy trained with full collisions is adopted, or terrain features (curbs, steps) make
lower-shin contact a meaningful failure signal.
