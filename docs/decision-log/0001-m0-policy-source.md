# ADR 0001 — M0 policy source, MuJoCo pin and package layout

- Status: accepted
- Date: 2026-11-21
- Milestone: M0 (`.kiro/specs/m0-walking-go1/`)

## Context

M0 must run an existing pretrained Go1 walking policy through native MuJoCo on a Mac without CUDA
(`delivery.md` §3). The steering files set constraints that this choice departs from:

- `tech.md` §2 names MuJoCo 3.17 as the compatibility baseline.
- `structure.md` §1 places the Python package at a top-level `robustlab/` directory.
- `product.md` and `tech.md` assume a baseline policy trained in-house with Brax PPO.

`delivery.md` is the execution authority until submission. It asks for a public, redistributable,
already-trained Go1 policy with matching assets, actor-only weights in a safe format, and a
`src/robustlab/` layout (§7).

The Task 1 spike (`docs/kiro_build_log.md`) ran weeble's upstream runner with `go1-walk-s0` and
observed 9.263 m forward displacement in 10 s, max tilt 3.90°, no non-finite steps.

## Decision

1. **Policy source:** weeble commit `60878341072256ccc68f6118ec23a298c55e1bb9`, policy
   `go1-walk-s0`, JSON export (actor weights and observation normalizer; no pickle or ONNX runtime).
2. **Scene:** the exact-export `assets/go1/walk.xml` that the policy was exported with, derived from
   MuJoCo Playground commit `d59156e099a8785dd58fbce2e25a6ee1f7b5800c` and Menagerie commit
   `1b86ece576591213e2b666ebf59508454200ca97`. The observation and action contract stays as
   exported; we do not rewrite it for a different scene.
3. **MuJoCo pin:** `mujoco==3.15.0`, the version the XML was exported and the policy validated
   with, instead of `tech.md`'s 3.17.
4. **Layout:** `src/robustlab/` per `delivery.md` §7, instead of `structure.md`'s top-level
   `robustlab/`.
5. **Interpreter:** the local Python 3.12.2 (`--python <local python3.12>` with
   `UV_PYTHON_DOWNLOADS=never`), because uv-managed CPython downloads timed out on this network.

## Consequences

- The scene is feet-only: non-foot geoms do not collide with the floor. Fall detection in M0 relies
  on height, tilt and non-finite checks; non-foot contact detection is deferred to M1.
- The policy was trained with MJX-Warp and runs here in C MuJoCo. Parity is not assumed; the M0
  spec plans parity tests of our runner against weeble's reference runner (not yet implemented).
- The policy JSON has no separate license file. We assume it is covered by the weeble repository's
  MIT license and record that assumption in `THIRD_PARTY_NOTICES.md`.
- The project depends on an older MuJoCo than `tech.md` specifies. Any later browser/WASM work must
  match 3.15.0 or migrate first.
- Setup instructions must document the local-interpreter workaround.

**Migration trigger:** revisit the 3.15.0 pin if MuJoCo 3.17 parity is needed (for example, browser
WASM 3.17 or a dependency requiring it). A migration needs a re-run of the walking and parity tests
on 3.17 with recorded tolerances.

## Alternatives considered

- **MuJoCo Playground `rss2025` `go1_policy.onnx` with the rough-terrain scene.** Kept as fallback.
  Not chosen because it needs an ONNX runtime and the spike showed the weeble pair works.
- **Train our own policy (Brax PPO).** Out of scope before submission per `delivery.md` (no GPU
  training, no JAX/MJX/Brax runtime).
