# Technical Architecture and Engineering Standards

> **Deadline execution override:** Implement only the technical components required by `delivery.md` M0–M7. The JAX/MJX/Brax/PPO stack and full 80-slot simulator are *future architecture*, not required for the 2026-12-04 hackathon. Use native MuJoCo + NumPy and externally sourced, verified Go1 actor weights in this release. The CPU residual adapter is a separately named, bounded controller augmentation, not a substitute claimed to be equivalent to PPO. Do not install contradictory dependency families for a single demo.


> Kiro steering — technical design decisions, invariants, interfaces, runtime, and tooling. Implement this system from an empty repository. `product.md` states what users need; `structure.md` defines where modules belong. Choices below are defaults for the **new** codebase; validate dependency compatibility against a committed lockfile before production use.

## 1. Architectural shape and trust boundaries

```text
Browser / CLI
    | typed HTTPS requests, SSE events, immutable artifacts
FastAPI application layer -----------------------------------> job metadata + audit logs
    |                             |                                 |
    | invokes providers          | schedules isolated GPU work      | persists immutable manifests
    v                             v                                 v
AI provider adapters        simulation / PPO workers              artifacts + policy registry
    |                             |
    v                             v
validated Program ------> compiler(theta) ---> MuJoCo/MJX -> batched rollout metrics
                                  ^                    |                 |
                                  |                    |                 v
                            scenario optimizer <-------+          counterexample bank
                                                                     |
                                                               PPO fine-tuning
                                                                     |
                                                             independent gate
                                                                     |
                                                              atomic promotion
```

**Decisions**

- `T-001` Keep policy inference, physics, search and promotion deterministic with respect to explicit inputs; AI output is an untrusted proposal.
- `T-002` Use a modular Python monolith for the research kernel, a separate HTTP app, an isolated training process, and a TypeScript SPA. Avoid microservices until a measured operational need exists.
- `T-003` Separate *control plane* (requests, jobs, providers, lineage, reports) from *data plane* (JAX arrays, MJX models, policy parameters, optimizer batches). Do not import FastAPI from the physics kernel.
- `T-004` Prefer immutable, content-addressed experiment inputs and outputs; mutable 'latest' is only a pointer to an accepted immutable version.
- `T-005` Bound expensive tasks with one GPU execution slot and one training/promotion slot in single-machine mode. Only one candidate can be promoted atomically at a time.
- `T-006` Provider adapters consume/emit typed domain DTOs; they never mutate `theta`, model state or evaluation metrics.
- `T-007` Treat reusable compiled computation shapes as an interface contract: same canonical dimension, fixed number of free parameters, stable bank capacity and fixed evaluator record variants.

## 2. Technology selection and compatibility baseline

| Layer | Initial choice | Reason and constraints |
| --- | --- | --- |
| Main language | Python 3.12 | Supports scientific stack, typing and orchestration; use exactly one interpreter range in lockfile |
| Environment/package manager | `uv`, `pyproject.toml`, `uv.lock` | Reproducible dependency resolution and isolated environments |
| Numerical arrays | NumPy | Host-side metadata, serialized vectors, simple CPU operations |
| Accelerator computing | JAX 0.9.x + XLA | JIT, functional PRNG, `vmap`, `lax.scan`, compiled batched rollouts |
| Physics | MuJoCo 3.17 compatibility baseline and MJX | Contact/collision simulation, GPU-capable vectorization; pin compatible native and browser versions |
| Robot locomotion environments | MuJoCo Playground | Existing joystick locomotion tasks and robot assets; inspect individual asset licenses |
| RL trainer | Brax PPO | Actor/critic training, checkpoint callback, observation statistics |
| Derivative-free optimization | `evosax` CMA-ES | Mixed physics is not reliably differentiable through falls/collisions |
| Schema validation | Pydantic v2 | Discriminated-union programs, forbidden unknown keys, generated JSON Schema |
| Model API abstraction | provider-specific adapters + OpenAI-compatible SDK where suitable | Interchangeable hosted or local language/vision models |
| HTTP client | `httpx` | Timeouts, retries and non-SDK upload/segmentation endpoints |
| Backend | FastAPI + Uvicorn + `python-multipart` | Typed endpoints, file uploads, SSE, static build serving |
| Media processing | FFmpeg, Pillow, image/video utilities | Render clips, thumbnails, mobile/social exports, EXIF normalization |
| Web | React 18, TypeScript 5.6, Vite 6, React Router 6 | Browser SPA with typed view models and route-level features |
| Styling | Tailwind CSS 4 + Lucide | Consistent layout and icons with minimal bespoke styling |
| 3D and live physics | Three.js + matching MuJoCo WebAssembly | Prerecorded 3D replay and actual on-page policy simulation are separate modes |
| Testing/lint | pytest, Ruff, TypeScript compiler | Unit, property-based and integration verification; add a GPU validation target |
| Initial persistence | JSON/JSONL, NumPy `.npy`, model checkpoints, immutable manifest files | Transparent single-machine experimentation |
| Durable jobs (recommended) | SQLite WAL once API jobs survive restart | Avoid in-memory-only job state and non-atomic multi-file updates |

**Upstream compatibility warning (November 2026):** MuJoCo Playground 1.0 moved its primary environment authoring/training stack toward PyTorch, `reaf-sim` and MuJoCo Warp, while the older JAX/MJX/Brax locomotion environments remain in the legacy `rss2025` lineage. The JAX/Brax version table below is a **legacy compatibility target**, not a recommendation to install the latest `playground` release. At M1/M2 verify the exact `Go1JoystickFlatTerrain` imports, asset revisions, PPO config and checkpoint path against a pinned legacy checkout or compatible released version. Commit a lockfile and `docs/compatibility.md`; do not mix the two generations without an explicit migration ADR and parity evidence.

**Execution-order guardrail:** `delivery.md` governs milestone sequence. Python native MuJoCo simulation begins at M0; the canonical 80-slot compiler is built in M5 after a working Go1 policy, deterministic shove and reproducible evaluation exist. `tech.md` numerical contracts describe the end-state, not obligations to pre-scaffold code.

**Version guardrail:** Some Brax releases call JAX APIs that later JAX releases removed. For the initial verified dependency envelope target `jax>=0.9,<0.10`, with CUDA-enabled JAX wheels on Linux, and separately lock Brax/MuJoCo/MJX/Playground to tested compatible versions. The proposed versions are a bootstrap compatibility target, **not** a declaration that all latest upstream packages will be compatible. The browser physics package and Python MuJoCo must use an export-compatible model format and tested matching release (start with 3.17). Do not upgrade one component casually.

Supported runtime: Linux + NVIDIA GPU for training; WSL2 may work after driver verification. CPU-only mode must support schema tests, compiler tests and small native MuJoCo checks. A laptop GPU is not required to run the server/frontend test suite. CUDA builds use an explicit matching driver and toolkit baseline; don't silently assume the workstation is accelerator-capable.

## 3. Configuration philosophy

- `T-008` One typed `Settings` root validates configuration at startup; load environment from `.env` for local development. Do not scatter `os.getenv` through numerical modules.
- `T-009` Split `PhysicsConfig`, `SearchConfig`, `TrainingConfig`, `ModelRoles`, `ServerConfig`, `QuotaConfig` and `ExperimentConfig`. Export the effective config to every run manifest.
- `T-010` Model roles are `text_compiler`, `investigator`, `vision_analyst`, `segmenter`, `transcriber`, `report_writer`. A role selects a provider, model ID, endpoint, timeout, maximum tokens, sampling settings and ability flags.
- `T-011` Never hardcode credentials. Resolve keys from environment or a secrets manager; redact Authorization headers, raw keys and private media from logs.
- `T-012` Profile examples: `local` (1,024 training envs / 256 eval envs), `large_gpu` (8,192 / 2,048). Profile selection affects capacity, not physics semantics.
- `T-013` Establish persistent JAX compilation cache, e.g. `$HOME/.cache/robustness_studio_jax`. Set cache before loading any simulator/training modules. Cache compiled functions taking **policy params and `theta` as data arguments**, not static new Python objects per user scenario.
- `T-014` Record the simulator implementation (`jax`/other), backend, precision policy, host GPU, library versions, and layout version. A backend change is an explicit experiment dimension.

Proposed `.env.example` (names are project-owned and provider-neutral):

```dotenv
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000
APP_DATA_DIR=runs
APP_PROFILE=local
APP_ROBOT=go1
SIM_BACKEND=jax
JAX_CACHE_DIR=~/.cache/robustness_studio_jax
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
PATCH_WORKER_MEM_FRACTION=0.6

TEXT_COMPILER_PROVIDER=disabled
TEXT_COMPILER_MODEL=
INVESTIGATOR_PROVIDER=disabled
INVESTIGATOR_MODEL=
VISION_ANALYST_PROVIDER=disabled
VISION_ANALYST_MODEL=
SEGMENTER_PROVIDER=disabled
SEGMENTER_MODEL=
TRANSCRIBER_PROVIDER=disabled
TRANSCRIBER_MODEL=
REPORT_WRITER_PROVIDER=disabled
REPORT_WRITER_MODEL=

OPENAI_COMPAT_BASE_URL=
OPENAI_COMPAT_API_KEY=
LOCAL_MODEL_BASE_URL=http://127.0.0.1:8080/v1

TRAINING_MAX_STEPS=8000000
TRAINING_MIN_STEPS=2000000
SEARCH_MAX_GENERATIONS=12
SEARCH_POPULATION=24
SEARCH_SEEDS=8
AI_MONTHLY_BUDGET_USD=0
AI_STOP_AT_USD=0
PUBLIC_COMMUNITY_ENABLED=false
AUTONOMOUS_INVESTIGATION_ENABLED=false
```

**Important:** a 'disabled' provider is intentional. The deterministic core must still run. Per-role configuration overrides shared provider defaults; one model may serve several roles if its capabilities permit. A segmentation service with a proprietary API is an optional adapter, never a required foundational dependency.

## 4. Robot registry — metadata vs resolved simulator IDs

Define an immutable `RobotDefinition` containing **names and physical parameters only**. Separate `ResolvedRobotSpec` maps those names into actual integer body, geom, site, actuator, joint and collision-pair IDs after the MJCF model loads. This prevents hardcoded IDs from breaking when model compilation inserts geometry.

Required fields: `name`, Playground environment key, ordered feet, per-leg joint names, ordered actuators, trunk and foot site names, root minimum height, maximum tilt, foot-friction strategy (per-geom or explicit contact pairs), nominal friction/cap, collision shape selectors, additional primitive fall geoms, actor observation block offsets, privileged critic offsets, command bounds, nominal command, payload envelope, force envelope, base DR ranges, hip height and available terrain box count.

Starting robot presets:

| Parameter | Quadruped `go1` | Humanoid `g1` |
| --- | --- | --- |
| Source locomotion task | `Go1JoystickFlatTerrain` | `G1JoystickFlatTerrain` |
| Named feet | `FR, FL, RR, RL` | `L, R` |
| Lower-body joint families | hip, thigh, calf (4 × 3) | hip pitch/roll/yaw, knee, ankle pitch/roll (2 × 6) |
| Trunk body | `trunk` | `torso_link` |
| Root fall height | 0.18 m | 0.45 m |
| Maximum trunk tilt | 45° | 45° |
| Actor base observation | 48 values | 103 values |
| Observation history | 3 frames (144 for quadruped) | 3 frames (309 for humanoid) |
| Nominal commanded forward speed | 1.0 m/s | 0.8 m/s |
| Forward/side/yaw bounds | ±1.5 m/s, ±0.8 m/s, ±1.2 rad/s | ±1.0 m/s, ±0.5 m/s, ±1.0 rad/s |
| Max payload | 5 kg | 10 kg |
| Max force per component | ±150 N | ±300 N |
| Terrain boxes | 8 | Initially 0 (flat) |
| Ground friction mechanism | Foot geom priority; neutral effective cap 0.6 | Explicit foot-floor contact pairs |
| Extra collision geoms | trunk and leg bodies selected | low-mass/massless pelvis, torso, head, shin primitives as needed |

These robot defaults are **configuration examples** to validate experimentally; verify actual model names and observation offsets from the installed Playground version. Do not assume every robot fits the same kinematic or collision model.

### Contact and fall detection decisions

- `T-015` Ground-contact semantics are part of the experiment: distinguish actual contact, fall geometry, and ordinary environment termination.
- `T-016` Some default locomotion scenes allow only feet to touch the floor; this makes an apparent fallen body pass through the ground. Add collision-capable non-foot body geoms where necessary without changing the robot's mass/inertia.
- `T-017` For per-foot friction, the floor's contact priority can override foot friction. Copy the nominal floor contact parameters to foot geoms, raise foot priority, and cap effective contact friction at the trained nominal value when necessary. Add a neutral-rollout parity test.
- `T-018` For explicit contact pairs, set the pair's friction directly rather than applying the per-geom workaround.
- `T-019` Terrain body collisions can create enormous contact arrays. Let feet contact terrain collision boxes; detect non-foot contact with raised terrain geometrically rather than enabling all body-to-box collision pairs, and document approximations.
- `T-020` For geometric fall checking, sample collision primitives across surfaces / capsule axes at no more than 0.02 m between points; evaluate intersection against floor and terrain surfaces. Combine this with trunk orientation, root height and non-finite checks.
- `T-021` A 45° tilt is evaluated using the trunk up-vector, i.e. `up_z < cos(pi/4)`, not by comparing arbitrary Euler angles.
- `T-022` Raw task termination may penalize foot or limb contacts while the robot is upright. Use a stricter physical fall metric for reports. Training may still respect raw termination as an episode boundary.
- `T-023` Disable underlying environment's random external pushes in deterministic attack evaluation, otherwise the experiment contains undocumented forces.

## 5. The canonical perturbation layout

`theta` is a float32 vector of fixed dimension `D=80`; unused dimensions stay zero/neutral. A `Slot` has `{name, length, neutral, lower, upper, unit, doc, neutral_off}`. A `Layout` constructs contiguous slices, `NEUTRAL`, `LO`, `HI`, `SPAN`, and slot names for each robot. Export a human-readable schema and per-robot `layout_version` in every run.

**Slot order is a compatibility interface. Never insert earlier slots or reuse an existing meaning. Add at the end, bump `layout_version`, supply migration, and snapshot old vectors.** For robot-specific shapes, pad to the same dimension; the used slice count may differ. Assertions must reject a layout whose used count exceeds 80.

### Ordered slot catalog

| Group | Slot | Width | Neutral | Bounded interpretation |
| --- | --- | ---: | --- | --- |
| Contact | `foot_friction` | number of feet | robot nominal μ | 0.05–1.5 μ, effective contact cap may apply |
| Contact | `ice_mu` | 1 | 1.0 | 0.05–1.0 μ within patch |
| Contact | `ice_start` | 1 | 0 | 0–6 m along initial heading |
| Contact | `ice_length` | 1 | 0 | 0–4 m; zero disables patch |
| Body | `payload_mass` | 1 | 0 | 0–robot max kg |
| Body | `payload_offset` | 3 | 0 | robot-specific XYZ relative to trunk CoM |
| World | `slope` | 2 | 0 | pitch/roll each −0.3..0.3 rad |
| Actuation | `motor_strength` | number of joints | 1 | 0.3–1.0 gain and torque-limit multipliers |
| Actuation | `joint_friction` | 1 | 1 | 1–4 friction-loss multiplier |
| Actuation | `latency` | 1 | 0 | 0–2 control steps, quantized |
| External | `push1_force` | 3 | 0 | ± robot max per XYZ component, N |
| External | `push1_start` | 1 | 0 | 0–20 s |
| External | `push1_duration` | 1 | 0 | 0–0.5 s; zero disables |
| External | `push2_force` | 3 | 0 | second force vector |
| External | `push2_start` | 1 | 0 | second start time |
| External | `push2_duration` | 1 | 0 | second duration |
| Sensing | `obs_noise` | 1 | 0 | additional proprioceptive Gaussian std 0–0.3 |
| Sensing | `joint_offset` | number of joints | 0 | encoder bias −0.05..0.05 rad |
| Sensing | `imu_bias` | 2 | 0 | roll/pitch bias −0.08..0.08 rad |
| Sensing | `linvel_bias` | 3 | 0 | body linear-velocity bias −0.3..0.3 m/s |
| Task | `command_mode` | 1 | 0 | 1 enables user-specified joystick profile |
| Task | `command_a` | 3 | 0 | initial vx,vy,yaw in robot bounds |
| Task | `command_b` | 3 | 0 | post-switch vx,vy,yaw |
| Task | `command_switch` | 1 | 20 | time of command switch in seconds |
| Terrain* | `stairs_height` | 1 | 0 | 0–0.15 m riser; zero disables |
| Terrain* | `stairs_depth` | 1 | 0 (off marker) | 0.2–0.6 m when active |
| Terrain* | `stairs_count` | 1 | 0 | integer-rounded 0–8 |
| Terrain* | `stairs_start` | 1 | 0 (off marker) | 0.5–3 m when active |
| Course* | `ramp_angle` | 1 | 0 | 0–15 degrees |
| Course* | `ramp_start` | 1 | 0 (off marker) | 0.5–6 m |
| Course* | `ramp_length` | 1 | 0 (off marker) | 0.5–2.2 m |
| Course* | `stairs_landing` | 1 | 0 (off marker) | 1.5–3 m, zero disables descent |
| Course* | `ice2_slip` | 1 | 0 | `1 - mu`, 0–0.95, zero disables second patch |
| Course* | `ice2_start` | 1 | 0 | 0–6 m |
| Course* | `ice2_length` | 1 | 0 | 0–4 m; zero disables second patch |

`*` Terrain slots exist only for robot definitions that support preallocated terrain collision boxes. Some inactive terrain slots have zero as a neutral **off marker outside their active lower bound**. Exclude those from magnitude comparisons and give their normalized critic input a stable off-value instead of dividing by a fake span.

Implementation requirements:

- `T-024` A `theta` value is not a learned neural-network weight; it is a declarative physical parameter.
- `T-025` `layout.normalized(theta)` transforms physical coordinates for the privileged critic, in stable order and with documented padding behavior.
- `T-026` `theta` arrays are always validated for finite values, shape, dtype, robot mapping and version before being used.
- `T-027` For neutral `theta`, the patched simulation should agree with the original simulation within a declared numerical tolerance under the same seed.
- `T-028` Never infer whether terrain is present from an inactive slot whose neutral is an out-of-range off marker; use its activation condition (step count+height, ramp angle, patch length).

## 6. Scenario DSL and deterministic compiler

Use Pydantic discriminated unions, `extra='forbid'`, robot-specific dynamic enums, and generated JSON Schema in the AI prompt. `ScenarioProgram` fields: `name: str`, `hypothesis: str`, `ops: list[Op]` (1–6). All numeric operation fields use `Number = fixed float | [lower, upper]` subject to bounds. Search programs can include intervals; direct user attacks and deployment plans default to **fixed values only**.

Implement these **16** primitives:

1. `friction(feet[], mu)` — selected feet.
2. `ice_patch(mu, start, length)` — distance-localized friction.
3. `payload(mass, offset_x, offset_y, offset_z)` — adds mass, CoM and inertia effects.
4. `slope(pitch, roll)` — ground incline relative to initial heading.
5. `motor(legs[], joints[], strength)` — joint-specific gain/torque multiplier.
6. `joint_friction(scale)` — friction-loss multiplier.
7. `latency(steps)` — quantized action delay.
8. `push(force_x, force_y, force_z, start, duration)` — timed trunk force.
9. `sensor_noise(std)` — observation perturbation.
10. `encoder_offset(legs[], joints[], offset)` — joint zero error.
11. `imu_bias(roll, pitch)` — gravity/tilt sensor bias.
12. `velocity_bias(x, y, z)` — body velocity-estimate bias.
13. `command(a=[vx,vy,yaw], b?, switch?)` — task command and optional mid-episode switch.
14. `stairs(height, depth, count, start)` — straight rising flight.
15. `curb(height, start)` — single step with landing.
16. `course(segments[])` — ordered ground, stair, descent, ramp, ice and wet route.

A `course` has 1–6 segments, each with a `kind` discriminant. Segment choices: `stairs_up`, `stairs_down`, `curb`, `ramp`, `ice`, `wet`. Positions are laid out by a **cursor** along the initial heading, with `gap` defined from the prior segment's end (not the origin). Maintain a small state machine: ground/top/ramp. Validate raised walkway lengths, return-to-ground constraints, max one up-flight and one descent, ramp placement after other raised structures, geometric overlap and finite episode distance. Limit a route to at most two slippery patches, one ramp, and terrain-box capacity. `stairs_down` must follow a valid upward flight/curb. With a ramp, reserve two boxes; an up/down flight requires two boxes per riser.

**Compiler details:**

```text
K = 10 free scalar dimensions, padded
Compiled = {
  base_theta: float32[D],       # fixed values and neutral defaults
  map: float32[K,D],            # 0/1 mapping from free scalar to target slots
  lower: float32[K],
  upper: float32[K],
  free_names: string[],
  layout_version: int
}

theta(u) = where(map.sum(axis=0) > 0,
                 (lower + (upper - lower)*u) @ map,
                 base_theta)
```

- `T-029` A single searched value may set several related joints or feet; duplicate **free parameter** writes to the same slot must fail validation.
- `T-030` Do not allow more than ten free parameters, two push ops, one terrain op, or two low-friction patches.
- `T-031` Distinguish schema-invalid from physically-out-of-envelope. **Preferred new-project policy:** reject out-of-range input with actionable error instead of silently clipping; use clipping only as an explicit optimizer projection whose effect is recorded.
- `T-032` Validate positive segment gaps, monotonic course cursor, patch placement and terrain occupancy before compilation.
- `T-033` Use one source of truth for units and sign conventions in schema, API, compiler, rendered overlays and reports.
- `T-034` The provider sees only the schema, robot capabilities and bounds needed for proposal; no arbitrary code generation capability.

## 7. Physics implementation and terrain

### Static physics vs per-step physics vs observations

`apply_static(model, theta, initial_yaw, robot_spec)` returns an MJX model with friction, masses/inertia, motor parameters, joint friction and slope/gravity changes. `apply_step(state, theta, t, origin)` applies push windows, chosen command, friction patches, action delay and terrain poses. `perturb_observation(obs, theta, rng, robot_spec)` applies noise, biases and encoder offsets without changing ground truth. Keep these functions pure and JAX traceable.

- `T-035` Modify payload mass, inertia and center-of-mass together; changing mass alone misrepresents load dynamics.
- `T-036` Encode pushes in start-heading coordinates and rotate into the world frame once per step. Push is active only in `[start, start+duration)`.
- `T-037` Quantize 0–40 ms action latency into 0–2 control steps at 20 ms; carry a fixed-size action history buffer (3 actions).
- `T-038` Use an explicitly split/folded JAX PRNG for observation noise, episode resets and training randomization. No hidden host-level random sampling inside compiled steps.
- `T-039` Apply friction patch conditions by projected position along original heading, not current yaw, to avoid the patch following the robot during a turn.
- `T-040` Generate course geometries from a shared pure numerical module usable by NumPy (for export) and JAX (for simulation), rather than writing separate geometry code in the frontend.

### Preallocated terrain technique

- `T-041` Prebuild a fixed number of massless collision boxes on mocap bodies at environment creation. Park unused boxes far below the ground (e.g. z = −10 m).
- `T-042` Move their `mocap_pos/mocap_quat` at reset/step; do not create or remove geoms per scenario. This preserves model/contact shapes and avoids per-request XLA recompilation.
- `T-043` Keep terrain boxes at a shared fixed half-size, initially `(1.2, 1.5, 0.75)` m, spanning 2.4 m along the heading, 3.0 m wide, and 1.5 m vertically. Render actual visible upper surfaces rather than hidden solids to prevent coincident-face flicker.
- `T-044` Stairs: riser i begins at `start+i*depth`, height `(i+1)*height`, with boxes extending under the final landing. Curb = one riser.
- `T-045` Descent: use two boxes per riser (front/back), with a 1.5–3 m landing. Check capacity before placing.
- `T-046` Ramp: reserve final two boxes for incline and landing, angle in degrees, rise = horizontal_run × tan(angle); cap horizontal run at 2.2 m under the selected geometry.
- `T-047` Only feet physically collide with these terrain boxes to keep contact arrays small; non-foot intersections are fall-detected by sampled geometry. This is a deliberate approximation and requires targeted physics regression tests.
- `T-048` Do not add terrain collision geoms to an existing model without a version bump and parity retests; even extra unused contact allocations can change floating-point behavior.

## 8. Environment wrapper and observation integrity

Implement a `RobustnessEnvWrapper` below Brax's episode/auto-reset/vectorization wrappers. Do **not** also attach an upstream randomization wrapper that bypasses the custom wrapper by calling the underlying environment directly. Provide a per-episode sampler for both base domain randomization and attack `theta`.

Required per-environment state: `theta`, independent base-DR sample, perturbation RNG, elapsed control steps, delay buffer, original position/yaw, actor observation history, and a `resample` flag (false during evaluation). Actor observes 3 most recent state frames; privileged critic observes its normal fields plus normalized canonical `theta`. For `go1`, actor dimension is 3 × 48 = 144.

- `T-049` Base environment DR can include DOF friction-loss scale, armature scale, body mass scale, trunk center-of-mass shift and trunk mass offset. Sample once per episode.
- `T-050` Example DR envelopes: friction-loss multiplier 0.9–1.1 (quadruped) or 0.5–2.0 (humanoid); armature 1.0–1.05; body mass scale 0.9–1.1; CoM ±0.05 m; trunk mass offset ±1 kg. Terrain support bodies are not randomized.
- `T-051` Ensure `theta` is re-sampled after auto-reset for training but pinned for the entire evaluation episode.
- `T-052` Preserve informative negative rewards in difficult scenarios; do not clip all negative outcomes to zero. Include a calibrated fall penalty; document interaction with base reward scales.
- `T-053` Ensure training actor input doesn't expose privileged simulation truth through a convenience field, debug channel or exported model.
- `T-054` On checkpoint restoration, privileged `theta` critic-normalizer dimensions from nominal-only pretraining may have near-zero variance. Reset/pin their mean=0 and std=1 before fine-tuning so a real attack doesn't explode the normalized input.
- `T-055` Use observation history in fixed order (newest first) and test that browser inference implements the identical order.

## 9. Batched evaluator: algorithm and metrics

- `T-056` Define a pure single-rollout function `evaluate_one(policy_params, theta, rng, record)`; construct batch evaluation with `jax.vmap` over `(theta, seed)` pairs and use `jax.jit` with `record` as a static variant.
- `T-057` Use `jax.lax.scan` for the horizon. Do not auto-reset after a fall during evaluation; keep last known upright pose for progress and first-fall time.
- `T-058` Pass `theta` and policy parameters as **dynamic data inputs**. Never embed a user scenario or version-specific weights into a JIT closure that forces recompilation.
- `T-059` Record `fell`, `time_to_fall`, `track_err`, `progress`, `robustness`, `success`, `fall_gravity`, and optional `qpos` for recorded trajectories.
- `T-060` Tracking error is mean planar velocity distance from the commanded XY velocity over alive steps; tolerance starts at 0.4 m/s.
- `T-061` Progress is forward displacement in the initial-heading frame up to the last upright state; do not credit sliding or continued motion after falling.
- `T-062` Use a continuous robustness objective: surviving trials score [0,1] based on smallest upright margin and tracking score; failed trials score `−1 + first_fall_step/horizon` (earlier falls are worse).
- `T-063` Success = stayed upright AND tracking error below tolerance. A policy that simply stands still may be fall-free but unsuccessful.
- `T-064` Calculate fall direction from the gravity vector in the trunk frame at first fall. Save raw and categorized signatures.
- `T-065` Pair seeds across policy comparisons. Record evaluated seed list, batch dimensions, horizon, run/build fingerprint, and effective `theta`.
- `T-066` Keep recorded trajectories separate from metric evaluation because full pose tensors are costly. Compilation may have one `record=false` and one `record=true` executable.

## 10. Search, scoring, confirmation and shrinking

Use CMA-ES with a normalized `K=10` search interface. Work in unconstrained `z` coordinates and map via `u = sigmoid(z)` so every optimizer sample remains within `[0,1]`; convert `u` into the scenario program's numeric ranges and canonical `theta`. Include a random-search baseline.

Starting budgets: population 24; maximum generations 12; 8 seeds/candidate; fixed search slots per batch; stop when no fitness improvement ≥0.001 for four generations; optionally shorten agent searches to eight generations.

Default objective, minimized:

```text
fall_rate = mean(fell over search seeds)
plausibility_cost = sqrt(sum(normalized_active_deviation^2) / active_dimension_count)
fitness = -2 + plausibility_cost      if fall_rate >= 0.75
          mean(robustness)            otherwise
```

A NaN fitness maps to `+infinity`. The cost measures *normalized deviation* in the configured envelope; it is not a probability of occurrence in the real world. Exclude task commands, push timing, patch positions and terrain layout-only dimensions from severity cost; keep actual physical magnitudes included. Group causes by dominant normalized family; archive fall directions separately.

- `T-067` Batch multiple scenario programs into fixed numbers of slots and pad absent/completed programs with neutral vectors, excluding padding from accounting.
- `T-068` Final confirmation uses 32 fresh seeds and archive threshold 0.5; search-stage threshold 0.75 is different and must remain visible in metrics.
- `T-069` Shrinking removes or neutralizes perturbation groups iteratively while preserving at least 0.8 measured failure rate on shrink seeds; persist original and shrunken forms.
- `T-070` Measure safety margins using bisection or 11-point sweeps between nominal and failure values; indicate non-monotonic outcomes and uncertainty rather than fabricating a threshold.
- `T-071` Archive refutations and invalid hypotheses, not just wins. Distinct discoveries are keyed by scenario semantics + physical failure signature, not prompt wording alone.

## 11. Counterexample bank and retraining

### Curation

For a bank scenario, estimate current fall probability `p` using a fresh test batch (16 seeds starting point). Quarantine candidates where `p >= 0.95` as potentially unlearnable, but retain them in the research archive and holdout reports. Weight kept scenarios by `max(p(1-p), 0.03)` to emphasize learnable near-boundary cases while still replaying solved examples. Normalize family weights so friction does not swamp sensing, actuation and other failure families. Keep the archive separate from the ephemeral fixed-shape training bank.

### Fixed-shape bank

- `T-072` Bank capacity 512 physical vectors. For each new counterexample include the original and 48 locally jittered variants.
- `T-073` Jitter only active non-neutral slots with Gaussian sigma = 0.08 × slot span; do not jitter the command enable flag. Project to physical bounds after jitter and preserve the original.
- `T-074` Half the *sampling mass* of the bank is reserved for new failures, half for older replay if any. Pad unused rows by repeated valid rows with **zero probability**. If no replay exists, new examples get all non-nominal bank weight.
- `T-075` Per training episode, sample: 50% nominal, 40% weighted bank, 10% uniform over a 25%-wide physical envelope around nominal. Base DR remains separately controlled and logged.
- `T-076` Save the bank vectors and weights with a hash and the active layout version; validate before every worker consumes them.

### PPO and checkpoint rules

- `T-077` Use the robot's task-specific Playground PPO config as a starting point. Copy its effective hyperparameters into `config.json`, not just its name.
- `T-078` When restoring a policy, verify matching actor/critic network architecture, observation sizes, simulator task and normalization state.
- `T-079` Cap forced PPO environment resets per evaluation interval so the controller actually experiences late-episode events. Short training runs otherwise reset before attacks fire.
- `T-080` Start fine-tuning at learning rate 1e-4, max eight million simulated steps, checkpoint tests every one million steps and minimum two million steps.
- `T-081` Early stop only if every new target is handled and nominal movement is intact; an example threshold is at most 2/16 falls on the primary target. Always run final independent gate afterward.
- `T-082` Store every PPO checkpoint under `runs/<id>/checkpoints/<zero-padded-step>.pkl`; treat Python pickle as trusted **internal** checkpoint data, never an untrusted uploaded format.
- `T-083` Save `config.json`, `metrics.jsonl`, bank reference, seed, source policy manifest and final gate decision. Flush metrics after each evaluation callback.
- `T-084` Run PPO in a subprocess with a separately constrained GPU memory fraction and newline-delimited structured events. Do not block the FastAPI event loop with long GPU calls or media encoding.

### Promotion gate

Run candidate and parent on the **same 16 fixed seeds** for new attack suite, nominal forward task and held-out gauntlet. Apply the exact gate in `product.md`; save all before/after metrics and accepted flag. Training must be complete before the gate; the gate must not consume training-variant seeds as if they were holdout. One accepted candidate increments the lineage; rejected checkpoints go under a separate `rejected/` tree.

## 12. Provider-agnostic AI interface

Define protocols rather than direct SDK calls inside application logic:

```python
class TextModel(Protocol):
    def structured(self, prompt: Prompt, schema: dict, *, budget: CallBudget) -> ModelResult: ...
    def tools(self, messages: list[Message], tools: list[ToolSpec], *, budget: CallBudget) -> ToolResult: ...

class VisionModel(Protocol):
    def inspect(self, media: MediaRef, schema: dict, *, budget: CallBudget) -> ModelResult: ...

class Segmenter(Protocol):
    def segment(self, image: MediaRef, concepts: list[str]) -> SegmentationResult: ...

class Transcriber(Protocol):
    def transcribe(self, audio: MediaRef) -> Transcript: ...
```

Provider roles must declare whether they support JSON Schema, tool calling, images, video, audio, prompt caching and streaming. Fallbacks are explicit: a role lacking segmentation may produce *no mask* but still generate a scenario from text/vision; an unavailable vision provider must not silently invent visual measurements.

- `T-085` On structured responses, validate parsed JSON using the local Pydantic model regardless of provider advertised schema support.
- `T-086` Allow one feedback-based repair of invalid model output; keep an audit of the original, validator errors, repair and final accepted program.
- `T-087` Use request-content SHA-256 cache keys including model ID, settings, prompt/schema revision and media digest; never cache across users when private material would leak.
- `T-088` Store model id, latency, token usage, reasoning/token counts if supplied, cache hit, approximate billed cost and logical role. Do not log API keys.
- `T-089` SDK retries/timeouts are bounded (e.g. two retries; 300 s remote and larger local timeout only where justified). Retry only idempotent calls; timeouts and budget exhaustion appear in job results.
- `T-090` Local inference may use `llama.cpp`/vLLM with OpenAI-compatible endpoints. Capability maps handle differences between Chat Completions and Responses-like APIs; do not presume identical media/tool semantics.
- `T-091` For photo understanding, normalize EXIF rotation, limit maximum resolution (e.g. 1600 px side), and store input digest. Optional masks target named objects such as steps, ramps and wet/icy areas; reject low-confidence detections (start at 0.5) for automatic geometry counts.
- `T-092` Sample a limited set of frames for video providers without native video support, and label the resulting analysis as sampled-frame inference.
- `T-093` Optional speech transcription must preserve original transcript, time and provider; a local ASR engine is an allowed fallback.
- `T-094` Tool-calling AI may request only registered tools; no tool yields arbitrary code execution or unchecked shell/HTTP actions.

## 13. Backend API and events

A typed FastAPI service uses `/api` prefixes, snake_case JSON field names, stable IDs and Pydantic DTOs. Initial endpoints:

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/api/scenarios/compile` | Validate program and preview canonical vector + warnings |
| POST | `/api/attacks` | Submit text/JSON attack; return job ID |
| POST | `/api/attacks/voice` | Audio transcription then attack |
| POST | `/api/scenes` | Image/video analysis and scenario generation |
| GET | `/api/attacks/{id}` | Experiment record + metrics, subject to authorization |
| GET | `/api/attacks` | Search/filter archive by robot, family, version and status |
| POST | `/api/searches` | CMA-ES discovery for validated program |
| POST | `/api/patches` | Start isolated fine-tune/promotion job |
| GET | `/api/policies` | Version lineage and active pointer |
| GET | `/api/policies/{id}` | Manifest and gate results |
| POST | `/api/investigations` | Bounded autonomous tool loop |
| POST | `/api/deployments` | Generate site-based test suite |
| POST | `/api/reports` | Reproducible readiness review |
| GET | `/api/jobs/{id}` | Current job state and result |
| GET | `/api/jobs/{id}/events` | EventSource stream and replay by Last-Event-ID |
| GET | `/api/replays/{id}` | Prerecorded poses/metadata |
| GET | `/api/clips/{id}` | Validated generated video |
| GET | `/api/play/model` | Exported matching browser model |
| GET | `/api/play/policy/{version}` | Immutable browser actor binary |
| POST | `/api/play/shoves` | Server-verify a bounded browser-generated shove |
| GET | `/api/system/limits` | Active quotas, budget mode and GPU queue status |

Return job IDs immediately for expensive operations. Event schema:

```json
{
  "seq": 18,
  "job_id": "job_...",
  "kind": "patch",
  "stage": "attempt",
  "timestamp": "2026-11-21T00:00:00Z",
  "correlation_id": "...",
  "data": {"checkpoint_steps": 3000000, "fall_rate": 0.125, "nominal_ok": true}
}
```

Jobs transition `queued -> running -> done|failed|cancelled`; events are append-only. SSE sends an increasing ID and heartbeats; replay missed events using `Last-Event-ID`. Use a 15-second keepalive, 2-second recommended browser retry, and no proxy buffering. A simple server may poll state twice a second. Never return success before the worker has committed its artifact/result.

- `T-095` Keep heavy GPU work behind a scheduler interface; use `asyncio.to_thread` for bounded native calls and subprocess for PPO.
- `T-096` Offload rendering to a small thread pool; no blocking MP4 conversion on the event loop.
- `T-097` Gzip ordinary API/static text but exclude `application/wasm` when a reverse proxy handles the large WASM binary, to avoid long synchronous compression.
- `T-098` For the built SPA, serve `index.html` on client-side routes but do not swallow real 404s for asset URLs containing extensions.
- `T-099` Validate file suffix **and** actual media type, cap voice at e.g. 4 MiB and image/video at e.g. 25 MiB, and prevent path traversal by generating internal IDs.
- `T-100` All user-defined IDs in file paths must match a tight allow-list; never interpolate user strings into subprocess shell command lines.
- `T-101` Source-generated social images/video must escape or render user text through trusted drawing libraries before passing assets to FFmpeg; moderation affects shareable artifacts and cache keys.

## 14. Lineage, files, reproducibility, and write integrity

Initial storage may use:

```text
runs/
  evaluations/<id>/{manifest.json,metrics.json,trajectories/,clips/}
  searches/<id>/{manifest.json,history.jsonl,best.npy,confirmation.json}
  training/<id>/{config.json,metrics.jsonl,checkpoints/,bank.npy,weights.npy,gate.json}
  serve/{jobs.sqlite,attacks.jsonl,policy_versions.json,active_policy.json}
  artifacts/policies/<version>/{manifest.json,checkpoint/,browser/}
  artifacts/rejected/<id>/...
  llm/{cache/,usage.jsonl}
```

- `T-102` Every file payload has a schema version and an immutable content hash in its manifest.
- `T-103` Write version metadata through a temporary file followed by atomic rename; guard active-pointer updates with a promotion lock.
- `T-104` New policy versions are monotonic and never reuse identifiers, including after rejected attempts.
- `T-105` Store parent version, trained-on attacks, data hashes, PPO config, gate evaluation, training seed and code revision in manifest.
- `T-106` Replay bank and status maps update only after acceptance; use transactional storage for multi-file consistency, or a journal/recovery strategy if using files.
- `T-107` Use a durable SQLite/WAL job/event store in the API release so restart does not erase in-flight history. Jobs interrupted during restart move to an explicit recoverable/failed state; never pretend they completed.
- `T-108` Immutable artifacts served by ID can be cached; mutable active pointers and quota status must not be cached accidentally.
- `T-109` Track baseline checkpoints separately from accepted versions. Fresh clones do **not** assume pretrained weights exist; train or import a verified licensed checkpoint.
- `T-110` Metrics logs use newline-delimited JSON with timestamps, run id, job id, step, elapsed time and observable metrics; privacy-sensitive text and media are stored separately.

## 15. Web architecture and browser inference

Two sharply different 3D experiences:

1. **Replay:** server runs physics, exports timestep-by-timestep `qpos` plus scene and event metadata; Three.js interpolates and renders poses. Playback alone does not simulate contacts or train anything.
2. **Live playground:** load a robot `.mjb` model into MuJoCo WASM, construct the same observations as the Python actor, run a serialized feed-forward actor/normalizer in JavaScript, apply actions at 50 Hz, and let the WASM engine advance physics. The browser may send a bounded experiment to the server for authoritative reevaluation.

- `T-111` Use immutable policy binaries with explicit magic header, format version, tensor shapes, dtype, layer order, activation, normalizer, observed state order and checksum. Avoid JSON-serialized neural tensors in hot paths.
- `T-112` Export actor-only data, never privileged critic inputs. Ensure normalizer is applied identically in Python and JS.
- `T-113` Write a parity test: fixed observation vectors produce matching actor actions in Python and JS within a declared floating-point tolerance.
- `T-114` Couple browser MuJoCo release with exported model release and verify loadability/physics timestep before serving assets.
- `T-115` Browser shoves must clamp each force axis and duration to server limits, but server must independently enforce them too.
- `T-116` Treat all visual interpretations (gait labels, mask overlays, generated captions) as annotations; store numeric evaluation evidence separately.
- `T-117` The frontend should not embed secret model credentials, hidden prompt text, raw training checkpoints or privileged simulation state.

## 16. Quotas, budgets, moderation, operations

- `T-118` Bound per-visitor bursts and rolling quotas by task cost: typed attack, voice/photo, search, investigation, deployment brief, patch, report, envelope sweep, browser shove and challenge round have separate buckets.
- `T-119` Support an application-wide model usage ledger with token/audio/image billing adapters, unknown-price fallback, daily/hourly budget and stop threshold. Local inference still consumes GPU quota even when monetary API cost is zero.
- `T-120` Example public strategy: six-hour quota windows plus per-minute bursts; a 'clamped' budget mode reduces expensive calls, a 'stopped' mode denies new paid provider calls while keeping local replays visible. Make threshold amounts configurable, not vendor-specific constants.
- `T-121` Use hashed visitor identifiers if there are no accounts; do not pretend this is strong abuse prevention. Respect proxy trust configuration rather than treating arbitrary forwarded headers as IP truth.
- `T-122` Public strings and AI-generated summaries need fail-closed moderation before other users see them; unchecked content remains private. Admin overrides are logged and can hide content even after a share preview was cached.
- `T-123` Prefer local/on-disk modes for research and disable public/community routes by default; avoid sending user media to cloud models without explicit configured consent and provider disclosure.
- `T-124` Scrub user media, transcripts, full prompts, policy artifacts and personal data from ordinary application logs; enforce retention policies for uploaded files and clips.
- `T-125` Surface worker memory, queue length, compilation cache hits/misses, simulator steps/sec, failure counts, search progress, cost, and retry statuses to operators.

## 17. Development commands and conventions

These commands are **to be implemented** as part of the initial scaffold; do not assume they already work:

```bash
uv sync --extra server --group dev              # install Python runtime and tooling
uv run pytest -q                                 # CPU unit and schema tests
uv run pytest -q -m integration                  # native MuJoCo integration
uv run pytest -q -m gpu                          # explicit accelerator tests
uv run ruff check . && uv run ruff format --check .

uv run python -m robustlab.cli train-baseline --robot go1 --out runs/training/baseline
uv run python -m robustlab.cli evaluate --robot go1 --policy <path> --scenario <file>
uv run python -m robustlab.cli search --robot go1 --policy <path> --program <file>
uv run python -m robustlab.cli patch --robot go1 --policy <path> --attacks <file>
uv run uvicorn api.app:app --host 127.0.0.1 --port 8000

cd web && npm ci && npm run dev
cd web && npm run build && npm run typecheck
```

Use line length 100 for Python, static type hints at module boundaries, descriptive function names, stable DTOs, immutable dataclasses for robot metadata and pure core functions. Keep scientific algorithms and array operations readable; unit tests should cover sign conventions, numerical ranges and boundary conditions, not just happy-path mocks. Use JSONL for structured event streams and run metrics, with logs emitted at steps, not arbitrary print spam.

## 18. Mandatory test matrix

| Category | Test obligations |
| --- | --- |
| Schema | Every op, invalid extra fields, malformed ranges, duplicate ops, robot-specific names, max free dimensions |
| Compiler | Neutral vector, scalar/vector mapping, deterministic unit conversion, shared-slot collision, padding, layout migration |
| Physics | Neutral parity, correct friction authority, payload inertia/CoM, left/right shove direction, exact force timing, latency quantization |
| Terrain | Stair/curb geometry, ramp units, course cursor, collision-box capacity, non-foot terrain fall, parked boxes |
| Observations | Actor/critic separation, history ordering, sensor offset placement, normalized theta, reloaded normalizer stability |
| Evaluator | Same-seed repeatability, no hidden env pushes, raw termination ≠ fall, upright but no progress ≠ success, NaN handling |
| Optimizer | Sigmoid bounded samples, CMA-ES seed reproducibility, fitness phases, padding accounting, confirmation and shrink thresholds |
| Bank/PPO | Exactly 512 rows, probabilities sum to 1, padded rows unselectable, every-episode resample, restore compatibility, checkpoint callbacks |
| Gate | All conditions independently tested, numerical epsilon/edge values, accepted promotion, rejected artifact non-promotion |
| API/jobs | Input validation, FIFO/concurrency, SSE resume, restart recovery, events on exception, media limits, path traversal |
| AI | Provider-off workflow, invalid JSON retry, missing role capability, prompt injection, budget exhaustion, per-role traceability |
| Browser | Native vs JS action parity, model version compatibility, bounded shove replay, asset/cache isolation |
| Experiments | Holdout checksum, matching seeds/computational budget, benchmark distribution shift, evidence provenance |

CI runs CPU unit tests, static type/build checks, and a small simulator test on a compatible runner; GPU tests execute on a tagged runner or manually on compatible hardware. Keep long training benchmarks out of ordinary PR CI but publish their manifests and results for reproducibility.

## 19. Technical decision register: rationale and tradeoffs

| Decision | Why | Known limitation / trigger to revisit |
| --- | --- | --- |
| Python/JAX over a single Rust/Go physics stack | Strongest available locomotion/RL ecosystem | Can isolate high-throughput workers later; do not rewrite physics prematurely |
| MuJoCo/MJX rather than hand-written physics | Standard articulated/contact simulation | Contact modeling and sim-to-real mismatch remain |
| Existing locomotion tasks instead of custom PPO task | Faster verified learning baseline | Baseline reward and termination may need overriding |
| Fixed `theta` vector | Stable JIT shapes, shared search/training/evaluation protocol | Hard layout capacity and versioning requirements |
| Declarative DSL | Validation, provenance, model-provider independence | Less expressive than arbitrary code by design |
| Max ten free parameters | Bounded low-dimensional search and fixed JAX batch | Larger attacks need decomposition and separate search strategy |
| One terrain route op | Constrains geometry to fixed collision capacity | Cannot represent arbitrary 3D scenes faithfully |
| Preallocated terrain boxes | Avoid JIT recompilation and contact shape drift | Simplified contact geometry; explicit collision approximations |
| Three-frame observations | History reveals slip/latency without hidden physics truth | Adds latency and policy input dimension |
| Privileged critic, nonprivileged actor | Easier training without impossible deployment sensors | Requires careful observation and normalization separation |
| CMA-ES vs gradient optimization | Falls/contact are non-smooth and black-box | Simulation calls expensive; test random-search baseline |
| Failure-first, plausibility-second fitness | Avoid trivial robust-but-never-failing candidates | Normalized magnitude is not actual occurrence probability |
| Fresh-seed confirmation | Prevent overfitting search randomness | 32 seeds remain a finite confidence sample |
| Replay bank + nominal mixture | Reduce forgetting during fine-tuning | Balance and quarantine heuristics need ablations |
| PPO warm-start | Preserves useful walking behavior; avoids cold-start cost | Can get stuck in local optimum; compare against scratch and DR baselines |
| Regression gate | Controls promotion and preserves provenance | Aggregate +3 percentage points permits some regressions |
| Subprocess for PPO | GPU memory isolation and nonblocking service | Single-host worker lifecycle must be managed |
| SSE rather than WebSocket for job updates | Unidirectional append-only progress and native reconnection | Separate endpoint required for interactive commands/cancellation |
| JSON/NumPy artifacts initially | Easy inspection and research iteration | SQLite/object storage when persistence or scale grows |
| Local WASM playground | Responsive tactile interaction without server roundtrip | JS/native parity and physics compatibility challenges |
| Provider abstraction | Model substitutions without rewiring physics or tool contracts | Capability matrix and per-role test coverage required |
| Optional share/challenge layer | Makes failures easier to explore and communicate | Privacy, moderation and cost must be managed |

## 20. Engineering assistant instructions

Treat every decision listed here as an explicit starting requirement unless a code-level experiment disproves it. When changing one: state the prior constraint, the failure or measurement motivating change, the new invariant, migration requirements and tests. Do not use the same config name for two different concepts. Never conflate a simulation outcome with an AI diagnosis, a checkpoint with an accepted version, a browser experiment with a server-confirmed result, or a physical approximation with a real-world measurement.
