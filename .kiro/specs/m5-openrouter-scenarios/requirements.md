# Requirements Document

## Introduction

Milestone M5: multi-axis fixed-parameter scenarios (trunk push, effective foot-ground friction,
trunk payload) in one validated schema, and an OpenRouter natural-language scenario compiler.
Two stages: **M5a** physics, schema, caching, tests and measurements (verified first), then
**M5b** OpenRouter compiler and the Scenario Designer web tab. Physics runs only through the M1
runner/fall monitor and M2 `run_trial`/`run_suite`. The M4 optimizer is not changed. No
simulator code is generated or executed.

Measured before design (M0–M4 scenes): no `<pair>` elements, no contact override flag, pyramidal
cone. The floor geom has `priority=1`, `condim=3`, friction (0.6, 0.005, 0.0001); the foot spheres
have `priority=0`, `condim=1`, friction 1.0. MuJoCo uses the higher-priority geom's friction and
condim, so every floor contact resolves to sliding friction 0.6 (observed in `data.contact`).
The feet's declared 1.0 is never used.

## Glossary

- **Scenario**: a schema-validated JSON document (`kind: robustlab-scenario`, `schema_version 1`).
- **Param**: a numeric field written as a fixed number or `{"range": [lo, hi]}`. Ranges are
  accepted by the parser for future search and rejected by the M5 executor.
- **Requested friction**: `environment.ground_friction_mu` in the Scenario (null = trained nominal).
- **Configured friction**: the floor geom's sliding friction in the compiled model.
- **Resolved friction**: the friction MuJoCo would use for a foot-floor contact by its priority rule.
- **Observed friction**: `data.contact[i].friction[0]` of foot-floor contacts in a running simulation.
- **Payload**: a uniform-density rigid box of 0.10 × 0.08 × 0.04 m (x × y × z), mass
  `payload.mass_kg`, welded (no joint) to the trunk with its centre at the trunk CoM plus
  `payload.offset_xyz_m` in trunk body axes.
- **Scenario_Model**: the compiled MjModel for a (scene, friction, payload) signature.
- **Matched baseline**: same friction, payload, command, horizon and initial state; push removed.
- **Original nominal**: the trained condition: nominal friction, no payload, no push.

## Requirements

### Requirement 1: Scenario schema (M5a)

1. THE Scenario SHALL contain `schema_version`, `kind`, `name`, `command` (vx, vy, yaw),
   `horizon_s`, `scene`, `environment.ground_friction_mu` (Param or null), `payload.mass_kg`,
   `payload.offset_xyz_m` (3 Params) and `push` (null or `force_n`, `direction_deg`, `start_s`,
   `duration_s`).
2. THE validator SHALL reject unknown keys, wrong types, non-finite numbers and values outside:
   μ ∈ [0.05, 1.50]; mass ∈ [0, 5] kg; offset x ∈ [−0.10, 0.10], y ∈ [−0.08, 0.08],
   z ∈ [0, 0.08] m; force ∈ [0, 150] N; start ∈ [0, horizon − duration]; duration ∈ [0, 0.5] s;
   command within the policy's trained ranges; horizon a multiple of 0.02 s in (0, 60] s.
3. THE validator SHALL report every error with its field path, never clamp an out-of-range value,
   and quantize `start_s` and `duration_s` to the 4 ms grid and `direction_deg` to [0, 360),
   disclosing each change as an adjustment.
4. WHERE the execution mode is fixed, THE validator SHALL reject any `range` Param.
5. A push with `force_n = 0` or `duration_s = 0` SHALL be treated as no push and disclosed.

### Requirement 2: Friction (M5a)

1. WHEN `ground_friction_mu` is set, THE Scenario_Model SHALL set only the floor geom's sliding
   friction to it; torsional and rolling coefficients, condim, priority and all other geoms stay.
2. THE Scenario_Model SHALL report requested, configured and resolved friction and the resolution
   rule, and SHALL refuse to build if any `<pair>` element or the contact override flag exists.
3. THE runner SHALL report observed foot-floor contact friction measured from MuJoCo contacts at
   the reset state; previews SHALL label the value as observed at reset, not over the episode.
4. A test SHALL verify that every observed floor contact during an episode has friction equal to the
   configured value, that the contact tangential-to-normal force ratio never exceeds it, and that a
   lower μ increases foot slip under the same disturbance.

### Requirement 3: Payload (M5a)

1. WHEN `payload.mass_kg > 0`, THE Scenario_Model SHALL add one jointless child body of the trunk
   with an explicit inertial (mass, box inertia diag) and one box geom with `contype = 0`,
   `conaffinity = 0`, group 4 and no mass of its own.
2. THE Scenario_Model SHALL leave nq, nv, nu, joint, actuator, sensor and existing body and geom
   ids and arrays unchanged, and total mass SHALL increase by exactly `mass_kg`.
3. THE Scenario_Model SHALL report the composite trunk+payload mass, CoM and inertia about the
   composite CoM, and that inertia SHALL be symmetric positive definite.
4. WHEN `mass_kg == 0` and friction is null, THE Scenario_Model SHALL be the unmodified base model
   object (no compilation) and nominal trajectories SHALL equal the M0 golden trajectories.

### Requirement 4: Model cache (M5a)

1. THE cache SHALL key models by scene name, scene transform version, base XML SHA-256, payload
   model version, requested μ, payload mass and the three offsets, and hold at most 8 models (LRU).
2. Cached models SHALL be treated as read-only; every episode SHALL use its own MjData.

### Requirement 5: Execution and trial semantics (M5a)

1. A single run SHALL simulate the scenario episode; WHEN a push is present also the matched
   baseline (via `run_trial`, with pre-push parity); and always the original nominal.
2. A suite SHALL require a push and vary only the push start by M2 stratified sampling (or grid)
   over a stated interval; all other parameters fixed. Without a push THE suite SHALL be refused,
   since repeated deterministic episodes are not independent trials.
3. THE result SHALL record both baselines with their exact physical parameters, label which one is
   the primary comparison (matched when a push exists, otherwise original nominal), and list the
   physical parameters that differ from the original nominal.
4. THE result SHALL record outcomes, fall reasons and times, progress, max tilt, foot slip, total
   mass, friction (requested/configured/resolved/observed) and digests, and persist M1-format
   trajectories for replay.
5. THE `robustlab scenario` CLI SHALL run single and suite modes and write the result.

### Requirement 6: OpenRouter compiler (M5b)

1. THE compiler SHALL call `POST https://openrouter.ai/api/v1/chat/completions` with httpx, the key
   from `OPENROUTER_API_KEY` only, the model from `OPENROUTER_MODEL` (with a documented default),
   `response_format` json_schema with `strict: true`, `provider.require_parameters: true`,
   `temperature 0` and a bounded `max_tokens`.
2. THE compiler SHALL validate the returned JSON with the same validator as manual scenarios and
   return the proposal, the validated scenario or field errors, adjustments, model id and usage.
3. THE compiler SHALL bound prompt length (1,000 characters), request time (30 s) and request rate
   (a process-wide limit and a per-client limit), and map missing key, timeout, upstream 429, 5xx,
   malformed JSON and refusals to explicit structured errors.
4. WHEN `OPENROUTER_API_KEY` is unset, THE Backend SHALL report `configured: false`, the compile
   endpoint SHALL return 503, and every other feature SHALL keep working. No fake response is ever
   produced.
5. THE key SHALL never be logged, returned or sent to the browser.

### Requirement 7: Scenario Designer (M5b)

1. THE Backend SHALL expose scenario config, validate (preview), compile, run, status and replay
   endpoints, using the existing one-expensive-job guard.
2. THE Viewer SHALL add a Scenario Designer tab with the prompt box and configuration status,
   example prompts, manual controls and presets for friction, payload and push, a preview of
   validated parameters, adjustments, errors and friction/mass properties before running, and
   single or 16-trial runs.
3. THE replay SHALL label the ghost (matched baseline or original nominal) with its parameters,
   tint the floor when friction differs from nominal, and draw the payload box on each robot that
   carries it.

### Requirement 8: Tests and evidence

1. Hypothesis tests SHALL cover schema validation, physical bounds, 4 ms quantization, payload
   inertia positive definiteness and mass accounting, friction configuration identity and contact
   friction bounds, and nominal-default identity.
2. OpenRouter tests SHALL use mocked HTTP for valid output, out-of-range rejection, malformed JSON,
   timeout, 5xx, upstream 429, local rate limit and missing key; a real smoke test SHALL run only
   when a key is present, recording the model used.
3. Multi-rollout physics tests SHALL be marked `integration`; the Stop hook suite stays under 30 s.
4. `docs/milestones/M5.md` SHALL record nominal, low-friction, payload-only and combined
   measurements and whether each change altered observed behaviour.
