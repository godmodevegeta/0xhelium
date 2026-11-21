# Design Document: M5 — Multi-axis scenarios and OpenRouter designer

## M5a modules

```text
robustlab/scenarios/schema.py      Scenario parsing/validation -> ValidatedScenario (+ errors, adjustments)
robustlab/physics/scenario_model.py ScenarioModel build (MjSpec), friction/payload reports, LRU cache
robustlab/experiments/scenario_run.py run_single / run_scenario_suite / foot_slip / result records
robustlab/cli.py  `robustlab scenario --file F --out D [--suite --trials 16 --interval 2.0,3.024 --seed S]`
```

### Schema (v1)

```json
{
  "schema_version": 1, "kind": "robustlab-scenario", "name": "icy-heavy-shove",
  "command": [1.0, 0.0, 0.0], "horizon_s": 5.0, "scene": "fall-contact",
  "environment": {"ground_friction_mu": 0.3},
  "payload": {"mass_kg": 3.0, "offset_xyz_m": [0.0, 0.0, 0.05]},
  "push": {"force_n": 100, "direction_deg": 270, "start_s": 2.0, "duration_s": 0.1}
}
```

Every numeric leaf is `number | {"range": [lo, hi]}` (Param). `validate(raw, mode="fixed",
command_ranges=...)` returns `ValidatedScenario` or raises `ScenarioError(errors=[{field,
reason}...])` with all errors. Adjustments: `{field, requested, applied, reason}` (4 ms
quantization, direction modulo 360, zero push dropped). Nothing is clamped.

### Scenario model

```text
signature = (scene, SCENE_TRANSFORM_VERSION, base_xml_sha256, PAYLOAD_MODEL_VERSION,
             mu or None, mass_kg, ox, oy, oz)
base signature (mu None, mass 0) -> the Stack's existing model object (no compile)
otherwise: MjSpec.from_file -> scene transform (shared with build_scene) -> floor.friction[0] = mu
           -> payload body (explicit inertial, box geom contype/conaffinity 0, group 4, mass 0)
           -> compile -> apply policy model_parameters -> check_model -> resolve_ids
cache: OrderedDict LRU, max 8, lock; models read-only after build
```

`build_scene(xml, scene, *, friction_mu=None, payload=None)` gains the two optional
modifications so the fall-contact transform has one implementation. `SceneInfo.describe()` adds a
`modifications` key only when one is applied, so M0–M4 descriptions are unchanged.

Friction report: requested; configured = `geom_friction[floor, 0]`; resolved per foot (priority
rule: higher priority wins, equal → element-wise max); `observed_at_reset` from `mj_forward` at the
policy reset pose (4 foot contacts); rule text. Build refuses `npair > 0` or the override flag.

Payload report: box size, inertia diag `m/12 (b²+c², a²+c², a²+b²)`, body position, composite
mass, composite CoM (trunk body frame), composite inertia about it (parallel-axis theorem with
trunk principal inertia rotated by `body_iquat`), its eigenvalues, total model mass.

### Execution

`scenario_stack(base_stack, model_entry)` = a Stack sharing the worker's policy and actor (same
thread) with the scenario model and ids, so `run_trial` and `run_suite` run unchanged.

- Single: if push: `run_trial` on the scenario stack → attacked + matched baseline + parity; else
  one episode. Plus the original nominal episode on the base stack. Primary ghost: matched if a push
  exists, else original nominal.
- Suite: push required; M2 `SuiteConfig` built from the scenario (force, direction vector,
  duration, interval, trials, seed, method) → `run_suite(config, scenario_stack, out)`; nominal in
  that report is the matched baseline. Original nominal run once as a reference.
- Metrics: outcome/reason/time (FallResult), progress, max tilt, foot slip = mean horizontal speed
  of each foot site over Ticks where that foot is in contact on consecutive Ticks (post-hoc
  `mj_kinematics` on recorded qpos), observed friction at reset.
- Result `scenario_result.json` (schema 1): scenario, model reports, cases {scenario, matched?,
  original}, primary_baseline, changed_parameters, trajectories paths, provenance, results digest.

## M5b modules

```text
robustlab/llm/openrouter.py   OpenRouterCompiler (httpx client injectable), RateLimiter, errors
robustlab/llm/prompt.py       system prompt + strict JSON schema (fixed-value subset)
robustlab/api/app.py          /api/scenarios/* endpoints
web/src/components/ScenarioDesigner.tsx, ReplayViewer (payload box, floor tint, ghost label)
```

LLM output schema (strict: all keys required, nullables via type unions, additionalProperties
false): `{name, rationale, command[3], ground_friction_mu|null, payload_mass_kg,
payload_offset_xyz_m[3], push|null{force_n, direction_deg, start_s, duration_s}}` → mapped to a
Scenario → same validator. Default model `openai/gpt-4o-mini` (structured outputs supported;
overridable by `OPENROUTER_MODEL`).

Errors → HTTP: not configured 503 `not_configured`; prompt too long 422; local rate limit 429
`rate_limited`; upstream 429 → 429 `upstream_rate_limited`; timeout → 504; upstream 5xx / network
→ 502; malformed JSON / missing content / refusal → 502 `invalid_model_output`; schema/physics
errors → 200 with `valid: false` and field errors (the model answered; the scenario is invalid).

Rate limits: token buckets, 6 per minute per client IP and 20 per minute process-wide. max_tokens
500, timeout 30 s. Compile log `runs/web/llm/compiles.jsonl` (prompt, model, usage, latency,
validity; never the key).

API:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/scenarios/config` | bounds, presets, examples, OpenRouter `configured` and model |
| POST | `/api/scenarios/validate` | preview: validated scenario, adjustments, friction & payload reports |
| POST | `/api/scenarios/compile` | `{prompt}` → proposal + same preview |
| POST | `/api/scenarios/runs` | `{scenario, mode: single|suite, trials, interval_s, seed}` → 202 |
| GET | `/api/scenarios/runs/{id}` | status / result |
| GET | `/api/scenarios/runs/{id}/replay?trial=&ghost=matched|original` | replay with payload boxes |

Replay: `replay_payload` gains per-case models and, when a case model has a payload geom, a
`payload_box {size, pos, quat}` from `mj_kinematics`; plus `ghost {kind, label, parameters}` and
`environment {mu, nominal_mu}`.
