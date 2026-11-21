# Repository Structure, Module Boundaries, and Implementation Rules

> **Deadline implementation order:** `delivery.md` supersedes the long-term section on build order. Start with walking policy + native MuJoCo, then deterministic shove, experiments, a thin web UI, failure search, validated text scenarios, and only then optional CPU adapter hardening. The directory tree below describes **future ownership**, not files to create now.


> Kiro steering — this is the canonical proposed file organization for an independently initialized repository. Directories and modules below are **targets to create**, not claims that code is already present. See `product.md` for user goals and `tech.md` for numerical and infrastructure decisions.

## 1. Repository organization

Use one repository containing a Python research package, a thin HTTP application, a TypeScript frontend, experimental protocols, and a single location for project-wide tooling. Do not place physics and model-training logic in HTTP handlers or React components.

```text
robustness-studio/
├── .kiro/
│   └── steering/
│       ├── product.md                   # Product objectives and acceptance requirements
│       ├── tech.md                      # Numerical contracts and technology standards
│       ├── structure.md                 # Module ownership and eventual layout
│       └── delivery.md                  # Authoritative incremental milestone sequence
├── .github/
│   └── workflows/
│       ├── ci.yml                       # CPU tests, lint, TypeScript build and smoke test
│       └── gpu-validation.yml           # Optional GPU-specific physics/RL parity tests
├── .env.example                         # Provider-neutral settings, no secrets
├── .gitignore
├── .python-version                      # 3.12
├── README.md                            # How to set up and run the new project
├── pyproject.toml                       # Python package, dependencies, scripts, lint/test config
├── uv.lock                              # Committed Python lockfile
├── package.json                         # Optional root scripts only if truly useful
├── scripts/
│   ├── setup_linux.sh                   # Drivers/env checks, dependency setup
│   ├── dev.sh                           # Local API + frontend convenience commands
│   ├── serve.sh                         # Production-ish startup, bounded GPU memory
│   ├── train_baseline.sh                # Explicit baseline creation
│   ├── run_cpu_tests.sh                 # Fast deterministic suite
│   ├── run_gpu_tests.sh                 # Accelerator validation
│   ├── smoke.sh                         # Compile/run/evaluate sanity path
│   └── check_artifacts.py               # Validate saved manifests, hashes, layouts and files
├── robustlab/                            # Pure Python core (never imports FastAPI)
│   ├── __init__.py
│   ├── config.py                         # Typed config + JAX cache bootstrap
│   ├── cli.py                            # Deterministic train/evaluate/search/patch commands
│   ├── types.py                          # Shared domain IDs/types not tied to HTTP
│   ├── robots/
│   │   ├── __init__.py
│   │   ├── registry.py                   # Immutable RobotDefinition data for supported robots
│   │   ├── resolve.py                    # Bound names to MjModel IDs; ResolvedRobotSpec
│   │   ├── go1.py                        # Quadruped-specific names, envelopes, observations
│   │   └── g1.py                         # Humanoid-specific data; initially feature-gated
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── loader.py                      # Load/rebuild Playground MJCF; parity tests
│   │   ├── contacts.py                    # Foot contact priority, non-foot geom selection
│   │   ├── base_randomization.py          # Per-episode dynamics randomization
│   │   └── wrapper.py                     # Per-env theta, state, actor history, critic context
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── slots.py                       # Slot definitions, neutral, bound, layout version
│   │   ├── schema.py                      # Pydantic operation variants and program validation
│   │   ├── compiler.py                    # Program -> CompiledProgram -> theta
│   │   ├── terrain.py                     # Shared NumPy/JAX route geometry and collision math
│   │   ├── physics.py                     # Apply model dynamics / per-step effects / obs noise
│   │   ├── frames.py                      # Initial-heading transforms and physical units
│   │   └── examples/                     # Small valid JSON scenarios for each primitive
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── evaluator.py                   # JAX-vmap/scan evaluation; authoritative metrics
│   │   ├── metrics.py                     # Fall rate, tracking, progress, robustness signatures
│   │   ├── recording.py                   # Saved qpos + replay metadata
│   │   ├── render.py                      # Native offline frames/clips, CPU/GPU constraints
│   │   └── manifests.py                   # Deterministic trial manifest + checksum
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── cmaes.py                       # Bounded CMA-ES search and batched candidates
│   │   ├── random_baseline.py             # Matched-budget unguided baseline
│   │   ├── objective.py                   # Failure-first fitness and severity cost
│   │   ├── confirm.py                     # Fresh seed reproducibility and acceptance rules
│   │   ├── shrink.py                      # Delta-debug physical-condition simplification
│   │   ├── margins.py                     # Single-condition threshold/sweep estimates
│   │   └── families.py                    # Physical cause and behavior signatures
│   ├── training/
│   │   ├── __init__.py
│   │   ├── ppo.py                         # Brax adapter, checkpoints, warm starts
│   │   ├── observation_stats.py           # Critic normalization compatibility handling
│   │   ├── bank.py                        # Curate and sample learnable/replay scenarios
│   │   ├── jitter.py                      # Nearby variation generation
│   │   ├── patch.py                       # Pure training process orchestration
│   │   └── gate.py                        # Independent regression gate
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── checkpoint.py                  # Load/validate immutable actor+critic checkpoints
│   │   ├── manifest.py                    # Version ancestry and artifact digests
│   │   ├── registry.py                    # Active version pointer and atomic promotion
│   │   ├── browser_export.py              # Serialize actor-only inference representation
│   │   └── parity.py                      # Reference observation->action fixtures
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── contracts.py                   # Model/segmenter/transcriber Protocols and DTOs
│   │   ├── factory.py                     # Role -> configured provider resolver
│   │   ├── prompts.py                     # Versioned schema-guided prompt constructors
│   │   ├── translation.py                 # Text -> validated ScenarioProgram
│   │   ├── hypotheses.py                  # Candidate program proposer and repair loop
│   │   ├── vision.py                      # Image/video assumptions and diagnostic DTOs
│   │   ├── segmentation.py                # Concept masks, geometry count checks
│   │   ├── transcription.py               # Speech -> text role adapter
│   │   ├── investigator.py                # Tool loop, turn/compute budget, tool validation
│   │   ├── reports.py                     # Structured narratives from measured evidence
│   │   ├── caching.py                     # Request digest and privacy-aware cached results
│   │   ├── ledger.py                      # Usage/cost accounting and hard budget guard
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── disabled.py                # Deterministic unavailable-provider errors
│   │       ├── openai_compatible.py       # Hosted/local HTTP-compatible text/vision adapter
│   │       ├── local_inference.py          # Local model-specific capability/fallback logic
│   │       ├── segmenter_http.py          # Optional external segmentation provider
│   │       └── local_asr.py               # Optional on-machine transcription
│   ├── evidence/
│   │   ├── __init__.py
│   │   ├── archive.py                     # Attack records, confirmation, per-version status
│   │   ├── lineage.py                     # Version status and historical re-tests
│   │   ├── deployment.py                  # Place-based structured test plans and grading
│   │   ├── reporting.py                   # Measured readiness summaries
│   │   └── formats.py                     # Versioned persistent record DTOs
│   └── export/
│       ├── __init__.py
│       ├── trajectory.py                  # Native qpos -> renderer-neutral replay JSON
│       ├── terrain_overlays.py             # Authoritative geometry for visual overlays
│       ├── media.py                        # FFmpeg/Pillow output assembly
│       └── browser_model.py                # Native .mjb + metadata export
├── api/                                  # Transport only; depends on robustlab
│   ├── __init__.py
│   ├── app.py                            # FastAPI creation, middleware, static mount
│   ├── dependencies.py                    # State, typed settings, services, auth/visitor identity
│   ├── schemas.py                         # Request/response DTOs and error codes
│   ├── jobs.py                            # Durable job/event store, SSE resume
│   ├── scheduler.py                       # GPU/train locks and process/worker dispatch
│   ├── patch_worker.py                    # CLI subprocess entrypoint for PPO, JSONL events
│   ├── versions.py                        # API-facing registry use and status responses
│   ├── uploads.py                         # Media validation, storage and lifecycle
│   ├── quotas.py                          # Per-role/task limits and shared spend budget
│   ├── moderation.py                      # Public text policy, redact and admin decisions
│   ├── media_tasks.py                     # Render queue; never block HTTP event loop
│   ├── observability.py                   # Structured logging, correlation and metrics
│   └── routes/
│       ├── __init__.py
│       ├── scenarios.py                   # Compile / validate / preview
│       ├── attacks.py                     # Typed attack, audio and media workflows
│       ├── searches.py                    # Adversarial search jobs
│       ├── patches.py                     # Retraining jobs and gate results
│       ├── policies.py                    # Immutable lineage and live pointer
│       ├── investigations.py              # Agent orchestration jobs
│       ├── deployments.py                 # Site-specific test plans and re-grading
│       ├── reports.py                     # Readiness generation
│       ├── jobs.py                        # Job inspect/events/cancellation
│       ├── replays.py                     # Pose metadata and clip download
│       ├── playground.py                  # Browser physics assets and shove verification
│       ├── limits.py                      # Rate quota information
│       ├── challenges.py                  # Optional fair human-vs-agent mode
│       └── shares.py                      # Optional sanitized OG/card/video sharing
├── web/
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── public/                           # Static assets, no model secrets
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── app/
│       │   ├── routes.tsx                 # React Router definitions
│       │   ├── layout.tsx                 # Shared shell + navigation
│       │   └── config.ts                  # Public runtime config only
│       ├── api/
│       │   ├── client.ts                  # Typed fetch, retry and error handling
│       │   ├── schemas.ts                 # API DTOs / versioned runtime parsing
│       │   ├── events.ts                  # SSE subscription + reconnect / state transitions
│       │   └── urls.ts                    # Stable asset URL construction
│       ├── features/
│       │   ├── attacks/                   # Prompt/editor, comparison and results
│       │   ├── scenes/                    # Image upload, masks, uncertainty labels
│       │   ├── investigations/            # Live autonomous hypothesis timeline
│       │   ├── training/                  # Retrain progress, gate and timelapse
│       │   ├── policies/                  # Version tree, side-by-side selection
│       │   ├── deployment/                # Scenario grid and re-grade
│       │   ├── reports/                   # Readiness reports
│       │   ├── playground/                # Live WASM physics and in-page actor
│       │   └── challenges/                # Optional duel, leaderboard and share views
│       ├── graphics/
│       │   ├── replay/                    # Three.js playback of recorded poses
│       │   ├── robot/                     # Robot meshes, coordinate transforms
│       │   ├── overlays/                  # Force/friction/terrain rendering
│       │   └── camera/                    # Controls and scene framing
│       ├── wasm/
│       │   ├── mujoco.ts                   # Version-checked WASM initialization
│       │   ├── observations.ts             # Browser-equivalent actor observation pipeline
│       │   ├── policy.ts                   # Binary parser / deterministic inference
│       │   ├── simulator.ts                # Fixed-timestep simulation loop
│       │   └── controls.ts                 # Validated bounded browser shove
│       ├── components/
│       │   ├── ui/                         # Accessible generic controls
│       │   ├── layout/                     # Cards, grid and headers
│       │   └── visualization/              # Charts, timelines and badges
│       ├── hooks/                          # Shared typed hooks, no physics equations
│       ├── styles/
│       └── test/                            # Browser fixtures, unit tests, parity tooling
├── experiments/
│   ├── README.md                           # Protocol, evaluation splits and reproducibility
│   ├── protocols/
│   │   ├── baseline.yaml                   # Nominal training and evaluation protocol
│   │   ├── gauntlet-v1.yaml                # Committed holdout suite definition
│   │   ├── discover-v1.yaml                # Search, seeds and budgets
│   │   ├── patch-v1.yaml                   # Replay/PPO/gate defaults
│   │   └── comparisons-v1.yaml             # Matched-compute baseline experiments
│   ├── gauntlets/
│   │   ├── hand-authored/                   # Physically curated regression scenarios
│   │   ├── realistic/                       # Deployment-like combinations
│   │   └── generated/                       # Explicitly separate from immutable holdout
│   ├── baselines/
│   │   ├── domain_randomization.py
│   │   ├── random_programs.py
│   │   └── structured_shuffle.py
│   ├── runners/
│   │   ├── matched_compute.py
│   │   ├── discovery_curve.py
│   │   ├── ablations.py
│   │   └── robustness_sweep.py
│   └── results/                             # Small committed summaries + artifact links
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── scenarios/                        # Valid and invalid programs
│   │   ├── thetas/                           # Neutral / attack golden vectors
│   │   ├── policies/                         # Small test policies, no private real checkpoints
│   │   └── observations/                     # Python-JS inference parity vectors
│   ├── unit/
│   │   ├── test_schema.py
│   │   ├── test_slots.py
│   │   ├── test_compiler.py
│   │   ├── test_course.py
│   │   ├── test_frames.py
│   │   ├── test_objective.py
│   │   ├── test_bank.py
│   │   ├── test_gate.py
│   │   ├── test_model_adapters.py
│   │   ├── test_job_events.py
│   │   └── test_quota_moderation.py
│   ├── integration/
│   │   ├── test_mujoco_loader.py
│   │   ├── test_contact_friction.py
│   │   ├── test_terrain_collisions.py
│   │   ├── test_wrapper_fall.py
│   │   ├── test_evaluator.py
│   │   ├── test_checkpoint_restore.py
│   │   ├── test_patch_worker.py
│   │   ├── test_version_atomicity.py
│   │   └── test_api_lifecycle.py
│   ├── gpu/
│   │   ├── test_jit_shapes.py
│   │   ├── test_seed_consistency.py
│   │   ├── test_rollout_speed.py
│   │   └── test_training_smoke.py
│   └── parity/
│       ├── test_native_browser_actor.py
│       └── test_exported_model.py
├── docs/
│   ├── architecture.md                     # Dataflow, component and trust boundaries
│   ├── scenario-language.md                # Full public typed DSL reference
│   ├── coordinate-systems.md               # Sign conventions and transformations
│   ├── model-compatibility.md              # Provider role capability matrix
│   ├── operator-guide.md                   # GPU, caches, logs, quotas, recovery
│   ├── runbook.md                          # Failure modes and rollout diagnosis
│   ├── scientific-method.md                # Seeds, holdouts, matched budget and uncertainty
│   ├── decision-log/                       # Future reviewed architectural changes (ADRs)
│   └── examples/                           # Reproducible tutorial scenarios and notebooks
└── runs/                                   # GENERATED; ignored by Git except placeholders
    ├── evaluations/
    ├── searches/
    ├── training/
    ├── artifacts/
    ├── serve/
    ├── llm/
    └── media/
```

## 2. Dependency direction: enforceable layering

```text
physical data + types
  ├── scenarios/slots + schema + compiler
  ├── robots registry -> robots resolution -> environments -> simulation
  ├── optimization -> simulation + scenarios
  ├── training -> simulation + environments + scenarios + policies
  ├── evidence -> scenario/manifests/metrics/lineage
  ├── AI contracts and adapters -> typed domain DTOs + evidence references
  └── export -> models + simulation records + trajectories

api -> robustlab core + job/persistence abstractions
web -> public API + immutable artifacts (not Python imports)
experiments -> robustlab core (never the reverse)
tests -> everything (production code never imports tests)
```

The **dependency rule** is more important than the precise directory names. `robustlab/scenarios/compiler.py` must not import a provider client; `robustlab/simulation/evaluator.py` must not import the web server; `robustlab/training/gate.py` must not ask an LLM whether to promote. APIs adapt transport to kernel operations, not vice versa.

### Module ownership and forbidden couplings

| Domain concern | Only authoritative owner | Must not be duplicated in |
| --- | --- | --- |
| Robot joint/foot names, limits | `robots/registry.py` and robot presets | Prompts, React forms, API route constants |
| MuJoCo integer model IDs | `robots/resolve.py` | Canonical slot definitions, UI components |
| Physical units and initial-heading coordinates | `scenarios/frames.py` | Arbitrary API handlers and graphics math |
| Canonical 80-slot map and migrations | `scenarios/slots.py` | Compiler literals, optimizer, web |
| Operation schemas and bounds | `scenarios/schema.py` | Provider-specific prompts written separately |
| DSL-to-theta translation | `scenarios/compiler.py` | Agent tool handler / frontend |
| Terrain math, placement and activation | `scenarios/terrain.py` | Renderer-specific approximations |
| Simulator perturbations | `scenarios/physics.py` | LLM tools and web |
| Physical fall metric | `environments/wrapper.py` + `simulation/metrics.py` | Vision diagnosis, UI's own heuristic |
| Authoritative rollout aggregation | `simulation/evaluator.py` | Search optimizer, frontend |
| Search scoring/severity | `optimization/objective.py` | Duel scoring and report generator (import it) |
| Bank sampling | `training/bank.py` | API patch route |
| Promotion acceptance | `training/gate.py` | Model summaries, web event handler |
| Policy active pointer | `policies/registry.py` | API route, worker, frontend |
| Role-to-model choices | `ai/factory.py` and config | Hardcoded vendor names in prompts or feature code |
| Usage accounting | `ai/ledger.py` | Ad hoc tokens/cost code in API routes |
| Job sequencing and durable events | `api/jobs.py` | Global in-memory list attached to route handler |
| HTTP request validation | `api/schemas.py` and route DTOs | Core numerical module |
| Native/browser actor serialization | `policies/browser_export.py` + `web/src/wasm/policy.ts` | JSON model blobs in React components |

When introducing a new feature, find the authoritative owner first. Add a small adapter instead of duplicating the calculation in another layer.

## 3. File- and data-contract rules

### Code conventions

- Python package name: `robustlab`, lowercase and importable by `uv run python -m robustlab.cli`.
- Python files: `snake_case.py`; classes: `PascalCase`; pure functions: descriptive verb names; constants: `UPPER_SNAKE_CASE`.
- React: components `PascalCase.tsx`, hooks `useThing.ts`, pure data helpers `camelCase.ts`, CSS scoped per feature where possible.
- No environment variables or global mutable client state in `scenarios`, `simulation`, `optimization`, or `training` functions; inject typed config and explicit PRNG seeds.
- A Pydantic model is for external validation; a dataclass or typed array bundle is for internal numerical work; convert explicitly at boundaries.
- Avoid circular imports: robot definitions don't load MuJoCo models on import, and provider clients don't initialize until a role is requested.
- Don't import heavy JAX/MuJoCo modules into fast API modules at module import time if a lazy worker or setup path would do.
- Use narrow `__init__.py` exports; do not glob-import all scientific modules.
- Use separate code paths for producer and consumer of binary actor models, with shared golden fixtures.

### Data formats

`ScenarioProgram` — validated JSON with `schema_version`, `robot`, `name`, `hypothesis`, `ops`.

`CompiledScenario` — `theta: float32[80]`, layout version, program digest, program source, fixed/free slot mapping and compiler warnings.

`TrialManifest` — scenario digest, policy artifact digest, robot/model fingerprint, numerical library versions, seed list, horizon, control timestep, backend, command, run mode, git revision and exact effective settings.

`EvaluationMetrics` — raw per-seed fall boolean, first fall time, displacement and velocity error, plus aggregate numerator/denominator, confidence intervals where available, tracking success and failure direction.

`SearchManifest` — program, candidate range map, CMA-ES seed, population, generations, batch slots, candidate history, top result, fresh-seed confirmation and shrink history.

`PolicyManifest` — immutable ID, parent ID, checkpoint digest, robot, actor/critic architecture, normalization state, training run, training bank digest, replay policy, holdout suite digest, gate result and creation time.

`JobEvent` — monotonically increasing `seq`, `job_id`, `timestamp`, `kind`, `stage`, validated data DTO and optional trace id. Use event IDs to resume SSE after disconnect.

`ReplayManifest` — model and layout version, robot pose format, fixed timestep, start-heading transform, scene geometry, source trial, fall time, optional force annotations and trajectory location.

All persistent schemas have a `schema_version`; consumers reject unknown **major** versions. Save atomic JSON or transactionally store records. Never silently coerce a policy checkpoint trained for a different observation dimension or vector layout.

### Artifacts and secrets

- `runs/` contains generated artifacts and is Git-ignored. Do not commit training logs, private uploaded media, large checkpoints, credentials, or packaged model weights accidentally.
- Config templates may be committed but contain only placeholder values.
- Lockfiles, fixtures, small sample JSON programs, benchmark protocol manifests and source assets with approved redistribution terms may be committed.
- Use explicit manifest digests to detect corrupted or partially written artifacts.
- Preserve a reproducible original train/eval run rather than overwriting it with a subsequent patch.

## 4. Lifecycle tracing: where every step lives

### A. Plain-language attack

```text
web/features/attacks
  -> POST /api/attacks
  -> api/routes/attacks.py
  -> robustlab/ai/translation.py
  -> robustlab/scenarios/schema.py
  -> robustlab/scenarios/compiler.py
  -> api/scheduler.py [GPU slot]
  -> robustlab/simulation/evaluator.py [baseline and active]
  -> robustlab/evidence/archive.py [immutable measurements]
  -> robustlab/export/trajectory.py [replay]
  -> api/jobs.py [event completion]
  -> web/features/attacks [measured comparison]
```

The simulator owns numerical outcomes. The model may generate `ScenarioProgram` only. The backend must not pass free-form text into physical simulation without validation.

### B. Autonomous discovery

```text
web/features/investigations
  -> api/routes/investigations.py
  -> robustlab/ai/investigator.py [bounded tool loop]
  -> robustlab/ai/hypotheses.py [candidate program]
  -> robustlab/scenarios/compiler.py
  -> robustlab/optimization/cmaes.py
  -> robustlab/simulation/evaluator.py
  -> robustlab/optimization/confirm.py + shrink.py
  -> robustlab/evidence/archive.py [confirmed/refuted]
  -> optional vision analysis + structured summary
```

Do not let the investigator import or mutate `training/gate.py`; it may call only a controlled patch tool that invokes the gate independently.

### C. Hardening a controller

```text
web/features/training
  -> POST /api/patches
  -> api/scheduler.py [exclusive training job]
  -> api/patch_worker.py [new process, bounded GPU memory]
  -> robustlab/training/jitter.py + bank.py
  -> robustlab/training/ppo.py [warm start + checkpoints]
  -> robustlab/simulation/evaluator.py [checkpoint attempts]
  -> robustlab/training/gate.py [paired holdout comparison]
  -> robustlab/policies/registry.py [promote OR reject]
  -> robustlab/evidence/lineage.py [version status updates]
  -> api/jobs.py [SSE, final gate evidence]
```

Training must not bypass the independent gate because of UI requests, agent tool decisions, or early-stop criteria.

### D. Visual scene to simulated place

```text
web/features/scenes [photo or short video]
  -> api/uploads.py [type/size/EXIF limits]
  -> robustlab/ai/vision.py [uncertain assumptions]
  -> robustlab/ai/segmentation.py [optional concept masks]
  -> robustlab/ai/translation.py [bounded structured program]
  -> compiler -> evaluator -> archive -> replay
```

Vision outputs are not authoritative physical measurements. Segmentation is optional and cannot change the meaning of the schema.

### E. Browser physical playground

```text
web/features/playground
  -> GET /api/play/model [matching .mjb + model metadata]
  -> GET /api/play/policy/{version} [actor-only binary]
  -> web/wasm/observations.ts + policy.ts + simulator.ts
  -> local visual shove
  -> POST /api/play/shoves [bounded force, server re-evaluation]
  -> normal attack archive / training workflow
```

Do not equate in-browser visual frames with server-confirmed simulation data.

## 5. Repository creation / build order — incremental milestones

**`delivery.md` is the authoritative source for build sequence, acceptance criteria, and learning objectives.** Build the smallest demonstrable capability at each milestone. The directory tree in §1 is a future target, not a command to create empty placeholders. In particular, *the 80-dimensional compiler does not precede the first successful native robot simulation*.

| Milestone | Added capability | Demonstration/exit gate |
| --- | --- | --- |
| M0 | Native MuJoCo box under gravity | CPU command writes reproducible physical trace |
| M1 | Inspect and actuate Go1 | Named joints/actuators resolve and move as commanded |
| M2 | Load or train walking policy | Valid checkpoint truly walks on nominal ground |
| M3 | Timed deterministic shove + strict fall check | Heading-relative force and real fall metrics are verified |
| M4 | Paired multi-seed experiment runner | Nominal-versus-attack metrics and manifests reproduce; **first MVP** |
| M5 | Versioned 80-slot typed scenario compiler | Push, friction, payload validated one at a time |
| M6 | Non-AI adversarial search | Reproducible random baseline, CMA-ES and fresh-seed confirmation |
| M7 | Counterexample PPO retraining | Unpromoted candidate checkpoint with before/after metrics |
| M8 | Independent regression gate + lineage | Accepted/rejected versions handled atomically; **research core complete** |
| M9 | Typed API, durable jobs, recorded browser replay | End-to-end submit → simulate → measure → replay |
| M10 | Model-independent assisted input | Validated text-to-program; optional vision/speech adapters |
| M11 | Bounded autonomous investigation | Hypothesis/tool/evidence loop respects budgets and gate |
| M12 | Independent full-product subreleases | Browser live physics, terrain, second robot, benchmarks and scale |

The first two milestones run on macOS CPU with no checkpoint, API key or GPU. A GPU-trained checkpoint is a **hard requirement for claiming M2 walking success**, but CPU work on M3 code and testing can proceed with a clearly labelled synthetic controller. Keep all past commands and tests passing after each stage.

Kiro should not implement M1 while M0 is unverified. Kiro should not implement M5's full physics envelope in one pull request: build its compiler push case, then friction, then payload, then expand only as needed. M12 is split into M12a–M12d in `delivery.md` and has independent acceptance criteria.

## 6. Decision process for new dependencies or abstractions

Before adding a library, new schema coordinate, training algorithm or API boundary, answer:

1. What precise problem does it solve, and can the current standard library or existing dependency solve it simply?
2. Does it introduce a second implementation of a physical rule or another source of truth?
3. Does it change array shapes, XLA compilation behavior, robot model IDs or serialized checkpoints?
4. What is the dependency's version/license/distribution constraint and required runtime hardware?
5. Is this capability required for the scientific core, or can it be a role-specific optional adapter?
6. What golden vector, unit test, integration test, benchmark or migration proves correctness?
7. Can the next developer inspect a saved experiment and reconstruct it from explicit inputs?

Record nontrivial changes as an ADR in `docs/decision-log/` and update the relevant steering file. Keep the implementation aligned with the user-visible product contract.

## 7. Required tests next to each boundary

| Component changed | Minimum evidence |
| --- | --- |
| Robot definition | Resolve names and IDs in actual MjModel; confirm feet, actuators, collision geoms |
| DSL/schema | Valid and invalid JSON; exact bound and duplicate cases; generated schema snapshot |
| Slot layout | Golden neutral/active vectors, dimension/ordering, version upgrade test |
| Physics effect | Unit/scene test demonstrating correct force, sensor, inertia or collision change |
| Terrain placement | Static geometry test and dynamic fall parity; unchanged neutral scene |
| Observation stack | Shape, ordering, actor/critic non-leakage, deterministic PRNG |
| Policy serialization | Fixture-based Python/JS action match; checksum/corruption rejection |
| Batch evaluator | Paired seeds, no accidental reset, correct metric schema and NaN behavior |
| Search objective | Known failure threshold, severity exclusion, deterministic early-stop logic |
| Retraining bank | Family-balancing, quarantine, zero-weight padding, mixture proportions |
| PPO restore | Compatible checkpoint load and normalized critic input |
| Promotion | Every gate predicate and atomic active-pointer behavior |
| AI adapter | Mock provider, schema error/repair, disabled role, budget exceed, media constraints |
| Jobs/SSE | Ordered IDs, resume, end marker, exception event, restart state |
| Public media | Path/input validation, moderation and cache invalidation |

## 8. Kiro working conventions

- Begin each implementation session by checking `product.md`, `tech.md` and this file for the relevant subsystem, then identify the smallest executable slice to deliver.
- Write a short implementation plan specifying input/output contracts, tests, dependencies and risk to existing numerical shapes.
- Prefer test-first development when altering physical math or serialization. Implement one operation, assert its intended effect, then add another operation.
- When blocked by unavailable CUDA or model credentials, finish CPU validation and document the exact GPU/remote integration test that remains; do not mark it passing.
- Keep Python physics and training modules importable without starting an API, browser, provider service or external database.
- Don't invent policy checkpoints, research results, real-world surface coefficients, human-reviewed safety decisions or successful tests.
- Treat all model-generated text, arguments and scene readings as untrusted. Use local schemas, reject unsafe or out-of-bound values, and persist repairs as auditable events.
- Favor reproducibility and independently testable numerical behavior over UI polish, higher model capability or complicated distributed architecture.
- Name every newly created model role and feature by its **function**, not by a specific model vendor.
- Do not copy unexplained constants into multiple files; declare them in settings, robot presets, slot metadata or experiment manifests.
- Before claiming a slice complete, execute its tests and include the command + observed outcome in the implementation summary.

## 9. First files the coding assistant should create

**Begin with M0 and nothing further.** The initial source files are `pyproject.toml`, `robustlab/__init__.py`, `robustlab/cli.py`, `robustlab/simulation/recording.py`, `examples/falling_body.xml`, `tests/test_falling_body.py`, and `README.md`. The exact filenames may vary only if the M0 spec documents a simpler arrangement.

Do not create `scenarios/slots.py`, `scenarios/schema.py`, `scenarios/compiler.py`, `web/`, or `api/` during M0. They are deliberate future milestones, not prerequisites to showing a box fall in MuJoCo. At M1 add the Go1 model registry/loader; at M2 add the actual policy load/inference contract; then follow `delivery.md` strictly.
