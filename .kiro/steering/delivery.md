# Hackathon Delivery Contract — Robustness Studio

> **Execution authority until submission:** This file supersedes the old M0–M12 build order. `product.md`, `tech.md`, and `structure.md` describe the longer-term intended platform, **not** requirements to implement before the deadline. All work must remain part of a new, independently authored repository. The deliverable is an honest CPU-first, model-assisted *simulation robustness prototype*, not a replacement for a GPU-trained robotics research system.
>
> **Submission deadline:** Friday, **2026-12-04, 2:00 PM Pacific Time**. Target a feature freeze by **2026-11-28** and reserve **2026-11-29 through 2026-12-03** for resilience, deployment, video, and contingency. Do not plan critical work for deadline day.

## 1. Scope and proof of value

**The one-minute user story:** Open an interactive robot-simulation website; choose a Go1 walking controller; describe or select a disturbance; observe the baseline robot and its measured outcome; ask the system to search for a meaningful failure; optionally train a *small bounded CPU reflex adapter* around the frozen policy; see paired, independent before/after test results and whether the candidate is accepted or rejected.

**Minimum honest demo:** one pre-trained Go1 policy, one native MuJoCo simulation runtime, a push applied at a defined time and in a specified coordinate frame, deterministic seeded test runs, a failure metric with recorded evidence, a simple web replay, and AI producing a validated structured attack. Additional hardening/search are **conditional** on measured feasibility; a working baseline demo must never be sacrificed to pursue them.

**Important language:** The CPU adapter, if built, is *low-dimensional policy adaptation*. It is **not PPO**, **not neural network fine-tuning**, and **not a guarantee of increased robustness**. Never present a policy as improved unless an independent validation gate with predefined thresholds passes. If no candidate improves, show the rejection honestly.

**Out of scope before submission:** GPU training, JAX/MJX/Brax runtime integration, production queues, 80-slot end-state compiler, photoreal scene reconstruction, segmentation, microphone/speech, G1, multi-robot comparison, stairs, WASM physics parity, domain-randomization research, quantitative sim-to-real claims, long-lived autonomous GPU jobs. Do not create placeholder modules for these features. They remain future scope in other steering files.

## 2. Technical execution profile

- **Physics:** pinned native `mujoco` Python package, NumPy, MuJoCo XML and matching Go1 assets; headless CPU worker. Use `uv`/`pyproject.toml` and a committed lockfile. No accelerator dependencies in default install.
- **Controller:** prefer a *public, redistributable, already-trained Go1 walking policy* with its compatible assets and documented observation/action/actuator conventions. Prefer pre-exported, safe, actor-only weights (e.g. JSON/NPZ or ONNX) and validate their provenance; never unpickle an untrusted checkpoint. A checkpoint's mere existence is not proof it works in our scene. Record upstream revision, artifact hashes, rights/license, and any asset license notices.
- **Runtime:** an authoritative native MuJoCo evaluation path for **all measured results**. The web UI may animate recorded native trajectories through Three.js. It must not silently substitute browser-only physics for the scored evaluator.
- **Backend/UI:** FastAPI plus a small Python worker/executor, React/TypeScript/Vite and Three.js for a trajectory viewer. Keep API endpoints short, return IDs for expensive tasks, use bounded concurrency and either polling or SSE. SQLite is optional; persisted manifests+artifact files are enough for a single-instance hackathon service. If jobs must survive restarts, add a minimal SQLite job table rather than a multi-component queue.
- **AI:** a single provider-neutral text-to-scenario adapter, defaulting to an OpenAI-compatible HTTP API if configured. OpenRouter can be the first provider. Model output must be parsed, validated, bounded, and used only as *data*; never evaluate generated Python, shell commands or XML. Absence of an API key must leave the rest of the app fully functional through presets.
- **Optimization:** bounded CPU sweeps first. `CMA-ES` may be integrated only after benchmarking its objective and the cost per rollout. Parameter bounds, seed lists and run budgets are explicit. Optimize on *development seeds*, validate on *unseen seeds*.
- **Optional adapter:** frozen original actor plus bounded correction `a = clip(a_base + delta_beta(obs), action_min, action_max)`. First try 2–4 interpretable gains; allow a 16-parameter linear feature adapter **only if** measured searches are tractable. Build from four observable signals (e.g. projected-gravity components and body angular velocities), normalized and clipped; compose through a documented fixed, symmetric joint-action basis. Set explicit action correction, rate and actuator limits. The baseline's observation history and action mapping stay unchanged.
- **Contracts:** immutable run manifest (policy/asset hashes, versions, seed, scenario, time step, coordinate frame, evaluation code revision, outcomes). Runtime-specific numeric tolerances rather than unsupported byte-identical cross-platform claims.
- **Repository:** `.kiro/steering` committed; source repo must have a visible open-source license, setup/test/run commands, public access to required deployable assets or documented retrieval, public live app URL, and a public video under three minutes. Prepare submission evidence from day 1.

## 3. Eight daily milestones (M0–M7)

### M0 — Day 1, Nov 21: A walking Go1 in a reproducible local process

**User-visible result:** One command runs the actual trained Go1 controller on flat ground and records a walking trajectory (including an optional video/PNG). It must run on the intended development machine without CUDA.

**Implementation:** Create minimal Python package and CLI; pin MuJoCo/NumPy. Integrate robot MJCF, actuator control loop, and existing compatible actor export. Verify exact observation assembly, normalizer, action scale, joint order, policy control period vs physics substeps, robot initial state, and command semantics. Create a baseline manifest and a short replay trace. A falling-box or individual-joint check may be a 15-minute diagnostic **inside this milestone**, not a stand-alone release.

**Exit:** test imported policy on known observations; on a nominal forward command it remains upright and produces meaningful forward progress in the same scene and settings as the reference policy. Record actual distance, duration, fall flag, ground contacts, latency. Tests pass. If not, never label it walking.

**Stop-loss:** spend at most an initial focused integration window on a candidate checkpoint; if incompatible, switch to another *verified public* policy+scene pair. Keep original matching scene rather than rewriting the observation contract. If no walking policy works by end of Day 2, prioritize a physically honest deterministic robot demonstrator with explicit reduced scope and do not pretend random weights work.

**Learn:** MJCF; `MjModel` vs `MjData`; `qpos/qvel`; action normalization; controller tick; stable closed-loop inference.

### M1 — Day 2, Nov 22: One shove, one authoritative fall detector

**User-visible result:** Run the baseline twice: normal walking and a scripted sideways shove. The saved outputs show the trajectory, push vector and applied interval, fall reason and fall time.

**Implementation:** Add `PushScenario` with strength in N, onset and duration in simulated seconds, direction in an *initial-heading* frame (record transform to world frame). Apply and clear force at every physics substep. Turn off unrelated random impulses and perturbations. Use an explicit `FallResult` with cause/first timestamp; validate non-foot contacts against proper collision geoms, plus height and tilt checks. Record sensor/physics state at a bounded sampling rate; no rendering inside scored timings.

**Exit:** zero push equals nominal within tolerance; 0-duration push changes nothing; known sign/direction tests pass; severe shove produces a real measured perturbation; fall detection is unit-tested on synthetic states and checked against simulation. Don't force a fall verdict to make the demo dramatic.

**Learn:** local/world frames, forces vs impulses, contact geometry, fall metrics.

### M2 — Day 3, Nov 23: A reproducible experiment runner

**User-visible result:** CLI compares two conditions on a fixed seed set and prints measured fall rate, forward distance and time-to-fall, with saved run manifests and trajectories.

**Implementation:** Add a typed **minimal** scenario schema (`push` + command + horizon); only the currently implemented operation is accepted. Run matched seeds on baseline/perturbed conditions. Support repeat command, a manifest schema version and SHA-256 hashes. Compute aggregate results from per-episode records, not UI state. Include `--dry-run` to show parameters and resource budget. Add basic property-based tests for malformed push bounds and episode timing.

**Exit:** same manifest produces matching metrics within declared tolerance on the same build; report counts and the exact denominator; failure cases are classified, not silently retried; API-free CLI already useful. Store a golden demo experiment.

**Learn:** experimental controls, random seeds, fair comparisons, deterministic vs stochastic execution.

### M3 — Day 4, Nov 24: A working, judge-clickable browser UI

**User-visible result:** A deployed beta page lets someone choose a scenario, click **Run**, watch an authentic recorded Go1 trajectory, and inspect experiment metrics. The public URL can be tested from a fresh device.

**Implementation:** thin FastAPI endpoints for presets/run/status/results/trajectory; bounded single-worker task execution (never block the event loop on simulation); timeouts and cancellations; basic React UI and a Three.js replay of saved `qpos`/pose data. Show clear nominal vs attacked result and a playback scrubber. If rendering is a blocker, first use pre-recorded genuine videos generated by the same simulator with a visible note and keep metrics live. Distinguish recorded replay from live physics.

**Exit:** production build and cold-start tested; one-click demo works without login or model key; backend failures are visible; no leaked server paths or secret keys. Freeze the working demo route now.

**Learn:** API contracts, async orchestration, 3D playback, deployment failure modes.

### M4 — Day 5, Nov 25: Automatically find a failure

**User-visible result:** Click **Find breaking point**. System probes push amplitude and reports a measured failure frontier/lowest confirmed tested level, with replay of a discovered failure.

**Implementation:** begin with bounded force sweep or binary-style refinement only when the response is sufficiently monotonic; otherwise report a tested interval, not a true mathematical threshold. Cache scored outcomes by full manifest/hash; cap number of trials and total wall time. Include multiple seeds where budget allows and validation seeds excluded from selection. An optional small CMA-ES search over force + timing comes **after** the sweep works.

**Exit:** optimization does not exceed its run budget; it reports every tested parameter/result; results can be replicated; a failed search is an honest 'not found in budget,' not a fabricated success. Benchmark `ms per physics step`, `seconds per rollout`, and `seconds per search` on target Mac before promising live latencies.

**Learn:** objective functions, optimization, non-monotonicity, generalization from test seeds.

### M5 — Day 6, Nov 26: Natural-language scenario compiler (OpenRouter optional)

**User-visible result:** Type a physical situation like "sideways push while walking forward"; a configured model proposes a concise JSON scenario, displays it for review, then the user can run it. Without a key, scenario presets and manual fields still work.

**Implementation:** provider-neutral `ScenarioProposer` interface; first adapter to configured OpenAI-compatible endpoint; structured JSON response where supported; retry only on parsing/network failures and within cost/time bounds. Pydantic validation forbids unknown keys and nonphysical ranges. Separate natural-language *intent* from approved numeric defaults; ask for review before running costly searches. Include examples of rejected model output and prompt-injection-like content. Store the model ID and validated proposal, not raw secrets. No autonomously generated code.

**Exit:** 10 representative user prompts yield valid bounded schemas or explicit errors; malformed/hallucinated model fields cannot alter the simulation; model timeout leaves the app usable. Do not imply photo understanding if no vision adapter exists.

**Learn:** LLM tools/structured output, schema validation, trust boundaries and prompt variability.

### M6 — Day 7, Nov 27: Optional CPU policy-adapter hardening

**User-visible result:** For one previously discovered *trainable* attack, click **Try hardening** and see a bounded search for a small reflex correction to the existing frozen controller. The feature always distinguishes an accepted improvement, rejected candidate, and timeout/no result.

**Implementation:** save unchanged original actor and define a small transparent residual adapter. Start with **2–4 trainable parameters**, not an unexplained 16-D magic layer. Features: observable body tilt proxy and roll/pitch rates; correct signs and gains through fixed tests. Clamp residual magnitude/actions, use seed-stable rollouts, early stop on true falls, cache evaluations, and constrain wall time. Optimize on train seeds only using grid/random search; add CMA-ES or expand to 16 parameters if measured beneficial. Test that zero adapter yields exactly the baseline actions. Avoid PPO/retraining terminology.

**Exit:** optimization budget enforced; candidate saved with params and provenance; independent validation required before any 'better' claim. If improvements cannot be demonstrated in the allocated time, retain **Find breaking point** and omit hardening from featured demo rather than faking results.

**Learn:** additive feedback control, optimizer sample efficiency, overfitting to attacks.

### M7 — Day 8, Nov 28: Independent gate, comparisons, feature freeze

**User-visible result:** A side-by-side report explains baseline versus candidate under identical scenarios, with a visible accept/reject verdict based on metrics rather than model opinion. Replays and exportable manifests are available from the demo route.

**Implementation:** locked validation seeds/scenarios, policy-regression checks for normal walking and a small range of untouched disturbance cases. Distinguish exploratory/selection/holdout data. Enforce nonnegotiable nominal stability; require statistically defensible positive robustness improvement for a claimed win, preferably paired episode outcomes with confidence intervals or sufficiently large seed counts. For small samples, present exploratory comparisons, not proof. Never update the active controller if a gate fails. Keep a pointer to latest accepted immutable candidate only. Record failure classes and evaluation denominator.

**Exit:** nominal demo, attacked demo, failure search, AI proposal, and (only if validated) bounded hardening work from one deployed URL. Re-run all smoke and contract tests, package reproducible example data, pin dependencies, record demo sequence. **Feature freeze.**

**Learn:** train/validation leakage, fair comparisons, promotion safety gates, repeatable scientific demos.

## 4. Nov 29–Dec 3: Buffer and submission, not new architecture

- **Nov 29:** End-to-end testing on laptop and deployment; collect latency evidence, mobile/desktop screenshot and browser compatibility; fix crashers.
- **Nov 30:** UI readability, first-run sample, response to no model key, accessibility, plot/replay polish. Preserve the core scientific flow.
- **Dec 1:** Confirm hosted URL, public repository, open-source LICENSE, complete `.kiro` committed, third-party notices, setup instructions, clean clone smoke-test. Draft submission. Resolve any Kiro credit eligibility by organizer's published cutoff.
- **Dec 2:** Record **<3 minute** public demo showing actual simulator results, how Kiro produced a tested implementation, and failure/recovery handling. No results hardcoded as live.
- **Dec 3:** Submit early, verify links from incognito, verify public YouTube video, final regression smoke. Keep Dec 4 strictly as contingency before 2 PM Pacific.

## 5. Contingency ladder and time-box policy

1. **Non-negotiable:** real simulation + verified walking policy + deterministic push + measured metrics + accessible app + documentation and demo. Protect these first.
2. **High-value but cuttable:** automated push search; model-assisted typed scenario generation. If online providers fail, presets support the demo. If search is slow, use a bounded asynchronous job and an authentic previously saved result with a label.
3. **Optional:** CPU adapter hardening and gated promotion. Include only when a candidate *actually passes* tests.
4. **Never before deadline:** real PPO fine-tuning, complete 80-D DSL, browser-only physics, photos, G1, full autonomous investigation.

A single optimization run taking 3–5 seconds is **not a design assumption**. The total cost scales roughly as `population × generations × seeds × episode_steps × physics_substeps`. Benchmark on target hardware and enforce a configurable timeout (e.g. preview and extended budgets), with honest partial results.

**Stop rules:** if a milestone slips more than half a day, split its minimum meaningful behavior and defer optional additions. If baseline weights fail, spend no more than one additional workday integrating an alternative verified checkpoint; use a simpler validated scene and preserve the web flow. Do not launch more than one risky research effort at a time. Do not require human review between every tiny subtask; request review at **M2 (science), M3 (web), and M7 (final)**.

## 6. Kiro working agreement and quality practices

1. Work **only on the current milestone**, plus its direct prerequisites. At the start show: what the user will be able to do, physical/ML concepts introduced, expected file changes, assumptions, and exact exit tests.
2. Before implementing each new subsystem, inspect existing proven local/upstream code and its licensing; reuse libraries and authorized compatible components, while writing project-specific integration, verification, and attribution. Never blindly paste source with unclear rights.
3. Create a lightweight Kiro spec per substantial vertical slice (requirements, design, task checklist), not a 100-page plan or dozens of empty modules. Commit `.kiro/` and a brief `docs/kiro_build_log.md` mapping Kiro outputs to observed tests.
4. Use Kiro hooks (where supported) for `ruff`, `pytest`, schema/PBT checks, and frontend typechecking on relevant file changes; avoid GPU/network calls in hooks. Keep tests fast enough that developers will run them.
5. Put source of truth in simulator and test files. React components cannot declare a policy 'safe' or compute the authoritative fall verdict. The model cannot bypass scenario validation, optimize policy parameters directly, or promote a candidate.
6. Record each milestone in `docs/milestones/M*.md`: command used, actual observed result, screenshot/trace path, tests passed, unresolved limitations, and the next dependency. The progress log must separate proposed features from working ones.
7. Show provenance: policy and robot asset origins, dependency revision, relevant licenses, scene modifications, and measured parity. Attribution in third-party notices does not create an affiliation between the product and other projects.
8. Favor one working deployment and one reliable demo path over extra settings, frameworks, infrastructures, or advanced numerical methods.

## 7. Suggested minimal repository evolution

```
.kiro/steering/                # commit all four files
.kiro/specs/                   # current slice's requirements/design/tasks
src/robustlab/
  physics/                     # Go1 load, native MuJoCo stepping
  policy/                      # actor export, observation/action contract
  scenarios/                   # push DTO, frame transforms, validation
  experiments/                 # runner, evaluator, traces and manifests
  search/                      # push sweep, optional bounded optimizer
  adapt/                       # OPTIONAL reflex adapter and promotion gate
  providers/                   # text-to-scenario adapter (M5)
  api/                         # thin FastAPI routes/job execution (M3)
web/                           # React + Three.js replay (M3)
assets/                        # model and documented third-party notices
examples/                      # versioned scenario presets
runs/                          # generated, gitignored experiment evidence
scripts/                       # fetch/verify external assets and demo
tests/                         # simulation, schema, API, and UI tests
 docs/milestones/              # actual milestone observations
```

The target end-state module tree in `structure.md` remains useful for later releases; **do not pre-create it**. When delivery and structure disagree about *when* a package appears, delivery wins. When technical contracts differ because of a deliberate prototype substitution (CPU adapter vs PPO), record an ADR and tell the truth in the UI.

## 8. First command to Kiro after installing this file

> Read `.kiro/steering/{product,tech,structure,delivery}.md`. Treat `delivery.md` as the time-bound execution authority. Implement **M0 only**: run a verified, existing pretrained Go1 walking policy through native MuJoCo on this Mac and record a valid trajectory. Before touching code, inspect public policy/model assets, their observation and action contracts, and licenses; choose the least risky compatible combination. Run actual tests and a numerical walking rollout. Produce `docs/milestones/M0.md` with commands, timings, checksums, observed distance and fall criteria. Stop after M0. Do not install JAX/CUDA, train a policy, build the UI or create future module stubs.
