# Product: Robustness Studio

> **Deadline delivery mode (November–December 2026):** `delivery.md` defines the narrow, measurable hackathon release. The complete product vision below remains a future roadmap. CPU residual-adapter optimization, if included, is explicitly not PPO retraining. Do not describe unimplemented features as complete.


> Kiro steering — product definition. This document defines the product to be built, not a description of software that already exists. The working name may be changed without changing the architecture. Read alongside `tech.md` and `structure.md`.

## 1. Vision

Build a simulation-first robustness engineering platform for legged robots. Engineers should be able to describe a difficult physical situation, convert it into a bounded and reproducible simulation, discover whether a walking policy fails, diagnose the failure, train a candidate improvement, and promote the candidate **only after independent regression evaluation**.

A natural-language or visual AI model **proposes experiments**; it does not control the robot, author arbitrary simulator code, declare physics outcomes, or decide whether a trained policy is safe. The physics simulator, deterministic validation, quantitative evaluator, and promotion gate are the sources of truth.

The platform must have a useful core even with every generative AI provider disabled: users can author structured experiments, replay failures, search perturbation parameters, retrain policies, and compare versions.

### Product hypothesis

Manually testing locomotion policies requires expertise in low-level simulator configuration and may miss interacting failure conditions. A reusable scenario language plus reproducible search and regression-gated training can make robustness testing more systematic, interpretable, and accessible.

### Guiding principles

1. **Physics before prose.** All factual claims about success, falls, timings, and margins must be derived from recorded simulator measurements.
2. **Reproducibility before spectacle.** Every result identifies the policy checkpoint, robot model, environment, perturbation layout version, simulator version, seeds, horizon, code revision, and evaluation settings.
3. **Failures are valuable data.** Discoveries are preserved as counterexamples with provenance, measurements, and policy-version status.
4. **Improve without forgetting.** Training must be evaluated on the failure, nominal movement, and a fixed regression suite. Rejection is a first-class outcome.
5. **No privileged knowledge at inference.** The deployed actor can consume only deployable observations; simulation-only parameters may be supplied to the training critic.
6. **Model providers are replaceable.** Separate role contracts for language reasoning, vision, segmentation, and speech. Behavior must not depend on any one vendor.
7. **Bound the action space.** Every experiment compiles from a validated declarative program into named, unit-bearing physical parameters; never execute AI-generated code.
8. **Honest epistemics.** A visual estimate of friction or a simulated fall is not an experimentally measured real-world safety guarantee.
9. **Start locally; scale deliberately.** A single-machine research implementation comes first. Concurrency, persistence, and workers are explicit interfaces so infrastructure can evolve.
10. **Treat assumptions as hypotheses.** Default numerical settings are reference starting points subject to benchmark tests, not universal physical truths.

## 2. Users and jobs to be done

| Primary user | Problem | Desired outcome |
| --- | --- | --- |
| Robotics / controls researcher | Controller fails under unknown dynamics; manual sweeps are cumbersome | Reproducible failure discovery, margins and comparative measurements |
| RL engineer | A controller improves on one case but degrades elsewhere | Replay-based fine-tuning with an auditable acceptance gate |
| Simulation engineer | Scenario configuration is expensive and error-prone | Typed scenario programs and stable simulator integration |
| Deployment engineer | Has a description or image of a location, but no systematic test plan | Approximate site-specific hazards and evidence-backed deployment caveats |
| Technical educator / evaluator | Wants to see why a controller failed | Interactive replay, diagnostics, version comparison and experiment provenance |

Non-primary audience: visitors who use an optional challenge mode to attempt to defeat a robot. Community features must never dictate engineering acceptance or be mistaken for measured safety.

## 3. Terminology / domain model

- **Robot definition**: metadata and physical envelope for a robot family, including named feet, joints, actuator mappings, nominal gait command, fall geometry and available terrain features.
- **Resolved robot specification**: the definition bound to concrete MuJoCo body, geom, actuator, joint, site and contact-pair identifiers of a particular model.
- **Policy**: neural actor mapping recent observations to motor commands, plus associated normalization and training checkpoint metadata.
- **Policy version**: immutable trained artifact with an identifier, parent version, creation time, training evidence, gate result and manifest.
- **Scenario program**: validated JSON containing a name, physical hypothesis and list of typed operations; may include fixed values or numeric search intervals.
- **Canonical perturbation**: a float32 vector `theta` (initially dimension 80) with a declared, versioned mapping from coordinates to physical quantities.
- **Attack**: a scenario executed against one or more policy versions with saved simulator measurements; the term describes adversarial testing, not physical violence against people.
- **Episode**: one simulated trial under one policy, perturbation vector and seed.
- **Counterexample**: a reproducible scenario where the current policy fails or its task performance falls below the defined threshold.
- **Search run**: optimization history for a scenario program, including candidate vectors, seeds, fitness, failure rates and costs.
- **Failure family**: coarse physical cause such as low friction, payload, slope, actuation, external force, latency, sensing or terrain.
- **Behavior signature**: observed direction or mechanism of falling, combined with the inferred perturbation family.
- **Counterexample bank**: versioned collection of scenarios and sampling weights used to avoid forgetting.
- **Patch / hardening run**: PPO fine-tuning of a policy using new counterexamples, variants and replay; produces a candidate checkpoint, not an automatic deployment.
- **Regression gauntlet**: immutable, versioned collection of scenarios and seeds withheld from training, used for policy promotion.
- **Job**: long-running task with state, event sequence, correlation ID, input manifest and result location.
- **Readiness report**: engineering summary of measured robustness and remaining unknowns for a particular policy version; not a safety certification.

## 4. Critical workflows

### W1. Describe and execute an experiment

1. User chooses a robot (`go1` first; humanoid support later) and active policy version.
2. User enters a plain-English scenario, edits structured JSON, records a spoken description, or uploads a scene image/video.
3. A model adapter may generate one program with fixed physical values. The backend validates fields, units, operation counts, robot capabilities, and envelope bounds; on error it returns exact repairable diagnostics.
4. The compiler emits an 80-element `theta` vector and an immutable scenario manifest, not Python code.
5. An evaluator runs the **baseline** and **selected/live** policy against the exact same evaluation seeds, with no undocumented simulator perturbations.
6. The system reports falls, fall rate, time to first fall, command-tracking error, progress, robustness margin, and optional joint trajectories. Display numerator/denominator for every fall rate.
7. A 3D replay and comparison view shows the two rollouts. A model may annotate video only after the simulator result exists.
8. The result and input provenance are stored; only approved data may appear in public listings.

**Acceptance:** deterministic structured scenario produces the same compiler output and test configuration independent of AI availability; policy comparison uses paired seeds; a model narration cannot overwrite numeric results.

### W2. Discover a reproducible weakness automatically

1. Researcher selects policy, robot, physical families and search budget.
2. A reasoning model proposes testable hypotheses as scenario programs with bounded numeric ranges. It sees the robot envelope, schema, previous successes and failures, and its allowed tool budget.
3. Invalid proposals are rejected and may receive **one** validation-feedback retry. Exhaustion yields a visible error, not silent coercion into a different attack.
4. A derivative-free optimizer searches only the free parameters from the program, not all coordinates of `theta`.
5. Candidate batches are simulated, scored by failure reproducibility and normalized perturbation magnitude.
6. The best candidate is re-evaluated on **fresh seeds**. Optionally simplify it one condition family at a time, preserving reproducibility.
7. Archive a discovered counterexample with its hypothesis, optimized values, complete search history, confirmation evidence and failure signature.
8. If no confirmed failure exists, archive a **refuted/unsupported hypothesis** with results so the agent avoids circular repetition.

**Acceptance:** a failure label requires a deterministic simulator threshold and measured seeds; a model cannot mark a result confirmed; the search must work without an LLM when supplied a program directly.

### W3. Harden a policy

1. Engineer selects one or more confirmed failures that are eligible for training.
2. System checks whether a reference-stronger controller also fails very frequently. If so, mark the scenario *possibly infeasible* and require explicit override; never assert physical impossibility from this heuristic.
3. Build a fixed-shape training bank of new scenarios, nearby jittered variations and previous accepted examples, with zeros for unused probability mass.
4. Restore the selected policy's actor, critic, and observation-normalizer state; validate architecture and layout compatibility.
5. Train PPO with a defined mixture of nominal, replay-bank, and mild-uniform-randomization episodes; instrument every checkpoint.
6. At each checkpoint, measure the new attack and nominal walking on consistent seeds. Early stopping requires a minimum training budget and repeated evidence.
7. At the end, evaluate old and candidate policies on the attack, normal walking, and locked gauntlet using paired seeds.
8. Promote only if all configured criteria pass. Otherwise quarantine the checkpoint in rejected artifacts and preserve failure diagnostics.
9. On promotion, append an immutable policy lineage entry, update replay history and re-evaluate historic attacks on the new live version.

**Acceptance:** training completion is never equivalent to promotion; rejected candidates cannot be loaded as live by mistake; no earlier success is forgotten silently.

### W4. Investigate a real-place description

1. User supplies a text description or a still/video of a place.
2. A segmentation adapter may outline steps, ramps and low-friction-looking regions; a vision adapter estimates uncertain geometry and surface properties.
3. Treat all inferred coefficients, scales, depths and slopes as **assumptions with uncertainty**. Images are not reliable friction sensors or calibrated depth maps.
4. Generate a small deployment-specific test plan containing fixed structured scenarios, plausibility explanations and severity levels.
5. Run every situation on baseline/live policy, optionally refine only within user-approved physical envelopes, and rank measured failures.
6. Label diagnostics, train against learnable cases if requested, then repeat the full suite after promotion.
7. Present recommendations as simulated evidence with explicit conditions and limitations.

**Acceptance:** model-only visual estimates never become ground-truth physical measurements; uploaded media has size/type limits and is handled as untrusted content.

### W5. Autonomous investigation

Use a bounded tool-calling agent with exactly these conceptual capabilities:

| Tool | Observable action |
| --- | --- |
| `test_hypothesis` | Validate program, optimize parameters, confirm and store result |
| `measure_margin` | Sweep/bisect one named physical condition against a fixed baseline |
| `watch` | Request descriptive analysis of a saved clip, clearly separated from simulator metrics |
| `patch` | Request controlled training and gated comparison of eligible failures |
| `finish` | Return an evidence-linked summary, including negative findings |

The agent must have a maximum number of tool calls, a separate expensive-search quota, a total time/cost budget, cancellation handling, and an append-only user-visible event history. Tool arguments are validated independently of what the model promised.

### W6. Interactive browser playground

- Load a stripped MuJoCo robot model plus a serialized actor and its normalizer; simulate the control loop inside the browser with MuJoCo WebAssembly.
- Display original and latest versions, with consistent motion controls and clear version labels.
- Allow bounded shoves by drag or explicit force/duration inputs. The browser simulation is educational; replay a user-selected experiment on the server before admitting it to the evidence archive.
- Do not treat a browser-only collision or render result as an authoritative server evaluation.
- Verify parity between native and browser inference for observation order, scaling, activation functions, action output and control frequency.

### W7. Optional challenge/community experience

- A human and an agent submit independent attacks against the same version. In the same round the model must commit to its proposed scenario before seeing the user's candidate.
- Both candidates use paired seeds. Score a successful attack by fall rate and subtlety, not simply raw force; a brute-force attack that defeats the designated reference policy earns no competitive credit.
- Include a leaderboard, hall of failure cases, shareable clips/cards and optional community training queue only after the core testing and versioning system is reliable.
- Public user text, handles and generated captions require moderation; moderation failures remain hidden from public pages, without blocking a user's private ability to run an experiment.

### W8. Readiness review

For one policy checkpoint, generate a reproducible report containing: the test suite hash, number of unique scenarios, measured seed counts, failures by family, before/after comparisons, safety-margin sweeps, highest-severity remaining failures, regressions, rollout examples, policy ancestry, and a plain-language explanation grounded in numbers. No unsupported 'safe to deploy' verdicts.

## 5. Product requirements and invariants

### Physical truth and scenario semantics

- `P-001` Simulator outcomes, not model judgments, define falls and task success.
- `P-002` A fall includes measured non-foot body contact with a ground/step, excessive trunk tilt, root below a robot-specific threshold, or non-finite simulation state.
- `P-003` An ordinary environment termination is not necessarily a fall; distinguish training termination from physical fall.
- `P-004` A robot that remains upright but does not follow the commanded velocity is not counted as a successful task completion.
- `P-005` Perturbations use SI units except explicitly documented angular terrain inputs; angle unit conversions are centralized.
- `P-006` Force directions are relative to the robot's **initial-heading frame**: +x forward, +y left, +z up. A shove *from* the left pushes toward negative y.
- `P-007` Ranged fields represent optimizer freedoms; fixed fields are immutable during optimization.
- `P-008` Program compilation uses stable shapes and a versioned schema. A feature addition must not silently change the interpretation of an older vector.
- `P-009` Each robot has its own physically meaningful bounds and capabilities; unsupported operations must fail validation.
- `P-010` Terrain is a simplified collision representation, not a metrically recovered digital twin.

### Evaluation integrity

- `P-011` Record trial seeds, total trials and reference policy for every aggregate metric.
- `P-012` Evaluate policy comparisons with common random numbers (same seeds) and identical perturbation vectors.
- `P-013` Separate optimization/evaluation seeds from holdout confirmation and promotion-gate seeds.
- `P-014` Never use promotion-gauntlet scenarios to tune training hyperparameters without explicitly retiring and replacing that holdout suite.
- `P-015` Track time to fall, fall direction, tracking error and distance progress in addition to the binary label.
- `P-016` Treat non-finite simulation values as failures and log the physical parameters that generated them.
- `P-017` Numerical differences between different batch shapes or backends may exist; test reproducibility within declared tolerance rather than claiming universal bitwise identity.
- `P-018` The summary report must distinguish observed evidence from model inference and engineering judgment.

### Policy safety and provenance

- `P-019` Each policy version is immutable and has an explicit parent.
- `P-020` The promotion gate is separate from PPO and is executable without the frontend.
- `P-021` Promotion is atomic: never publish a partially written checkpoint, model binary or manifest.
- `P-022` Preserve rejected candidates and reasons for rejection.
- `P-023` Actor observations may not contain the ground-truth perturbation vector or hidden simulation values used only by the critic.
- `P-024` Prior failures remain eligible for regression testing even when excluded from current training by a learnability heuristic.
- `P-025` A reference-stronger policy failing frequently is *evidence of difficulty*, not a proof no possible controller can succeed.

### Platform and security

- `P-026` No model-generated Python, MJCF/XML, shell commands or arbitrary URLs are executed as part of attack translation.
- `P-027` Fail closed on invalid schemas, unknown names, out-of-envelope values and missing model permissions.
- `P-028` API and background tasks use structured identifiers and state transitions; every client-visible failure is explicit.
- `P-029` Long-running jobs stream append-only, ordered events with reconnection support.
- `P-030` Enforce media upload size and content-type validation; never trust the uploaded extension alone.
- `P-031` Public text is escaped, moderated and optionally redacted at rendering time; user-generated media never becomes a shell command.
- `P-032` Bound GPU slots, per-user requests, AI token budget, worker memory and concurrent training processes.
- `P-033` Provider errors and temporary model unavailability must not corrupt simulator or policy state.
- `P-034` Self-hosted and hosted model options expose the same typed application-level interfaces where capability permits.

## 6. Default quantitative contract

These are **initial research defaults**. Centralize them in one versioned configuration file; change them only with experiments and compatibility tests.

| Quantity | Starting value | Meaning |
| --- | --- | --- |
| Control period | 0.02 s (50 Hz) | Policy action update frequency |
| Standard episode | 5 s (250 steps) | Typical attack trial duration |
| General evaluator horizon | up to 300 steps | Configurable longer evaluation |
| Canonical vector | 80 float32 coordinates | Fixed-shape perturbation interface |
| Free search parameters | up to 10 | Padded optimizer input |
| Program operations | 1–6 | Bound scenario complexity |
| Push operations | at most 2 | Reuse two fixed force slots |
| Low-friction patches | at most 2 | Reuse two terrain-friction slots |
| Terrain route program | at most 1 | Shared fixed terrain layout |
| Search population | 24 candidates | Batched derivative-free search |
| Search generations | up to 12 | Ordinary search budget |
| Search seeds | 8 per candidate | Cheap inner-loop estimate |
| Confirmation seeds | 32 fresh | Independent reproducibility check |
| Search failure threshold | 75% | Failure fitness phase |
| Archive eligibility | at least 50% on fresh seeds | Candidate bank threshold |
| Shrink preservation | at least 80% | Avoid brittle simplifications |
| Quarantine heuristic | at least 95% current-policy falls | Possibly unlearnable; review needed |
| Bank capacity | 512 rows | Stable training batch shape |
| Local jitter cloud | 48 variants/new failure | Generalization near counterexample |
| Jitter standard deviation | 0.08 of coordinate envelope | Active coordinates only |
| Training episode mixture | 50% nominal / 40% bank / 10% mild uniform | Protect ordinary walking |
| Uniform envelope for fine-tuning | 25% of allowed span | Mild randomization |
| Fine-tune learning rate | 1e-4 | PPO warm start |
| Checkpoint/test interval | 1 million environment steps | Progress observation |
| Minimum fine-tuning | 2 million environment steps | Avoid lucky early promotion |
| Maximum fine-tuning | 8 million environment steps | Bound resource usage |
| Attack and gate seeds | 16 paired seeds | Compare candidate and parent |
| Early-stop attack fall rate | at most 2/16 | Plus nominal walking requirement |
| Default tracking-error tolerance | 0.4 m/s | Upright alone is insufficient |
| Regression aggregate tolerance | +0.03 absolute fall-rate | Explicitly not zero-regression guarantee |

The promotion gate's default exact conditions are:

```text
new_attack_after < new_attack_before
AND new_attack_after <= max(0.50, new_attack_before - 0.25)
AND nominal_after_fall_rate == 0.0
AND nominal_after_forward_progress >= 0.90 * nominal_before_forward_progress
AND gauntlet_after_fall_rate <= gauntlet_before_fall_rate + 0.03
```

`0.03` means **three percentage points**, not 3% relative. The aggregate gauntlet rule does not guarantee that each individual scenario is non-regressing. Expose individual-case changes in reports, and require stricter per-case checks for serious deployment applications.

## 7. Modes and release scope

### Release A — research kernel (mandatory)

Structured scenario editor, first quadruped, canonical vector, simulation, reproducible batch evaluator, non-AI CMA-ES attack search, counterexample archive, PPO training, promotion gate, CLI, unit/integration tests and run manifests. **No model API required.**

### Release B — assisted investigation

Provider-agnostic text-to-program compiler, tool-calling investigator, optional vision and segmentation, voice transcription, progress streaming, side-by-side replay, policy lineage, regression review and deployment brief.

### Release C — interactive experience

Browser physics playground, interactive shoves, shareable results, challenge mode, moderation, quotas, generated summaries, cost controls, and richer reports.

### Release D — extended robotics

Humanoid robot, multiple locomotion policy families, additional terrain templates, stronger evaluators, scalable worker scheduling, experimental reproducibility dashboard, and optional hardware-in-the-loop integration under independent safety controls.

## 8. Non-goals and explicit limitations

- Do not build a model that drives a physical robot directly through an LLM's free-form instructions.
- Do not claim a single simulation engine, one robot model or one domain-randomization envelope captures all real-world conditions.
- Do not claim estimated friction from a photo is measured truth.
- Do not use benchmark results from another project as the new project's own evidence.
- Do not assume adversarial training always outperforms well-tuned domain randomization; run matched-compute baselines.
- Do not claim 'safe for deployment' solely because a locked gauntlet passed.
- Do not build community gamification ahead of the reproducible scientific core.
- Do not silently enable expensive cloud models, media providers or remote logging.

## 9. Research evaluation design

Before asserting a research benefit, run matched-compute comparisons between: baseline locomotion controller; full-envelope domain randomization; several envelope widths including a tuned width; uniform structured random search; shuffled structured-program search; and model-assisted program search. Match policy training steps, simulator backend, seed protocol and test distributions. Report uncertainty across at least three independent training seeds where practical, and reserve fixed holdout tests that are not used for choosing hyperparameters. Count **distinct confirmed failure modes**, not just raw candidate failures; report both per-family and aggregate rates to expose distribution shift.

## 10. Definition of done for a user-visible robustness claim

A result is shippable only when: (1) input scenario validates, (2) numerical vector can be reconstructed, (3) exact policy and model can be loaded, (4) seeds and horizon are recorded, (5) simulator metrics are saved, (6) outcome is reproduced on holdout seeds when marked confirmed, (7) any generated explanation cites those metrics, (8) regression and ancestry are linked for trained versions, and (9) the interface labels limitations clearly.

## 11. Instructions to the coding assistant

When implementing features, preserve the distinction among **hypothesis → program → compiled perturbation → measured rollout → confirmation → training candidate → regression decision → version**. Do not merge steps, invent measurements, bypass the gate for convenience, or assume a particular AI provider. Implement each stage independently and add a small executable verification before integrating the next stage. Where these steering files define a default but code does not exist yet, treat it as a build requirement rather than a claim of completed functionality.
