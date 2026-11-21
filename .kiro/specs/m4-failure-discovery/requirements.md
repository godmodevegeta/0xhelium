# Requirements Document

## Introduction

Milestone M4: a bounded, reproducible, CPU-only search for low-impulse trunk shoves that make the
walking Go1 fall, with counterexample shrinking, integrated into the M3 web app. Every candidate is
simulated by the M1 `EpisodeRunner` and judged by the M1 `FallMonitor`; confirmations and replays
use the M2 `run_trial`. Two searchers, Random Search and CMA-ES (pycma), get identical rollout
budgets. The target is the lowest-impulse confirmed physical failure found within the budget. It
is not a global minimum and carries no real-world plausibility claim.

Out of scope: GPU, JAX, PPO, training, LLM providers, friction, payload, warm starts.

## Glossary

- **Search_Space**: `force_n` ∈ [10, 150] N (0.1 N resolution); `direction_deg` ∈ [0, 360)
  (0.1° resolution) in the initial-heading horizontal plane, 0° forward, 90° left;
  `start_substep` ∈ {500, …, 755} (start 2.000–3.020 s, 4 ms grid); `duration_substeps` ∈
  {10, …, 50} (0.040–0.200 s). Fixed: command (1, 0, 0), 5 s horizon, policy, nominal reset,
  `fall-contact` scene.
- **Candidate**: one point of the Search_Space; impulse = `force_n × duration_s` (N·s).
- **Evaluation**: one simulated shoved episode of a Candidate (one physics rollout).
- **Outcome**: `fell` (FallResult primary event physical), `survived`, or `numerical`.
- **Candidate_Failure**: an Evaluation with Outcome `fell`.
- **Confirmed_Failure**: a Candidate_Failure re-simulated as a paired trial through `run_trial`
  whose shoved trajectory digest equals the Evaluation's and whose pre-shove parity holds.
- **Shrunk_Failure**: a Confirmed_Failure produced by the Shrinker with lower impulse than its
  starting point.
- **Budget**: the number of Evaluations a searcher may simulate; confirmation and shrinking
  rollouts are counted separately.
- **Discovery_Record**: `discovery.json`, the full record of one search.

## Requirements

### Requirement 1: Search space and quantization

1. THE Search_Space SHALL map every Candidate to a valid M1 `PushScenario` with direction
   `(cos θ, sin θ, 0)` in the initial-heading frame.
2. THE Search_Space SHALL quantize force to 0.1 N, direction to 0.1° in [0, 360), start and
   duration to whole 4 ms substeps, and clamp every value to its bounds.
3. THE CMA-ES decoder SHALL map the internal vector `(x_fx, x_fy, u_start, u_duration)` with
   `(fx, fy) = 150 N × (x_fx, x_fy)` projected radially onto the annulus 10 ≤ |f| ≤ 150 N,
   report `force_n = |f|` and `direction_deg = atan2(fy, fx)` mod 360, and map `u ∈ [0, 1]` to
   substeps by floor.
4. THE Random Search sampler SHALL draw force uniformly on [10, 150] N, direction uniformly on
   [0, 360), and start and duration substeps uniformly from their integer ranges.

### Requirement 2: Objective

1. THE objective (minimized) SHALL be `−2 + impulse/30` for `fell`, a robustness score in [0, 1]
   for `survived`, and `+2` for `numerical`.
2. THE robustness score SHALL be `min(tilt margin, height margin)` from the episode's minimum
   `up_z` and minimum trunk height relative to the 45° and 0.18 m fall thresholds, normalized to
   [0, 1].
3. THE objective SHALL order every `fell` below every `survived` below every `numerical`, and
   lower impulse below higher impulse among `fell`.
4. A `numerical` Evaluation SHALL never count as a failure, be selected as best, or be shrunk.

### Requirement 3: Searchers and budget

1. THE Random Search and CMA-ES searchers SHALL evaluate at most Budget Evaluations, with Budget a
   multiple of 8 in [8, 256].
2. CMA-ES SHALL use pycma with population 8, initial mean `(0, 0, 0.5, 0.5)`, σ₀ = 0.4, box bounds
   `[-1, 1]² × [0, 1]²`, and a private NumPy generator seeded from the search seed.
3. THE searchers SHALL count every simulated Evaluation, including numerical ones, and SHALL not
   re-simulate a quantized Candidate already evaluated in the same search (recorded as a cache
   hit, not counted against Budget).
4. THE searchers SHALL start without prior failure information; warm starts are not implemented.
5. THE engine SHALL stop on Budget exhaustion, cancellation, a wall-clock timeout or the CMA-ES
   iteration cap, and SHALL record the reason.

### Requirement 4: Confirmation and shrinking

1. WHEN a search ends with at least one Candidate_Failure, THE engine SHALL confirm the
   lowest-impulse one with `run_trial`, and if that fails, the next lowest, up to 3 attempts.
2. THE Shrinker SHALL start from the Confirmed_Failure with direction and start fixed and test,
   in ascending impulse order, a coarse 5 × 5 force-duration lattice below the current impulse,
   then a finer lattice next to the best point, then (one lower force step) neighbouring starts
   ±1, ±2 substeps and directions ±5°.
3. THE Shrinker SHALL replace the best failure only with a newly Confirmed_Failure of lower
   impulse, stop each phase at its first confirmed improvement, and respect its own rollout budget.
4. THE Shrinker SHALL record every tested point, the surviving points with lower impulse than the
   result, and the untested regions, and SHALL state that the result is not shown to be minimal.

### Requirement 5: Reproducibility and records

1. THE Discovery_Record SHALL contain the algorithm, seed, Budget, bounds, quantization, internal
   parameterization, pycma version and options, every Evaluation in order (candidate, impulse,
   outcome, reason, objective, stance at the shove, digest), confirmation and shrinking records,
   rollout accounting, provenance (policy, scene, versions, git) and a `results_sha256`.
2. WHEN the same configuration and seed run twice on the same build, THE engine SHALL reproduce the
   Evaluation sequence, results and `results_sha256`.
3. THE Discovery_Record SHALL export the final failure as an M2 suite config that `robustlab
   experiment` can rerun.

### Requirement 6: CLI and comparison

1. THE Robustlab_CLI SHALL provide `robustlab discover --algorithm {random,cmaes} --budget N --seed
   S --shrink-budget M --out DIR`.
2. `scripts/compare_search.py` SHALL run both searchers on matched seeds and budgets and report
   per-seed results, found counts, evaluations to first failure, lowest impulses, wall time and a
   paired comparison.

### Requirement 7: Web integration

1. THE Backend SHALL accept `POST /api/discovery/runs` with algorithm, Budget, optional seed and
   shrink budget, validate them, and run the search on the experiment worker.
2. THE Backend SHALL allow one expensive job (live stress run or discovery) at a time and answer
   409 otherwise.
3. THE Backend SHALL support cancellation and a server-side wall-clock limit.
4. THE Viewer SHALL add a Failure Discovery tab with algorithm, Budget, seed and shrink-budget
   inputs, live progress and best confirmed impulse, distinct marks for candidate, confirmed and
   shrunk failures, an explicit "no failure found" outcome, the committed matched comparison
   labelled precomputed, and "Replay Discovered Failure" in the existing viewer.
5. WHEN a discovered failure is replayed, THE Backend SHALL re-simulate it with `run_trial` and
   report whether the shoved digest matches the record.

### Requirement 8: Tests

1. Hypothesis tests SHALL cover parameter bounds and angular wrapping, time quantization,
   determinism, budget accounting, fitness ordering, numerical-failure exclusion, shrinking never
   accepting an unconfirmed candidate, and the exported scenario matching the discovered one.
2. Integration tests SHALL re-simulate a discovered failure and compare digests.
3. THE Kiro Stop hook's fast suite SHALL keep finishing within its 30 s timeout.
