# Requirements Document

## Introduction

Milestone M2 of `delivery.md`: turn the single deterministic M1 shove into a reproducible experiment
suite that measures how often the Go1 falls across different shove timings. The M1 runner, fall
monitor, push implementation and scene are reused unchanged; there is one physics implementation.

The runner is deterministic, so running the same episode with different seed labels would give the
same trajectory every time. M2 therefore varies one physical quantity: the shove start time, drawn
from a documented interval that spans more than two gait cycles (the nominal gait period is 0.44 s).
Force, direction, duration, command, policy, scene, horizon and initial state stay fixed. Start times
are derived from the experiment seed and quantized to the 4 ms physics step.

Out of scope: web UI, LLM providers, CMA-ES or other search, PPO, reflex adapters, friction, payload,
other attack dimensions.

## Glossary

- **Suite_Config**: a JSON experiment definition validated by `robustlab.experiments.suite`.
- **Trial**: one pair of episodes, nominal and attacked, sharing initial state, command and horizon.
- **Start_Interval**: the half-open interval `[low_s, high_s)` of shove start times, on the 4 ms
  substep grid. Its substeps are `L .. H-1` with `L = low_s/0.004`, `H = high_s/0.004`, `W = H - L`.
- **Stratum**: for `n` trials, trial `i` owns substeps `[L + i·W/n, L + (i+1)·W/n)`.
- **Stratified_Sampling**: random method; trial `i` starts at one substep drawn uniformly within its
  stratum from the seed. Requires `W` divisible by `n` so all strata are equal.
- **Grid_Sweep**: deterministic method; trial `i` starts at the first substep of its stratum. The
  seed is not used. With `n = W` it enumerates every start substep (exhaustive).
- **Test_Distribution**: start time uniform over the `W` substeps of the Start_Interval, every other
  parameter fixed. Fall-rate estimates refer only to this distribution.
- **Outcome**: per episode, exactly one of `survived`, `fell` (the FallResult primary event is
  physical) or `numerical` (the primary event is numerical).
- **Experiment_Report**: `experiment.json` in the experiment run directory.

## Requirements

### Requirement 1: Typed Suite_Config

**User Story:** As the developer, I want an experiment defined in one validated file, so that it can
be rerun exactly.

#### Acceptance Criteria

1. THE Suite_Config SHALL contain `schema_version`, `name`, `policy`, `scene`, `command`,
   `horizon_s`, `trials`, `seed` and `attack` with `kind: "push"`, `force_n`, `direction`,
   `duration_s` and `start` (`method`, `low_s`, `high_s`).
2. IF the Suite_Config has an unknown or missing key, a wrong type, an unknown method or scene, or a
   value out of range, THEN THE loader SHALL reject it with an error naming the field.
3. THE loader SHALL validate force, direction and duration with the M1 `PushScenario` rules, require
   `low_s` and `high_s` to be multiples of 0.004 s with `0 <= low_s < high_s`, require
   `1 <= trials <= 1024` and `W >= trials`, and require `W % trials == 0` for Stratified_Sampling.
4. IF the latest possible shove ends after `horizon_s`, THEN THE loader SHALL reject the config.
5. IF the command is outside the policy's trained range, THEN THE suite SHALL reject it before
   simulating.
6. THE Robustlab_CLI SHALL allow `--trials` and `--seed` to override the file, and SHALL record the
   effective config.

### Requirement 2: Shove timing derivation

**User Story:** As the developer, I want trial timings that differ and are reproducible, so that
trials are not identical simulations under different labels.

#### Acceptance Criteria

1. THE sampler SHALL return integer start substeps; each start time SHALL equal substep × 0.004 s.
2. Every start substep SHALL lie in its trial's Stratum, so all trials have distinct start times.
3. FOR Stratified_Sampling, THE sampler SHALL take 32-bit words from
   `numpy.random.SeedSequence(seed).generate_state(trials, uint32)` and set the offset within
   stratum `i` to `(word_i × size) >> 32`.
4. WHEN the same seed and interval are used, THE sampler SHALL return the same start substeps.
5. FOR Grid_Sweep, THE sampler SHALL ignore the seed.
6. THE Experiment_Report SHALL record the method, seed, generator description, interval, strata and
   each trial's actual start substep and time.

### Requirement 3: Paired trials

**User Story:** As the reviewer, I want each attacked run compared against a nominal run that
differs only by the shove, so that differences are attributable to the shove.

#### Acceptance Criteria

1. FOR every Trial, THE suite SHALL run a nominal episode (no push) and an attacked episode (push at
   the trial's start) with the same reset state, command, horizon, scene and policy, through the M1
   `EpisodeRunner`.
2. FOR every Trial, THE suite SHALL compare the two trajectories at every Tick whose time is at or
   before the shove start and record the maximum absolute difference of qpos, qvel and action.
3. IF that difference is non-zero, THEN THE suite SHALL mark the Trial `pairing_ok: false` and the
   CLI SHALL exit non-zero.
4. THE suite SHALL record a digest of each trajectory and report how many distinct nominal
   trajectories occurred.

### Requirement 4: Measurements and aggregation

**User Story:** As the reviewer, I want fall counts with denominators and times, so that the result
is a measured frequency, not an impression.

#### Acceptance Criteria

1. FOR every episode, THE suite SHALL record the Outcome, the primary reason, the first fall time,
   the time from shove start to fall, the first non-foot contact body, and forward progress up to
   the fall (M1 `evaluate`).
2. FOR each case, THE aggregation SHALL report `total_trials`, `survived`, `physical_falls`,
   `numerical_failures`, `valid_trials = total_trials − numerical_failures`, and
   `fall_rate = physical_falls / valid_trials` (null when `valid_trials` is 0).
3. THE aggregation SHALL satisfy `physical_falls + survived + numerical_failures == total_trials` and
   `0 <= fall_rate <= 1`.
4. THE aggregation SHALL NOT count a numerical failure as a fall or as a survival.
5. THE aggregation SHALL report fall times and time-after-shove (min, median, max), mean forward
   progress per case, the mean paired progress difference over pairs where both episodes are
   valid, and a 3×3 table of (nominal Outcome, attacked Outcome) counts.
6. WHERE the method is Stratified_Sampling, THE aggregation SHALL report a Wilson 95% interval for
   the attacked fall rate, labelled as treating trials as independent draws. WHERE the method is
   Grid_Sweep, THE report SHALL give no interval.

### Requirement 5: Interpretation labels

**User Story:** As the reviewer, I want the report to say exactly what the number means, so that it
is not read as a general robustness claim.

#### Acceptance Criteria

1. THE Experiment_Report SHALL describe the attacked rate as a fall-rate estimate over the
   Test_Distribution for Stratified_Sampling, and as a failure frequency over the grid for
   Grid_Sweep (an exact frequency over the Test_Distribution when the grid is exhaustive).
2. THE Experiment_Report SHALL state that nominal trials are repeated identical simulations when
   they produce one distinct trajectory, and SHALL NOT present them as independent observations.
3. THE Experiment_Report SHALL list the parameters held fixed.

### Requirement 6: Persistence

**User Story:** As the developer, I want every trial stored with its provenance, so that M3 can
replay any trial and the result can be audited.

#### Acceptance Criteria

1. THE suite SHALL write `experiment.json` last, atomically, under the `--out` directory, containing
   the effective config and its SHA-256, the policy identity, scene description, asset lock commit,
   Python/NumPy/MuJoCo versions, platform, git revision, sampling record, every trial's parameters
   and results, the aggregates and the interpretation.
2. THE suite SHALL persist per-trial trajectories in the existing M1 format (`--save-trajectories`,
   enabled by default for at most 32 trials) under `trials/<NN>/{nominal,attacked}/` so M3's web
   viewer can replay any passing or failing trial directly.
3. THE suite SHALL refuse to replace an existing `experiment.json` unless `--overwrite` is given.
4. THE `read_run` function SHALL keep reading M0 (schema 1) and M1 (schema 2) run directories.

### Requirement 7: Determinism and CLI

**User Story:** As the developer, I want one command and a reproducibility check, so that the result
can be regenerated and compared.

#### Acceptance Criteria

1. THE Robustlab_CLI SHALL provide `robustlab experiment --config FILE --out DIR` with `--trials`,
   `--seed`, `--save-trajectories/--no-save-trajectories`, `--overwrite` and `--dry-run` (print the
   sampled start times and rollout count without simulating).
2. THE Experiment_Report SHALL contain a `results_sha256` computed over the canonical JSON of the
   sampling record, trials and aggregates, excluding wall-clock times and paths.
3. WHEN the same effective config runs twice on the same build and machine, THE suite SHALL produce
   the same start times, outcomes and `results_sha256`.
4. THE CLI SHALL print a per-trial table (start time, stance feet at the shove, nominal and attacked
   outcomes, fall reason and time, progress) and the case aggregates with their denominators.

### Requirement 8: Tests

**User Story:** As the reviewer, I want the statistics and pairing tested, so that the report can be
trusted.

#### Acceptance Criteria

1. THE test suite SHALL include Hypothesis property-based tests verifying aggregation invariants
   (`physical_falls + survived + numerical_failures == total_trials`, `0.0 <= fall_rate <= 1.0`),
   4 ms timestep quantization, and seed determinism.
2. THE test suite SHALL cover paired construction, pre-attack trajectory parity, invalid configs,
   distinct trial timings, rerun determinism and M0/M1 trajectory compatibility.
3. THE Repository SHALL contain no web UI, LLM, optimizer, PPO, adapter, friction or payload code.
