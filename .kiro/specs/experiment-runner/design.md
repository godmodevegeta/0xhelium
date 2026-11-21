# Design Document: M2 — Reproducible experiment runner

## Overview

```text
Suite_Config (JSON) --load/validate--> SuiteConfig
        |                                  |
        | sample_start_steps(seed, interval, n, method)
        v                                  v
   start substeps k_0..k_{n-1}      Stack (scene, policy, actor)  <- shared with `walk`
        |                                  |
        +--> for each trial i: EpisodeRunner.run(nominal), EpisodeRunner.run(push at k_i)
                 evaluate() both, pre-attack parity, digests, optional write_run() x2
        |
   aggregate(trials) -> aggregates; interpretation(); results_sha256
        |
   experiment.json (written last)
```

All physics goes through the M1 `EpisodeRunner`, `FallMonitor`, `PushScenario` and `build_scene`.

## Why shove timing, and the interval chosen

The simulation is deterministic: no observation noise, no randomized reset. Re-running one episode
with a new seed label yields the same trajectory, so "16 trials" of it would be one observation
counted 16 times. The robot's state when the shove arrives depends on the gait phase, so the start
time is a physically meaningful variable. Measured nominal gait period at command (1, 0, 0): 0.44 s
for every foot (touchdown-to-touchdown, 2–50 s of `runs/m1/latency`).

Default interval: `[2.000, 3.024)` s = substeps 500..755, `W = 256` = 1.024 s ≈ 2.3 gait cycles,
divisible by 16 (and 32, 64, 128, 256). The latest shove ends at 3.020 + 0.1 = 3.120 s, leaving
1.88 s of the 5 s horizon to observe a fall. Falls later than the horizon are not counted.

## Sampling

```text
L = low_s/dt, H = high_s/dt, W = H - L, n trials
stratum i = [L + floor(i*W/n), L + floor((i+1)*W/n))
grid:       k_i = stratum start
stratified: words = SeedSequence(seed).generate_state(n, uint32); k_i = start_i + (words[i]*size_i >> 32)
```

Integer arithmetic only. `SeedSequence.generate_state` is a fixed hashing algorithm, independent of
NumPy's Generator stream; NumPy's version is recorded regardless. Stratified sampling with equal
strata is unbiased for the Test_Distribution fall frequency, and its variance is no larger than
simple random sampling with the same n (proportional allocation). With `n = W` the grid is the
full enumeration.

## Components

### `experiments/stack.py`

`load_stack(root, policy_path, scene) -> Stack` (moved from `cli._load_stack`; `walk` uses it too).
`Stack` holds `scene_info`, `ids`, `policy`, `actor`, `load_s`, `lock`, `provenance`, `policy_info`
and `library_versions`, and `manifest_for(record, run_id, command, seconds, seed, outcome)` builds an
M1 manifest via `trace.build_manifest`.

### `experiments/suite.py`

- `SuiteConfigError(field, reason)`; `SuiteConfig`, `AttackConfig`, `StartConfig` frozen
  dataclasses; `load_suite_config(path_or_dict, overrides)` with strict key checking.
- `sample_start_steps(method, seed, low_step, high_step, n) -> list[(k, (lo, hi))]`.
- `classify(fall) -> "survived" | "fell" | "numerical"`.
- `aggregate(case_results, push_start_s)`, `wilson_interval(k, n)`, `paired_table(...)`.
- `run_suite(config, stack, out_dir, save_trajectories) -> report` and `write_report`.
- `results_digest(report)`: SHA-256 of canonical JSON of `sampling`, `trials` (paths removed) and
  `aggregates`.

### Trial record (in `experiment.json`)

```text
{trial, start_substep, start_s, stratum_substeps: [lo, hi), stance_at_start: {FR..RL: bool},
 nominal:  {outcome, reason, fall_time_s, forward_progress_m, first_nonfoot_body, digest},
 attacked: {... same ..., time_after_start_s, events: {reason: time_s}},
 paired:   {pre_attack_ticks, pre_attack_max_abs_diff, pairing_ok, progress_difference_m},
 trajectories: {nominal: "trials/00/nominal", attacked: "trials/00/attacked"} | null}
```

`stance_at_start` comes from the nominal run's foot-contact flags at the last Tick at or before the
shove start. Pre-attack parity covers Ticks with `time <= start_s` (the push first acts during the
substep starting at `start_s`, so these states cannot be affected). Digest = SHA-256 of qpos, qvel
and action bytes.

### CLI

`robustlab experiment --config F --out D [--trials N] [--seed S] [--save-trajectories |
--no-save-trajectories] [--overwrite] [--dry-run] [--root R]`. Exit 0 on success, 2 on invalid input,
3 on write errors, 4 if any trial fails pairing.

## Correctness properties

| # | Property (Hypothesis) | Requirements |
| --- | --- | --- |
| 17 | Aggregation: falls + survived + numerical == total; 0 ≤ rate ≤ 1; rate null iff valid = 0; Wilson bounds in [0, 1] and contain the rate; paired table sums to total | 4.2–4.6 |
| 18 | Quantization: every start is an integer substep in its stratum, start_s = k·0.004, all distinct, within [low, high) | 2.1, 2.2 |
| 19 | Seed determinism: same seed → same starts; grid ignores the seed; exhaustive grid = every substep | 2.3–2.5 |

Hypothesis runs derandomized with no example database, so CI results do not depend on local state.

## Error handling

Config and command errors exit 2 before any simulation. A numerical failure is an outcome, never an
exception. `experiment.json` is written last, so a directory without it is incomplete.
