# Requirements: M7 — Independent regression gate and policy promotion

M7 evaluates the M6 CPU-reflex candidates **once** against data that M6 never used for tuning (the
locked test set plus gate-defined nominal and regression suites), under acceptance criteria that
are fixed and committed before any evaluation runs. A candidate is promoted only if every
criterion passes; otherwise the frozen `go1-walk-s0` baseline stays active and the rejection is
recorded. The gate is not tuned on its own results, and no candidate is retuned after it.

Glossary: *baseline* = frozen go1-walk-s0 actor with zero residual; *candidate* = baseline + M6
reflex parameters; *episode* = one deterministic CPU MuJoCo rollout (fall-contact scene, nominal
reset). Episodes are deterministic, so "paired" means the same episode under both controllers.

## R1 — Pre-registration
1.1 THE gate configuration (`examples/m7/gate-v1.json`) SHALL fix, before evaluation: the candidate
registry (ids and record SHA-256), the primary candidate and the rule that chose it, every episode
in every suite, the acceptance criteria with thresholds, and the statistical test.
1.2 THE primary candidate SHALL be chosen from M6 evidence only (training and validation preview):
rank by (no validation-preview nominal fall, validation-preview attacks survived, lower training
objective); only the primary candidate is eligible for promotion. Other registered candidates are
evaluated and reported, so the holdout is not used to pick a winner.
1.3 THE gate SHALL refuse to run if the config, a candidate record, the calibration, the base policy
or the locked set do not match their recorded hashes, or if an evaluation of this gate version
already exists (no re-evaluation; a new gate version needs a new locked set).

## R2 — Suites (all episodes run for the baseline and every candidate)
2.1 *Locked holdout*: the 13 episodes of the M6 locked test set (10 attacks, 3 nominal), sha
`599888a34071…`, evaluated for the first time in M7.
2.2 *Nominal walking*: the 3 locked nominal episodes plus gate-defined nominal conditions not used
in M6 tuning (10 s forward and zero command as in M0, sidestep, backward, μ 0.1/0.2/1.5, 5 kg
forward/left payload) and the validation-preview μ 0.3 + 3 kg condition.
2.3 *Regression gauntlet*: every 4 ms shove start in [2.000, 3.024) s (256) for five recorded
conditions: 100 N (M2), 150 N (M3), μ 0.3 at 100 N, 3 kg at 100 N, μ 0.3 + 3 kg at 100 N (M5);
1280 episodes, toward the robot's right for 0.1 s. Includes M6 training and validation timings;
used for regression detection, not as independent evidence of improvement.
2.4 *Known failures*: the 34 confirmed M4 discovery finals (any force/direction/timing).

## R3 — Acceptance criteria (all SHALL pass)
3.1 G1 nominal: candidate physical falls = 0 and numerical failures = 0 on every nominal episode,
and for each the final horizontal displacement in the initial-heading frame is within
max(0.10 m, 10%) of the baseline's.
3.2 G2 holdout: on the 10 locked attacks, candidate falls < baseline falls, and candidate fall
rate ≤ max(0.50, baseline rate − 0.25), and the exact one-sided McNemar test on paired outcomes
gives p ≤ 0.05.
3.3 G3 regression: on the gauntlet, the candidate fall rate is ≤ baseline + 0.03 (absolute)
overall and in each of the five conditions; every individual regression (baseline survived,
candidate fell) is listed.
3.4 G4 numerical: zero numerical failures for the candidate in any suite.
3.5 Known failures are reported (recovered / still failing) and are not a pass criterion, because
they include M6 training data.

## R4 — Decision and registry
4.1 IF the primary candidate passes every criterion, THE system SHALL write an immutable promoted
version manifest and atomically move `policies/registry/active.json` to it; otherwise it SHALL
leave the active pointer on the baseline and write a rejection record with the failed criteria.
4.2 THE gate result SHALL record per-episode outcomes for every controller, aggregates with
denominators and Wilson 95% intervals, individual regressions, the criteria with measured values,
the decision, rollout counts, timing and provenance.
4.3 No code path SHALL promote a candidate other than the gate.

## R5 — Application
5.1 THE API SHALL expose the gate result and active pointer, and a re-simulated replay of any gate
episode (including locked ones, now opened) for baseline vs a registered candidate.
5.2 THE Harden and Results modes SHALL show the gate verdict (promoted / rejected, failed criteria)
instead of "pending M7" once the gate has run; Stress test, Discover failure and Design scenario
continue to evaluate the frozen baseline and say so.
