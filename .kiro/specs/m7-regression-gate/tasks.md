# Implementation Plan: M7 — Independent regression gate and policy promotion

- [x] 1. Preconditions: UI fixes verified; tag `m6-ui-baseline`; merge into `main`; branch `m7-regression-gate`
- [x] 2. Pre-registration
  - [x] 2.1 Commit the five M6 candidate records as a hashed preset
  - [x] 2.2 `scripts/make_gate_config.py` → `examples/m7/gate-v1.json` (suites, criteria, primary rule); commit before any evaluation
- [x] 3. Gate code: config verification, suite runner, criteria, McNemar/Wilson, registry, CLI; unit tests with a fake evaluator (no locked-set rollouts in tests)
- [x] 4. Run the gate once; commit the result preset and registry state
- [x] 5. API and UI: gate result, verdict in Harden/Results, gate replays; tests and browser check
- [x] 6. `docs/milestones/M7.md`, README, build log; milestone check; stop for review
