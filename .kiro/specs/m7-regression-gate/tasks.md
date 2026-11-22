# Implementation Plan: M7 — Independent regression gate and policy promotion

- [x] 1. Preconditions: UI fixes verified; tag `m6-ui-baseline`; merge into `main`; branch `m7-regression-gate`
- [ ] 2. Pre-registration
  - [ ] 2.1 Commit the five M6 candidate records as a hashed preset
  - [ ] 2.2 `scripts/make_gate_config.py` → `examples/m7/gate-v1.json` (suites, criteria, primary rule); commit before any evaluation
- [ ] 3. Gate code: config verification, suite runner, criteria, McNemar/Wilson, registry, CLI; unit tests with a fake evaluator (no locked-set rollouts in tests)
- [ ] 4. Run the gate once; commit the result preset and registry state
- [ ] 5. API and UI: gate result, verdict in Harden/Results, gate replays; tests and browser check
- [ ] 6. `docs/milestones/M7.md`, README, build log; milestone check; stop for review
