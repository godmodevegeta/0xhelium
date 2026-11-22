# Robustness Studio

A CPU-first simulation robustness prototype for legged robots. Milestone M0 runs an existing
pretrained Go1 walking policy in native MuJoCo and records a trajectory with a run manifest.

Status: M0–M6.1 approved; M7 (pre-registered regression gate and policy registry) evaluated once
locally, awaiting review, not deployed. Every command below was run, and its observed output is in
`docs/milestones/M0.md` to `M7.md`.
The pretrained policy (`go1-walk-s0` from [weeble](https://github.com/mottopanikeiku/weeble)) passed
the fixed walking criteria: a 10 s forward run at 0.944 m/s mean over 2–10 s, and a zero-command run
that stayed within 0.016 m. A 100 N shove toward its right for 0.1 s, started at each of the 256
4 ms steps in [2.000, 3.024) s, made it fall in 11 of 256 cases (4.3%). Results come from a
flat-ground simulation only; no general robustness or real-robot claim is made.

## Setup

Requirements: macOS or Linux, Python 3.12, [uv](https://docs.astral.sh/uv/). No GPU or CUDA.

```bash
uv sync
```

On networks where uv cannot download its managed Python (on the development machine the GitHub
release-asset download timed out), use a local Python 3.12 instead:

```bash
UV_PYTHON_DOWNLOADS=never uv sync --python /opt/homebrew/Caskroom/miniconda/base/bin/python3.12
```

Replace the path with any local Python 3.12 interpreter. See `docs/kiro_build_log.md` for details.

## Test

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                  # fast unit and property tests
uv run pytest -q -m integration   # native MuJoCo closed-loop tests
uv run pytest -q -m smoke         # the Stop hook's ~4 s contract subset
scripts/milestone_check.sh        # mandatory before a milestone review: all of the above + web build
```

## Run

```bash
uv run robustlab doctor           # print Python, NumPy and MuJoCo versions
uv run robustlab verify-assets    # check pinned asset hashes
uv run robustlab walk --policy policies/go1-walk-s0.json --command 1,0,0 --seconds 10 \
    --out runs/m0/forward          # forward walk, 10 s
uv run robustlab walk --policy policies/go1-walk-s0.json --command 0,0,0 --seconds 10 \
    --out runs/m0/zero             # zero command, stand in place
uv run robustlab walk --policy policies/go1-walk-s0.json --command 1,0,0 --seconds 51 \
    --out runs/m0/latency          # latency benchmark: 50 warm-up + 2,500 measured ticks
```

M1 experiments: normal walking, and the same walk with a sideways shove on the trunk.

```bash
uv run robustlab walk --policy policies/go1-walk-s0.json --command 1,0,0 --seconds 5 \
    --out runs/m1/nominal
uv run robustlab walk --policy policies/go1-walk-s0.json --command 1,0,0 --seconds 5 \
    --out runs/m1/shove-right-100N \
    --push-force 100 --push-direction right --push-start 2.0 --push-duration 0.1
```

The push flags go together: force in N (0–150), direction in the robot's initial-heading frame
(`forward`, `backward`, `left`, `right` or `x,y,z`; the name is where the force points, so `right`
is a shove from the robot's left), start and duration in seconds (multiples of 0.004 s, duration
≤ 0.5 s). `--scene fall-contact` (default) lets the trunk, hips, thighs and knees touch the floor;
`--scene feet-only` runs the pinned M0 `walk.xml` unchanged. The summary prints whether the robot
fell, the first reason (`nonfoot_contact`, `tilt`, `low_height`) and its time, and reports
numerical failures (`nonfinite_state`, `mujoco_warning`) separately.

M2 experiment suite: paired nominal and shoved runs, varying only the shove start time.

```bash
uv run robustlab experiment --config examples/m2/sideways-shove-stratified-16.json \
    --out runs/m2/stratified-16                  # 16 stratified random start times
uv run robustlab experiment --config examples/m2/sideways-shove-grid-all-256.json \
    --out runs/m2/grid-all-256                   # every 4 ms start time in the interval
```

`experiment` writes `experiment.json` (every trial's timing, outcome, fall reason and time, paired
difference, aggregates with denominators, provenance, and what the numbers do and do not mean) and,
for up to 32 trials, each trial's nominal and shoved trajectories in the `walk` format. Options:
`--trials`, `--seed`, `--no-save-trajectories`, `--overwrite`, `--dry-run`.

M3 web app (needs Node 20+ for the build):

```bash
uv sync --extra server
(cd web && npm ci && npm run build)
uv run robustlab serve                           # http://127.0.0.1:8000
```

Two presets (since M6.1 they appear in the **Results** and **Stress test** modes): **Hidden Phase Vulnerability** shows the precomputed M2 sweep (11/256 falls at
100 N) and re-simulates any timing for a 3D replay. **Live Stress Test** runs real 16-trial
experiments at 150 N on request and replays each trial. The browser only plays back poses computed
by MuJoCo on the server.

M4 failure discovery (pycma is in the `server` and `search` extras):

```bash
uv run robustlab discover --algorithm cmaes --budget 64 --seed 0 --shrink-budget 32 \
    --out runs/m4/cmaes-s0                       # or --algorithm random
uv run python scripts/compare_search.py --seeds 0-19 --budget 64 --shrink-budget 32
```

The search looks for the lowest-impulse shove (force 10–150 N, any horizontal direction, start
2.000–3.020 s, duration 40–200 ms) that makes the robot fall. Each failure is confirmed with a
paired re-simulation, then shrunk locally. The web app's **Discover failure** mode runs the same
search with live progress and replays the result next to the nominal robot.

M5 multi-axis scenarios (floor friction, a payload box welded to the trunk, and the M1 push):

```bash
uv run robustlab scenario --file examples/m5/icy-heavy-shove.json --out runs/m5/x
uv run robustlab scenario --file examples/m5/icy-heavy-shove.json --out runs/m5/x-suite \
    --suite --trials 16 --seed 7                 # varies only the push start
```

A single run records the scenario, a matched baseline (same friction and payload, push removed)
and the original trained condition. Values outside the bounds (μ 0.05–1.5, payload 0–5 kg, push
0–150 N) are rejected, never clamped. The web app's **Design scenario** mode edits, validates,
runs and replays scenarios. With `OPENROUTER_API_KEY` set on the server (optional
`OPENROUTER_MODEL`, default `openai/gpt-4o-mini`), it can also turn a description into proposed
values, which go through the same validator. Without a key everything else still works.

M6 candidate hardening: a bounded 10-parameter residual reflex around the frozen policy, tuned by
seeded CMA-ES on CPU MuJoCo (not PPO, not network fine-tuning). The result is a candidate record;
`harden` never promotes anything.

```bash
uv run robustlab harden --seed 0 --out runs/m6/seed0            # about 20 s on 8 processes
uv run robustlab harden --seed 0 --focus m2-100N-500 --out runs/m6/focus
uv run python scripts/hardening_grids.py runs/m6/seed0/candidate.json   # exploratory grids
```

`harden` writes an immutable `candidate.json`. It contains the base-policy hash, the reflex
parameters, the calibration, the training/validation-preview sets (the locked M7 set is never
evaluated), the optimizer settings and seed, rollout counts, timings, the stop reason, and baseline
vs candidate measurements per episode. Options: `--generations`, `--popsize`, `--max-rollouts`,
`--wall-clock`, `--workers` (default: CPU count minus 2, at most 8), `--focus`.

The web app (dark theme since M6.1) has five modes: **Design scenario**, **Stress test**,
**Discover failure**, **Harden controller** and **Results**. Each shows its compact controls and
one primary action on the left, and the shared 3D replay on the right. The replay shows in-scene
robot labels, a HUD (shove, friction, payload) with the source of the replay, and a transport with
the shove interval and fall markers. Detailed evidence is in expandable sections below. The app
opens on a labelled sample replay. The **Harden controller** mode runs the same job, shows CPU
optimization progress, and replays the frozen baseline and the candidate on identical shoves.
`uv run python scripts/ui_checks.py` checks the layout and the main flows against a running server
at 1440×900, 1280×800, 1024×768 and 390×844 (add `--nokey-base` for the no-OpenRouter fallback).

M7 regression gate: `examples/m7/gate-v1.json` was committed before evaluation. It pins the five
M6 candidates, the suites (10 locked attacks, 13 nominal walks, a 1280-episode regression gauntlet,
34 known failures), the criteria G1–G4 and the one eligible primary candidate. It was evaluated
once.

```bash
uv run robustlab gate             # refuses: gate-v1 already has a result (presets/m7-gate-v1/)
uv run robustlab gate --verify    # recomputes all 8022 rollouts (~100 s on 8 processes) and compares
```

Result: the primary `cand-ea7b9dbc0bbd` passed G1–G4, with 16 individual low-friction gauntlet
regressions listed in `M7.md`, and was promoted. `policies/registry/active.json` points at
`v1-cand-ea7b9dbc0bbd`: the frozen policy plus that reflex. Only the gate writes the registry. In
the web app, **Results** shows the gate evidence and replays any gate episode, and **Harden
controller** shows the gate verdict for the candidate shown. Stress, Discover and Design still
evaluate the frozen baseline v0.

`walk` checks asset hashes first, prints a summary table and writes `trajectory.json` (50 Hz) and
`manifest.json` into the `--out` directory. It refuses to replace an existing run unless
`--overwrite` is given. Commands are body-frame `vx,vy,yaw` (m/s, m/s, rad/s) within the policy's
trained ranges; write negative values as `--command=-0.5,0,0`.

Outputs are written under `runs/`, which is not tracked by git. Pinned upstream files are already
committed; `scripts/fetch_assets.py` re-downloads them from the pinned weeble commit if needed.

## Documentation

- `.kiro/steering/` — product, technical, structure and delivery steering files
- `.kiro/specs/m0-walking-go1/` — M0 requirements, design and tasks
- `docs/kiro_build_log.md` — Kiro tasks mapped to observed commands and results
- `docs/milestones/M0.md` — M0 evidence: hashes, criteria, measured values, parity, latency,
  limitations
- `.kiro/specs/fall-detection/` — M1 requirements, design and tasks
- `docs/milestones/M1.md` — M1 evidence: shove and fall results, scene comparison with M0, tests
- `.kiro/specs/experiment-runner/` — M2 requirements, design and tasks
- `docs/milestones/M2.md` — M2 evidence: paired suites, fall frequencies, what they establish
- `.kiro/specs/web-replay/` — M3 requirements, design and tasks
- `docs/milestones/M3.md` — M3 evidence: force sweep, presets, live runs, replay, endpoint timings
- `.kiro/specs/m4-failure-discovery/` — M4 requirements, design and tasks
- `docs/milestones/M4.md` — M4 evidence: Random Search vs CMA-ES, shrinking, web runs
- `.kiro/specs/m5-openrouter-scenarios/` — M5 requirements, design and tasks
- `docs/milestones/M5.md` — M5 evidence: friction and payload measurements, OpenRouter prompt
  results, Scenario Designer runs
- `.kiro/specs/m6-cpu-hardening/` — M6 requirements, design and tasks
- `docs/milestones/M6.md` — M6 evidence: reflex design, calibration, data sets, five-seed results,
  grids, timings, UI checks and screenshots
- `.kiro/specs/m6-ui-redesign/` and `docs/milestones/M6_UI.md` — M6.1 simulation-first UI: audit,
  design, before/after screenshots, automated UI checks
- `.kiro/specs/m7-regression-gate/` — M7 requirements, design and tasks
- `docs/milestones/M7.md` — M7 evidence: pre-registration, suites, criteria, all five candidates,
  individual regressions, registry, API/UI, limitations
- `docs/decision-log/` — architecture decision records

## License

MIT, see `LICENSE`. Third-party files and their licenses are listed in `THIRD_PARTY_NOTICES.md`.
