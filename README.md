# Robustness Studio

A CPU-first simulation robustness prototype for legged robots. Milestone M0 runs an existing
pretrained Go1 walking policy in native MuJoCo and records a trajectory with a run manifest.

Status: M0–M2 approved; M3 (web app with 3D replays) implemented locally, awaiting review, not
deployed. Every command below was run, and its observed output is in `docs/milestones/M0.md` to
`M3.md`.
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

Two presets: **Hidden Phase Vulnerability** shows the precomputed M2 sweep (11/256 falls at
100 N) and re-simulates any timing for a 3D replay. **Live Stress Test** runs real 16-trial
experiments at 150 N on request and replays each trial. The browser only plays back poses computed
by MuJoCo on the server.

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
- `docs/decision-log/` — architecture decision records

## License

MIT, see `LICENSE`. Third-party files and their licenses are listed in `THIRD_PARTY_NOTICES.md`.
