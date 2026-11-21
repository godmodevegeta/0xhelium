# Robustness Studio

A CPU-first simulation robustness prototype for legged robots. Milestone M0 runs an existing
pretrained Go1 walking policy in native MuJoCo and records a trajectory with a run manifest.

Status: M0 complete, awaiting review. Every command below was run, and its observed output is in
`docs/milestones/M0.md`. The pretrained policy (`go1-walk-s0` from
[weeble](https://github.com/mottopanikeiku/weeble)) passed the fixed walking criteria: a 10 s forward
run at 0.944 m/s mean over 2–10 s, and a zero-command run that stayed within 0.016 m. Results come
from a flat-ground simulation only; no robustness or real-robot claim is made.

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
- `docs/decision-log/` — architecture decision records

## License

MIT, see `LICENSE`. Third-party files and their licenses are listed in `THIRD_PARTY_NOTICES.md`.
