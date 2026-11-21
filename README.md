# Robustness Studio

A CPU-first simulation robustness prototype for legged robots. Milestone M0 runs an existing
pretrained Go1 walking policy in native MuJoCo and records a trajectory with a run manifest.

Status: M0 in progress. The commands below are the planned interface and are finalized in M0
Task 7.3. A command is not claimed to work until `docs/milestones/M0.md` records its observed output.

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
    --out runs/m0/forward
```

Outputs are written under `runs/`, which is not tracked by git.

## Documentation

- `.kiro/steering/` — product, technical, structure and delivery steering files
- `.kiro/specs/m0-walking-go1/` — M0 requirements, design and tasks
- `docs/kiro_build_log.md` — Kiro tasks mapped to observed commands and results

## License

MIT, see `LICENSE`. Third-party files and their licenses are listed in `THIRD_PARTY_NOTICES.md`.
