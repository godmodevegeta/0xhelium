# Kiro build log

Maps Kiro spec tasks to observed commands and results. Proposed work is not listed here until it has
been run.

## M0 Task 1 — Upstream compatibility spike (go/no-go)

Spec: `.kiro/specs/m0-walking-go1/` (tasks 1.1–1.3). Time-box: 30 min; used ≈9 min.

**Source.** weeble commit `60878341072256ccc68f6118ec23a298c55e1bb9`, resolved with `git ls-remote`
in 0.56 s. Shallow clone of that commit into `$TMPDIR`, outside the workspace. Nothing from upstream
was copied into this repository.

**Install.**

- `uv sync --frozen --only-group eval` failed after 123.89 s. uv could not download its managed
  CPython 3.12.10 from GitHub release assets (timeout). curl measured about 2.4 KB/s from
  `release-assets.githubusercontent.com`; PyPI downloads were fast.
- Workaround, succeeded in 8.33 s:
  `UV_PYTHON_DOWNLOADS=never uv sync --frozen --only-group eval --python /opt/homebrew/Caskroom/miniconda/base/bin/python3.12`
  (Python 3.12.2).
- Installed: mujoco 3.15.0, numpy 2.5.4, pytest 9.1.1. No jax, torch or mjx.

**Rollout.** Upstream `eval/sim.py`, policy `go1-walk-s0`, command (1.0, 0, 0), 500 control ticks
(10 s). Command: `PYTHONPATH=. uv run --no-sync python $TMPDIR/m0_spike_rollout.py`.

| Metric | Observed |
| --- | --- |
| Forward x displacement (qpos) | 9.263 m |
| Lateral displacement | 0.264 m |
| Mean forward speed, t = 2–10 s | 0.943 m/s |
| Max trunk tilt | 3.90° at t = 0.12 s |
| Min trunk z | 0.278 m (initial pose) |
| Final trunk z | 0.310 m |
| Non-finite steps | 0 |
| MuJoCo warnings | 0 |
| Rollout wall time | 0.070 s (0.14 ms per control tick) |

**Decision: GO** with weeble `go1-walk-s0`. The fallback (Playground rss2025 `go1_policy.onnx`) is
not needed.

These numbers come from upstream's runner, not from this project's implementation. The acceptance
criteria in `requirements.md` are evaluated later by our own code (Tasks 5–7).

**Carry-forward risk.** uv-managed Python downloads are unreliable on this network. Task 2.2 should
use the local Python 3.12.2 (`--python /opt/homebrew/Caskroom/miniconda/base/bin/python3.12` with
`UV_PYTHON_DOWNLOADS=never`) and document this in the README.
