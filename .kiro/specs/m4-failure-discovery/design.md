# Design Document: M4 — Failure discovery and shrinking

## Overview

```text
robustlab.search.space      Candidate, bounds, quantization, CMA decoder, random sampler
robustlab.search.objective  outcome -> objective (pure)
robustlab.search.engine     Evaluator (M1 runner), searchers, confirmation, record, digest
robustlab.search.shrink     coarse-to-fine shrinker (pure loop over injected evaluate/confirm)
robustlab.cli discover      CLI
scripts/compare_search.py   matched-seed comparison
robustlab.api               /api/discovery/* on the existing JobManager experiment worker
web/src/components/Discovery.tsx   new tab; reuses ReplayViewer
```

The searchers and shrinker take `evaluate(candidate) -> EvalResult` and `confirm(candidate) ->
Confirmation` callables, so the search logic is testable with fake physics. Production wiring
uses `EpisodeRunner.run` (attacked episode only) for Evaluations and `suite.run_trial` (paired
nominal + attacked) for confirmations and replays.

## Parameterization

External: `(force_n, direction_deg, start_substep, duration_substeps)`.

CMA-ES internal vector `x = (x_fx, x_fy, u_s, u_d)` in the box `[-1, 1]² × [0, 1]²` (pycma
`BoundTransform`):

```text
f = 150 · (x_fx, x_fy);  r = |f|
r < 10  -> f scaled to radius 10 (r = 0 -> direction 0°)
r > 150 -> f scaled to radius 150
force_n = round(r, 0.1 N) clamped [10, 150]; direction_deg = round(atan2(fy, fx) mod 360, 0.1°) mod 360
start_substep = 500 + min(255, floor(256 · u_s)); duration_substeps = 10 + min(40, floor(41 · u_d))
```

Cartesian force avoids the 0°/360° seam: nearby directions are nearby in `x`. The projection makes
the box corners (|f| up to 212 N) collapse onto the 150 N circle; the quantized values are what
gets simulated and recorded.

Random Search: `numpy.random.Generator(PCG64(SeedSequence(seed)))`; force `uniform(10, 150)`,
direction `uniform(0, 360)`, `integers(500, 756)`, `integers(10, 51)`, then the same quantization.
Its force prior is uniform in magnitude; CMA-ES's initial distribution is Gaussian around the
origin of the force plane (σ₀ = 0.4 → about 60 N), so the two priors differ and this is documented.

## Objective

```text
fell:      −2 + impulse / 30          ∈ [−1.987, −1.0]   (30 N·s = 150 N × 0.2 s)
survived:  min(m_tilt, m_height)      ∈ [0, 1]
           m_tilt   = (min up_z − cos 45°) / (1 − cos 45°)
           m_height = (min trunk z − 0.18) / (0.28 − 0.18)     (0.28 m ≈ nominal trunk height)
numerical: +2
```

## Engine

1. Reference nominal episode (1 rollout): its digest and its foot contacts give each candidate's
   stance at the shove.
2. Search loop until Budget, cancel, timeout (default 300 s) or, for CMA-ES, 4 × Budget / 8
   generations. Random asks one candidate at a time; CMA-ES asks 8, evaluates uncached ones, and
   tells the full generation (cache hits reuse their objective). A partially evaluated final
   generation is never told.
3. Confirmation of the lowest-impulse Candidate_Failure (≤ 3 attempts, 2 rollouts each).
4. Shrinking (own budget, default 32 Evaluations plus confirmation rollouts).
5. Final failure = shrunk if any improvement was confirmed, else the confirmed search failure.

Accounting: `search_rollouts` (≤ Budget), `cache_hits`, `reference_rollouts` (1),
`confirmation_rollouts`, `shrink_rollouts`, `shrink_confirmation_rollouts`, `total_rollouts`.

`results_sha256` = SHA-256 of canonical JSON of config, evaluations, confirmation, shrink and final
(wall times excluded).

## Shrinker

```text
best = confirmed start (F, θ, k, D)
phase coarse: forces linspace(10, F, 5), durations unique(round(linspace(10, D, 5)));
              points with impulse < best, ascending impulse; first `fell` -> confirm -> accept & stop phase
phase fine:   forces linspace(F_lo, F*, 4)[1:-1] ∪ {F*}, durations D_lo..D* (≤ 4 values),
              F_lo/D_lo = next lower coarse level (or the bound); same rule
phase release: one fine force step below F* (≥ 10 N) at starts k* ± {1, 2} and directions θ* ± 5°
              (8 points); same rule
```

Each phase stops at its first confirmed improvement because remaining points have higher impulse.
An unconfirmed `fell` is recorded as `unconfirmed` and the search continues. Recorded: every tested
point; survivors with impulse below the final failure; untested regions (between lattice levels;
other directions and starts; release phase tested only one force level); `budget_exhausted`; and
the statement "not shown to be minimal".

## API

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/api/discovery` | bounds, defaults, precomputed comparison, recent runs |
| POST | `/api/discovery/runs` | `{algorithm, budget, seed?, shrink_budget}` → 202 `{run_id}`; 409 busy; 422 invalid |
| GET | `/api/discovery/runs/{id}` | status, evaluations so far, best, final record when done |
| POST | `/api/discovery/runs/{id}/cancel` | sets the cancel flag |
| GET | `/api/discovery/runs/{id}/replay?which=final\|search` | `run_trial` re-simulation + digest check |

Run id `ds-YYYYMMDDTHHMMSSZ-<algorithm>-<seed>`; directory `runs/web/discovery/<id>/` with
`request.json` and `discovery.json` (written last).

## Viewer

Discovery tab: controls; a scatter of impulse against evaluation index (green survived, red fell,
purple numerical; ring = confirmed; diamond = shrunk); best-confirmed line; summary table; replay
buttons; comparison card from `presets/discovery-comparison-v1/` (precomputed).
